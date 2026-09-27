"""agent.py — Agent class.

Owns the ReAct loop and all 5 tools as private methods.
"""

import re

import ollama

from src.rag.config import LANGUAGE_MODEL
from src.rag.vector_store import VectorStore


class Agent:
    """Owns the ReAct reasoning loop and all 5 tools as private methods.

    Tools are private methods — not separate classes — because they have no state
    of their own and exist only as implementation details of the agent loop.

    State:
        store:             VectorStore reference for rag_search and sentiment tools
        messages:          ReAct message history (system + user + tool exchanges)
        collected_context: accumulated search results used in final synthesis
        max_steps:         step limit to prevent infinite loops (default 8)

    Public API:
        run(user_query, streamlit_mode) — execute the ReAct loop and return result dict
    """

    AGENT_SYSTEM_PROMPT = """You are an AI agent. You must ONLY respond with tool calls — no explanations, no extra text.

Available tools:
1. rag_search - search the knowledge base for information
2. calculator - evaluate a math expression
3. summarise  - summarise a piece of text
4. sentiment  - analyse the sentiment/tone of a passage or topic from the documents
5. translate  - translate text to any language. Format: "TargetLanguage: text to translate"
6. finish     - return the final answer to the user

You MUST respond in EXACTLY this format with NO other text before or after:
TOOL: tool_name(your argument here)

Examples:
TOOL: rag_search(NLP experience)
TOOL: calculator(16 * 365)
TOOL: summarise(cats sleep a lot and are nocturnal hunters...)
TOOL: sentiment(customer reviews)
TOOL: translate(Spanish: The candidate has ten years of experience in machine learning.)
TOOL: translate(French: What are the main findings of this report?)
TOOL: finish(Yes, the candidate has NLP experience including POS tagging and language modeling.)

Rules:
- Never write anything except a single TOOL: line
- Always end with TOOL: finish(your final answer)
- Use rag_search first to find information before answering
- Do not explain yourself or add any commentary
- The finish argument must be a clean, direct answer in plain English — NEVER paste raw bullet points or document chunks into finish
- rag_search arguments must be SHORT, SIMPLE keyword phrases — NEVER use boolean operators like AND, OR, quotes, or complex syntax
- For simple math questions, call calculator once then finish
- For simple factual questions, call rag_search once then finish
- For sentiment questions (e.g. "what is the tone", "is this positive", "sentiment of"), call sentiment with a keyword or passage then finish
- For translation questions (e.g. "translate to", "in Spanish", "en français"):
  * If translating document content: call rag_search first, then translate with the result
  * If translating a given passage: call translate directly
  * Format: translate(TargetLanguage: text)
- For summarisation or comprehensive tasks (e.g. "summarise", "tell me about", "what is in"):
  * Make multiple SEPARATE rag_search calls, one per topic
  * For a resume: search "work experience", then "education", then "skills", then "projects" as separate calls
  * Collect all results, then call finish with a complete summary
"""

    def __init__(self, store: VectorStore):
        """Bind VectorStore and initialise per-run state (reset on each run() call).

        Args:
            store: an initialised VectorStore with build_or_load() already called.
        """
        self.store             = store
        self.messages          = []
        self.collected_context = []
        self.max_steps         = 8

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, user_query: str, streamlit_mode: bool = False) -> dict:
        """Run the ReAct agent loop for a user query.

        Args:
            user_query:     The user's question or task.
            streamlit_mode: If True, suppress terminal prints and return structured result dict.

        Returns:
            dict with keys: answer (str), steps (list of step dicts with tool/arg/result).
        """
        self.messages          = [
            {'role': 'system', 'content': self.AGENT_SYSTEM_PROMPT},
            {'role': 'user',   'content': user_query},
        ]
        self.collected_context = []

        steps            = []
        answer           = None
        bad_format_count = 0

        # Detect summarisation queries
        _q_lower = user_query.lower()
        is_summarise = (
            any(s in _q_lower for s in
                ['summarise', 'summarize', 'summerise', 'summerize', 'summary',
                 'overview', 'tell me about', 'describe', 'what is in'])
            or _q_lower.startswith('summ')
        )

        # Fast path: for summarise queries do multi-search directly — no agent loop needed
        if is_summarise:
            return self._fast_path_summarise(user_query, streamlit_mode)

        # Fast path: for sentiment queries — search then analyse directly
        is_sentiment = any(s in _q_lower for s in
                           ['sentiment', 'tone', 'feeling', 'positive', 'negative', 'neutral',
                            'emotion', 'attitude', 'mood'])
        if is_sentiment:
            return self._fast_path_sentiment(user_query, streamlit_mode)

        for step in range(self.max_steps):
            resp     = ollama.chat(model=LANGUAGE_MODEL, messages=self.messages)
            raw_text = resp['message']['content'].strip()
            tool_name, tool_arg = self._parse_tool_call(raw_text)

            if not tool_name:
                bad_format_count += 1
                if bad_format_count <= 2:
                    if not streamlit_mode:
                        print(f"\n  [Agent] Bad format (attempt {bad_format_count}/2), retrying...")
                    self.messages.append({'role': 'assistant', 'content': raw_text})
                    self.messages.append({'role': 'user', 'content':
                        'Wrong format. You must respond with ONLY this format — nothing else:\n'
                        'TOOL: tool_name(argument)\n'
                        'Example: TOOL: rag_search(cat sleep hours)'})
                    continue
                else:
                    answer = raw_text
                    steps.append({'step': step+1, 'tool': 'none', 'arg': '', 'result': raw_text})
                    break

            bad_format_count = 0

            if tool_name == 'finish':
                # Synthesize final answer from ALL collected context — ignore model's raw arg
                if self.collected_context:
                    all_context = '\n'.join(self.collected_context)
                    answer = self._synthesize_final_answer(user_query, all_context)
                else:
                    answer = tool_arg
                steps.append({'step': step+1, 'tool': 'finish', 'arg': answer, 'result': answer})
                break

            result = self._dispatch_tool(tool_name, tool_arg)

            steps.append({'step': step+1, 'tool': tool_name, 'arg': tool_arg, 'result': result})

            if not streamlit_mode:
                print(f"\n  [Agent Step {step+1}] {tool_name}({tool_arg[:60]}...)"
                      if len(tool_arg) > 60 else f"\n  [Agent Step {step+1}] {tool_name}({tool_arg})")
                print(f"  → {result[:120]}..." if len(result) > 120 else f"  → {result}")

            if tool_name == 'calculator' and not result.startswith('Error'):
                answer = f"{tool_arg} = {result}"
                steps.append({'step': step+2, 'tool': 'finish', 'arg': answer, 'result': answer})
                if not streamlit_mode:
                    print(f"\n  [Agent Step {step+2}] finish({answer})")
                break

            # For simple (non-summarise) queries: auto-finish after first rag_search
            if tool_name == 'rag_search' and not is_summarise:
                answer = self._synthesize_final_answer(user_query, result)
                steps.append({'step': step+2, 'tool': 'finish', 'arg': answer, 'result': answer})
                if not streamlit_mode:
                    print(f"\n  [Agent Step {step+2}] finish({answer[:120]}..."
                          if len(answer) > 120 else f"\n  [Agent Step {step+2}] finish({answer})")
                break

            self.messages.append({'role': 'assistant', 'content': raw_text})
            self.messages.append({'role': 'user', 'content':
                f"Tool result: {result}\n\n"
                f"Original task: {user_query}\n\n"
                f"If you now have enough information to answer the original task, call:\n"
                f"TOOL: finish(your answer)\n\n"
                f"Otherwise call the next tool. Respond ONLY with a single TOOL: line."})

        if answer is None:
            answer = "Agent reached max steps without a final answer."

        return {'answer': answer, 'steps': steps}

    # ── Private — loop ───────────────────────────────────────────────────────

    def _parse_tool_call(self, response_text: str) -> tuple:
        """Extract the tool name and argument from a raw LLM response string.

        Tries two patterns in order:
          1. With parentheses:    TOOL: tool_name(argument)
          2. Without parentheses: TOOL: tool_name argument

        Pattern 1 is preferred — it preserves nested parentheses in expressions
        like "calculator(7+(9+8)-2*6)". Pattern 2 is a fallback for models that
        omit the parentheses.

        Args:
            response_text: The raw string from the language model.

        Returns:
            Tuple of (tool_name: str, tool_arg: str), or (None, None) if neither
            pattern matches — which triggers a bad-format retry in the agent loop.
        """
        # Greedy (.+) + no DOTALL: captures up to the LAST ')' on the line,
        # so nested parens in expressions like "7+(9+8)-2*6" are preserved.
        match = re.search(r'(?i)TOOL:\s*(\w+)\s*\(\s*(.+)\s*\)', response_text)
        if match:
            return match.group(1).strip().lower(), match.group(2).strip()
        match = re.search(r'(?i)TOOL:\s*(\w+)\s+(.+)', response_text)
        if match:
            return match.group(1).strip().lower(), match.group(2).strip()
        return None, None

    def _dispatch_tool(self, tool_name: str, tool_arg: str) -> str:
        """Route a parsed tool call to the correct private tool method.

        rag_search and sentiment results are also appended to collected_context
        so the final synthesis at finish-time has the full accumulated evidence,
        not just the last result.

        Args:
            tool_name: Lowercase tool name (e.g. 'rag_search', 'calculator').
            tool_arg:  The argument string extracted by _parse_tool_call.

        Returns:
            String result from the tool, or an error message for unknown tools.
        """
        if tool_name == 'rag_search':
            result = self._tool_rag_search(tool_arg)
            self.collected_context.append(f"[Search: {tool_arg}]\n{result}")
        elif tool_name == 'calculator':
            result = self._tool_calculator(tool_arg)
        elif tool_name == 'summarise':
            result = self._tool_summarise(tool_arg)
        elif tool_name == 'sentiment':
            result = self._tool_sentiment(tool_arg)
            self.collected_context.append(f"[Sentiment analysis: {tool_arg}]\n{result}")
        elif tool_name == 'translate':
            result = self._tool_translate(tool_arg)
        else:
            result = (f"Unknown tool '{tool_name}'. "
                      f"Available: rag_search, calculator, summarise, sentiment, translate, finish")
        return result

    def _synthesize_final_answer(self, query: str, context: str) -> str:
        """Takes the raw retrieved context and asks the LLM to produce a clean answer."""
        prompt = (
            "You are a helpful assistant. Answer the question below using ONLY the "
            "provided context. Be concise and direct. Do not repeat the context — "
            "just answer the question. Cite the source filename at the end.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        try:
            resp = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0}
            )
            return resp['message']['content'].strip()
        except Exception:
            return context

    def _fast_path_summarise(self, query: str, streamlit_mode: bool = False) -> dict:
        """For summarise queries: 4-term multi-search → synthesize. No agent loop."""
        search_terms = ['work experience', 'education', 'skills projects', 'summary contact']
        fast_steps   = []
        collected    = []
        for i, term in enumerate(search_terms):
            r = self._tool_rag_search(term)
            collected.append(f"[Search: {term}]\n{r}")
            fast_steps.append({'step': i+1, 'tool': 'rag_search', 'arg': term, 'result': r})
            if not streamlit_mode:
                print(f"\n  [Agent Step {i+1}] rag_search({term})")
                print(f"  → {r[:120]}..." if len(r) > 120 else f"  → {r}")
        all_context = '\n\n'.join(collected)
        answer = self._synthesize_final_answer(query, all_context)
        fast_steps.append({'step': len(search_terms)+1, 'tool': 'finish',
                           'arg': answer, 'result': answer})
        return {'answer': answer, 'steps': fast_steps}

    def _fast_path_sentiment(self, query: str, streamlit_mode: bool = False) -> dict:
        """For sentiment queries: search then analyse directly."""
        _q_lower = query.lower()
        # Strip sentiment-related words to get the search subject
        search_query = re.sub(
            r'\b(what is the|what\'s the|analyse|analyze|analysis|check|tell me the|'
            r'of the|of|sentiment|tone|feeling|emotion|attitude|mood|the|is|a|an)\b',
            '', _q_lower, flags=re.IGNORECASE
        ).strip() or query
        if not streamlit_mode:
            print(f"\n  [Agent Step 1] rag_search({search_query})")
        raw = self._tool_rag_search(search_query)
        if not streamlit_mode:
            print(f"  → {raw[:120]}..." if len(raw) > 120 else f"  → {raw}")
            print(f"\n  [Agent Step 2] sentiment({search_query})")
        # Strip chunk metadata labels (e.g. "- [source L1-3]") for cleaner sentiment input
        clean_text = re.sub(r'-\s*\[[^\]]+\]\s*', '', raw).strip()
        sentiment_result = self._tool_sentiment(clean_text or raw)
        if not streamlit_mode:
            print(f"  → {sentiment_result[:120]}..."
                  if len(sentiment_result) > 120 else f"  → {sentiment_result}")
        sentiment_steps = [
            {'step': 1, 'tool': 'rag_search',  'arg': search_query,     'result': raw},
            {'step': 2, 'tool': 'sentiment',    'arg': search_query,     'result': sentiment_result},
            {'step': 3, 'tool': 'finish',       'arg': sentiment_result, 'result': sentiment_result},
        ]
        return {'answer': sentiment_result, 'steps': sentiment_steps}

    # ── Private — tools ──────────────────────────────────────────────────────

    def _tool_rag_search(self, query: str) -> str:
        """Returns retrieved chunks with source labels for grounded synthesis."""
        queries  = self.store._expand_query(query)
        retrieved = self.store._hybrid_retrieve(queries, top_n=5)
        reranked  = self.store._rerank(query, retrieved, top_n=3)
        lines = []
        for e, sim, _ in reranked:
            label = self.store._source_label(e)
            lines.append(f"- [{e['source']} {label}] {e['text']}")
        return '\n'.join(lines)

    def _tool_calculator(self, expression: str) -> str:
        """Safely evaluate a mathematical expression and return the result as a string.

        Only digits, arithmetic operators (+, -, *, /), parentheses, dots, and spaces
        are allowed. Any other character is rejected to prevent code injection via eval().
        Percentage expressions are normalised before evaluation:
          "15% of 85000" → "(15/100*85000)"
          "15%"          → "(15/100)"

        Args:
            expression: A mathematical expression string (e.g. "15% of 85000").

        Returns:
            The numeric result as a string, or an "Error: ..." message if the
            expression is unsafe, malformed, or causes a runtime error.
        """
        try:
            # Normalise percentage expressions before safety check
            # "15% of 85000" → "(15/100*85000)"
            # "15%" alone → "(15/100)"
            expr = re.sub(
                r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)',
                r'(\1/100*\2)',
                expression,
                flags=re.IGNORECASE,
            )
            expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
            allowed = set('0123456789+-*/(). ')
            if not all(c in allowed for c in expr):
                return "Error: unsafe expression"
            return str(eval(expr))
        except Exception as e:
            return f"Error: {e}"

    def _tool_summarise(self, text: str) -> str:
        """Summarise a passage using an adaptive length hint based on word count.

        Shorter passages get a tighter summary (2-3 sentences) to avoid padding;
        longer passages get a fuller treatment (6-8 sentences) so key points
        are not dropped.

        Length thresholds:
            < 100 words  → "2-3 sentences"
            100-299 words → "4-5 sentences"
            300+ words   → "6-8 sentences covering all key points"

        Args:
            text: The passage to summarise.

        Returns:
            A summary string produced by the language model.
        """
        word_count = len(text.split())
        if word_count < 100:
            length_hint = "2-3 sentences"
        elif word_count < 300:
            length_hint = "4-5 sentences"
        else:
            length_hint = "6-8 sentences covering all key points"
        resp = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user',
                       'content': f"Summarise this in {length_hint}:\n{text}"}]
        )
        return resp['message']['content'].strip()

    def _tool_sentiment(self, text_or_query: str) -> str:
        """Analyse the sentiment and tone of a text passage.

        If the input is a short keyword or query (fewer than 10 words), the
        knowledge base is searched first and the retrieved content is analysed.
        This prevents the LLM from hallucinating sentiment for a topic it hasn't
        retrieved context for.

        Output format (always 4 fields):
            Sentiment: <Positive / Negative / Neutral / Mixed>
            Tone: <one short phrase>
            Key phrases: <2-4 phrases>
            Explanation: <1-2 sentences>

        Args:
            text_or_query: Either a full passage to analyse, or a short search query.

        Returns:
            Structured sentiment analysis string with the 4 fields above.
        """
        if len(text_or_query.split()) < 10:
            retrieved = self._tool_rag_search(text_or_query)
            text_to_analyse = retrieved if retrieved.strip() else text_or_query
        else:
            text_to_analyse = text_or_query

        prompt = (
            "Analyse the sentiment and tone of the following text.\n\n"
            "Respond in this exact format:\n"
            "Sentiment: <Positive / Negative / Neutral / Mixed>\n"
            "Tone: <one short phrase describing the tone, e.g. 'professional and confident'>\n"
            "Key phrases: <2-4 phrases from the text that drove this assessment>\n"
            "Explanation: <1-2 sentences explaining the sentiment>\n\n"
            f"Text:\n{text_to_analyse}"
        )
        try:
            resp = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0}
            )
            return resp['message']['content'].strip()
        except Exception as e:
            return f"Sentiment analysis error: {e}"

    def _tool_translate(self, language_and_text: str) -> str:
        """Translate text to any target language.

        Accepts input in one of two forms:
        - "TargetLanguage: text to translate"  → translate the text directly
        - "TargetLanguage: short keyword query" → search the knowledge base first,
          then translate the retrieved content

        If no colon separator is found, the whole argument is treated as the text
        to translate and the language defaults to English.

        Args:
            language_and_text: A string in the format "Language: text",
                               e.g. "Spanish: The candidate has ten years of experience."

        Returns:
            The translated text as a plain string, with a note of the source language
            detected by the model.
        """
        # Split on the first colon to get target language and the content to translate
        if ':' in language_and_text:
            target_language, content = language_and_text.split(':', 1)
            target_language = target_language.strip()
            content         = content.strip()
        else:
            # No language prefix — default to English
            target_language = 'English'
            content         = language_and_text.strip()

        # If the content looks like a short search query (under 15 words),
        # retrieve relevant document text first so we translate real content
        if len(content.split()) < 15:
            retrieved = self._tool_rag_search(content)
            # Only use retrieved text if the search returned something useful
            if retrieved.strip():
                content = retrieved

        prompt = (
            f"Translate the following text to {target_language}.\n"
            f"Return ONLY the translation — no explanation, no original text, "
            f"no notes. Just the translated text.\n\n"
            f"Text to translate:\n{content}"
        )
        try:
            resp = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0},
            )
            return resp['message']['content'].strip()
        except Exception as error:
            return f"Translation error: {error}"
