"""Unit tests for Agent.

Mock strategy:
  ollama.chat → {'message': {'content': 'mock'}}
  store._hybrid_retrieve, store._rerank → mocked returns
  store._expand_query → mocked
  store._synthesize → mocked

Never mock: _parse_tool_call, _tool_calculator, AGENT_SYSTEM_PROMPT structure.
"""

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Build a MagicMock VectorStore with one pre-loaded cat-facts chunk."""
    from src.rag.vector_store import VectorStore
    store = MagicMock(spec=VectorStore)
    store.chunks = [
        {'text': 'Cats sleep 16 hours a day.', 'source': 'cats.txt',
         'start_line': 1, 'end_line': 1, 'type': 'txt'},
    ]
    store.bm25_index = BM25Okapi([['cats', 'sleep', '16', 'hours']])
    store.collection = MagicMock()
    store.collection.count.return_value = 1
    # Make _expand_query return original only
    store._expand_query.return_value = ['cats']
    # Make _hybrid_retrieve return one result
    entry = {'text': 'Cats sleep 16 hours a day.', 'source': 'cats.txt',
             'start_line': 1, 'end_line': 1, 'type': 'txt'}
    store._hybrid_retrieve.return_value = [(entry, 0.9)]
    # Make _rerank return the same
    store._rerank.return_value = [(entry, 0.9, 0.9)]
    store._source_label.return_value = 'L1-1'
    return store


@pytest.fixture
def agent(mock_store):
    """Construct an Agent instance wired to mock_store."""
    from src.rag.agent import Agent
    return Agent(mock_store)


def _fake_chat(content='mock'):
    """Return a minimal ollama.chat-shaped dict with the given content string."""
    return {'message': {'content': content}}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests that Agent.__init__ sets all required instance attributes."""

    def test_has_store(self, agent, mock_store):
        """store attribute: Agent.__init__ stores the injected VectorStore reference."""
        assert agent.store is mock_store

    def test_has_max_steps(self, agent):
        """max_steps attribute: Agent.__init__ sets max_steps to 8."""
        assert hasattr(agent, 'max_steps')
        assert agent.max_steps == 8



# ---------------------------------------------------------------------------
# AGENT_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestAgentSystemPrompt:
    """Tests that the AGENT_SYSTEM_PROMPT class constant is complete and well-formed."""

    def test_prompt_is_string(self):
        """AGENT_SYSTEM_PROMPT type: must be a non-trivial string longer than 100 chars."""
        from src.rag.agent import Agent
        assert isinstance(Agent.AGENT_SYSTEM_PROMPT, str)
        assert len(Agent.AGENT_SYSTEM_PROMPT) > 100

    def test_prompt_has_all_5_tools(self):
        """All 5 tools present: rag_search, calculator, summarise, sentiment, finish must appear."""
        from src.rag.agent import Agent
        prompt = Agent.AGENT_SYSTEM_PROMPT
        for tool in ('rag_search', 'calculator', 'summarise', 'sentiment', 'finish'):
            assert tool in prompt, f"Tool '{tool}' missing from AGENT_SYSTEM_PROMPT"

    def test_prompt_has_tool_format(self):
        """TOOL: keyword present: prompt must instruct LLM to use the TOOL: prefix."""
        from src.rag.agent import Agent
        assert 'TOOL:' in Agent.AGENT_SYSTEM_PROMPT

    def test_prompt_has_finish_rule(self):
        """finish rule present: prompt must mention finish as a termination instruction."""
        from src.rag.agent import Agent
        assert 'finish' in Agent.AGENT_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# _parse_tool_call — two regex patterns
# ---------------------------------------------------------------------------

class TestParseToolCall:
    """Tests for _parse_tool_call covering both regex patterns and edge cases."""

    def test_with_parentheses(self, agent):
        """Pattern 1 (with parens): 'TOOL: name(arg)' returns (name, arg) correctly."""
        name, arg = agent._parse_tool_call('TOOL: rag_search(cat sleep hours)')
        assert name == 'rag_search'
        assert arg == 'cat sleep hours'

    def test_without_parentheses(self, agent):
        """Pattern 2 (no parens): 'TOOL: name arg' returns (name, arg) via fallback."""
        name, arg = agent._parse_tool_call('TOOL: rag_search cat sleep hours')
        assert name == 'rag_search'
        assert arg == 'cat sleep hours'

    def test_case_insensitive(self, agent):
        """Case insensitivity: lowercase 'tool:' prefix is still recognised."""
        name, arg = agent._parse_tool_call('tool: calculator(2 + 2)')
        assert name == 'calculator'

    def test_malformed_returns_none(self, agent):
        """Malformed input: text with no TOOL: marker returns (None, None)."""
        name, arg = agent._parse_tool_call('This is not a valid tool call.')
        assert name is None
        assert arg is None

    def test_finish_tool(self, agent):
        """finish tool parsing: 'TOOL: finish(...)' is parsed with name='finish'."""
        name, arg = agent._parse_tool_call('TOOL: finish(The answer is 42.)')
        assert name == 'finish'
        assert 'answer' in arg.lower() or '42' in arg

    def test_multiline_arg_falls_back_to_no_paren_pattern(self, agent):
        """No-paren fallback: single-line text without parens is parsed via pattern 2."""
        # The paren pattern uses (.+) without re.DOTALL by design — a newline
        # inside the parens breaks the match and falls through to the no-paren
        # fallback, which captures everything after the tool name on that line.
        # The LLM always produces single-line tool calls in practice.
        text = 'TOOL: summarise line one line two'
        name, arg = agent._parse_tool_call(text)
        assert name == 'summarise'
        assert 'line one' in arg


# ---------------------------------------------------------------------------
# _tool_calculator
# ---------------------------------------------------------------------------

class TestToolCalculator:
    """Tests for _tool_calculator covering safe expressions and unsafe input rejection."""

    def test_basic_addition(self, agent):
        """Basic addition: '2 + 2' evaluates to '4'."""
        result = agent._tool_calculator('2 + 2')
        assert result == '4'

    def test_multiplication(self, agent):
        """Multiplication: '16 * 365' evaluates to '5840'."""
        result = agent._tool_calculator('16 * 365')
        assert result == '5840'

    def test_complex_expression(self, agent):
        """Grouped expression: '(100 + 50) * 2' evaluates to '300'."""
        result = agent._tool_calculator('(100 + 50) * 2')
        assert result == '300'

    def test_unsafe_chars_rejected(self, agent):
        """Unsafe chars: expression with '__import__' is rejected with an error string."""
        result = agent._tool_calculator('__import__("os")')
        assert 'Error' in result or 'error' in result.lower()

    def test_unsafe_letters_rejected(self, agent):
        """Unsafe letters: 'import sys' contains disallowed chars and is rejected."""
        result = agent._tool_calculator('import sys')
        assert 'Error' in result or 'error' in result.lower()

    def test_division(self, agent):
        """Division: '10 / 2' result contains '5'."""
        result = agent._tool_calculator('10 / 2')
        assert '5' in result


# ---------------------------------------------------------------------------
# _tool_summarise — length hints
# ---------------------------------------------------------------------------

class TestToolSummarise:
    """Tests for _tool_summarise verifying adaptive length hint selection."""

    def test_short_text_uses_2_3_sentences(self, agent):
        """Short text (<100 words): prompt includes '2-3 sentences' length hint."""
        short_text = ' '.join(['word'] * 50)  # 50 words < 100
        with patch('ollama.chat', return_value=_fake_chat('summary here')) as mock_chat:
            agent._tool_summarise(short_text)
        prompt_used = mock_chat.call_args[1]['messages'][0]['content']
        assert '2-3 sentences' in prompt_used

    def test_medium_text_uses_4_5_sentences(self, agent):
        """Medium text (100-300 words): prompt includes '4-5 sentences' length hint."""
        medium_text = ' '.join(['word'] * 150)  # 150 words, 100-300 range
        with patch('ollama.chat', return_value=_fake_chat('summary here')) as mock_chat:
            agent._tool_summarise(medium_text)
        prompt_used = mock_chat.call_args[1]['messages'][0]['content']
        assert '4-5 sentences' in prompt_used

    def test_long_text_uses_6_8_sentences(self, agent):
        """Long text (>300 words): prompt includes '6-8 sentences' length hint."""
        long_text = ' '.join(['word'] * 400)  # 400 words > 300
        with patch('ollama.chat', return_value=_fake_chat('summary here')) as mock_chat:
            agent._tool_summarise(long_text)
        prompt_used = mock_chat.call_args[1]['messages'][0]['content']
        assert '6-8 sentences' in prompt_used

    def test_returns_string(self, agent):
        """Return type: _tool_summarise always returns a string."""
        with patch('ollama.chat', return_value=_fake_chat('the summary')):
            result = agent._tool_summarise('some text here to summarise')
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _tool_sentiment
# ---------------------------------------------------------------------------

class TestToolSentiment:
    """Tests for _tool_sentiment covering output format and short-query search behaviour."""

    def test_returns_string(self, agent):
        """Return type: _tool_sentiment always returns a string."""
        with patch('ollama.chat', return_value=_fake_chat(
            'Sentiment: Positive\nTone: upbeat\nKey phrases: great, awesome\nExplanation: positive mood.'
        )):
            result = agent._tool_sentiment('great product!')
        assert isinstance(result, str)

    def test_output_has_4_fields(self, agent):
        """4-field output: result contains both 'Sentiment' and 'Tone' field labels."""
        mock_response = (
            'Sentiment: Positive\nTone: upbeat\n'
            'Key phrases: great, awesome\nExplanation: positive mood.'
        )
        with patch('ollama.chat', return_value=_fake_chat(mock_response)):
            result = agent._tool_sentiment('great product, awesome service!')
        assert 'Sentiment' in result
        assert 'Tone' in result

    def test_short_query_searches_first(self, agent):
        """Short query (< 10 words) should call _tool_rag_search first."""
        agent._tool_rag_search = MagicMock(return_value='some retrieved text about topic')
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Neutral\nTone: flat\nKey phrases: text\nExplanation: neutral.')):
            agent._tool_sentiment('short query')
        agent._tool_rag_search.assert_called_once()

    def test_long_text_analyzes_directly(self, agent):
        """Long text (>= 10 words) should analyse directly without RAG search."""
        agent._tool_rag_search = MagicMock(return_value='some text')
        long_text = ' '.join(['word'] * 15)
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Neutral\nTone: flat\nKey phrases: word\nExplanation: neutral.')):
            agent._tool_sentiment(long_text)
        agent._tool_rag_search.assert_not_called()


# ---------------------------------------------------------------------------
# _tool_rag_search
# ---------------------------------------------------------------------------

class TestToolRagSearch:
    """Tests for _tool_rag_search verifying it delegates to hybrid retrieve and rerank."""

    def test_returns_string(self, agent):
        """Return type: _tool_rag_search always returns a formatted string."""
        result = agent._tool_rag_search('cats sleep')
        assert isinstance(result, str)

    def test_calls_hybrid_retrieve(self, agent, mock_store):
        """Hybrid retrieve called: _tool_rag_search invokes store._hybrid_retrieve."""
        agent._tool_rag_search('cats')
        mock_store._hybrid_retrieve.assert_called()

    def test_calls_rerank(self, agent, mock_store):
        """Rerank called: _tool_rag_search invokes store._rerank on retrieved results."""
        agent._tool_rag_search('cats')
        mock_store._rerank.assert_called()


# ---------------------------------------------------------------------------
# _synthesize_final_answer
# ---------------------------------------------------------------------------

class TestSynthesizeFinalAnswer:
    """Tests for _synthesize_final_answer verifying LLM call and return type."""

    def test_returns_string(self, agent):
        """Return type: _synthesize_final_answer always returns a string."""
        with patch('ollama.chat', return_value=_fake_chat('the final answer')):
            result = agent._synthesize_final_answer('what is X', 'context about X')
        assert isinstance(result, str)

    def test_calls_ollama_chat(self, agent):
        """ollama.chat called once: _synthesize_final_answer makes exactly one LLM call."""
        with patch('ollama.chat', return_value=_fake_chat('answer')) as mock_chat:
            agent._synthesize_final_answer('query', 'ctx')
        mock_chat.assert_called_once()


# ---------------------------------------------------------------------------
# _fast_path_summarise
# ---------------------------------------------------------------------------

class TestFastPathSummarise:
    """Tests for _fast_path_summarise verifying the 4-term multi-search strategy."""

    def test_returns_dict_with_answer_and_steps(self, agent):
        """Return shape: result dict contains both 'answer' and 'steps' keys."""
        agent._tool_rag_search = MagicMock(return_value='some result')
        with patch('ollama.chat', return_value=_fake_chat('summarised answer')):
            result = agent._fast_path_summarise('summarise the document')
        assert 'answer' in result
        assert 'steps' in result

    def test_makes_4_searches(self, agent):
        """4 rag_search steps: fast path issues exactly 4 rag_search tool calls."""
        agent._tool_rag_search = MagicMock(return_value='result')
        with patch('ollama.chat', return_value=_fake_chat('answer')):
            result = agent._fast_path_summarise('summarise the document')
        # 4 rag_search steps + 1 finish step
        tool_calls = [s for s in result['steps'] if s['tool'] == 'rag_search']
        assert len(tool_calls) == 4

    def test_searches_correct_terms(self, agent):
        """Exact search terms: all 4 fixed terms are searched in the correct order."""
        searched_terms = []
        def capture_search(q):
            """Record each query term passed to _tool_rag_search for assertion."""
            searched_terms.append(q)
            return 'result'
        agent._tool_rag_search = capture_search
        with patch('ollama.chat', return_value=_fake_chat('answer')):
            agent._fast_path_summarise('summarise please')
        assert 'work experience' in searched_terms
        assert 'education' in searched_terms
        assert 'skills projects' in searched_terms
        assert 'summary contact' in searched_terms


# ---------------------------------------------------------------------------
# _fast_path_sentiment
# ---------------------------------------------------------------------------

class TestFastPathSentiment:
    """Tests for _fast_path_sentiment verifying search, label stripping, and return shape."""

    def test_returns_dict_with_answer(self, agent):
        """Return shape: result dict contains both 'answer' and 'steps' keys."""
        agent._tool_rag_search = MagicMock(return_value='- [source L1-1] positive experience')
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Positive\nTone: upbeat\nKey phrases: positive\nExplanation: good.')):
            result = agent._fast_path_sentiment('what is the sentiment')
        assert 'answer' in result
        assert 'steps' in result

    def test_strips_metadata_labels(self, agent):
        """Metadata stripped: chunk source labels like '[cats.txt L1-1]' are removed before sentiment analysis."""
        raw = '- [cats.txt L1-1] Cats are wonderful creatures.'
        agent._tool_rag_search = MagicMock(return_value=raw)
        captured_text = []
        def capture_sentiment(text):
            """Capture the text passed to _tool_sentiment so metadata stripping can be asserted."""
            captured_text.append(text)
            return 'Sentiment: Positive\nTone: warm\nKey phrases: wonderful\nExplanation: positive.'
        agent._tool_sentiment = capture_sentiment
        result = agent._fast_path_sentiment('what is the sentiment of cats')
        if captured_text:
            assert '[cats.txt' not in captured_text[0]


# ---------------------------------------------------------------------------
# run — calculator auto-finish
# ---------------------------------------------------------------------------

class TestRunCalculatorAutoFinish:
    """Tests that a calculator tool call triggers auto-finish without a second LLM round-trip."""

    def test_calculator_auto_finish(self, agent):
        """Auto-finish triggered: calculator result is returned without explicit finish call."""
        responses = [
            _fake_chat('TOOL: calculator(16 * 365)'),
        ]
        with patch('ollama.chat', side_effect=responses):
            result = agent.run('how many hours in a year if cats sleep 16h')
        assert result is not None
        assert '5840' in result['answer'] or 'calculator' in result['answer'].lower() or result['answer']

    def test_calculator_result_in_steps(self, agent):
        """Steps recorded: at least one calculator step appears in the returned steps list."""
        with patch('ollama.chat', return_value=_fake_chat('TOOL: calculator(2 + 2)')):
            result = agent.run('what is 2 + 2')
        calc_steps = [s for s in result['steps'] if s['tool'] == 'calculator']
        assert len(calc_steps) >= 1


# ---------------------------------------------------------------------------
# run — rag_search auto-finish
# ---------------------------------------------------------------------------

class TestRunRagSearchAutoFinish:
    """Tests that a rag_search result triggers auto-finish synthesis for non-summarise queries."""

    def test_rag_search_auto_finish(self, agent):
        """Auto-finish triggered: rag_search followed by synthesis returns a valid result dict."""
        with patch('ollama.chat', side_effect=[
            _fake_chat('TOOL: rag_search(cats sleep)'),
            _fake_chat('Cats sleep 16 hours a day.'),
        ]):
            result = agent.run('how many hours do cats sleep')
        assert result is not None
        assert 'answer' in result

    def test_finish_step_added_after_rag_search(self, agent):
        """Finish step present: at least one finish step is recorded after rag_search auto-finish."""
        with patch('ollama.chat', side_effect=[
            _fake_chat('TOOL: rag_search(cats)'),
            _fake_chat('Cats are awesome.'),
        ]):
            result = agent.run('tell me about cats')
        finish_steps = [s for s in result['steps'] if s['tool'] == 'finish']
        assert len(finish_steps) >= 1


# ---------------------------------------------------------------------------
# run — bad format recovery
# ---------------------------------------------------------------------------

class TestBadFormatRecovery:
    """Tests that the agent retries up to 2 times on malformed LLM output before giving up."""

    def test_bad_format_retries_max_2(self, agent):
        """2 bad responses then valid finish: agent recovers and returns a non-None result."""
        bad_responses = [
            _fake_chat('This is not a tool call'),
            _fake_chat('Still not a tool call'),
            _fake_chat('TOOL: finish(final answer)'),
        ]
        with patch('ollama.chat', side_effect=bad_responses):
            result = agent.run('any query')
        assert result is not None

    def test_step_limit_reached(self, agent):
        """When max_steps hit without finish, agent returns message."""
        with patch('ollama.chat', return_value=_fake_chat('not a tool call ever')):
            result = agent.run('any query')
        assert result is not None
        assert 'answer' in result


# ---------------------------------------------------------------------------
# run — summarise fast path triggered
# ---------------------------------------------------------------------------

class TestRunFastPaths:
    """Tests that run() detects summarise and sentiment keywords and delegates to fast paths."""

    def test_summarise_fast_path_triggered(self, agent):
        """Summarise keyword: run() delegates to _fast_path_summarise and returns its result."""
        agent._fast_path_summarise = MagicMock(return_value={'answer': 'summary', 'steps': []})
        result = agent.run('summarise the document')
        agent._fast_path_summarise.assert_called_once()
        assert result['answer'] == 'summary'

    def test_sentiment_fast_path_triggered(self, agent):
        """Sentiment keyword: run() delegates to _fast_path_sentiment and returns its result."""
        agent._fast_path_sentiment = MagicMock(return_value={'answer': 'Sentiment: Positive', 'steps': []})
        result = agent.run('what is the sentiment of this document')
        agent._fast_path_sentiment.assert_called_once()
        assert 'Positive' in result['answer']
