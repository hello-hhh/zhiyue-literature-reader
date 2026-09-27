"""test_translate_tool.py — Tests for Agent._tool_translate() — HF Space version.

Coverage:
    Unit:
        - Language:text format is parsed correctly
        - No colon → defaults to English target
        - Short content (< 15 words) triggers a rag_search before translating
        - Long content (≥ 15 words) is sent directly without searching
        - LLM result is returned as-is
        - LLM exception returns an error string (no crash)

    Functional:
        - translate dispatched correctly via _dispatch_tool
        - translate tool appears in AGENT_SYSTEM_PROMPT
        - unknown tool error message lists translate

    Regression:
        - AGENT_SYSTEM_PROMPT contains translate tool entry
        - Prompt says "Return ONLY the translation" (no extra text)
        - Format example "TargetLanguage: text" is in the system prompt

    Boundary / negative:
        - Empty string input returns error or empty (no crash)
        - Input with no colon still works (defaults to English)
        - Input with multiple colons uses only the first as separator
        - Very long text (> 200 words) is sent to LLM without truncation by tool
        - LLM returns empty string → tool returns empty string (no crash)
        - rag_search returns empty → falls back to original content

    Parametrized combination:
        - All supported language names work as target
        - Both short (search-first) and long (direct) paths produce string output

Mock strategy:
    HF differences from local version:
        - Agent calls store._llm_chat() instead of ollama.chat().
        - _tool_translate uses self.store._llm_chat() internally.
        - patch.object(store, '_llm_chat') is used instead of patch('ollama.chat').
        - _expand_query is disabled in HF — returns [original_query] always.
        - No Ollama or network calls are made.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

from tests.conftest import sample_chunks

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent():
    """Return an Agent with a minimal mock VectorStore — no HF inference needed.

    The VectorStore is mocked so that _llm_chat can be patched per test.
    Only the methods called inside _tool_translate (and _tool_rag_search
    for short-content routing) need real return values.
    """
    from src.rag.agent import Agent

    # Build a real store mock that satisfies Agent's constructor
    store = MagicMock()
    store._expand_query.return_value   = ['test query']
    store._hybrid_retrieve.return_value = []
    store._rerank.return_value          = []
    store._source_label.return_value    = 'L1-5'
    # _llm_chat is what _tool_translate calls in the HF version
    store._llm_chat.return_value       = 'mock LLM response'

    return Agent(store)


# ─────────────────────────────────────────────────────────────────────────────
# Unit — _tool_translate parsing and dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestToolTranslateParsing:
    """Unit tests for input parsing inside _tool_translate (HF version)."""

    def test_language_text_format_sends_correct_language(self):
        """'Spanish: hello world...' must send a prompt that names Spanish."""
        agent = _make_agent()
        prompts_sent = []

        def _fake_llm_chat(messages, **kw):
            # Capture the prompt content for assertion
            prompts_sent.append(messages[0]['content'])
            return 'Hola mundo'

        agent.store._llm_chat.side_effect = _fake_llm_chat
        agent._tool_translate(
            'Spanish: hello world this is a long enough text here yes indeed'
        )

        assert prompts_sent, "store._llm_chat was not called"
        assert 'Spanish' in prompts_sent[0], \
            f"Expected 'Spanish' in prompt, got: {prompts_sent[0][:200]}"

    def test_no_colon_defaults_to_english(self):
        """Input with no colon should still work — defaults to English."""
        agent = _make_agent()
        prompts_sent = []

        def _fake_llm_chat(messages, **kw):
            prompts_sent.append(messages[0]['content'])
            return 'some text'

        agent.store._llm_chat.side_effect = _fake_llm_chat
        # Provide enough words so rag_search is not triggered
        agent._tool_translate(
            'this is a sentence without a language prefix that is long enough to skip search'
        )

        assert 'English' in prompts_sent[0]

    def test_multiple_colons_uses_only_first_as_separator(self):
        """'French: text with: colons inside' → language=French, text=text with: colons inside."""
        agent = _make_agent()
        prompts_sent = []

        def _fake_llm_chat(messages, **kw):
            prompts_sent.append(messages[0]['content'])
            return 'translation'

        agent.store._llm_chat.side_effect = _fake_llm_chat
        agent._tool_translate(
            'French: text with: colons inside and enough words to skip search yes'
        )

        # Language should be French
        assert 'French' in prompts_sent[0]
        # Text after the first colon should contain the rest
        assert 'colons inside' in prompts_sent[0]

    def test_returns_llm_output_directly(self):
        """The LLM response text must be returned unchanged."""
        agent = _make_agent()
        expected = 'Bonjour le monde'

        agent.store._llm_chat.return_value = expected
        result = agent._tool_translate(
            'French: Hello world — this is long enough text for direct translation path yes'
        )

        assert result == expected

    def test_llm_exception_returns_error_string(self):
        """If store._llm_chat raises, _tool_translate must return an error string, not crash."""
        agent = _make_agent()

        agent.store._llm_chat.side_effect = RuntimeError('model not found')
        result = agent._tool_translate(
            'German: some text that is long enough to go direct without searching here'
        )

        assert 'error' in result.lower() or 'Error' in result, \
            f"Expected error message, got: {result}"

    def test_llm_returns_empty_string_no_crash(self):
        """LLM returning an empty string must not crash the tool."""
        agent = _make_agent()

        agent.store._llm_chat.return_value = ''
        result = agent._tool_translate(
            'Italian: some text long enough to go direct path without rag search yes'
        )

        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Unit — short vs long content routing
# ─────────────────────────────────────────────────────────────────────────────

class TestToolTranslateRouting:
    """Tests for the short-query (search first) vs long-text (direct) routing logic."""

    def test_short_content_triggers_rag_search(self):
        """Content under 15 words must call _tool_rag_search before translating."""
        agent = _make_agent()
        search_calls = []

        def _fake_search(query):
            search_calls.append(query)
            return 'retrieved document text about the topic here it is'

        with patch.object(agent, '_tool_rag_search', side_effect=_fake_search):
            agent.store._llm_chat.return_value = 'translated'
            agent._tool_translate('Spanish: machine learning')  # 2 words — short

        assert search_calls, "Expected _tool_rag_search to be called for short content"

    def test_long_content_skips_rag_search(self):
        """Content of 15 or more words must NOT call _tool_rag_search."""
        agent = _make_agent()
        search_calls = []

        def _fake_search(query):
            search_calls.append(query)
            return 'retrieved text'

        long_text = 'this is a long passage that is definitely more than fifteen words total right here yes'
        with patch.object(agent, '_tool_rag_search', side_effect=_fake_search):
            agent.store._llm_chat.return_value = 'translated'
            agent._tool_translate(f'French: {long_text}')

        assert not search_calls, "Expected _tool_rag_search NOT to be called for long content"

    def test_short_content_uses_retrieved_text_for_translation(self):
        """When rag_search returns content, that content (not the query) is translated."""
        agent = _make_agent()
        retrieved_text = 'The cat sleeps 16 hours per day according to research.'
        prompts_sent   = []

        def _fake_search(query):
            return retrieved_text

        def _fake_llm_chat(messages, **kw):
            prompts_sent.append(messages[0]['content'])
            return 'translated'

        with patch.object(agent, '_tool_rag_search', side_effect=_fake_search):
            agent.store._llm_chat.side_effect = _fake_llm_chat
            agent._tool_translate('Spanish: cat facts')

        assert prompts_sent, "store._llm_chat not called"
        assert retrieved_text in prompts_sent[0], \
            "Retrieved text should be in the translation prompt"

    def test_short_content_falls_back_when_search_empty(self):
        """When rag_search returns empty, the original content is used for translation."""
        agent = _make_agent()
        prompts_sent = []

        def _fake_search(query):
            return ''   # empty search result

        def _fake_llm_chat(messages, **kw):
            prompts_sent.append(messages[0]['content'])
            return 'translated'

        with patch.object(agent, '_tool_rag_search', side_effect=_fake_search):
            agent.store._llm_chat.side_effect = _fake_llm_chat
            agent._tool_translate('German: cat')   # short — triggers search, but search empty

        assert prompts_sent
        # The original word 'cat' should appear in the prompt since search was empty
        assert 'cat' in prompts_sent[0]


# ─────────────────────────────────────────────────────────────────────────────
# Functional — dispatch and agent loop
# ─────────────────────────────────────────────────────────────────────────────

class TestToolTranslateFunctional:
    """Functional tests — translate called via _dispatch_tool."""

    def test_dispatch_tool_routes_to_translate(self):
        """_dispatch_tool('translate', ...) must call _tool_translate."""
        agent = _make_agent()
        translate_calls = []

        def _fake_translate(arg):
            translate_calls.append(arg)
            return 'translated result'

        with patch.object(agent, '_tool_translate', side_effect=_fake_translate):
            result = agent._dispatch_tool('translate', 'Spanish: hello world')

        assert translate_calls, "_tool_translate was not called"
        assert result == 'translated result'

    def test_dispatch_unknown_tool_message_mentions_translate(self):
        """The unknown-tool error message must list translate as an available tool."""
        agent  = _make_agent()
        result = agent._dispatch_tool('nonexistent_tool', 'some arg')
        assert 'translate' in result.lower(), \
            f"Expected 'translate' in unknown-tool message, got: {result}"

    def test_translate_in_system_prompt(self):
        """AGENT_SYSTEM_PROMPT must mention the translate tool."""
        from src.rag.agent import Agent
        assert 'translate' in Agent.AGENT_SYSTEM_PROMPT.lower()

    def test_system_prompt_shows_translate_format_example(self):
        """The system prompt must include a TOOL: translate(...) example."""
        from src.rag.agent import Agent
        assert 'TOOL: translate(' in Agent.AGENT_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Regression — prompt text is locked down
# ─────────────────────────────────────────────────────────────────────────────

class TestToolTranslateRegression:
    """Regression tests — the translation prompt must contain the key instruction."""

    def test_prompt_says_return_only_translation(self):
        """The LLM prompt must say 'Return ONLY the translation' to suppress extra text."""
        agent = _make_agent()
        prompts_sent = []

        def _fake_llm_chat(messages, **kw):
            prompts_sent.append(messages[0]['content'])
            return 'translated'

        agent.store._llm_chat.side_effect = _fake_llm_chat
        agent._tool_translate(
            'French: this is long enough text to go direct without search path yes'
        )

        assert prompts_sent
        assert 'ONLY' in prompts_sent[0] or 'only' in prompts_sent[0], \
            "Prompt must instruct LLM to return ONLY the translation"

    def test_system_prompt_has_6_tools(self):
        """AGENT_SYSTEM_PROMPT must list exactly 6 tools (including translate)."""
        from src.rag.agent import Agent
        prompt = Agent.AGENT_SYSTEM_PROMPT
        # Count numbered tool entries: "1.", "2.", ..., "6."
        tool_entries = [line for line in prompt.splitlines()
                        if line.strip() and line.strip()[0].isdigit()
                        and '. ' in line]
        assert len(tool_entries) == 6, \
            f"Expected 6 numbered tools in prompt, found {len(tool_entries)}: {tool_entries}"

    def test_system_prompt_translate_rule_mentions_format(self):
        """The translate rule in AGENT_SYSTEM_PROMPT must show the TargetLanguage: format."""
        from src.rag.agent import Agent
        assert 'TargetLanguage:' in Agent.AGENT_SYSTEM_PROMPT or \
               'TargetLanguage' in Agent.AGENT_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / negative
# ─────────────────────────────────────────────────────────────────────────────

class TestToolTranslateBoundary:
    """Boundary and negative tests for _tool_translate (HF version)."""

    def test_empty_string_does_not_crash(self):
        """Empty string input must return a string, not raise an exception."""
        agent = _make_agent()
        with patch.object(agent, '_tool_rag_search', return_value=''):
            agent.store._llm_chat.return_value = ''
            result = agent._tool_translate('')
        assert isinstance(result, str)

    def test_only_language_prefix_no_text(self):
        """'Spanish:' with no text after the colon must not crash."""
        agent = _make_agent()
        with patch.object(agent, '_tool_rag_search', return_value='retrieved text'):
            agent.store._llm_chat.return_value = 'translated'
            result = agent._tool_translate('Spanish:')
        assert isinstance(result, str)

    def test_whitespace_only_content_does_not_crash(self):
        """'French:   ' (spaces only after colon) must not crash."""
        agent = _make_agent()
        with patch.object(agent, '_tool_rag_search', return_value='retrieved text'):
            agent.store._llm_chat.return_value = 'translated'
            result = agent._tool_translate('French:   ')
        assert isinstance(result, str)

    def test_very_long_text_sent_to_llm_without_truncation(self):
        """_tool_translate must not silently truncate long input before the LLM call."""
        agent = _make_agent()
        # Build 50-word text
        long_text = ' '.join(['word'] * 50)
        prompts_sent = []

        def _fake_llm_chat(messages, **kw):
            prompts_sent.append(messages[0]['content'])
            return 'translated'

        agent.store._llm_chat.side_effect = _fake_llm_chat
        agent._tool_translate(f'German: {long_text}')

        # All 50 words should appear in the prompt
        assert long_text in prompts_sent[0], \
            "Long text was truncated before being sent to the LLM"


# ─────────────────────────────────────────────────────────────────────────────
# Parametrized — all common target languages work
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("language", [
    "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Chinese", "Korean", "Arabic", "Hindi",
    "Russian", "Dutch", "Polish", "Turkish", "Swedish",
])
class TestToolTranslateLanguages:
    """Parametrized test — every common target language must produce a string result."""

    def test_translate_to_language_returns_string(self, language):
        """_tool_translate to any language must return a non-empty string."""
        agent = _make_agent()
        long_text = 'this is a long enough sentence to skip the rag search path entirely yes'

        agent.store._llm_chat.return_value = f'Translated to {language}'
        result = agent._tool_translate(f'{language}: {long_text}')

        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Parametrized — short vs long content routing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("content,expect_search", [
    ("machine learning",                                                        True),
    ("cat facts",                                                               True),
    ("short",                                                                   True),
    ("this is a sentence that has at least fifteen words so it skips search yes", False),
    ("this is a long passage with many more words that will definitely go direct path", False),
])
class TestToolTranslateShortLongRouting:
    """Parametrized test — short inputs search first, long inputs go direct."""

    def test_routing_based_on_content_length(self, content, expect_search):
        """Verify rag_search is called iff content is under 15 words."""
        agent = _make_agent()
        search_called = []

        def _fake_search(q):
            search_called.append(q)
            return 'retrieved text about the topic in question'

        with patch.object(agent, '_tool_rag_search', side_effect=_fake_search):
            agent.store._llm_chat.return_value = 'translated'
            agent._tool_translate(f'Spanish: {content}')

        if expect_search:
            assert search_called, f"Expected search for short content: '{content}'"
        else:
            assert not search_called, f"Expected no search for long content: '{content}'"
