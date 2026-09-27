"""test_agent_tools.py — Agent tool-level tests for the HF Space version.

Covers:
  - TestParseToolCall:   Both regex patterns (with and without parentheses)
  - TestCalculatorTool:  Safe eval with allowed-chars whitelist
  - TestSummariseTool:   Adaptive length hints based on word count
  - TestSentimentTool:   Short queries search first; long text analysed directly

ReAct loop, fast paths, dispatch routing, system prompt integrity, and
app handler tests are in test_agent.py.

Mock strategy:
  - conftest.py patches _get_st_model globally (fake 384-dim embeddings).
  - conftest.py patches _llm_call globally (safe mock string).
  - Agent LLM calls use store._llm_chat — patched via patch.object(store, '_llm_chat').
  - Never mock: _parse_tool_call, _tool_calculator, AGENT_SYSTEM_PROMPT structure.

HF differences from local:
  - Agent calls store._llm_chat() instead of ollama.chat().
  - _expand_query is disabled — returns [original_query] always.
  - No ollama.chat to mock in agent tests — use patch.object(store, '_llm_chat').
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

from tests.conftest import make_store_with_chunks, sample_chunks

# ═══════════════════════════════════════════════════════════════════════════════
# Shared agent factory
# ═══════════════════════════════════════════════════════════════════════════════

def _make_agent(n_chunks: int = 3):
    """Build a minimal Agent backed by an in-memory VectorStore.

    Args:
        n_chunks: Number of sample cat-facts chunks to index.

    Returns:
        Agent instance with a fully initialised VectorStore.
    """
    chunks = sample_chunks(n_chunks)
    from src.rag.vector_store import VectorStore
    store = VectorStore()
    store.build_or_load(chunks)
    from src.rag.agent import Agent
    return Agent(store)


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_tool_call — two regex patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseToolCall:
    """Tests for Agent._parse_tool_call — both regex patterns (with and without parentheses)."""

    def setup_method(self):
        """Build a minimal Agent backed by a small in-memory VectorStore."""
        self.agent = _make_agent(3)

    def test_parse_with_parentheses(self):
        """Pattern 1: TOOL: name(arg) → (name, arg)."""
        name, arg = self.agent._parse_tool_call("TOOL: rag_search(cat sleep hours)")
        assert name == 'rag_search'
        assert arg == 'cat sleep hours'

    def test_parse_without_parentheses(self):
        """Pattern 2 fallback: TOOL: name arg → (name, arg)."""
        name, arg = self.agent._parse_tool_call("TOOL: rag_search cat nap duration")
        assert name == 'rag_search'
        assert 'cat' in arg

    def test_parse_case_insensitive(self):
        """TOOL keyword and tool name are case-insensitive."""
        name, arg = self.agent._parse_tool_call("tool: Calculator(2 + 2)")
        assert name == 'calculator'

    def test_parse_finish(self):
        """finish tool with a full-sentence argument is parsed correctly."""
        name, arg = self.agent._parse_tool_call("TOOL: finish(The answer is 42.)")
        assert name == 'finish'
        assert arg == 'The answer is 42.'

    def test_parse_malformed_returns_none(self):
        """Text without TOOL: prefix returns (None, None)."""
        name, arg = self.agent._parse_tool_call("This is not a tool call")
        assert name is None
        assert arg is None

    def test_parse_summarise_no_parens(self):
        """No-paren fallback correctly parses a multi-word summarise argument."""
        name, arg = self.agent._parse_tool_call("TOOL: summarise line one line two")
        assert name == 'summarise'
        assert 'line one' in arg


# ═══════════════════════════════════════════════════════════════════════════════
# _tool_calculator — safe eval with allowed-chars whitelist
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculatorTool:
    """Tests for Agent._tool_calculator — safe eval with allowed-chars whitelist."""

    def setup_method(self):
        """Build a minimal Agent to exercise the calculator tool."""
        self.agent = _make_agent(2)

    def test_basic_addition(self):
        """Simple integer addition returns string result."""
        assert self.agent._tool_calculator("2 + 2") == "4"

    def test_complex_expression(self):
        """Nested parentheses and mixed operators evaluate correctly."""
        result = self.agent._tool_calculator("(10 + 5) * 2 / 3")
        assert float(result) == pytest.approx(10.0)

    def test_unsafe_chars_rejected(self):
        """Expressions with disallowed chars (letters, quotes) return an error string."""
        result = self.agent._tool_calculator("__import__('os').system('rm -rf')")
        assert "Error" in result

    def test_unsafe_letters_rejected(self):
        """exec() — contains letters → rejected by whitelist before eval."""
        result = self.agent._tool_calculator("exec('bad')")
        assert "Error" in result

    def test_division(self):
        """Float division returns decimal string result."""
        assert self.agent._tool_calculator("10 / 4") == "2.5"

    def test_float_expression(self):
        """Float literals in expression are handled correctly."""
        result = float(self.agent._tool_calculator("3.14 * 2"))
        assert result == pytest.approx(6.28)

    def test_multiplication(self):
        """Integer multiplication: '16 * 365' evaluates to '5840'."""
        assert self.agent._tool_calculator("16 * 365") == "5840"


# ═══════════════════════════════════════════════════════════════════════════════
# _tool_summarise — adaptive length hints
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummariseTool:
    """Tests for Agent._tool_summarise — adaptive length hints based on word count."""

    def setup_method(self):
        """Build a minimal Agent to exercise the summarise tool."""
        self.agent = _make_agent(2)

    def test_short_text_hint(self):
        """< 100 words → 2-3 sentences hint."""
        captured = {}

        def capture_llm(messages, **kwargs):
            """Record the prompt so we can assert on the length hint."""
            captured['msg'] = messages[0]['content']
            return "summary"

        with patch.object(self.agent.store, '_llm_chat', side_effect=capture_llm):
            self.agent._tool_summarise("Short text " * 5)
        assert "2-3 sentences" in captured['msg']

    def test_medium_text_hint(self):
        """100-299 words → 4-5 sentences hint."""
        captured = {}

        def capture_llm(messages, **kwargs):
            """Record the prompt so we can assert on the length hint."""
            captured['msg'] = messages[0]['content']
            return "summary"

        with patch.object(self.agent.store, '_llm_chat', side_effect=capture_llm):
            self.agent._tool_summarise("word " * 150)
        assert "4-5 sentences" in captured['msg']

    def test_long_text_hint(self):
        """>=300 words → 6-8 sentences hint."""
        captured = {}

        def capture_llm(messages, **kwargs):
            """Record the prompt so we can assert on the length hint."""
            captured['msg'] = messages[0]['content']
            return "summary"

        with patch.object(self.agent.store, '_llm_chat', side_effect=capture_llm):
            self.agent._tool_summarise("word " * 350)
        assert "6-8 sentences" in captured['msg']

    def test_returns_string(self):
        """_tool_summarise always returns a string."""
        with patch.object(self.agent.store, '_llm_chat', return_value="the summary"):
            result = self.agent._tool_summarise("some text to summarise here")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# _tool_sentiment — short queries search first; long text analysed directly
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentimentTool:
    """Tests for Agent._tool_sentiment — short queries search first, long text analysed directly."""

    def setup_method(self):
        """Build a minimal Agent to exercise the sentiment tool."""
        self.agent = _make_agent(3)

    def test_short_query_searches_first(self):
        """< 10 words → rag_search called first."""
        searched = {'called': False}

        def mock_search(q):
            """Record that rag_search was invoked."""
            searched['called'] = True
            return "The text is very positive and uplifting."

        with patch.object(self.agent, '_tool_rag_search', side_effect=mock_search), \
             patch.object(self.agent.store, '_llm_chat',
                          return_value="Sentiment: Positive\nTone: uplifting\n"
                                       "Key phrases: positive\nExplanation: test"):
            self.agent._tool_sentiment("resume tone")
        assert searched['called']

    def test_long_text_analyses_directly(self):
        """>=10 words → no rag_search, analyse directly."""
        searched = {'called': False}

        def mock_search(q):
            """Record that rag_search was invoked (should NOT be called)."""
            searched['called'] = True
            return ""

        with patch.object(self.agent, '_tool_rag_search', side_effect=mock_search), \
             patch.object(self.agent.store, '_llm_chat', return_value="Sentiment: Neutral"):
            self.agent._tool_sentiment("This is a very long text that has many words in it.")
        assert not searched['called']

    def test_sentiment_prompt_has_4_fields(self):
        """Sentiment prompt instructs the LLM to return all 4 structured fields."""
        captured = {}

        def capture_llm(messages, **kwargs):
            """Record the prompt to assert on structured output fields."""
            captured['msg'] = messages[0]['content']
            return ("Sentiment: Positive\nTone: upbeat\n"
                    "Key phrases: great\nExplanation: Good vibes.")

        with patch.object(self.agent.store, '_llm_chat', side_effect=capture_llm):
            self.agent._tool_sentiment("This is a long enough text to analyse directly now.")
        prompt = captured['msg']
        assert "Sentiment:" in prompt
        assert "Tone:" in prompt
        assert "Key phrases:" in prompt
        assert "Explanation:" in prompt
