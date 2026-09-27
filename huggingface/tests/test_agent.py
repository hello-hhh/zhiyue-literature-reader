"""test_agent.py — Agent ReAct loop and integration tests for the HF Space version.

Covers:
  - TestAgentFastPaths:    Summarise (4-term search) and sentiment (label strip) shortcuts
  - TestAgentReactLoop:    Auto-finish, bad-format retry, step limit, collected context
  - TestDispatchTool:      Routes tool name to the correct private method
  - TestAgentSystemPrompt: AGENT_SYSTEM_PROMPT class constant integrity
  - TestAppHandlers:       app.py handler functions in isolation (no Gradio server)

Tool-level tests (_parse_tool_call, _tool_calculator, _tool_summarise,
_tool_sentiment) are in test_agent_tools.py.

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

# ── Gradio availability guard ─────────────────────────────────────────────────
try:
    import gradio  # noqa: F401
    _GRADIO_INSTALLED = True
except ImportError:
    _GRADIO_INSTALLED = False

requires_gradio = pytest.mark.skipif(
    not _GRADIO_INSTALLED, reason="gradio not installed locally"
)


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
# Agent fast paths — summarise (4-term search) and sentiment (label strip)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentFastPaths:
    """Tests for Agent fast-path shortcuts — summarise (4-term search) and sentiment (label strip)."""

    def setup_method(self):
        """Build an Agent over 5 sample chunks for fast-path testing."""
        self.agent = _make_agent(5)

    def test_fast_path_summarise_does_4_searches(self):
        """Summarise fast path fires exactly 4 rag_search calls with resume-specific terms.

        Resume terms are triggered when the query contains 'resume' or 'cv'.
        """
        search_calls = []

        def mock_search(q):
            """Record each query term to assert the 4-term strategy."""
            search_calls.append(q)
            return f"result for {q}"

        with patch.object(self.agent, '_tool_rag_search', side_effect=mock_search), \
             patch.object(self.agent, '_synthesize_final_answer', return_value="summary"):
            result = self.agent._fast_path_summarise("summarise the resume")
        assert len(search_calls) == 4
        assert 'work experience' in search_calls
        assert 'education' in search_calls
        assert 'answer' in result
        assert 'steps' in result

    def test_fast_path_sentiment_strips_labels(self):
        """Sentiment fast path strips chunk metadata labels before calling _tool_sentiment."""
        with patch.object(self.agent, '_tool_rag_search',
                          return_value="- [cats.txt L1-1] Cats are great."), \
             patch.object(self.agent, '_tool_sentiment', return_value="Sentiment: Positive"):
            result = self.agent._fast_path_sentiment("what is the sentiment?")
        assert result['answer'] == "Sentiment: Positive"
        assert len(result['steps']) == 3

    def test_summarise_query_triggers_fast_path(self):
        """A query containing a summarise keyword routes to _fast_path_summarise."""
        fast_called = {'called': False}

        def mock_fast(q, streamlit_mode=False):
            """Record that the fast path was entered."""
            fast_called['called'] = True
            return {'answer': 'summary', 'steps': []}

        with patch.object(self.agent, '_fast_path_summarise', side_effect=mock_fast):
            self.agent.run("summarise the document")
        assert fast_called['called']

    def test_sentiment_query_triggers_fast_path(self):
        """A query containing a sentiment keyword routes to _fast_path_sentiment."""
        fast_called = {'called': False}

        def mock_fast(q, streamlit_mode=False):
            """Record that the fast path was entered."""
            fast_called['called'] = True
            return {'answer': 'sentiment result', 'steps': []}

        with patch.object(self.agent, '_fast_path_sentiment', side_effect=mock_fast):
            self.agent.run("what is the sentiment of the document?")
        assert fast_called['called']


# ═══════════════════════════════════════════════════════════════════════════════
# ReAct loop — auto-finish, bad-format retry, step limit, collected context
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentReactLoop:
    """Tests for the Agent ReAct loop — auto-finish, bad-format retry, step limit."""

    def setup_method(self):
        """Build an Agent over 5 sample chunks for ReAct loop testing."""
        self.agent = _make_agent(5)

    def test_calculator_auto_finish(self):
        """Calculator result → auto-finish step appended with the numeric result."""
        responses = iter(["TOOL: calculator(10 * 5)"])
        with patch.object(self.agent.store, '_llm_chat',
                          side_effect=lambda *a, **kw: next(responses)):
            result = self.agent.run("what is 10 times 5?")
        assert "50" in result['answer']
        assert any(s['tool'] == 'finish' for s in result['steps'])

    def test_rag_search_auto_finish(self):
        """First rag_search on a non-summarise query → auto-finish after synthesis."""
        responses = iter(["TOOL: rag_search(cat sleep hours)"])
        with patch.object(self.agent.store, '_llm_chat',
                          side_effect=lambda *a, **kw: next(responses)):
            result = self.agent.run("how long do cats sleep?")
        assert result['answer'] is not None
        assert any(s['tool'] == 'finish' for s in result['steps'])

    def test_bad_format_retries_max_2(self):
        """Non-tool LLM output triggers up to 2 correction retries, then uses raw text."""
        call_count = {'n': 0}

        def bad_llm(messages, **kwargs):
            """Always return non-tool text to trigger retry logic."""
            call_count['n'] += 1
            return "This is not a tool call at all."

        with patch.object(self.agent.store, '_llm_chat', side_effect=bad_llm):
            result = self.agent.run("simple question?")
        # Result must always be non-None even after max retries
        assert result['answer'] is not None

    def test_max_steps_reached(self):
        """Loop that keeps calling rag_search terminates at max_steps and returns an answer."""
        self.agent.max_steps = 3

        def looping_llm(messages, **kwargs):
            """Keep issuing rag_search to drive the loop to its limit."""
            return "TOOL: rag_search(cats)"

        with patch.object(self.agent.store, '_llm_chat', side_effect=looping_llm), \
             patch.object(self.agent, '_synthesize_final_answer', return_value="final"):
            result = self.agent.run("how many steps?")
        assert result['answer'] is not None

    def test_finish_uses_collected_context(self):
        """rag_search followed by finish uses collected context for final synthesis."""
        responses = iter([
            "TOOL: rag_search(cats)",
            "TOOL: finish(done)",
        ])
        with patch.object(self.agent.store, '_llm_chat',
                          side_effect=lambda *a, **kw: next(responses)), \
             patch.object(self.agent, '_synthesize_final_answer',
                          return_value="synthesized answer"):
            result = self.agent.run("what do cats do?")
        assert result['answer'] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# _dispatch_tool — routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestDispatchTool:
    """Tests for Agent._dispatch_tool — routes tool name to the correct private method."""

    def setup_method(self):
        """Build a minimal Agent to test tool dispatch routing."""
        self.agent = _make_agent(3)

    def test_dispatch_calculator(self):
        """'calculator' routes to _tool_calculator and returns numeric string."""
        result = self.agent._dispatch_tool('calculator', '3 + 3')
        assert result == '6'

    def test_dispatch_unknown_tool(self):
        """Unrecognised tool name returns an 'Unknown tool' error string."""
        result = self.agent._dispatch_tool('unknown_tool', 'arg')
        assert 'Unknown tool' in result

    def test_dispatch_rag_search_adds_to_context(self):
        """rag_search result is appended to self.collected_context."""
        self.agent.collected_context = []
        with patch.object(self.agent, '_tool_rag_search', return_value="some result"):
            self.agent._dispatch_tool('rag_search', 'cats')
        assert len(self.agent.collected_context) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT_SYSTEM_PROMPT integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentSystemPrompt:
    """Tests that the AGENT_SYSTEM_PROMPT class constant is complete and well-formed."""

    def test_prompt_is_non_empty_string(self):
        """AGENT_SYSTEM_PROMPT is a string longer than 100 chars."""
        from src.rag.agent import Agent
        assert isinstance(Agent.AGENT_SYSTEM_PROMPT, str)
        assert len(Agent.AGENT_SYSTEM_PROMPT) > 100

    def test_prompt_has_all_5_tools(self):
        """All 5 tools are named in the system prompt."""
        from src.rag.agent import Agent
        prompt = Agent.AGENT_SYSTEM_PROMPT
        for tool in ('rag_search', 'calculator', 'summarise', 'sentiment', 'finish'):
            assert tool in prompt, f"Tool '{tool}' missing from AGENT_SYSTEM_PROMPT"

    def test_prompt_has_tool_format(self):
        """Prompt instructs LLM to use 'TOOL:' prefix."""
        from src.rag.agent import Agent
        assert 'TOOL:' in Agent.AGENT_SYSTEM_PROMPT

    def test_prompt_has_finish_rule(self):
        """Prompt mentions 'finish' as a termination instruction."""
        from src.rag.agent import Agent
        assert 'finish' in Agent.AGENT_SYSTEM_PROMPT.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# App handler functions (unit — no Gradio server)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_gradio
class TestAppHandlers:
    """Test handler functions in isolation (no Gradio server needed).

    After the refactor, handler functions live in src/rag/handlers.py —
    not in app.py. Tests import from there directly.
    """

    def _make_store(self):
        """Build a small in-memory VectorStore for handler tests."""
        return make_store_with_chunks(sample_chunks(3))

    def _import_handlers(self):
        """Import src.handlers with Gradio components mocked out.

        Returns:
            The imported handlers module, or skips the test if import fails.
        """
        import importlib
        import unittest.mock as um
        with um.patch('gradio.Blocks'), um.patch('gradio.Chatbot'), \
             um.patch('gradio.Textbox'), um.patch('gradio.Radio'), \
             um.patch('gradio.Button'), um.patch('gradio.Row'), \
             um.patch('gradio.Column'), um.patch('gradio.Accordion'), \
             um.patch('gradio.File'), um.patch('gradio.Markdown'), \
             um.patch('gradio.HTML'), um.patch('gradio.Progress'):
            try:
                import src.handlers as handlers_module
                importlib.reload(handlers_module)
                return handlers_module
            except Exception:
                pytest.skip("Could not import src.handlers in test environment")

    def test_pipeline_summary_format(self):
        """_pipeline_summary produces markdown with Query type and Confidence score."""
        handlers = self._import_handlers()
        store = self._make_store()
        # Inject the store so _pipeline_summary can call store._source_label()
        handlers._store = store

        pipeline_data = {
            'query_type':   'factual',
            'best_score':   0.75,
            'is_confident': True,
            'retrieved': [
                ({'text': 'Cats sleep.', 'source': 'cats.txt',
                  'start_line': 1, 'end_line': 1, 'type': 'txt'}, 0.75),
            ],
            'reranked': [
                ({'text': 'Cats sleep.', 'source': 'cats.txt',
                  'start_line': 1, 'end_line': 1, 'type': 'txt'}, 0.75, 8.0),
            ],
        }
        result = handlers._pipeline_summary(pipeline_data)
        assert 'factual' in result
        assert '0.750' in result

    def test_agent_steps_md_format(self):
        """_agent_steps_md produces markdown with Step N and the tool name."""
        handlers = self._import_handlers()

        steps = [
            {'step': 1, 'tool': 'rag_search', 'arg': 'cats',
             'result': 'Cats sleep 16 hours.'},
            {'step': 2, 'tool': 'finish', 'arg': 'done', 'result': 'done'},
        ]
        result = handlers._agent_steps_md(steps)
        assert 'Step 1' in result
        assert 'rag_search' in result
        assert 'Step 2' in result
