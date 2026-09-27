"""test_contracts_pipeline.py — Contract tests for run_pipeline() and agent.run() (HF Space).

Split from test_contracts.py to keep each file under 500 lines (per CLAUDE.md).

Verifies output *shape*, required keys, and value types — not exact values.

HF differences from local:
  - LLM calls via _llm_call / _llm_chat (patched in conftest.py).
  - Embeddings via sentence-transformers (patched in conftest.py).
  - _rerank uses cross-encoder (patched per test).
  - ChromaDB is EphemeralClient (in-memory).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

from tests.conftest import make_store_with_chunks, sample_chunks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """VectorStore backed by 5 cat-facts chunks.

    conftest.py autouse fixtures patch _get_st_model and _llm_call globally,
    so no additional patching is needed here.
    """
    chunks = sample_chunks(5)
    return make_store_with_chunks(chunks)


# ---------------------------------------------------------------------------
# 1. run_pipeline contract
# ---------------------------------------------------------------------------

class TestRunPipelineContract:
    """run_pipeline() returns a dict with all required keys and correct types."""

    PIPELINE_KEYS = {
        'response':     str,
        'query_type':   str,
        'queries':      list,
        'is_confident': bool,
        'best_score':   float,
        'retrieved':    list,
        'reranked':     list,
    }

    def test_all_keys_present(self, store):
        """run_pipeline result contains all 7 required keys."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        for key in self.PIPELINE_KEYS:
            assert key in result, f"Missing key '{key}'"

    def test_key_types_correct(self, store):
        """Each key has the expected Python type."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        for key, expected_type in self.PIPELINE_KEYS.items():
            assert isinstance(result[key], expected_type), (
                f"Key '{key}': expected {expected_type.__name__}, "
                f"got {type(result[key]).__name__}"
            )

    def test_no_none_values(self, store):
        """No required key has a None value."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        for key in self.PIPELINE_KEYS:
            assert result[key] is not None, f"Key '{key}' is None"

    def test_response_is_non_empty(self, store):
        """response is a non-empty string."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        assert len(result['response']) > 0

    def test_query_type_is_valid(self, store):
        """query_type is one of the four valid classification labels."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        assert result['query_type'] in ('factual', 'comparison', 'general', 'summarise')

    def test_retrieved_entries_are_2_tuples(self, store):
        """Each item in retrieved is a (dict, float) 2-tuple."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        for item in result['retrieved']:
            assert len(item) == 2
            assert isinstance(item[0], dict)
            assert isinstance(item[1], float)

    def test_reranked_entries_are_3_tuples(self, store):
        """Each item in reranked is a (dict, float, float) 3-tuple."""
        with patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            result = store.run_pipeline('How long do cats sleep?')
        for item in result['reranked']:
            assert len(item) == 3
            assert isinstance(item[0], dict)
            assert isinstance(item[1], float)
            assert isinstance(item[2], float)

    def test_low_confidence_contract(self, store):
        """Low-confidence path still returns all required keys with correct types."""
        with patch('src.rag.vector_store.SIMILARITY_THRESHOLD', 2.0), \
             patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.1] * 10
            result = store.run_pipeline('zzzxxx unknownquery', streamlit_mode=True)
        for key in self.PIPELINE_KEYS:
            assert key in result, f"Low-confidence path missing key '{key}'"
        assert result['is_confident'] is False


# ---------------------------------------------------------------------------
# 2. agent.run() contract
# ---------------------------------------------------------------------------

class TestAgentRunContract:
    """agent.run() returns dict with answer(str) and steps(list of step dicts)."""

    @pytest.fixture
    def agent(self, store):
        """Agent wired to the shared store fixture."""
        from src.rag.agent import Agent
        return Agent(store)

    def _run(self, agent, query):
        """Run agent with _llm_chat mocked to emit a finish tool call."""
        with patch.object(agent.store, '_llm_chat',
                          return_value='TOOL: finish(Cats sleep 16 hours a day.)'), \
             patch('src.rag.vector_store._get_cross_encoder') as mock_ce:
            mock_ce.return_value.predict.return_value = [0.8] * 10
            return agent.run(query, streamlit_mode=True)

    def test_result_has_answer_key(self, agent):
        """run() result contains 'answer' key."""
        result = self._run(agent, 'How long do cats sleep?')
        assert 'answer' in result

    def test_result_has_steps_key(self, agent):
        """run() result contains 'steps' key."""
        result = self._run(agent, 'How long do cats sleep?')
        assert 'steps' in result

    def test_answer_is_non_empty_str(self, agent):
        """answer is a non-empty string."""
        result = self._run(agent, 'How long do cats sleep?')
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

    def test_steps_is_list(self, agent):
        """steps is a list."""
        result = self._run(agent, 'How long do cats sleep?')
        assert isinstance(result['steps'], list)

    def test_each_step_has_required_keys(self, agent):
        """Every step dict has keys: step, tool, arg, result."""
        result = self._run(agent, 'How long do cats sleep?')
        for i, step in enumerate(result['steps']):
            for key in ('step', 'tool', 'arg', 'result'):
                assert key in step, f"Step {i} missing key '{key}'"

    def test_step_types(self, agent):
        """step(int), tool(str), arg(str), result(str) in every step dict."""
        result = self._run(agent, 'How long do cats sleep?')
        for i, step in enumerate(result['steps']):
            assert isinstance(step['step'],   int), f"Step {i}: 'step' not int"
            assert isinstance(step['tool'],   str), f"Step {i}: 'tool' not str"
            assert isinstance(step['arg'],    str), f"Step {i}: 'arg' not str"
            assert isinstance(step['result'], str), f"Step {i}: 'result' not str"

    def test_no_none_in_step_keys(self, agent):
        """No required step key is None."""
        result = self._run(agent, 'How long do cats sleep?')
        for i, step in enumerate(result['steps']):
            for key in ('step', 'tool', 'arg', 'result'):
                assert step[key] is not None, f"Step {i} key '{key}' is None"
