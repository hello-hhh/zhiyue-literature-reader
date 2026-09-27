"""test_contracts_pipeline.py — Contract tests for run_pipeline() and agent.run().

Split from test_contracts.py to keep each file under 500 lines (per CLAUDE.md).

Verifies output *shape*, required keys, and value types — not exact values.
This file covers the two highest-level public methods that callers depend on most.

Mock strategy (per CLAUDE.md):
  Always mock: ollama.embed, ollama.chat, chromadb → EphemeralClient
  Never mock:  BM25Okapi, classification, source label, hallucination filter
"""

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Shared helpers (duplicated from test_contracts.py so each file is standalone)
# ---------------------------------------------------------------------------

CHUNK_KEYS = {
    'text':       str,
    'source':     str,
    'start_line': int,
    'end_line':   int,
    'type':       str,
}


def _assert_chunk_contract(chunk: dict, label: str = '') -> None:
    """Assert that *chunk* satisfies the 5-key chunk-dict contract.

    Args:
        chunk: The chunk dict to validate.
        label: Optional identifier printed in assertion messages.
    """
    prefix = f"[{label}] " if label else ""
    for key, expected_type in CHUNK_KEYS.items():
        assert key in chunk, f"{prefix}Missing key '{key}'"
        assert chunk[key] is not None, f"{prefix}Key '{key}' is None"
        assert isinstance(chunk[key], expected_type), (
            f"{prefix}Key '{key}' expected {expected_type.__name__}, "
            f"got {type(chunk[key]).__name__}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    """Five minimal cat-facts txt chunks used across contract tests."""
    return [
        {
            'text':       f"Cats sleep about 16 hours a day chunk {i}.",
            'source':     'cats.txt',
            'start_line': i + 1,
            'end_line':   i + 1,
            'type':       'txt',
        }
        for i in range(5)
    ]


@pytest.fixture
def store(sample_chunks):
    """VectorStore wired to an in-memory EphemeralClient with 5 cat-facts chunks."""
    from src.rag.vector_store import VectorStore
    vs     = VectorStore()
    client = chromadb.EphemeralClient()
    col    = client.get_or_create_collection(
        'test_pipeline_contracts', metadata={'hnsw:space': 'cosine'}
    )
    ids    = [f'c{i}' for i in range(len(sample_chunks))]
    texts  = [c['text'] for c in sample_chunks]
    metas  = [
        {'source': c['source'], 'start_line': c['start_line'],
         'end_line': c['end_line'], 'type': c['type']}
        for c in sample_chunks
    ]
    embeds = [[0.1, 0.2, 0.3, 0.4]] * len(sample_chunks)
    col.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    vs.collection           = col
    vs.chunks               = sample_chunks
    vs.bm25_index           = BM25Okapi([c['text'].lower().split() for c in sample_chunks])
    vs.conversation_history = []
    return vs


# ---------------------------------------------------------------------------
# 1. run_pipeline contract
# ---------------------------------------------------------------------------

class TestRunPipelineContract:
    """run_pipeline() returns a dict with all required keys and correct value types."""

    # Required keys and their expected Python types
    PIPELINE_KEYS = {
        'response':     str,
        'query_type':   str,
        'queries':      list,
        'is_confident': bool,
        'best_score':   float,
        'retrieved':    list,
        'reranked':     list,
    }

    def _run(self, store, query):
        """Run pipeline with ollama fully mocked; returns the result dict."""
        def _chat_mock(*args, **kwargs):
            if kwargs.get('stream'):
                return [{'message': {'content': 'Cats sleep 16 hours a day.'}}]
            return {'message': {'content': '8'}}

        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_chat_mock):
            return store.run_pipeline(query, streamlit_mode=True)

    def test_all_keys_present(self, store):
        """run_pipeline result contains all 7 required keys."""
        result = self._run(store, 'How long do cats sleep?')
        for key in self.PIPELINE_KEYS:
            assert key in result, f"Missing key '{key}' in run_pipeline result"

    def test_key_types_correct(self, store):
        """Each key in the pipeline result has the expected Python type.

        is_confident may be np.bool_ (from chromadb / numpy internals).  numpy.bool_
        is not a subclass of Python bool, so we check it by verifying the value equals
        True or False rather than using isinstance.
        """
        result = self._run(store, 'How long do cats sleep?')
        for key, expected_type in self.PIPELINE_KEYS.items():
            value = result[key]
            if key == 'is_confident':
                # Accept Python bool or numpy.bool_ — both behave as booleans
                assert value in (True, False), (
                    f"Key 'is_confident': expected True or False, got {value!r} "
                    f"(type: {type(value).__name__})"
                )
            else:
                assert isinstance(value, expected_type), (
                    f"Key '{key}': expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

    def test_no_none_values(self, store):
        """No required key in the pipeline result has a None value."""
        result = self._run(store, 'How long do cats sleep?')
        for key in self.PIPELINE_KEYS:
            assert result[key] is not None, f"Key '{key}' is None"

    def test_response_is_non_empty_string(self, store):
        """response key is a non-empty string."""
        result = self._run(store, 'How long do cats sleep?')
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    def test_query_type_is_valid(self, store):
        """query_type is one of the four valid classification labels."""
        result = self._run(store, 'How long do cats sleep?')
        assert result['query_type'] in ('factual', 'comparison', 'general', 'summarise')

    def test_retrieved_entries_are_tuples(self, store):
        """Each item in retrieved is a (dict, float) 2-tuple."""
        result = self._run(store, 'How long do cats sleep?')
        for item in result['retrieved']:
            assert len(item) == 2
            assert isinstance(item[0], dict)
            assert isinstance(item[1], float)

    def test_reranked_entries_are_triples(self, store):
        """Each item in reranked is a (dict, float, float) 3-tuple."""
        result = self._run(store, 'How long do cats sleep?')
        for item in result['reranked']:
            assert len(item) == 3
            assert isinstance(item[0], dict)
            assert isinstance(item[1], float)
            assert isinstance(item[2], float)

    def test_low_confidence_contract(self, store):
        """Pipeline result under low-confidence path still satisfies full key contract."""
        with patch('ollama.embed', return_value={'embeddings': [[0.9, 0.9, 0.9, 0.9]]}), \
             patch('ollama.chat', return_value={'message': {'content': '1'}}):
            # Force similarity threshold very high so nothing passes confidence check
            with patch('src.rag.vector_store.SIMILARITY_THRESHOLD', 2.0):
                result = store.run_pipeline('zzzxxx unknown query', streamlit_mode=True)
        for key in self.PIPELINE_KEYS:
            assert key in result, f"Low-confidence path missing key '{key}'"
        assert result['is_confident'] is False


# ---------------------------------------------------------------------------
# 2. agent.run() contract
# ---------------------------------------------------------------------------

class TestAgentRunContract:
    """agent.run() returns dict with answer(str) and steps(list of step dicts)."""

    @pytest.fixture
    def mock_store(self):
        """MagicMock VectorStore with enough state for the Agent to initialise."""
        from src.rag.vector_store import VectorStore
        s = MagicMock(spec=VectorStore)
        s.chunks = [
            {'text': 'Cats sleep 16 hours a day.',
             'source': 'cats.txt', 'start_line': 1, 'end_line': 1, 'type': 'txt'},
        ]
        s.bm25_index = BM25Okapi([['cats', 'sleep', '16']])
        s.collection = MagicMock()
        s.collection.count.return_value = 1
        s._expand_query.return_value = ['cats sleep']
        entry = s.chunks[0]
        s._hybrid_retrieve.return_value = [(entry, 0.9)]
        s._rerank.return_value = [(entry, 0.9, 0.9)]
        s._source_label.return_value = 'L1-1'
        return s

    @pytest.fixture
    def agent(self, mock_store):
        """Agent wired to mock_store."""
        from src.rag.agent import Agent
        return Agent(mock_store)

    def _run(self, agent, query):
        """Run agent with ollama.chat mocked to emit a finish tool call."""
        def _chat(*args, **kwargs):
            return {'message': {'content': 'TOOL: finish(Cats sleep 16 hours a day.)'}}
        with patch('ollama.chat', side_effect=_chat):
            return agent.run(query, streamlit_mode=True)

    def test_result_has_answer_key(self, agent):
        """agent.run() result contains 'answer' key."""
        result = self._run(agent, 'How long do cats sleep?')
        assert 'answer' in result

    def test_result_has_steps_key(self, agent):
        """agent.run() result contains 'steps' key."""
        result = self._run(agent, 'How long do cats sleep?')
        assert 'steps' in result

    def test_answer_is_non_empty_str(self, agent):
        """answer value is a non-empty string."""
        result = self._run(agent, 'How long do cats sleep?')
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

    def test_steps_is_list(self, agent):
        """steps value is a list."""
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

    def test_no_none_in_required_keys(self, agent):
        """No required key is None in any step dict."""
        result = self._run(agent, 'How long do cats sleep?')
        for i, step in enumerate(result['steps']):
            for key in ('step', 'tool', 'arg', 'result'):
                assert step[key] is not None, f"Step {i} key '{key}' is None"
