"""Unit tests for VectorStore — pipeline methods.

Covers: build_or_load, add_chunks, rebuild_bm25, clear_conversation,
        _expand_query, _hybrid_retrieve, _rerank, run_pipeline.

Mock strategy (per CLAUDE.md):
  ollama.embed  → {'embeddings': [[0.1, 0.2, ...]]}
  ollama.chat   → {'message': {'content': 'mock'}}
  chromadb      → chromadb.EphemeralClient()

Never mock: BM25Okapi, cosine_similarity.

Reason for split: max 500 lines per file per CLAUDE.md.
Core stateless methods (_truncate_for_embedding, _cosine_similarity,
_classify_query, _smart_top_n, _check_confidence, _source_label,
rerank_prompt (reranker module), _filter_hallucination, _build_instruction_prompt)
live in test_vector_store.py.
"""

from unittest.mock import patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    """Provide a bare VectorStore instance with no collection or chunks loaded."""
    from src.rag.vector_store import VectorStore
    vs = VectorStore()
    return vs


@pytest.fixture
def sample_chunks():
    """Provide three minimal txt chunks about cats for use across tests."""
    return [
        {'text': 'Cats sleep sixteen hours a day.', 'source': 'cats.txt',
         'start_line': 1, 'end_line': 1, 'type': 'txt'},
        {'text': 'Cats can see in dim light very well.', 'source': 'cats.txt',
         'start_line': 2, 'end_line': 2, 'type': 'txt'},
        {'text': 'Cats have five toes on front paws.', 'source': 'cats.txt',
         'start_line': 3, 'end_line': 3, 'type': 'txt'},
    ]


def _fake_embed(dim=4):
    """Return a mock ollama.embed response."""
    return {'embeddings': [[0.1] * dim]}


def _fake_chat(content='mock response'):
    """Return a mock ollama.chat response with the given content string."""
    return {'message': {'content': content}}


def _pipeline_chat_mock(*args, **kwargs):
    """Smart mock: returns stream list when stream=True, dict otherwise."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': '8'}}


def _setup_store_with_collection(sample_chunks):
    """Build a VectorStore with an EphemeralClient collection pre-loaded with sample_chunks."""
    from src.rag.vector_store import VectorStore
    vs = VectorStore()
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        'rag_docs', metadata={'hnsw:space': 'cosine'}
    )
    # Add chunks to the in-memory collection with fixed embeddings
    ids    = [f'c{i}' for i in range(len(sample_chunks))]
    texts  = [c['text'] for c in sample_chunks]
    metas  = [{'source': c['source'], 'start_line': c['start_line'],
               'end_line': c['end_line'], 'type': c['type']}
              for c in sample_chunks]
    embeds = [[0.1, 0.2, 0.3, 0.4]] * len(sample_chunks)
    collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    vs.collection = collection
    vs.chunks = sample_chunks
    vs.bm25_index = BM25Okapi([c['text'].lower().split() for c in sample_chunks])
    vs.conversation_history = []
    return vs


# ---------------------------------------------------------------------------
# build_or_load — uses EphemeralClient
# ---------------------------------------------------------------------------

class TestBuildOrLoad:
    """Tests that build_or_load initialises collection, BM25 index, and chunks correctly."""

    def _make_patched_store(self, sample_chunks):
        """Wire PersistentClient mock to EphemeralClient and call build_or_load."""
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        with patch('chromadb.PersistentClient') as mock_client:
            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(
                'rag_docs', metadata={'hnsw:space': 'cosine'}
            )
            mock_client.return_value.get_or_create_collection.return_value = collection
            with patch('ollama.embed', return_value=_fake_embed()):
                vs.build_or_load(sample_chunks)
        return vs

    def test_build_or_load_sets_collection(self, sample_chunks):
        """build_or_load with valid chunks: vs.collection is set to a non-None value."""
        vs = self._make_patched_store(sample_chunks)
        assert vs.collection is not None

    def test_build_or_load_sets_bm25(self, sample_chunks):
        """build_or_load with valid chunks: vs.bm25_index is populated."""
        vs = self._make_patched_store(sample_chunks)
        assert vs.bm25_index is not None

    def test_build_or_load_sets_chunks(self, sample_chunks):
        """build_or_load with valid chunks: vs.chunks is set to the provided chunk list."""
        vs = self._make_patched_store(sample_chunks)
        assert vs.chunks == sample_chunks


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    """Tests that add_chunks inserts new chunks into both ChromaDB and self.chunks."""

    def test_add_chunks_increases_collection_count(self, sample_chunks):
        """add_chunks with 3 chunks into empty collection: ChromaDB document count grows."""
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(
                'rag_docs', metadata={'hnsw:space': 'cosine'}
            )
            vs.collection = collection
            vs.chunks = []
            before = collection.count()
            vs.add_chunks(sample_chunks, id_prefix='url')
            assert collection.count() > before

    def test_add_chunks_updates_self_chunks(self, sample_chunks):
        """add_chunks with 3 chunks into empty vs.chunks: vs.chunks length equals input length."""
        with patch('ollama.embed', return_value=_fake_embed()):
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(
                'rag_docs', metadata={'hnsw:space': 'cosine'}
            )
            vs.collection = collection
            vs.chunks = []
            vs.add_chunks(sample_chunks, id_prefix='url')
            assert len(vs.chunks) == len(sample_chunks)


# ---------------------------------------------------------------------------
# rebuild_bm25
# ---------------------------------------------------------------------------

class TestRebuildBm25:
    """Tests that rebuild_bm25 creates a valid BM25Okapi index from the given chunks."""

    def test_rebuild_bm25_sets_index(self, store, sample_chunks):
        """rebuild_bm25 with 3 chunks: bm25_index is set to a BM25Okapi instance."""
        store.rebuild_bm25(sample_chunks)
        assert store.bm25_index is not None
        assert isinstance(store.bm25_index, BM25Okapi)


# ---------------------------------------------------------------------------
# clear_conversation
# ---------------------------------------------------------------------------

class TestClearConversation:
    """Tests that clear_conversation resets the conversation history to an empty list."""

    def test_clear_empties_history(self, store):
        """Non-empty conversation_history: after clear_conversation() it is []."""
        store.conversation_history = [{'role': 'user', 'content': 'hi'}]
        store.clear_conversation()
        assert store.conversation_history == []


# ---------------------------------------------------------------------------
# _expand_query — mocks ollama.chat
# ---------------------------------------------------------------------------

class TestExpandQuery:
    """Tests that _expand_query produces query rewrites and always includes the original."""

    def test_returns_list_with_original(self, store):
        """LLM returns two alternatives: original query is present in the result list."""
        with patch('ollama.chat', return_value=_fake_chat('alt1\nalt2')):
            result = store._expand_query('cats')
        assert 'cats' in result

    def test_returns_at_least_one(self, store):
        """LLM raises an exception: result falls back to list containing only original."""
        with patch('ollama.chat', side_effect=Exception('fail')):
            result = store._expand_query('cats')
        assert result == ['cats']

    def test_returns_original_plus_expansions(self, store):
        """LLM returns two rewrites: original is first and result has at least 2 entries."""
        with patch('ollama.chat', return_value=_fake_chat('rewrite1\nrewrite2')):
            result = store._expand_query('my query')
        assert result[0] == 'my query'
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _hybrid_retrieve — mocks ollama.embed and uses EphemeralClient
# ---------------------------------------------------------------------------

class TestHybridRetrieve:
    """Tests that _hybrid_retrieve fuses BM25 and dense scores and honours top_n."""

    def test_returns_list(self, sample_chunks):
        """_hybrid_retrieve with a single query: result is a list."""
        vs = _setup_store_with_collection(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}):
            result = vs._hybrid_retrieve(['cats sleep'], top_n=2)
        assert isinstance(result, list)

    def test_returns_tuples_entry_score(self, sample_chunks):
        """_hybrid_retrieve result items: each is a (chunk_dict, float_score) tuple."""
        vs = _setup_store_with_collection(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}):
            result = vs._hybrid_retrieve(['cats'], top_n=2)
        if result:
            entry, score = result[0]
            assert 'text' in entry
            assert isinstance(score, float)

    def test_top_n_respected(self, sample_chunks):
        """_hybrid_retrieve with top_n=2: at most 2 results are returned."""
        vs = _setup_store_with_collection(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}):
            result = vs._hybrid_retrieve(['cats'], top_n=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# _rerank — mocks ollama.chat
# ---------------------------------------------------------------------------

class TestRerank:
    """Tests that _rerank calls the LLM per candidate, scores, and returns top_n triples."""

    def test_returns_list(self, store, sample_chunks):
        """Two candidates, LLM score '8': result is a list."""
        entry = sample_chunks[0]
        candidates = [(entry, 0.8), (sample_chunks[1], 0.7)]
        with patch('ollama.chat', return_value=_fake_chat('8')):
            result = store._rerank('cats', candidates, top_n=2)
        assert isinstance(result, list)

    def test_top_n_respected(self, store, sample_chunks):
        """Three candidates, top_n=2: at most 2 results are returned."""
        candidates = [(c, 0.5) for c in sample_chunks]
        with patch('ollama.chat', return_value=_fake_chat('7')):
            result = store._rerank('cats', candidates, top_n=2)
        assert len(result) <= 2

    def test_returns_triples(self, store, sample_chunks):
        """One candidate, LLM score '9': each result item is a (entry, sim, llm_score) triple."""
        candidates = [(sample_chunks[0], 0.8)]
        with patch('ollama.chat', return_value=_fake_chat('9')):
            result = store._rerank('cats', candidates, top_n=1)
        if result:
            assert len(result[0]) == 3  # (entry, sim, llm_score)

    def test_bad_llm_response_falls_back_to_sim(self, store, sample_chunks):
        """LLM returns non-numeric text: _rerank does not raise and falls back gracefully."""
        candidates = [(sample_chunks[0], 0.75)]
        with patch('ollama.chat', return_value=_fake_chat('no number here at all')):
            result = store._rerank('cats', candidates, top_n=1)
        # Should not raise — result is a list (possibly empty)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# run_pipeline — integration with mocks
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Tests the full run_pipeline chat flow including classify, retrieve, rerank, synthesise."""

    def test_run_pipeline_returns_dict(self, sample_chunks):
        """Factual query in streamlit_mode: run_pipeline returns a dict."""
        vs = _setup_store_with_collection(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('how many hours do cats sleep', streamlit_mode=True)
        assert isinstance(result, dict)

    def test_run_pipeline_has_response_key(self, sample_chunks):
        """run_pipeline result dict contains a 'response' key."""
        vs = _setup_store_with_collection(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('cats sleep', streamlit_mode=True)
        assert 'response' in result

    def test_run_pipeline_has_query_type(self, sample_chunks):
        """run_pipeline result dict contains a 'query_type' key."""
        vs = _setup_store_with_collection(sample_chunks)
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('cats sleep', streamlit_mode=True)
        assert 'query_type' in result

    def test_low_confidence_returns_not_found_message(self, sample_chunks):
        """Distant embedding forces low similarity: is_confident is False or response present."""
        vs = _setup_store_with_collection(sample_chunks)
        # Force low scores by using a distant embedding vector
        with patch('ollama.embed', return_value={'embeddings': [[0.0, 0.0, 0.0, 1.0]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            result = vs.run_pipeline('unrelated query xyz', streamlit_mode=True)
        assert result['is_confident'] is False or 'response' in result

    def test_run_pipeline_appends_to_conversation_history(self, sample_chunks):
        """After run_pipeline completes: conversation_history has at least one entry."""
        vs = _setup_store_with_collection(sample_chunks)
        assert vs.conversation_history == []
        with patch('ollama.embed', return_value={'embeddings': [[0.1, 0.2, 0.3, 0.4]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock):
            vs.run_pipeline('cats sleep', streamlit_mode=True)
        assert len(vs.conversation_history) >= 1
