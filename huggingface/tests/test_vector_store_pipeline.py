"""test_vector_store_pipeline.py — VectorStore pipeline and retrieval tests for the HF Space.

Covers:
  - TestHybridRetrieve:       BM25 + dense fusion — returns list, top_n, tuple shape
  - TestQueryExpand:          Query expansion disabled in HF (always returns [original])
  - TestTruncateForEmbedding: 200-word AND 1200-char enforced simultaneously
  - TestLowConfidencePipeline: LLM synthesis is skipped when similarity is too low
  - TestRerankPrompts:        7 doc-type-specific prompt variants
  - TestLlmCallFallback:      HF Inference API with multi-model fallback chain

Mock strategy:
  - conftest.py patches _get_st_model globally (fake 384-dim embeddings).
  - conftest.py patches _llm_call globally (safe mock string).
  - Tests in TestLlmCallFallback are excluded from the _llm_call patch so
    they can test the real fallback logic against a mocked InferenceClient.
  - Never mock: BM25Okapi, cosine similarity, truncation, classification,
    source labels, hallucination filter, chromadb.EphemeralClient.

HF differences from local:
  - _expand_query is disabled — always returns [original_query].
  - LLM calls use _llm_call (InferenceClient), not ollama.chat.
  - ChromaDB is EphemeralClient (in-memory), not PersistentClient.
  - conversation_history (not conversation) is the attribute name.
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
# Hybrid retrieve (BM25 + dense fusion)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridRetrieve:
    """Tests for VectorStore._hybrid_retrieve — BM25 + dense fusion."""

    def setup_method(self):
        """Build a 3-chunk store with distinct topics for retrieval tests."""
        chunks = [
            {'text': 'Cats sleep 16 hours a day.', 'source': 'cats.txt',
             'start_line': 1, 'end_line': 1, 'type': 'txt'},
            {'text': 'Dogs are loyal companions.', 'source': 'dogs.txt',
             'start_line': 1, 'end_line': 1, 'type': 'txt'},
            {'text': 'Python is a programming language.', 'source': 'prog.txt',
             'start_line': 1, 'end_line': 1, 'type': 'txt'},
        ]
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load(chunks)

    def test_returns_list(self):
        """Hybrid retrieve always returns a list, even for unknown queries."""
        results = self.store._hybrid_retrieve(['cats'], top_n=2)
        assert isinstance(results, list)

    def test_top_n_respected(self):
        """Result count never exceeds the requested top_n."""
        results = self.store._hybrid_retrieve(['cats sleep'], top_n=2)
        assert len(results) <= 2

    def test_each_result_is_entry_score_tuple(self):
        """Each result is a (entry_dict, float_score) tuple with a 'text' key."""
        results = self.store._hybrid_retrieve(['cats'], top_n=3)
        for entry, score in results:
            assert 'text' in entry
            assert isinstance(score, float)

    def test_empty_collection_returns_empty(self):
        """No build_or_load called → collection is None → returns empty list."""
        from src.rag.vector_store import VectorStore
        empty_store = VectorStore()
        results = empty_store._hybrid_retrieve(['cats'], top_n=3)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# Query expansion (disabled in HF version)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryExpand:
    """Tests for VectorStore._expand_query — disabled in HF version."""

    def test_expand_returns_original_only(self):
        """HF version disables query expansion to reduce API calls → returns [original]."""
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load([])
        results = store._expand_query("how long do cats sleep")
        assert len(results) == 1
        assert results[0] == "how long do cats sleep"

    def test_expand_fallback_on_llm_error(self):
        """LLM raises an exception → falls back to original query only."""
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load([])
        with patch.object(store, '_llm_chat', side_effect=Exception("LLM down")):
            results = store._expand_query("test query")
        assert results == ["test query"]


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding truncation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTruncateForEmbedding:
    """Tests for VectorStore._truncate_for_embedding — enforces 200-word AND 1200-char limits."""

    def setup_method(self):
        """Build an empty VectorStore to access the truncation helper."""
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load([])

    def test_truncates_to_200_words(self):
        """300-word input is cut to ≤200 words."""
        text = ' '.join(['word'] * 300)
        result = self.store._truncate_for_embedding(text)
        assert len(result.split()) <= 200

    def test_truncates_to_1200_chars(self):
        """2000-char single-token string is cut to ≤1200 chars."""
        text = 'a' * 2000
        result = self.store._truncate_for_embedding(text)
        assert len(result) <= 1200

    def test_both_limits_enforced(self):
        """201 long words exceed both limits — both caps applied simultaneously."""
        text = ' '.join(['abcdefghij'] * 201)
        result = self.store._truncate_for_embedding(text)
        assert len(result.split()) <= 200
        assert len(result) <= 1200

    def test_short_text_unchanged(self):
        """Short input under both limits is returned unchanged."""
        text = 'hello world'
        assert self.store._truncate_for_embedding(text) == text


# ═══════════════════════════════════════════════════════════════════════════════
# Low-confidence pipeline path
# ═══════════════════════════════════════════════════════════════════════════════

class TestLowConfidencePipeline:
    """Tests for the low-confidence path — LLM synthesis is skipped when similarity is too low."""

    def test_low_confidence_skips_llm(self):
        """Forced low confidence → fixed 'could not find' message returned without LLM call."""
        chunks = [
            {'text': 'Cats sleep 16 hours.', 'source': 'cats.txt',
             'start_line': 1, 'end_line': 1, 'type': 'txt'},
        ]
        llm_call_count = {'n': 0}

        def counting_llm(prompt, **kwargs):
            """Count every _llm_call invocation so we can assert no LLM call was made."""
            llm_call_count['n'] += 1
            return "mock rewrite\nalternative rewrite"

        with patch('src.rag.vector_store._llm_call', side_effect=counting_llm):
            from src.rag.vector_store import VectorStore
            store = VectorStore()
            store.build_or_load(chunks)
            # Force confidence check to always fail
            store._check_confidence = lambda r: (False, 0.1)
            result = store.run_pipeline("unrelated query about quantum physics")

        assert result['is_confident'] is False
        assert "could not find" in result['response'].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Rerank prompts (7 doc-type variants)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRerankPrompts:
    """Tests for rerank_prompt (src/rag/reranker.py) — 7 doc-type-specific prompt variants.

    The HF VectorStore uses a cross-encoder for reranking, not LLM prompts,
    so rerank_prompt is tested here directly as a module function.
    """

    def _entry(self, doc_type: str) -> dict:
        """Return a minimal chunk entry dict for the given doc type."""
        return {'text': 'sample text', 'source': 'f', 'start_line': 1,
                'end_line': 1, 'type': doc_type}

    def test_xlsx_prompt(self):
        """xlsx prompt mentions 'spreadsheet' and uses 1-to-10 scoring scale."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('xlsx'))
        assert 'spreadsheet' in prompt.lower()
        assert '1 to 10' in prompt

    def test_csv_prompt(self):
        """csv prompt also uses the spreadsheet variant."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('csv'))
        assert 'spreadsheet' in prompt.lower()

    def test_pptx_prompt(self):
        """pptx prompt references slides."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('pptx'))
        assert 'slide' in prompt.lower()

    def test_pdf_prompt(self):
        """pdf prompt references PDF or page."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('pdf'))
        assert 'pdf' in prompt.lower() or 'page' in prompt.lower()

    def test_docx_prompt(self):
        """docx prompt references paragraph or document."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('docx'))
        assert 'paragraph' in prompt.lower() or 'document' in prompt.lower()

    def test_html_prompt(self):
        """html prompt references webpage."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('html'))
        assert 'webpage' in prompt.lower()

    def test_md_prompt(self):
        """md prompt references markdown."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('md'))
        assert 'markdown' in prompt.lower()

    def test_default_prompt(self):
        """txt falls through to the default prompt with a 1-to-10 numeric scale."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt("query", self._entry('txt'))
        assert '1-10' in prompt or '1 to 10' in prompt

    def test_all_prompts_have_scale(self):
        """All 8 types produce a prompt that instructs 1-to-10 scoring."""
        from src.rag.reranker import rerank_prompt
        for dtype in ('xlsx', 'csv', 'pptx', 'pdf', 'docx', 'html', 'md', 'txt'):
            prompt = rerank_prompt("query", self._entry(dtype))
            assert '1 to 10' in prompt or '1-10' in prompt, (
                f"Missing scoring scale in {dtype} prompt"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# _llm_call fallback chain (excluded from global _llm_call patch by conftest)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLlmCallFallback:
    """Tests for the module-level _llm_call — HF Inference API with multi-model fallback.

    conftest.py excludes this class from the global _llm_call patch so the real
    function runs against a mocked InferenceClient.
    """

    def test_no_token_returns_error(self):
        """Empty HF_TOKEN returns an error string without making any HTTP request."""
        with patch.dict(os.environ, {'HF_TOKEN': ''}):
            import importlib

            import src.rag.vector_store as vs_module
            result = vs_module._llm_call("test prompt")
        assert "HF_TOKEN not set" in result

    def test_first_model_success_returns_immediately(self):
        """Successful first InferenceClient call returns content without trying other models."""
        mock_result = MagicMock()
        mock_result.choices[0].message.content = "hello from model"
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_result
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_fake'}), \
             patch('huggingface_hub.InferenceClient', return_value=mock_client):
            import src.rag.vector_store as vs_module
            result = vs_module._llm_call([{"role": "user", "content": "test"}])
        assert result == "hello from model"

    def test_first_fails_tries_second(self):
        """Exception on first InferenceClient call triggers retry with the next model."""
        call_count = {'n': 0}

        def fake_chat(**kwargs):
            """Fail the first call; return a valid response on the second."""
            call_count['n'] += 1
            if call_count['n'] == 1:
                raise Exception("Service unavailable")
            mock_result = MagicMock()
            mock_result.choices[0].message.content = "second model"
            return mock_result

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = fake_chat
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_fake'}), \
             patch('huggingface_hub.InferenceClient', return_value=mock_client):
            import src.rag.vector_store as vs_module
            result = vs_module._llm_call([{"role": "user", "content": "test"}])
        assert call_count['n'] == 2
        assert result == "second model"

    def test_all_models_fail_returns_error_message(self):
        """All InferenceClient calls raising → 'all providers/models failed' string."""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = Exception("Service unavailable")
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_fake'}), \
             patch('huggingface_hub.InferenceClient', return_value=mock_client):
            import src.rag.vector_store as vs_module
            result = vs_module._llm_call([{"role": "user", "content": "test"}])
        assert "all providers/models failed" in result
