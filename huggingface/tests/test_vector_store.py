"""test_vector_store.py — VectorStore initialisation and query analysis tests for the HF Space.

Covers:
  - TestVectorStoreBuild:    Initialisation, chunk addition, BM25 rebuild, clear conversation
  - TestQueryClassification: All four query types and priority ordering
  - TestSmartTopN:           Correct retrieval budget per query type
  - TestConfidenceCheck:     Threshold logic (empty, above, below)
  - TestSourceLabel:         All seven document-type label formats
  - TestHallucinationFilter: No-info + pivot phrase detection and truncation

Pipeline, retrieval, rerank prompts, and LLM fallback tests are in
test_vector_store_pipeline.py.

Mock strategy:
  - conftest.py patches _get_st_model globally (fake 384-dim embeddings).
  - conftest.py patches _llm_call globally (safe mock string).
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
# VectorStore initialisation and chunk management
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreBuild:
    """Tests for VectorStore initialisation, chunk addition, and BM25 rebuild."""

    def test_build_empty(self):
        """build_or_load([]) creates a collection with 0 chunks and no BM25 index."""
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load([])
        assert store.collection is not None
        assert store.collection.count() == 0
        assert store.bm25_index is None

    def test_build_with_chunks(self):
        """build_or_load with 5 chunks stores all 5 and creates a BM25 index."""
        chunks = sample_chunks(5)
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load(chunks)
        assert store.collection.count() == 5
        assert store.bm25_index is not None

    def test_add_chunks_increases_count(self):
        """add_chunks appends to the collection, raising count from 3 to 5."""
        chunks = sample_chunks(3)
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load(chunks)
        extra = sample_chunks(2, doc_type='pdf')
        store.add_chunks(extra, id_prefix='url')
        assert store.collection.count() == 5

    def test_rebuild_bm25_after_add(self):
        """rebuild_bm25 after add_chunks produces a non-None BM25 index."""
        chunks = sample_chunks(3)
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load(chunks)
        extra = sample_chunks(2)
        store.add_chunks(extra, id_prefix='file')
        store.rebuild_bm25(store.chunks)
        assert store.bm25_index is not None

    def test_clear_conversation(self):
        """clear_conversation resets conversation_history to an empty list."""
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        store.build_or_load([])
        store.conversation_history = [{'role': 'user', 'content': 'hi'}]
        store.clear_conversation()
        assert store.conversation_history == []

    def test_conversation_history_starts_empty(self):
        """VectorStore() initialises conversation_history as an empty list."""
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        assert store.conversation_history == []


# ═══════════════════════════════════════════════════════════════════════════════
# Query classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryClassification:
    """Tests for VectorStore._classify_query — all four query types."""

    def setup_method(self):
        """Build an empty VectorStore to access the classifier."""
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load([])

    def test_classify_summarise(self):
        """Summarise keywords trigger the 'summarise' query type."""
        assert self.store._classify_query("summarise this document") == 'summarise'
        assert self.store._classify_query("give me a summary") == 'summarise'
        assert self.store._classify_query("tell me about the resume") == 'summarise'

    def test_classify_comparison(self):
        """Comparison keywords ('compare', 'vs', 'difference') → 'comparison' type."""
        assert self.store._classify_query("compare Python vs Java") == 'comparison'
        assert self.store._classify_query("what is the difference between them") == 'comparison'

    def test_classify_factual(self):
        """'what is', 'how many', 'list' and similar → 'factual' type."""
        assert self.store._classify_query("what is the capital?") == 'factual'
        assert self.store._classify_query("how many employees?") == 'factual'
        assert self.store._classify_query("list all features") == 'factual'

    def test_classify_general(self):
        """Queries matching no signal → fallback 'general' type."""
        assert self.store._classify_query("random vague question") == 'general'

    def test_summarise_checked_first(self):
        """Summarise is checked before comparison, so 'summarise and compare' → 'summarise'."""
        assert self.store._classify_query("summarise and compare") == 'summarise'

    def test_resume_is_summarise(self):
        """Query mentioning 'resume' is classified as 'summarise'."""
        assert self.store._classify_query("show me the resume") == 'summarise'


# ═══════════════════════════════════════════════════════════════════════════════
# Smart top-N retrieval budgets
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmartTopN:
    """Tests for VectorStore._smart_top_n — correct retrieval budget per query type."""

    def setup_method(self):
        """Build an empty VectorStore to access the top-N selector."""
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load([])

    def test_factual_returns_5(self):
        """Factual queries retrieve the fewest chunks — enough for one precise answer."""
        assert self.store._smart_top_n('factual') == 5

    def test_comparison_returns_15(self):
        """Comparison queries retrieve the most chunks to cover both sides."""
        assert self.store._smart_top_n('comparison') == 15

    def test_general_returns_10(self):
        """General queries use a mid-range budget."""
        assert self.store._smart_top_n('general') == 10

    def test_summarise_returns_top_retrieve(self):
        """Summarise uses the full TOP_RETRIEVE budget to cover the whole document."""
        from src.rag.config import TOP_RETRIEVE
        assert self.store._smart_top_n('summarise') == TOP_RETRIEVE


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence check
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceCheck:
    """Tests for VectorStore._check_confidence threshold logic."""

    def setup_method(self):
        """Build an empty VectorStore to access the confidence checker."""
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load([])

    def test_empty_results_not_confident(self):
        """Empty result list → is_confident=False, best_score=0.0."""
        ok, score = self.store._check_confidence([])
        assert ok is False
        assert score == 0.0

    def test_above_threshold_confident(self):
        """Best score strictly above SIMILARITY_THRESHOLD → is_confident=True."""
        from src.rag.config import SIMILARITY_THRESHOLD
        entry = {'text': 'x', 'source': 'f', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        ok, score = self.store._check_confidence([(entry, SIMILARITY_THRESHOLD + 0.1)])
        assert ok is True

    def test_below_threshold_not_confident(self):
        """Best score strictly below SIMILARITY_THRESHOLD → is_confident=False."""
        from src.rag.config import SIMILARITY_THRESHOLD
        entry = {'text': 'x', 'source': 'f', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        ok, score = self.store._check_confidence([(entry, SIMILARITY_THRESHOLD - 0.1)])
        assert ok is False


# ═══════════════════════════════════════════════════════════════════════════════
# Source labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceLabel:
    """Tests for VectorStore._source_label — all seven document types."""

    def setup_method(self):
        """Build an empty VectorStore to access the label formatter."""
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load([])

    def _entry(self, doc_type: str, start: int = 3, end: int = 3) -> dict:
        """Build a minimal chunk entry dict for label tests."""
        return {'text': 'x', 'source': 'f', 'start_line': start,
                'end_line': end, 'type': doc_type}

    def test_pdf_label(self):
        """PDF chunks use 'p{page}' format."""
        assert self.store._source_label(self._entry('pdf', 2)) == 'p2'

    def test_xlsx_label(self):
        """XLSX chunks use 'row{n}' format."""
        assert self.store._source_label(self._entry('xlsx', 5)) == 'row5'

    def test_csv_label(self):
        """CSV chunks use 'row{n}' format (same as xlsx)."""
        assert self.store._source_label(self._entry('csv', 7)) == 'row7'

    def test_pptx_label(self):
        """PPTX chunks use 'slide{n}' format."""
        assert self.store._source_label(self._entry('pptx', 3)) == 'slide3'

    def test_html_label(self):
        """HTML chunks use 's{n}' (sentence window index) format."""
        assert self.store._source_label(self._entry('html', 4)) == 's4'

    def test_txt_label(self):
        """TXT chunks use 'L{start}-{end}' line range format."""
        assert self.store._source_label(self._entry('txt', 1, 2)) == 'L1-2'

    def test_md_label(self):
        """Markdown chunks use 'L{start}-{end}' line range format."""
        assert self.store._source_label(self._entry('md', 6, 6)) == 'L6-6'


# ═══════════════════════════════════════════════════════════════════════════════
# Hallucination filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestHallucinationFilter:
    """Tests for VectorStore._filter_hallucination — no-info + pivot detection."""

    def setup_method(self):
        """Build an empty VectorStore to access the filter."""
        from src.rag.vector_store import VectorStore
        self.store = VectorStore()
        self.store.build_or_load([])

    def test_clean_response_unchanged(self):
        """Response with no no-info phrase is returned unmodified."""
        resp = "The answer is 42."
        assert self.store._filter_hallucination(resp) == resp

    def test_truncates_at_pivot_after_no_info(self):
        """No-info phrase followed by 'however,' → text after pivot is removed."""
        resp = "There is no information about this. However, I can tell you that cats sleep a lot."
        result = self.store._filter_hallucination(resp)
        assert "cats sleep" not in result

    def test_no_pivot_leaves_response(self):
        """No-info phrase with no following pivot → full response preserved."""
        resp = "There is no information about this topic in the documents."
        result = self.store._filter_hallucination(resp)
        assert "There is no information" in result

    def test_truncates_but_i_can(self):
        """No-info phrase followed by 'but i can' → text after pivot removed."""
        resp = "There is no information about it. But I can provide some context."
        result = self.store._filter_hallucination(resp)
        assert 'but i can' not in result.lower()

    def test_truncates_that_said(self):
        """No-info phrase followed by 'that said,' → pivot truncation fires."""
        resp = "I could not find this. That said, here is what I know."
        result = self.store._filter_hallucination(resp)
        # "That said," should be cut
        assert 'that said,' not in result.lower()
