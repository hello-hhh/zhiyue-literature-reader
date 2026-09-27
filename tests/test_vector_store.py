"""Unit tests for VectorStore — core methods.

Covers: __init__, _truncate_for_embedding, _cosine_similarity, _classify_query,
        _smart_top_n, _check_confidence, _source_label, rerank_prompt (reranker module),
        _filter_hallucination, _build_instruction_prompt.

Mock strategy (per CLAUDE.md):
  ollama.embed  → {'embeddings': [[0.1, 0.2, ...]]}
  ollama.chat   → {'message': {'content': 'mock'}}
  chromadb      → chromadb.EphemeralClient()

Never mock: BM25Okapi, cosine_similarity, _truncate_for_embedding,
            _classify_query, _smart_top_n, _source_label, _filter_hallucination.

Reason for split: max 500 lines per file per CLAUDE.md.
Higher-level methods (build_or_load, add_chunks, rebuild_bm25,
clear_conversation, _expand_query, _hybrid_retrieve, _rerank, run_pipeline)
live in test_vector_store_pipeline.py.
"""

import pytest

# ---------------------------------------------------------------------------
# Fixture
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


def _fake_chat_stream(content='mock response'):
    """Return a list of one chunk (simulates stream=True)."""
    return [{'message': {'content': content}}]


def _pipeline_chat_mock(*args, **kwargs):
    """Smart mock: returns stream list when stream=True, dict otherwise."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': '8'}}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests that VectorStore.__init__ creates all required instance attributes."""

    def test_has_conversation_history(self, store):
        """VectorStore() → 'conversation_history' exists and starts as empty list."""
        assert hasattr(store, 'conversation_history')
        assert store.conversation_history == []


# ---------------------------------------------------------------------------
# _truncate_for_embedding
# ---------------------------------------------------------------------------

class TestTruncateForEmbedding:
    """Tests that _truncate_for_embedding enforces both the 200-word and 1200-char limits."""

    def test_short_text_unchanged(self, store):
        """Short input under both limits: text is returned unchanged."""
        text = 'hello world'
        assert store._truncate_for_embedding(text) == text

    def test_truncates_at_200_words(self, store):
        """300-word input: result is truncated to at most 200 words."""
        text = ' '.join(['word'] * 300)
        result = store._truncate_for_embedding(text)
        assert len(result.split()) <= 200

    def test_truncates_at_1200_chars(self, store):
        """2000-char input: result is truncated to at most 1200 characters."""
        text = 'a' * 2000
        result = store._truncate_for_embedding(text)
        assert len(result) <= 1200

    def test_both_limits_applied(self, store):
        # 150 words each 9 chars → under 200 words but over 1200 chars
        """150 nine-char words (under word limit but over char limit): char limit wins."""
        text = ' '.join(['abcdefghi'] * 150)
        result = store._truncate_for_embedding(text)
        assert len(result) <= 1200


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests that _cosine_similarity computes correct dot-product-based cosine similarity."""

    def test_identical_vectors(self, store):
        """Identical non-zero vectors: cosine similarity is 1.0."""
        a = [1.0, 0.0, 0.0]
        assert abs(store._cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self, store):
        """Orthogonal vectors: cosine similarity is 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(store._cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self, store):
        """Zero vector as first argument: result is 0.0 (no division by zero)."""
        assert store._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# _classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    """Tests that _classify_query assigns the correct query type from keyword signals."""

    def test_summarise_signal(self, store):
        """Query containing 'summarise': classified as 'summarise'."""
        assert store._classify_query('summarise the document') == 'summarise'

    def test_comparison_signal(self, store):
        """Query containing 'compare' and 'vs': classified as 'comparison'."""
        assert store._classify_query('compare method A vs method B') == 'comparison'

    def test_factual_signal(self, store):
        """Query starting with 'what is': classified as 'factual'."""
        assert store._classify_query('what is the boiling point of water') == 'factual'

    def test_general_fallback(self, store):
        """Query with no matching keywords: falls back to 'general'."""
        assert store._classify_query('something interesting happened') == 'general'

    def test_resume_is_summarise(self, store):
        """Query mentioning 'resume': classified as 'summarise' (document overview)."""
        assert store._classify_query('show me the resume') == 'summarise'


# ---------------------------------------------------------------------------
# _smart_top_n
# ---------------------------------------------------------------------------

class TestSmartTopN:
    """Tests that _smart_top_n returns the correct retrieval count for each query type."""

    def test_factual_returns_5(self, store):
        """Query type 'factual': top-N is 5 (precision-focused)."""
        assert store._smart_top_n('factual') == 5

    def test_comparison_returns_15(self, store):
        """Query type 'comparison': top-N is 15 (needs broad coverage)."""
        assert store._smart_top_n('comparison') == 15

    def test_general_returns_10(self, store):
        """Query type 'general': top-N is 10 (balanced)."""
        assert store._smart_top_n('general') == 10

    def test_summarise_returns_top_retrieve(self, store):
        """Query type 'summarise': top-N equals TOP_RETRIEVE constant (max coverage)."""
        from src.rag.config import TOP_RETRIEVE
        assert store._smart_top_n('summarise') == TOP_RETRIEVE


# ---------------------------------------------------------------------------
# _check_confidence
# ---------------------------------------------------------------------------

class TestCheckConfidence:
    """Tests that _check_confidence evaluates top similarity score against the threshold."""

    def test_empty_returns_false(self, store):
        """Empty results list: confident=False and score=0.0 are returned."""
        confident, score = store._check_confidence([])
        assert confident is False
        assert score == 0.0

    def test_above_threshold_is_confident(self, store):
        """Top score above SIMILARITY_THRESHOLD: confident=True is returned."""
        from src.rag.config import SIMILARITY_THRESHOLD
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        results = [(entry, SIMILARITY_THRESHOLD + 0.1)]
        confident, score = store._check_confidence(results)
        assert confident is True

    def test_below_threshold_not_confident(self, store):
        """Top score below SIMILARITY_THRESHOLD: confident=False is returned."""
        from src.rag.config import SIMILARITY_THRESHOLD
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        results = [(entry, SIMILARITY_THRESHOLD - 0.1)]
        confident, score = store._check_confidence(results)
        assert confident is False


# ---------------------------------------------------------------------------
# _source_label
# ---------------------------------------------------------------------------

class TestSourceLabel:
    """Tests that _source_label produces the correct format string for each document type."""

    def test_pdf_returns_page(self, store):
        """PDF chunk at page 3: label is 'p3'."""
        entry = {'type': 'pdf', 'start_line': 3, 'end_line': 3}
        assert store._source_label(entry) == 'p3'

    def test_xlsx_returns_row(self, store):
        """XLSX chunk at row 5: label is 'row5'."""
        entry = {'type': 'xlsx', 'start_line': 5, 'end_line': 5}
        assert store._source_label(entry) == 'row5'

    def test_csv_returns_row(self, store):
        """CSV chunk at row 2: label is 'row2'."""
        entry = {'type': 'csv', 'start_line': 2, 'end_line': 2}
        assert store._source_label(entry) == 'row2'

    def test_pptx_returns_slide(self, store):
        """PPTX chunk at slide 1: label is 'slide1'."""
        entry = {'type': 'pptx', 'start_line': 1, 'end_line': 1}
        assert store._source_label(entry) == 'slide1'

    def test_html_returns_s(self, store):
        """HTML chunk at sentence 4: label is 's4'."""
        entry = {'type': 'html', 'start_line': 4, 'end_line': 4}
        assert store._source_label(entry) == 's4'

    def test_txt_returns_line_range(self, store):
        """TXT chunk spanning lines 1–3: label is 'L1-3'."""
        entry = {'type': 'txt', 'start_line': 1, 'end_line': 3}
        assert store._source_label(entry) == 'L1-3'


# ---------------------------------------------------------------------------
# rerank_prompt — 7 variants (extracted to src/rag/reranker.py)
# ---------------------------------------------------------------------------

class TestRerankPrompt:
    """Tests that rerank_prompt generates type-specific prompts for all 7 document variants.

    rerank_prompt is a module-level function in src/rag/reranker.py — it is
    stateless and requires no VectorStore instance.
    """

    def _entry(self, doc_type, text='test text'):
        """Build a minimal chunk entry dict for the given document type."""
        return {'text': text, 'type': doc_type, 'source': 's',
                'start_line': 1, 'end_line': 1}

    def test_xlsx_prompt_contains_row_data(self):
        """XLSX type: prompt references row data or spreadsheet context."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('what is the salary', self._entry('xlsx'))
        assert 'row data' in prompt.lower() or 'spreadsheet' in prompt.lower()

    def test_pptx_prompt_contains_slide(self):
        """PPTX type: prompt references slide content."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('test', self._entry('pptx'))
        assert 'slide' in prompt.lower()

    def test_pdf_prompt_contains_page(self):
        """PDF type: prompt references pdf or page."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('test', self._entry('pdf'))
        assert 'pdf' in prompt.lower() or 'page' in prompt.lower()

    def test_docx_prompt_contains_paragraph(self):
        """DOCX type: prompt references paragraph or document."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('test', self._entry('docx'))
        assert 'paragraph' in prompt.lower() or 'document' in prompt.lower()

    def test_html_prompt_contains_webpage(self):
        """HTML type: prompt references webpage or content."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('test', self._entry('html'))
        assert 'webpage' in prompt.lower() or 'content' in prompt.lower()

    def test_md_prompt_contains_markdown(self):
        """Markdown type: prompt references markdown or section."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('test', self._entry('md'))
        assert 'markdown' in prompt.lower() or 'section' in prompt.lower()

    def test_txt_prompt_is_generic(self):
        """TXT type: prompt contains the 1-to-10 scoring scale markers."""
        from src.rag.reranker import rerank_prompt
        prompt = rerank_prompt('test', self._entry('txt'))
        assert '1' in prompt and '10' in prompt

    def test_prompt_ends_with_integer_instruction(self):
        """All 8 types: prompt always instructs the model to respond with a 1-to-10 score."""
        from src.rag.reranker import rerank_prompt
        for dtype in ('xlsx', 'csv', 'pptx', 'pdf', 'docx', 'html', 'md', 'txt'):
            prompt = rerank_prompt('query', self._entry(dtype))
            assert '1 to 10' in prompt or '1-10' in prompt


# ---------------------------------------------------------------------------
# _filter_hallucination
# ---------------------------------------------------------------------------

class TestFilterHallucination:
    """Tests that _filter_hallucination truncates pivot phrases after no-info signals."""

    def test_clean_response_unchanged(self, store):
        """Response with no hallucination markers: returned exactly as-is."""
        response = 'Cats sleep 16 hours a day. [cats.txt L1]'
        assert store._filter_hallucination(response) == response

    def test_truncates_at_however(self, store):
        """No-info phrase followed by 'however,': text after the pivot is removed."""
        response = "I couldn't find info about that. However, I can tell you cats are cool."
        result = store._filter_hallucination(response)
        assert 'however' not in result.lower()

    def test_truncates_at_but_i_can(self, store):
        """No-info phrase followed by 'but i can': text after the pivot is removed."""
        response = "There is no information about it. But I can provide some context."
        result = store._filter_hallucination(response)
        assert 'but i can' not in result.lower()

    def test_no_pivot_returns_full(self, store):
        """No-info phrase with no following pivot: full response is preserved."""
        response = "The provided documents do not contain this information."
        result = store._filter_hallucination(response)
        assert 'provided documents' in result.lower()


# ---------------------------------------------------------------------------
# _build_instruction_prompt
# ---------------------------------------------------------------------------

class TestBuildInstructionPrompt:
    """Tests that _build_instruction_prompt embeds context and anti-hallucination rules."""

    def test_contains_context(self, store):
        """Given context string: it appears verbatim inside the returned prompt."""
        prompt = store._build_instruction_prompt('some context here')
        assert 'some context here' in prompt

    def test_contains_strict_rules(self, store):
        """Prompt includes an instruction limiting the model to provided context only."""
        prompt = store._build_instruction_prompt('ctx')
        assert 'ONLY' in prompt or 'only' in prompt

    def test_contains_source_citation_instruction(self, store):
        """Prompt instructs the model to cite sources in its answer."""
        prompt = store._build_instruction_prompt('ctx')
        assert 'cite' in prompt.lower() or 'source' in prompt.lower()


# build_or_load, add_chunks, rebuild_bm25, clear_conversation,
# _expand_query, _hybrid_retrieve, _rerank, and run_pipeline tests
# are in test_vector_store_pipeline.py (split for 500-line limit).
