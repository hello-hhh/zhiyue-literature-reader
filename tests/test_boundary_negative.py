"""test_boundary_negative.py — Boundary and Negative tests for the local RAG suite.

Boundary tests exercise edge cases at input limits:
  - XLSX with only a header row → []
  - DOCX with zero paragraphs and no tables → []
  - CSV with only a header row → []
  - _chunk_txt with only blank lines → []
  - Agent run with max_steps=1 still returns a result dict
  - VectorStore run_pipeline with a single chunk in the store

Negative tests verify wrong/empty inputs are handled gracefully:
  - chunk_url with HTTP 404 (raise_for_status raises) → []
  - _tool_calculator('') → error string, no exception
  - _tool_sentiment('') → string, no exception
  - _chunk_csv with all-empty-value rows → list, no crash

Mock strategy (per CLAUDE.md):
  ollama.embed  → {'embeddings': [[0.1, ...]]}
  ollama.chat   → {'message': {'content': 'mock'}}
  chromadb      → patched via chromadb.PersistentClient + EphemeralClient
  requests.get  → Mock with .raise_for_status() raising HTTPError

Never mock: fitz, python-docx, openpyxl, xlrd, BM25Okapi, calculator eval.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

from src.rag.binary_chunkers import chunk_docx, chunk_xlsx
from src.rag.chunkers import chunk_csv, chunk_txt

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader():
    """Provide a fresh DocumentLoader instance for each test."""
    from src.rag.document_loader import DocumentLoader
    return DocumentLoader()


@pytest.fixture
def mock_store():
    """Build a MagicMock VectorStore pre-wired with one cat-facts chunk."""
    from src.rag.vector_store import VectorStore
    store = MagicMock(spec=VectorStore)
    store.chunks = [
        {'text': 'Cats sleep 16 hours a day.', 'source': 'cats.txt',
         'start_line': 1, 'end_line': 1, 'type': 'txt'},
    ]
    store.bm25_index = BM25Okapi([['cats', 'sleep', '16', 'hours']])
    store.collection = MagicMock()
    store.collection.count.return_value = 1
    store._expand_query.return_value = ['cats']
    entry = store.chunks[0]
    store._hybrid_retrieve.return_value = [(entry, 0.9)]
    store._rerank.return_value = [(entry, 0.9, 0.9)]
    store._source_label.return_value = 'L1-1'
    return store


@pytest.fixture
def agent(mock_store):
    """Construct an Agent instance wired to mock_store."""
    from src.rag.agent import Agent
    return Agent(mock_store)


def _fake_embed(dim: int = 4):
    """Return a minimal ollama.embed-shaped response dict."""
    return {'embeddings': [[0.1] * dim]}


def _fake_chat(content: str = 'mock response'):
    """Return a minimal ollama.chat-shaped response dict."""
    return {'message': {'content': content}}


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDARY — Empty / header-only file inputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryEmptyFiles:
    """Boundary: chunkers must return [] for files that have no data content.

    These test zero-data inputs that are still structurally valid files — e.g.
    an XLSX with only a header row, a DOCX with no paragraphs, or a CSV with
    only a header.  The chunkers must not crash and must return an empty list.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a fresh DocumentLoader for each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_xlsx_header_only_returns_empty(self) -> None:
        """XLSX with only a header row (no data rows) → returns []."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['Name', 'Age', 'City'])   # header only, no data rows
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as f:
            tmp_path = f.name
        try:
            wb.save(tmp_path)
            chunks = chunk_xlsx(tmp_path, 'header_only.xlsx')
            assert chunks == [], (
                f"Expected [] for header-only XLSX, got {len(chunks)} chunks"
            )
        finally:
            os.unlink(tmp_path)

    def test_docx_no_paragraphs_no_tables_returns_empty(self) -> None:
        """DOCX with zero paragraphs and no tables → returns []."""
        from docx import Document
        doc = Document()
        # A freshly created Document has one empty paragraph by default;
        # we clear it to simulate a truly empty document.
        for para in list(doc.paragraphs):
            p = para._element
            p.getparent().remove(p)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as f:
            tmp_path = f.name
        try:
            doc.save(tmp_path)
            chunks = chunk_docx(tmp_path, 'empty.docx')
            assert chunks == [], (
                f"Expected [] for empty DOCX, got {len(chunks)} chunks"
            )
        finally:
            os.unlink(tmp_path)

    def test_csv_header_only_returns_empty(self) -> None:
        """CSV with only a header row, no data rows → returns []."""
        content = "name,age,city\n"   # header only
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.csv', mode='w', encoding='utf-8'
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            chunks = chunk_csv(tmp_path, 'header_only.csv')
            assert chunks == [], (
                f"Expected [] for header-only CSV, got {len(chunks)} chunks"
            )
        finally:
            os.unlink(tmp_path)

    def test_txt_blank_lines_only_returns_empty(self) -> None:
        """TXT file containing only blank lines → returns []."""
        content = "\n\n\n   \n\t\n"   # whitespace and blank lines only
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.txt', mode='w', encoding='utf-8'
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            chunks = chunk_txt(tmp_path, 'blanks.txt')
            assert chunks == [], (
                f"Expected [] for blank-lines-only TXT, got {len(chunks)} chunks"
            )
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDARY — Agent with max_steps=1
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryAgentMaxStepsOne:
    """Boundary: Agent with max_steps=1 must still return a valid result dict."""

    def test_max_steps_1_returns_result_dict(self, agent) -> None:
        """Agent with max_steps=1 returns a dict with 'answer' and 'steps' keys."""
        agent.max_steps = 1
        # Any chat response that isn't a valid TOOL: line will exhaust the step budget
        with patch('ollama.chat', return_value=_fake_chat('not a tool call')):
            result = agent.run('what do cats eat')
        assert isinstance(result, dict), "Agent.run() must always return a dict"
        assert 'answer' in result, "Result dict must contain 'answer' key"
        assert 'steps' in result, "Result dict must contain 'steps' key"


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDARY — VectorStore pipeline with a single chunk
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryVectorStoreSingleChunk:
    """Boundary: run_pipeline must succeed when the store contains only one chunk.

    With a single chunk the BM25 index, dense retrieval, and reranking path
    all face edge-case inputs (one-element lists).  We use a unique ChromaDB
    collection name to prevent dimension conflicts with other tests in this
    process that use a different embed dimension.
    """

    def test_single_chunk_pipeline_returns_response(self) -> None:
        """run_pipeline with one chunk in the store returns a dict with 'response'."""
        import uuid

        import chromadb

        from src.rag.vector_store import VectorStore

        chunks = [
            {'text': 'Cats sleep 16 hours a day.',
             'source': 'cats.txt', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        ]

        def _fake_chat_smart(*args, **kwargs):
            # stream=True path returns a list; otherwise returns a dict
            if kwargs.get('stream'):
                return [{'message': {'content': 'Cats sleep 16 hours.'}}]
            return {'message': {'content': '1'}}   # reranker score

        # Use a unique collection name so the EphemeralClient singleton's existing
        # collections (created by test_vector_store.py with dim=4) are not reused.
        unique_name = f'test_boundary_{uuid.uuid4().hex}'
        with patch('ollama.embed', return_value=_fake_embed(4)), \
             patch('ollama.chat', side_effect=_fake_chat_smart), \
             patch('src.rag.vector_store.CHROMA_COLLECTION', unique_name), \
             patch('chromadb.PersistentClient') as mock_pc:
            # Wire PersistentClient to return a fresh EphemeralClient collection
            ephemeral = chromadb.EphemeralClient()
            collection = ephemeral.get_or_create_collection(
                unique_name, metadata={'hnsw:space': 'cosine'}
            )
            mock_pc.return_value.get_or_create_collection.return_value = collection
            vs = VectorStore()
            vs.build_or_load(chunks)
            result = vs.run_pipeline('How long do cats sleep?')

        assert isinstance(result, dict), "run_pipeline must return a dict"
        assert 'response' in result, "Result dict must contain 'response'"


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — chunk_url with HTTP 404
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeChunkUrl404:
    """Negative: chunk_url must return [] when requests.get raises an HTTPError."""

    def test_http_404_returns_empty_list(self, loader) -> None:
        """chunk_url with a 404 response (raise_for_status raises) → returns []."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error: Not Found"
        )

        with patch('requests.get', return_value=mock_response):
            result = loader.chunk_url('http://example.com/missing.html')

        assert result == [], (
            f"Expected [] for 404 URL, got {result!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — _tool_calculator with empty string
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeCalculatorEmpty:
    """Negative: _tool_calculator('') must return an error string, not crash.

    The allowed-chars whitelist rejects empty input before any eval() is called,
    so no exception should propagate.
    """

    def test_empty_string_returns_error_string(self, agent) -> None:
        """_tool_calculator with empty string returns a non-empty error string."""
        result = agent._tool_calculator('')
        assert isinstance(result, str), "_tool_calculator must always return a string"
        assert len(result) > 0, "_tool_calculator must return a non-empty error message"

    def test_empty_string_does_not_raise(self, agent) -> None:
        """_tool_calculator('') must not raise any exception."""
        try:
            agent._tool_calculator('')
        except Exception as exc:
            pytest.fail(f"_tool_calculator('') raised {type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — _tool_sentiment with empty string
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeSentimentEmpty:
    """Negative: _tool_sentiment('') must return a string, not crash."""

    def test_empty_string_returns_string(self, agent) -> None:
        """_tool_sentiment with empty string returns a string."""
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Neutral')):
            result = agent._tool_sentiment('')
        assert isinstance(result, str), "_tool_sentiment must always return a string"

    def test_empty_string_does_not_raise(self, agent) -> None:
        """_tool_sentiment('') must not raise any exception."""
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Neutral')):
            try:
                agent._tool_sentiment('')
            except Exception as exc:
                pytest.fail(
                    f"_tool_sentiment('') raised {type(exc).__name__}: {exc}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — _chunk_csv with all-empty-value rows
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeCsvAllEmpty:
    """Negative: _chunk_csv with rows that have only empty/whitespace values is handled gracefully."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a fresh DocumentLoader for each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_all_empty_values_returns_list_without_crashing(self) -> None:
        """CSV where all data cells are empty → returns a list (possibly []) without crashing."""
        content = "name,age,city\n,,\n  ,  ,  \n"   # header + two blank-value rows
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.csv', mode='w', encoding='utf-8'
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            chunks = chunk_csv(tmp_path, 'empty_values.csv')
            # Must be a list (not crash); empty or non-empty both acceptable here
            assert isinstance(chunks, list), (
                "_chunk_csv must return a list even for all-empty-value rows"
            )
        finally:
            os.unlink(tmp_path)
