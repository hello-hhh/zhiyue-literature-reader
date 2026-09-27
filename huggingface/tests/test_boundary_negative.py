"""test_boundary_negative.py — Boundary and Negative tests for the HF Space suite.

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

HF differences from local:
  - VectorStore uses sentence-transformers (_get_st_model) + _llm_call instead of ollama.
  - Agent calls store._llm_chat() instead of ollama.chat().
  - conftest.py autouse fixtures patch _get_st_model and _llm_call globally.

Mock strategy:
  - conftest.py autouse fixtures handle _get_st_model and _llm_call.
  - requests.get mocked via patch where needed.
  - store._llm_chat patched via patch.object for Agent tests.
  - Never mock: fitz, python-docx, openpyxl, xlrd, BM25Okapi, calculator eval.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

from src.rag.binary_chunkers import chunk_docx, chunk_xlsx
from src.rag.chunkers import chunk_csv, chunk_txt
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
    from src.rag.agent import Agent
    store = make_store_with_chunks(sample_chunks(n_chunks))
    return Agent(store)


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
    """Boundary: Agent with max_steps=1 must still return a valid result dict.

    The HF Agent uses store._llm_chat rather than ollama.chat, so we patch
    that method on the store object rather than at the ollama module level.
    """

    def test_max_steps_1_returns_result_dict(self) -> None:
        """Agent with max_steps=1 returns a dict with 'answer' and 'steps' keys."""
        agent = _make_agent(n_chunks=1)
        agent.max_steps = 1

        # Patch _llm_chat on the agent's store so any call returns a non-tool response;
        # this exhausts the one available step without triggering a tool dispatch.
        with patch.object(agent.store, '_llm_chat', return_value='not a tool call'):
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
    all face edge-case inputs (one-element lists).  The conftest autouse patches
    handle _get_st_model and _llm_call so no Ollama or HF Inference call is made.
    """

    def test_single_chunk_pipeline_returns_response(self) -> None:
        """run_pipeline with one chunk returns a dict containing 'response'."""
        chunks = [
            {'text': 'Cats sleep 16 hours a day.',
             'source': 'cats.txt', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        ]
        # _llm_call is patched by conftest to return a safe mock string.
        # We also patch _llm_call for the reranker branch which needs a numeric string.
        with patch('src.rag.vector_store._llm_call', return_value='1'):
            store = make_store_with_chunks(chunks)
            result = store.run_pipeline('How long do cats sleep?')

        assert isinstance(result, dict), "run_pipeline must return a dict"
        assert 'response' in result, "Result dict must contain 'response'"


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — chunk_url with HTTP 404
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeChunkUrl404:
    """Negative: chunk_url must return [] when requests.get raises an HTTPError."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a fresh DocumentLoader for each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_http_404_returns_empty_list(self) -> None:
        """chunk_url with a 404 response (raise_for_status raises) → returns []."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error: Not Found"
        )

        with patch('requests.get', return_value=mock_response):
            result = self.loader.chunk_url('http://example.com/missing.html')

        assert result == [], (
            f"Expected [] for 404 URL, got {result!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — _tool_calculator with empty string
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeCalculatorEmpty:
    """Negative: _tool_calculator('') must return an error string, not crash.

    The HF Agent shares the same _tool_calculator implementation as local;
    the allowed-chars whitelist rejects empty input before any eval() is called.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a minimal Agent for each test."""
        self.agent = _make_agent(n_chunks=1)

    def test_empty_string_returns_error_string(self) -> None:
        """_tool_calculator with empty string returns a non-empty error string."""
        result = self.agent._tool_calculator('')
        assert isinstance(result, str), "_tool_calculator must always return a string"
        assert len(result) > 0, "_tool_calculator must return a non-empty error message"

    def test_empty_string_does_not_raise(self) -> None:
        """_tool_calculator('') must not raise any exception."""
        try:
            self.agent._tool_calculator('')
        except Exception as exc:
            pytest.fail(f"_tool_calculator('') raised {type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATIVE — _tool_sentiment with empty string
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeSentimentEmpty:
    """Negative: _tool_sentiment('') must return a string, not crash.

    In the HF version the agent calls store._llm_chat for sentiment analysis.
    We patch _llm_chat on the store so the test does not hit the HF Inference API.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a minimal Agent and pre-patch _llm_chat for each test."""
        self.agent = _make_agent(n_chunks=1)

    def test_empty_string_returns_string(self) -> None:
        """_tool_sentiment with empty string returns a string."""
        with patch.object(self.agent.store, '_llm_chat', return_value='Sentiment: Neutral'):
            result = self.agent._tool_sentiment('')
        assert isinstance(result, str), "_tool_sentiment must always return a string"

    def test_empty_string_does_not_raise(self) -> None:
        """_tool_sentiment('') must not raise any exception."""
        with patch.object(self.agent.store, '_llm_chat', return_value='Sentiment: Neutral'):
            try:
                self.agent._tool_sentiment('')
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
