"""test_combinations_analysis.py — Parametrized analysis tests for the HF Space.

Split from test_combinations.py to keep each file under 500 lines (per CLAUDE.md).

Covers:
  - TestTruncationCombinations:          300-word / 1200-char boundary cases
  - TestQueryClassificationCombinations: all 4 query type labels
  - TestSourceLabelCombinations:         label format for all 8 doc types

HF differences:
  - conftest.py autouse fixtures patch _get_st_model and _llm_call globally.
  - No LLM or embedding calls are needed by any test in this file — pure logic.
"""

import os
import sys

import pytest

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

from src.rag.chunkers import truncate_chunk

# ---------------------------------------------------------------------------
# 1. Truncation boundary × parametrize
# ---------------------------------------------------------------------------

class TestTruncationCombinations:
    """Parametrized truncation tests covering all boundary cases.

    _truncate_chunk enforces two independent limits:
      - 300-word maximum
      - 1200-character maximum
    The shorter of the two applies.
    """

    @pytest.mark.parametrize("text,max_words,max_chars", [
        # Over word limit — 400 words should be cut to ≤ 300
        (" ".join(["word"] * 400),    300, 99999),
        # Over char limit — 50 × 30-char tokens ≈ 1550 chars, cut to ≤ 1200
        (" ".join(["a" * 30] * 50),  9999,  1200),
        # Under both limits — returned unchanged
        ("short text here",          9999, 99999),
        # Exactly at word limit — should be returned as-is (not truncated)
        (" ".join(["word"] * 300),    300, 99999),
    ])
    def test_truncate_chunk(self, text: str, max_words: int, max_chars: int) -> None:
        """_truncate_chunk respects 300-word and 1200-char limits.

        Args:
            text:      Input text to truncate.
            max_words: Result word count must be ≤ this.
            max_chars: Result char count must be ≤ this.
        """
        result = truncate_chunk(text)
        assert len(result.split()) <= max_words, (
            f"Word count {len(result.split())} exceeds limit {max_words}"
        )
        assert len(result) <= max_chars, (
            f"Char count {len(result)} exceeds limit {max_chars}"
        )


# ---------------------------------------------------------------------------
# 2. Query classification × all 4 types
# ---------------------------------------------------------------------------

class TestQueryClassificationCombinations:
    """_classify_query returns one of four valid labels for representative queries."""

    @pytest.fixture
    def store(self):
        """Minimal VectorStore for classification-only tests — no chunks needed."""
        from src.rag.vector_store import VectorStore
        return VectorStore()

    @pytest.mark.parametrize("query,expected_type", [
        ("summarise all documents please",         "summarise"),
        ("summarize the key findings",             "summarise"),
        # "tell me about" is in the summarise keyword list → classified as summarise
        ("tell me about animals",                  "summarise"),
        ("compare cats and dogs",                  "comparison"),
        ("what is the difference between X and Y", "comparison"),
        ("how many hours do cats sleep",           "factual"),
        ("what is the capital of France",          "factual"),
        ("what do you know about cats",            "general"),
    ])
    def test_query_type(self, store, query: str, expected_type: str) -> None:
        """_classify_query returns expected_type for each representative query.

        Args:
            query:         Representative user query string.
            expected_type: Expected classification label.
        """
        result = store._classify_query(query)
        assert result == expected_type, (
            f"Query {query!r}: expected {expected_type!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# 3. Source label × all doc types
# ---------------------------------------------------------------------------

class TestSourceLabelCombinations:
    """_source_label produces correct format prefix for every document type."""

    @pytest.fixture
    def store(self):
        """Minimal VectorStore for source-label tests."""
        from src.rag.vector_store import VectorStore
        return VectorStore()

    @pytest.mark.parametrize("doc_type,start_line,end_line,expected_prefix", [
        ("pdf",   3,  3, "p"),      # page number label
        ("xlsx",  5,  5, "row"),    # row number label
        ("csv",   2,  2, "row"),    # row number label
        ("pptx",  1,  1, "slide"),  # slide number label
        ("html",  4,  8, "s"),      # sentence window label
        ("txt",   1, 10, "L"),      # line range label
        ("docx",  2,  4, "L"),      # paragraph range label
        ("md",    1,  3, "L"),      # line range label
    ])
    def test_source_label_format(self, store, doc_type: str, start_line: int,
                                  end_line: int, expected_prefix: str) -> None:
        """_source_label returns a non-empty label with the expected prefix.

        Args:
            doc_type:         Document type value in the entry dict.
            start_line:       start_line in the entry dict.
            end_line:         end_line in the entry dict.
            expected_prefix:  Required leading characters for the label.
        """
        entry = {
            'text':       'Some text here.',
            'source':     'test.txt',
            'start_line': start_line,
            'end_line':   end_line,
            'type':       doc_type,
        }
        label = store._source_label(entry)
        assert isinstance(label, str), "source label must be a str"
        assert len(label) > 0,         "source label must be non-empty"
        assert label.startswith(expected_prefix), (
            f"doc_type={doc_type!r}: expected label starting with "
            f"{expected_prefix!r}, got {label!r}"
        )
