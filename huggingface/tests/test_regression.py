"""test_regression.py — Regression tests for the HF Space suite.

Regression tests lock down exact content that must never drift across refactors:
  - Hallucination filter phrase lists (_no_info_phrases, _hallucination_pivots)
  - AGENT_SYSTEM_PROMPT key rules and tool names
  - _rerank_prompt: all 7 doc-type variants contain a type-specific keyword
    and a numeric scale

Boundary and negative tests are in test_boundary_negative.py.

HF differences from local:
  - VectorStore uses sentence-transformers (_get_st_model) + _llm_call instead of ollama.
  - VectorStore uses chromadb.EphemeralClient (already in-memory — no PersistentClient).
  - Agent calls store._llm_chat() instead of ollama.chat().
  - conftest.py autouse fixtures patch _get_st_model and _llm_call globally.

Mock strategy:
  - conftest.py autouse fixtures handle _get_st_model and _llm_call globally.
  - Never mock: BM25Okapi, _truncate_chunk.
"""

import os
import sys

import pytest

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION — Hallucination filter phrase lists
# ═══════════════════════════════════════════════════════════════════════════════

class TestHallucinationFilterPhrases:
    """Regression: exact phrase lists in _filter_hallucination must never drift.

    These lists are critical for suppressing hallucinated continuations after
    the model admits it could not find information.  Any accidental deletion or
    typo would silently break the filter, so we lock down the exact strings here.

    HF note: same logic as local — both versions share identical phrase lists.
    """

    # The complete expected lists as documented in CLAUDE.md and copied verbatim
    # from huggingface/src/rag/vector_store.py.
    EXPECTED_NO_INFO = [
        "there is no information",
        "i couldn't find",
        "i could not find",
        "the provided context does not",
        "the provided documents do not",
        "no information in the provided",
        "not mentioned in the",
        "not found in the",
    ]

    EXPECTED_PIVOTS = [
        "however,",
        "but i can",
        "but,",
        "that said,",
        "nevertheless,",
        "i can tell you",
        "i can provide",
    ]

    def _get_source(self) -> str:
        """Return the source of _filter_hallucination from the HF VectorStore module."""
        import inspect

        from src.rag import vector_store as vs_module
        return inspect.getsource(vs_module.VectorStore._filter_hallucination)

    def test_all_no_info_phrases_present(self) -> None:
        """Every expected no-info phrase appears verbatim in _filter_hallucination source."""
        src = self._get_source()
        for phrase in self.EXPECTED_NO_INFO:
            assert phrase in src, (
                f"Missing no-info phrase: {phrase!r} — "
                "do not remove or alter _no_info_phrases in huggingface/src/rag/vector_store.py"
            )

    def test_all_pivot_phrases_present(self) -> None:
        """Every expected pivot phrase appears verbatim in _filter_hallucination source."""
        src = self._get_source()
        for phrase in self.EXPECTED_PIVOTS:
            assert phrase in src, (
                f"Missing pivot phrase: {phrase!r} — "
                "do not remove or alter _hallucination_pivots in huggingface/src/rag/vector_store.py"
            )

    def test_no_info_phrase_count(self) -> None:
        """Exactly 8 no-info phrases are defined."""
        src = self._get_source()
        found = [p for p in self.EXPECTED_NO_INFO if p in src]
        assert len(found) == 8, (
            f"Expected 8 no-info phrases, found {len(found)}. "
            "Check _no_info_phrases in huggingface/src/rag/vector_store.py."
        )

    def test_pivot_phrase_count(self) -> None:
        """Exactly 7 pivot phrases are defined."""
        src = self._get_source()
        found = [p for p in self.EXPECTED_PIVOTS if p in src]
        assert len(found) == 7, (
            f"Expected 7 pivot phrases, found {len(found)}. "
            "Check _hallucination_pivots in huggingface/src/rag/vector_store.py."
        )

    def test_filter_truncates_on_no_info_plus_pivot(self) -> None:
        """_filter_hallucination truncates a response that admits no-info then pivots.

        This is a functional regression test — if the filter logic changes, this
        test will catch it before any phrase-list drift does damage.
        The conftest autouse patches keep _llm_call safe.
        """
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        resp = (
            "There is no information in the provided documents. "
            "However, I can tell you that cats are mammals."
        )
        result = store._filter_hallucination(resp)
        # "However," is a pivot — the rest of the sentence should be dropped
        assert "I can tell you that cats are mammals" not in result
        assert "There is no information" in result

    def test_filter_leaves_clean_response_unchanged(self) -> None:
        """_filter_hallucination returns a clean response unchanged."""
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        resp = "Cats sleep approximately 16 hours per day."
        result = store._filter_hallucination(resp)
        assert result == resp


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION — Agent system prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentSystemPromptRegression:
    """Regression: AGENT_SYSTEM_PROMPT key rules must appear verbatim.

    The prompt drives the entire ReAct loop.  Losing 'TOOL:' or 'finish' would
    break every tool-call parse.  These tests ensure no accidental edits slip
    through — they apply equally to the HF Space version.
    """

    REQUIRED_SUBSTRINGS = [
        "TOOL:",
        "finish",
        "calculator",
        "rag_search",
        "summarise",
        "sentiment",
        "Never write anything except a single TOOL: line",
        "Always end with TOOL: finish",
        "Use rag_search first",
    ]

    @pytest.fixture(autouse=True)
    def _get_prompt(self) -> None:
        """Cache AGENT_SYSTEM_PROMPT for all tests in this class."""
        from src.rag.agent import Agent
        self.prompt = Agent.AGENT_SYSTEM_PROMPT

    def test_prompt_is_non_empty_string(self) -> None:
        """AGENT_SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(self.prompt, str)
        assert len(self.prompt) > 100

    def test_all_required_substrings_present(self) -> None:
        """Every required substring appears verbatim in AGENT_SYSTEM_PROMPT."""
        for sub in self.REQUIRED_SUBSTRINGS:
            assert sub in self.prompt, (
                f"Required string missing from AGENT_SYSTEM_PROMPT: {sub!r}"
            )

    def test_all_5_tool_names_listed(self) -> None:
        """All five tool names appear in the prompt."""
        for tool in ('rag_search', 'calculator', 'summarise', 'sentiment', 'finish'):
            assert tool in self.prompt, (
                f"Tool name {tool!r} missing from AGENT_SYSTEM_PROMPT"
            )

    def test_tool_format_line_present(self) -> None:
        """The exact TOOL: format line appears so the model knows the response format."""
        assert "TOOL: tool_name(your argument here)" in self.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION — Rerank prompt variants
# ═══════════════════════════════════════════════════════════════════════════════

class TestRerankPromptRegression:
    """Regression: rerank_prompt must produce a type-specific keyword and a numeric scale.

    Seven doc-type branches exist (pdf, docx, xlsx/csv, pptx, html, md, txt/default).
    Each prompt must mention a domain word unique to that type AND include the
    phrase '1 to 10'.  The HF version shares identical prompt text with local.

    rerank_prompt is a module-level function in src/rag/reranker.py — no VectorStore
    instance is needed.
    """

    # (doc_type, expected_keyword, description)
    VARIANTS = [
        ('pdf',   'PDF',          'PDF page extract'),
        ('docx',  'paragraph',    'document paragraph'),
        ('xlsx',  'spreadsheet',  'spreadsheet row'),
        ('csv',   'spreadsheet',  'csv uses same xlsx branch'),
        ('pptx',  'slide',        'presentation slide'),
        ('html',  'webpage',      'webpage content'),
        ('md',    'markdown',     'markdown document section'),
        ('txt',   '1-10',         'generic txt/fallback prompt'),
    ]

    def _make_entry(self, doc_type: str) -> dict:
        """Return a minimal chunk entry dict for the given doc_type."""
        return {
            'text': 'Sample text for reranking.',
            'source': f'doc.{doc_type}',
            'type': doc_type,
        }

    def test_each_variant_has_numeric_scale(self) -> None:
        """Every doc-type variant contains '1 to 10' to indicate the scoring scale."""
        from src.rag.reranker import rerank_prompt
        for doc_type, _, _ in self.VARIANTS:
            entry = self._make_entry(doc_type)
            prompt = rerank_prompt('test query', entry)
            assert '1 to 10' in prompt, (
                f"Numeric scale '1 to 10' missing from {doc_type!r} rerank prompt"
            )

    @pytest.mark.parametrize("doc_type,keyword,desc", VARIANTS)
    def test_variant_contains_type_keyword(
        self, doc_type: str, keyword: str, desc: str
    ) -> None:
        """Each doc-type prompt contains its expected type-specific keyword."""
        from src.rag.reranker import rerank_prompt
        entry = self._make_entry(doc_type)
        prompt = rerank_prompt('test query', entry)
        assert keyword in prompt, (
            f"Expected keyword {keyword!r} not found in {doc_type!r} rerank prompt. "
            f"Description: {desc}"
        )

    def test_query_embedded_in_all_prompts(self) -> None:
        """The original user query appears in every rerank prompt variant."""
        from src.rag.reranker import rerank_prompt
        query = 'unique_sentinel_query_42'
        for doc_type, _, _ in self.VARIANTS:
            entry = self._make_entry(doc_type)
            prompt = rerank_prompt(query, entry)
            assert query in prompt, (
                f"User query not embedded in {doc_type!r} rerank prompt"
            )
