"""test_config.py — tests for src/rag/config.py constants.

Verifies that every constant required by the HF Space version exists,
has the correct type, and satisfies sanity constraints.

Mock strategy: none — config.py has no I/O, no functions, no mocking needed.
"""

import os
import sys

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)


class TestConfig:
    """Tests that config.py exports all required constants with valid values.

    Every constant listed in __all__ is verified for type, presence, and
    basic sanity constraints (e.g. thresholds in (0, 1), counts > 0).
    """

    def test_all_constants_present(self):
        """All expected constants exist, are the right types, and have sensible values."""
        from src.rag.config import (
            CHROMA_COLLECTION,
            DOCX_CHUNK_PARAS,
            EMBEDDING_MODEL,
            EXT_TO_TYPE,
            HTML_CHUNK_SENTENCES,
            LANGUAGE_MODEL,
            LANGUAGE_MODEL_FALLBACKS,
            PDF_CHUNK_SENTENCES,
            PPTX_CHUNK_SLIDES,
            SIMILARITY_THRESHOLD,
            TOP_RERANK,
            TOP_RETRIEVE,
            TXT_CHUNK_OVERLAP,
            TXT_CHUNK_SIZE,
        )

        # Model names must be non-empty strings
        assert isinstance(EMBEDDING_MODEL, str) and EMBEDDING_MODEL
        assert isinstance(LANGUAGE_MODEL, str) and LANGUAGE_MODEL
        # Fallback list must contain at least one model name
        assert isinstance(LANGUAGE_MODEL_FALLBACKS, list) and len(LANGUAGE_MODEL_FALLBACKS) >= 1
        # Collection name must match the expected value across all code
        assert CHROMA_COLLECTION == 'rag_docs'
        # Similarity threshold must be a probability in (0, 1)
        assert 0 < SIMILARITY_THRESHOLD < 1
        # Retrieval counts must be positive integers
        assert TOP_RETRIEVE > 0
        assert TOP_RERANK > 0
        # Chunk sizes must be at least 1 line / 0 overlap
        assert TXT_CHUNK_SIZE >= 1
        assert TXT_CHUNK_OVERLAP >= 0

    def test_ext_to_type_covers_all_formats(self):
        """EXT_TO_TYPE maps all 13 expected extensions without gaps."""
        from src.rag.config import EXT_TO_TYPE
        required_exts = [
            '.pdf', '.txt', '.docx', '.doc', '.xlsx', '.xls',
            '.pptx', '.ppt', '.csv', '.md', '.markdown', '.html', '.htm',
        ]
        for ext in required_exts:
            assert ext in EXT_TO_TYPE, f"Missing extension: {ext}"

    def test_fallbacks_list_has_no_gated_models(self):
        """LANGUAGE_MODEL_FALLBACKS contains no gated models that require HF access approval."""
        from src.rag.config import LANGUAGE_MODEL_FALLBACKS

        # These prefixes require accepting licence agreements on HF Hub
        gated_prefixes = ['google/gemma', 'meta-llama/Llama-3.2', 'google/gemma-2']
        for model in LANGUAGE_MODEL_FALLBACKS:
            for prefix in gated_prefixes:
                assert not model.startswith(prefix), (
                    f"Gated model in LANGUAGE_MODEL_FALLBACKS: {model}"
                )

    def test_top_rerank_le_top_retrieve(self):
        """TOP_RERANK must not exceed TOP_RETRIEVE — reranking more than retrieved is nonsensical."""
        from src.rag.config import TOP_RERANK, TOP_RETRIEVE
        assert TOP_RERANK <= TOP_RETRIEVE

    def test_chunk_sizes_positive(self):
        """All per-format chunk size constants must be at least 1."""
        from src.rag.config import DOCX_CHUNK_PARAS, HTML_CHUNK_SENTENCES, PDF_CHUNK_SENTENCES, PPTX_CHUNK_SLIDES
        assert PDF_CHUNK_SENTENCES >= 1
        assert DOCX_CHUNK_PARAS >= 1
        assert PPTX_CHUNK_SLIDES >= 1
        assert HTML_CHUNK_SENTENCES >= 1

    def test_ext_to_type_values_are_known_types(self):
        """Every value in EXT_TO_TYPE is one of the 8 canonical type strings."""
        from src.rag.config import EXT_TO_TYPE
        known_types = {'pdf', 'txt', 'docx', 'xlsx', 'pptx', 'csv', 'md', 'html'}
        for ext, type_key in EXT_TO_TYPE.items():
            assert type_key in known_types, (
                f"Extension {ext!r} maps to unknown type {type_key!r}"
            )
