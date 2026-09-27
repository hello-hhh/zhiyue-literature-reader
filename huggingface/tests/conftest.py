"""conftest.py — global fixtures for HF Space test suite.

Patches sentence-transformers and _llm_call globally so tests don't need
sentence_transformers installed locally and don't hit the HF Inference API.

Shared helpers (_sample_chunks, _make_store_with_chunks) live here so they
are importable by all four test modules without duplication.
"""

import hashlib
import os
import random
import sys
from unittest.mock import MagicMock, patch

import pytest

HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

FAKE_DIM = 384


class _FakeVector(list):
    """List subclass with a .tolist() method to satisfy any code that calls it."""

    def tolist(self):
        """Return a plain list copy of this vector."""
        return list(self)


def _fake_embed(text, normalize_embeddings=True):
    """Produce a deterministic unit-normalised 384-dim vector from the input text.

    The seed is derived from the MD5 hash of the text so that the same text
    always produces the same vector, making tests reproducible.
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(FAKE_DIM)]
    norm = sum(x**2 for x in vec) ** 0.5 or 1.0
    return _FakeVector(x / norm for x in vec)


def _mock_st_model():
    """Return a MagicMock that mimics a SentenceTransformer with _fake_embed as encode."""
    m = MagicMock()
    m.encode.side_effect = _fake_embed
    return m


# ── Shared test helpers ───────────────────────────────────────────────────────

def sample_chunks(n=5, doc_type='txt'):
    """Return *n* minimal chunk dicts suitable for use as test fixtures.

    Args:
        n:        Number of chunks to generate.
        doc_type: Value for the 'type' key in every chunk.

    Returns:
        List of chunk dicts with keys: text, source, start_line, end_line, type.
    """
    return [
        {
            'text': f"Cats sleep about 16 hours a day chunk {i}.",
            'source': 'cats.txt',
            'start_line': i + 1,
            'end_line': i + 1,
            'type': doc_type,
        }
        for i in range(n)
    ]


def make_store_with_chunks(chunks):
    """Build a VectorStore with fake sentence-transformer embeddings for the given chunks.

    The module-level _llm_call and _get_st_model patches from the autouse fixtures
    are already active when this helper is called, so no additional patching is needed.

    Args:
        chunks: List of chunk dicts passed to VectorStore.build_or_load.

    Returns:
        A fully initialised VectorStore instance backed by an in-memory collection.
    """
    from src.rag.vector_store import VectorStore
    store = VectorStore()
    store.build_or_load(chunks)
    return store


# ── Auto-use fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_st_model():
    """Patch sentence-transformers globally for every test.

    Replaces _get_st_model with a mock whose .encode() returns deterministic
    384-dim unit vectors — no sentence-transformers install required.
    """
    with patch('src.rag.vector_store._get_st_model', return_value=_mock_st_model()):
        yield


@pytest.fixture(autouse=True)
def patch_llm_call(request):
    """Return a safe mock string from _llm_call for every test.

    Tests in TestLlmCallFallback are excluded from this patch so they can
    exercise the real fallback logic against a mocked InferenceClient.
    """
    if request.node.cls and request.node.cls.__name__ == 'TestLlmCallFallback':
        yield   # no patch — let the real _llm_call run
        return
    with patch('src.rag.vector_store._llm_call', return_value="mock rewrite\nalternative rewrite"):
        yield


@pytest.fixture(autouse=True)
def unique_chroma_collection(monkeypatch):
    """Use a unique ChromaDB collection name per test to prevent cross-test state.

    Each test gets a UUID-based collection name so VectorStore instances created
    within the test cannot accidentally share data with other tests.
    """
    import uuid
    monkeypatch.setattr('src.rag.vector_store.CHROMA_COLLECTION', f'test_{uuid.uuid4().hex}')
    # Also patch config so new VectorStore instances pick up the unique name
    yield
