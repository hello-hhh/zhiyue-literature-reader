"""test_doc_types_and_modes.py — Chat mode tests for all 8 document types
plus pipeline features (confidence, reranking, memory, source labels).

Mock strategy:
  Always mock: ollama.embed, ollama.chat, chromadb → EphemeralClient
  Never mock:  BM25Okapi, calculator eval, chunk parsing

Reason for split: max 500 lines per file per CLAUDE.md.
Agent mode tests (agent tools, bad format recovery) are in
test_doc_types_agent.py.
"""

import os
import uuid
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_embed(**kwargs):
    """Return a minimal fixed embedding vector for any input text."""
    return {'embeddings': [[0.1, 0.2, 0.3, 0.4]]}


def _fake_chat(**kwargs):
    """Return streaming list or plain dict depending on stream= kwarg."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': 'mock response'}}


def _make_store(chunks):
    """Build a VectorStore with EphemeralClient loaded with given chunks."""
    from src.rag.vector_store import VectorStore
    vs         = VectorStore()
    client     = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        f'test_{uuid.uuid4().hex}', metadata={'hnsw:space': 'cosine'}
    )
    if chunks:
        ids    = [f'c{i}' for i in range(len(chunks))]
        texts  = [c['text'] for c in chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c['type']} for c in chunks]
        embeds = [[0.1, 0.2, 0.3, 0.4]] * len(chunks)
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    vs.collection          = collection
    vs.chunks              = chunks
    vs.bm25_index          = BM25Okapi([c['text'].lower().split() for c in chunks]) if chunks else None
    vs.conversation_history = []
    return vs


def _chunk(text, source, doc_type, line=1):
    """Build a minimal chunk dict with all required fields for VectorStore."""
    return {'text': text, 'source': source,
            'start_line': line, 'end_line': line, 'type': doc_type}


PATCHES = [
    patch('ollama.embed',  side_effect=_fake_embed),
    patch('ollama.chat',   side_effect=_fake_chat),
]


def _apply_patches(func):
    """Decorator to apply ollama patches to a test method."""
    for p in reversed(PATCHES):
        func = p(func)
    return func


# ---------------------------------------------------------------------------
# 1. Chat mode — all 8 document types
# ---------------------------------------------------------------------------

class TestChatModeTxt:
    """Chat pipeline tests using plain-text (.txt) document chunks."""

    @_apply_patches
    def test_chat_txt_factual(self, mock_chat, mock_embed):
        """Factual query against a txt chunk: response key is present and non-empty."""
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert 'response' in result
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    @_apply_patches
    def test_chat_txt_query_type_detected(self, mock_chat, mock_embed):
        """Factual query against a txt chunk: query_type is one of the four known types."""
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert result.get('query_type') in ('factual', 'general', 'comparison', 'summarise')

    @_apply_patches
    def test_chat_txt_source_cited(self, mock_chat, mock_embed):
        """Factual query against a txt chunk: retrieved list includes the source filename."""
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert 'retrieved' in result
        assert any('dog-facts.txt' in e['source'] for e, _ in result['retrieved'])


class TestChatModePdf:
    """Chat pipeline tests using PDF document chunks."""

    @_apply_patches
    def test_chat_pdf_factual(self, mock_chat, mock_embed):
        """Factual query against a pdf chunk: response key is present and non-empty."""
        chunks = [_chunk('Bananas contain 89 calories per 100 grams.', 'nutrition.pdf', 'pdf')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many calories are in bananas?', streamlit_mode=True)
        assert 'response' in result
        assert len(result['response']) > 0

    @_apply_patches
    def test_chat_pdf_source_label(self, mock_chat, mock_embed):
        """PDF chunk source label: contains page indicator 'p' or generic 'L'."""
        chunks = [_chunk('Salmon is rich in omega-3 fatty acids.', 'nutrition.pdf', 'pdf')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'p' in label or 'L' in label


class TestChatModeDocx:
    """Chat pipeline tests using Word document (.docx) chunks."""

    @_apply_patches
    def test_chat_docx_factual(self, mock_chat, mock_embed):
        """Factual query against a docx chunk: response key is present in pipeline result."""
        chunks = [_chunk('Python was created by Guido van Rossum in 1991.', 'resume.docx', 'docx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_docx_reranked(self, mock_chat, mock_embed):
        """Two docx chunks: pipeline result includes a reranked list."""
        chunks = [_chunk('Python was created by Guido van Rossum in 1991.', 'resume.docx', 'docx'),
                  _chunk('Java was created by James Gosling in 1995.', 'resume.docx', 'docx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'reranked' in result


class TestChatModeXlsx:
    """Chat pipeline tests using Excel spreadsheet (.xlsx) chunks."""

    @_apply_patches
    def test_chat_xlsx_factual(self, mock_chat, mock_embed):
        """Factual query against a key=value xlsx chunk: response key is present."""
        chunks = [_chunk('country=Brazil | capital=Brasilia | population_millions=215',
                         'countries.xlsx', 'xlsx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_xlsx_source_label_row(self, mock_chat, mock_embed):
        """XLSX chunk source label: contains row indicator 'row' or generic 'L'."""
        chunks = [_chunk('country=Japan | capital=Tokyo', 'countries.xlsx', 'xlsx')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'row' in label or 'L' in label


class TestChatModeCsv:
    """Chat pipeline tests using CSV document chunks."""

    @_apply_patches
    def test_chat_csv_factual(self, mock_chat, mock_embed):
        """Factual query against a key=value csv chunk: response key is present."""
        chunks = [_chunk('language=Python | year_created=1991 | creator=Guido van Rossum',
                         'programming-languages.csv', 'csv')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_csv_source_label_row(self, mock_chat, mock_embed):
        """CSV chunk source label: contains row indicator 'row' or generic 'L'."""
        chunks = [_chunk('language=Rust | year_created=2010', 'langs.csv', 'csv')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'row' in label or 'L' in label


class TestChatModePptx:
    """Chat pipeline tests using PowerPoint (.pptx) document chunks."""

    @_apply_patches
    def test_chat_pptx_factual(self, mock_chat, mock_embed):
        """Factual query against a pptx chunk: response key is present in pipeline result."""
        chunks = [_chunk('The Berlin Wall fell on November 9 1989.', 'history.pptx', 'pptx')]
        store  = _make_store(chunks)
        result = store.run_pipeline('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_pptx_source_label_slide(self, mock_chat, mock_embed):
        """PPTX chunk source label: contains slide indicator 'slide' or generic 'L'."""
        chunks = [_chunk('World War Two ended in 1945.', 'history.pptx', 'pptx')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 'slide' in label or 'L' in label


class TestChatModeMd:
    """Chat pipeline tests using Markdown (.md) document chunks."""

    @_apply_patches
    def test_chat_md_factual(self, mock_chat, mock_embed):
        """Factual query against a md chunk: response key is present in pipeline result."""
        chunks = [_chunk('Coffee was first discovered in Ethiopia around 850 AD.',
                         'coffee-facts.md', 'md')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Where was coffee discovered?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_md_confidence(self, mock_chat, mock_embed):
        """Factual query against a md chunk: is_confident key is a boolean in pipeline result."""
        chunks = [_chunk('Coffee was first discovered in Ethiopia around 850 AD.',
                         'coffee-facts.md', 'md')]
        store  = _make_store(chunks)
        result = store.run_pipeline('Where was coffee discovered?', streamlit_mode=True)
        assert 'is_confident' in result
        assert result['is_confident'] in (True, False)


class TestChatModeHtml:
    """Chat pipeline tests using HTML document chunks."""

    @_apply_patches
    def test_chat_html_factual(self, mock_chat, mock_embed):
        """Factual query against an html chunk: response key is present in pipeline result."""
        chunks = [_chunk('A dog can understand about 165 words and gestures.', 'dog-facts.html', 'html')]
        store  = _make_store(chunks)
        result = store.run_pipeline('How many words can a dog understand?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_chat_html_source_label(self, mock_chat, mock_embed):
        """HTML chunk source label: contains section indicator 's' or generic 'L'."""
        chunks = [_chunk('Dogs were domesticated 15000 years ago.', 'dog-facts.html', 'html')]
        store  = _make_store(chunks)
        label  = store._source_label(chunks[0])
        assert 's' in label or 'L' in label


# ---------------------------------------------------------------------------
# 4. Chat mode — pipeline features across doc types
# ---------------------------------------------------------------------------

class TestChatPipelineFeatures:
    """Cross-doc-type tests for pipeline metadata: confidence, reranking, memory, source labels."""

    @_apply_patches
    def test_confidence_flag_present(self, mock_chat, mock_embed):
        """Pipeline result always includes an is_confident boolean key."""
        chunks = [_chunk('Python was created in 1991.', 'langs.csv', 'csv')]
        result = _make_store(chunks).run_pipeline('When was Python created?', streamlit_mode=True)
        assert 'is_confident' in result

    @_apply_patches
    def test_reranked_chunks_present(self, mock_chat, mock_embed):
        """Pipeline with two chunks: reranked list is present and has at least one entry."""
        chunks = [_chunk('Python was created in 1991.', 'langs.csv', 'csv'),
                  _chunk('Rust was created in 2010.', 'langs.csv', 'csv')]
        result = _make_store(chunks).run_pipeline('When was Python created?', streamlit_mode=True)
        assert 'reranked' in result and len(result['reranked']) >= 1

    @_apply_patches
    def test_conversation_memory(self, mock_chat, mock_embed):
        """After one pipeline turn: conversation_history has at least two entries (user + assistant)."""
        store = _make_store([_chunk('Brazil capital is Brasilia.', 'countries.xlsx', 'xlsx')])
        store.run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        assert len(store.conversation_history) >= 2

    @_apply_patches
    def test_clear_conversation(self, mock_chat, mock_embed):
        """After clear_conversation: conversation_history is reset to an empty list."""
        store = _make_store([_chunk('Brazil capital is Brasilia.', 'countries.xlsx', 'xlsx')])
        store.run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        store.clear_conversation()
        assert store.conversation_history == []

    @_apply_patches
    def test_query_type_factual(self, mock_chat, mock_embed):
        """Factual-style query: query_type is classified as 'factual' or 'general'."""
        chunks = [_chunk('Dogs sleep 12 hours a day.', 'dog-facts.txt', 'txt')]
        result = _make_store(chunks).run_pipeline('How many hours do dogs sleep?', streamlit_mode=True)
        assert result.get('query_type') in ('factual', 'general')

    @_apply_patches
    def test_source_label_xlsx_row(self, mock_chat, mock_embed):
        """XLSX chunk: _source_label contains 'row' indicator."""
        chunk = _chunk('country=Brazil | capital=Brasilia', 'countries.xlsx', 'xlsx')
        assert 'row' in _make_store([chunk])._source_label(chunk)

    @_apply_patches
    def test_source_label_pptx_slide(self, mock_chat, mock_embed):
        """PPTX chunk: _source_label contains 'slide' indicator."""
        chunk = _chunk('Berlin Wall fell in 1989.', 'history.pptx', 'pptx')
        assert 'slide' in _make_store([chunk])._source_label(chunk)

    @_apply_patches
    def test_source_label_pdf_page(self, mock_chat, mock_embed):
        """PDF chunk: _source_label contains 'p' page indicator."""
        chunk = _chunk('Bananas have 89 calories.', 'nutrition.pdf', 'pdf')
        assert 'p' in _make_store([chunk])._source_label(chunk)

    @_apply_patches
    def test_source_label_html_section(self, mock_chat, mock_embed):
        """HTML chunk: _source_label contains section indicator 's' or generic 'L'."""
        chunk = _chunk('Dogs understand 165 words.', 'dog-facts.html', 'html')
        label = _make_store([chunk])._source_label(chunk)
        assert 's' in label or 'L' in label
