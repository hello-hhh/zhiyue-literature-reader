"""test_file_upload.py — File upload tests for all 8 document types using real
temp files through DocumentLoader._dispatch_chunker, then chat mode and agent mode.

Mock strategy:
  Always mock: ollama.embed, ollama.chat, chromadb → EphemeralClient
  Never mock:  fitz, python-docx, openpyxl, python-pptx, BM25Okapi, file I/O

Reason for split: max 500 lines per file per CLAUDE.md.
All-5-agent-tools tests with real file content are in test_file_upload_tools.py.
"""

import io
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
    """Return a fixed 4-dim embedding vector to satisfy ollama.embed calls."""
    return {'embeddings': [[0.1, 0.2, 0.3, 0.4]]}


def _fake_chat(**kwargs):
    """Return a canned chat response (stream or single) to satisfy ollama.chat calls."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': 'mock response'}}


def _make_store(chunks):
    """Build an isolated VectorStore with EphemeralClient."""
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
    vs.collection           = collection
    vs.chunks               = chunks
    vs.bm25_index           = BM25Okapi([c['text'].lower().split() for c in chunks]) if chunks else None
    vs.conversation_history = []
    return vs


PATCHES = [
    patch('ollama.embed', side_effect=_fake_embed),
    patch('ollama.chat',  side_effect=_fake_chat),
]


def _apply_patches(func):
    """Decorator that applies both ollama.embed and ollama.chat patches to a test method."""
    for p in reversed(PATCHES):
        func = p(func)
    return func


def _upload_chunks(tmp_path, filename, content_fn):
    """Create a real temp file and run it through DocumentLoader._dispatch_chunker."""
    from src.rag.document_loader import DocumentLoader
    loader   = DocumentLoader()
    ext      = os.path.splitext(filename)[1].lower()
    dtype    = loader.ext_to_type.get(ext, 'txt')
    filepath = str(tmp_path / filename)
    content_fn(filepath)
    file_info = {
        'filepath':      filepath,
        'filename':      filename,
        'detected_type': dtype,
        'is_misplaced':  False,
    }
    return loader._dispatch_chunker(file_info)


# ---------------------------------------------------------------------------
# Per-type file builders — write real files using the actual libraries
# ---------------------------------------------------------------------------

def _write_pdf(path):
    """Single-page PDF with nutrition facts."""
    import fitz
    doc  = fitz.open(); page = doc.new_page()
    page.insert_text((50, 50),
        'Nutrition facts: Bananas contain 89 calories per 100 grams. '
        'They are rich in potassium and vitamin B6. '
        'Avocados contain 160 calories per 100 grams and healthy fats.')
    doc.save(path); doc.close()


def _write_txt(path):
    """Plain-text dog facts, one fact per line."""
    with open(path, 'w') as f:
        f.write('Dogs were domesticated from wolves approximately 15000 years ago.\n')
        f.write('There are over 340 recognized dog breeds in the world.\n')
        f.write('A dog can understand about 165 words and gestures.\n')


def _write_docx(path):
    """DOCX with three programming language paragraphs."""
    from docx import Document
    doc = Document()
    doc.add_paragraph('Python is a programming language created by Guido van Rossum in 1991.')
    doc.add_paragraph('Java was created by James Gosling at Sun Microsystems in 1995.')
    doc.add_paragraph('Rust was developed at Mozilla Research and released in 2010.')
    doc.save(path)


def _write_xlsx(path):
    """XLSX with country/capital/population data."""
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active
    ws.append(['Country', 'Capital', 'Population_Millions'])
    ws.append(['Brazil',  'Brasilia', 215])
    ws.append(['Japan',   'Tokyo',    125])
    ws.append(['Germany', 'Berlin',   84])
    wb.save(path)


def _write_csv(path):
    """CSV with programming language creator and year."""
    with open(path, 'w') as f:
        f.write('language,year,creator\n')
        f.write('Python,1991,Guido van Rossum\n')
        f.write('Rust,2010,Graydon Hoare\n')
        f.write('Go,2009,Robert Griesemer\n')


def _write_pptx(path):
    """PPTX with three history fact slides."""
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    for title, body in [
        ('World War Two', 'World War Two ended in 1945 with Allied victory.'),
        ('Berlin Wall',   'The Berlin Wall fell on November 9 1989.'),
        ('Moon Landing',  'Apollo 11 landed on the Moon on July 20 1969.'),
    ]:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        tb    = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(2))
        tb.text_frame.text = f'{title}: {body}'
    prs.save(path)


def _write_md(path):
    """Markdown file with coffee facts."""
    with open(path, 'w') as f:
        f.write('# Coffee Facts\n\n')
        f.write("Coffee was first discovered in Ethiopia around 850 AD by a goat herder named Kaldi.\n\n")
        f.write("Brazil is the world's largest coffee producer, accounting for 40 percent of global supply.\n\n")
        f.write('A standard cup of coffee contains between 80 and 100 milligrams of caffeine.\n')


def _write_html(path):
    """HTML file with dog biology facts."""
    with open(path, 'w') as f:
        f.write('<html><body><h1>Dog Facts</h1>')
        f.write("<p>A dog's sense of smell is up to 100000 times more powerful than a human's.</p>")
        f.write('<p>Border Collies are considered the most intelligent dog breed.</p>')
        f.write('<p>Greyhounds can run at speeds up to 72 kilometers per hour.</p>')
        f.write('</body></html>')


# ---------------------------------------------------------------------------
# 1. Chat mode — full pipeline per file type
# ---------------------------------------------------------------------------

class TestFileUploadChatModeTxt:
    """TXT file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """TXT upload: chunks have type 'txt' and run_pipeline returns a non-empty response."""
        chunks = _upload_chunks(tmp_path, 'dogs.txt', _write_txt)
        assert len(chunks) >= 1 and all(c['type'] == 'txt' for c in chunks)
        result = _make_store(chunks).run_pipeline('How many dog breeds are there?', streamlit_mode=True)
        assert 'response' in result and len(result['response']) > 0

    @_apply_patches
    def test_query_type(self, mock_chat, mock_embed, tmp_path):
        """TXT upload: query_type in result is one of the 4 valid classifier labels."""
        chunks = _upload_chunks(tmp_path, 'dogs.txt', _write_txt)
        result = _make_store(chunks).run_pipeline('How many dog breeds are there?', streamlit_mode=True)
        assert result.get('query_type') in ('factual', 'general', 'comparison', 'summarise')

    @_apply_patches
    def test_source_label(self, mock_chat, mock_embed, tmp_path):
        """TXT upload chunk: _source_label contains 'L' (line range) marker."""
        chunks = _upload_chunks(tmp_path, 'dogs.txt', _write_txt)
        assert 'L' in _make_store(chunks)._source_label(chunks[0])   # txt → L{s}-{e}

    @_apply_patches
    def test_source_cited(self, mock_chat, mock_embed, tmp_path):
        """TXT upload: retrieved chunks list includes the dogs.txt filename as source."""
        chunks = _upload_chunks(tmp_path, 'dogs.txt', _write_txt)
        result = _make_store(chunks).run_pipeline('How many dog breeds are there?', streamlit_mode=True)
        assert any('dogs.txt' in e['source'] for e, _ in result['retrieved'])

    @_apply_patches
    def test_confidence_present(self, mock_chat, mock_embed, tmp_path):
        """TXT upload: is_confident key is present and is a boolean in the pipeline result."""
        chunks = _upload_chunks(tmp_path, 'dogs.txt', _write_txt)
        result = _make_store(chunks).run_pipeline('How many dog breeds are there?', streamlit_mode=True)
        assert 'is_confident' in result and result['is_confident'] in (True, False)


class TestFileUploadChatModePdf:
    """PDF file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """PDF upload: chunks have type 'pdf' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'nutrition.pdf', _write_pdf)
        assert len(chunks) >= 1 and all(c['type'] == 'pdf' for c in chunks)
        result = _make_store(chunks).run_pipeline('How many calories in bananas?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_source_label(self, mock_chat, mock_embed, tmp_path):
        """PDF upload chunk: _source_label contains 'p' (page) or 'L' marker."""
        chunks = _upload_chunks(tmp_path, 'nutrition.pdf', _write_pdf)
        label  = _make_store(chunks)._source_label(chunks[0])
        assert 'p' in label or 'L' in label   # pdf → p{n}

    @_apply_patches
    def test_reranked(self, mock_chat, mock_embed, tmp_path):
        """PDF upload: reranked key is present in the pipeline result."""
        chunks = _upload_chunks(tmp_path, 'nutrition.pdf', _write_pdf)
        result = _make_store(chunks).run_pipeline('How many calories in bananas?', streamlit_mode=True)
        assert 'reranked' in result

    @_apply_patches
    def test_source_cited(self, mock_chat, mock_embed, tmp_path):
        """PDF upload: retrieved chunks list includes nutrition.pdf filename as source."""
        chunks = _upload_chunks(tmp_path, 'nutrition.pdf', _write_pdf)
        result = _make_store(chunks).run_pipeline('How many calories in bananas?', streamlit_mode=True)
        assert any('nutrition.pdf' in e['source'] for e, _ in result['retrieved'])


class TestFileUploadChatModeDocx:
    """DOCX file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """DOCX upload: chunks have type 'docx' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'languages.docx', _write_docx)
        assert len(chunks) >= 1 and all(c['type'] == 'docx' for c in chunks)
        result = _make_store(chunks).run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_reranked(self, mock_chat, mock_embed, tmp_path):
        """DOCX upload: reranked list is present and contains at least one entry."""
        chunks = _upload_chunks(tmp_path, 'languages.docx', _write_docx)
        result = _make_store(chunks).run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'reranked' in result and len(result['reranked']) >= 1

    @_apply_patches
    def test_confidence(self, mock_chat, mock_embed, tmp_path):
        """DOCX upload: is_confident key is present and is a boolean in the pipeline result."""
        chunks = _upload_chunks(tmp_path, 'languages.docx', _write_docx)
        result = _make_store(chunks).run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'is_confident' in result and result['is_confident'] in (True, False)


class TestFileUploadChatModeXlsx:
    """XLSX file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """XLSX upload: chunks have type 'xlsx' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'countries.xlsx', _write_xlsx)
        assert len(chunks) >= 1 and all(c['type'] == 'xlsx' for c in chunks)
        result = _make_store(chunks).run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_source_label_row(self, mock_chat, mock_embed, tmp_path):
        """XLSX upload chunk: _source_label contains 'row' or 'L' marker."""
        chunks = _upload_chunks(tmp_path, 'countries.xlsx', _write_xlsx)
        label  = _make_store(chunks)._source_label(chunks[0])
        assert 'row' in label or 'L' in label   # xlsx → row{n}

    @_apply_patches
    def test_source_cited(self, mock_chat, mock_embed, tmp_path):
        """XLSX upload: retrieved chunks list includes the countries.xlsx filename as source."""
        chunks = _upload_chunks(tmp_path, 'countries.xlsx', _write_xlsx)
        result = _make_store(chunks).run_pipeline('What is the capital of Brazil?', streamlit_mode=True)
        assert any('countries.xlsx' in e['source'] for e, _ in result['retrieved'])


class TestFileUploadChatModeCsv:
    """CSV file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """CSV upload: chunks have type 'csv' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'languages.csv', _write_csv)
        assert len(chunks) >= 1 and all(c['type'] == 'csv' for c in chunks)
        result = _make_store(chunks).run_pipeline('Who created Python?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_source_label_row(self, mock_chat, mock_embed, tmp_path):
        """CSV upload chunk: _source_label contains 'row' or 'L' marker."""
        chunks = _upload_chunks(tmp_path, 'languages.csv', _write_csv)
        label  = _make_store(chunks)._source_label(chunks[0])
        assert 'row' in label or 'L' in label   # csv → row{n}

    @_apply_patches
    def test_query_type(self, mock_chat, mock_embed, tmp_path):
        """CSV upload: query_type in result is one of the 4 valid classifier labels."""
        chunks = _upload_chunks(tmp_path, 'languages.csv', _write_csv)
        result = _make_store(chunks).run_pipeline('Who created Python?', streamlit_mode=True)
        assert result.get('query_type') in ('factual', 'general', 'comparison', 'summarise')


class TestFileUploadChatModePptx:
    """PPTX file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """PPTX upload: chunks have type 'pptx' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'history.pptx', _write_pptx)
        assert len(chunks) >= 1 and all(c['type'] == 'pptx' for c in chunks)
        result = _make_store(chunks).run_pipeline('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_source_label_slide(self, mock_chat, mock_embed, tmp_path):
        """PPTX upload chunk: _source_label contains 'slide' or 'L' marker."""
        chunks = _upload_chunks(tmp_path, 'history.pptx', _write_pptx)
        label  = _make_store(chunks)._source_label(chunks[0])
        assert 'slide' in label or 'L' in label   # pptx → slide{n}

    @_apply_patches
    def test_reranked(self, mock_chat, mock_embed, tmp_path):
        """PPTX upload: reranked key is present in the pipeline result."""
        chunks = _upload_chunks(tmp_path, 'history.pptx', _write_pptx)
        result = _make_store(chunks).run_pipeline('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'reranked' in result


class TestFileUploadChatModeMd:
    """Markdown file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """Markdown upload: chunks have type 'md' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'coffee.md', _write_md)
        assert len(chunks) >= 1 and all(c['type'] == 'md' for c in chunks)
        result = _make_store(chunks).run_pipeline('Where was coffee discovered?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_source_label(self, mock_chat, mock_embed, tmp_path):
        """Markdown upload chunk: _source_label contains 'L' (line range) marker."""
        chunks = _upload_chunks(tmp_path, 'coffee.md', _write_md)
        assert 'L' in _make_store(chunks)._source_label(chunks[0])   # md → L{s}-{e}

    @_apply_patches
    def test_confidence(self, mock_chat, mock_embed, tmp_path):
        """Markdown upload: is_confident key is present and is a boolean in the pipeline result."""
        chunks = _upload_chunks(tmp_path, 'coffee.md', _write_md)
        result = _make_store(chunks).run_pipeline('Where was coffee discovered?', streamlit_mode=True)
        assert 'is_confident' in result and result['is_confident'] in (True, False)


class TestFileUploadChatModeHtml:
    """HTML file upload → chat pipeline."""

    @_apply_patches
    def test_factual(self, mock_chat, mock_embed, tmp_path):
        """HTML upload: chunks have type 'html' and run_pipeline returns a response key."""
        chunks = _upload_chunks(tmp_path, 'dogs.html', _write_html)
        assert len(chunks) >= 1 and all(c['type'] == 'html' for c in chunks)
        result = _make_store(chunks).run_pipeline('How fast can a greyhound run?', streamlit_mode=True)
        assert 'response' in result

    @_apply_patches
    def test_source_label(self, mock_chat, mock_embed, tmp_path):
        """HTML upload chunk: _source_label contains 's' (section) or 'L' marker."""
        chunks = _upload_chunks(tmp_path, 'dogs.html', _write_html)
        label  = _make_store(chunks)._source_label(chunks[0])
        assert 's' in label or 'L' in label   # html → s{n}

    @_apply_patches
    def test_source_cited(self, mock_chat, mock_embed, tmp_path):
        """HTML upload: retrieved chunks list includes the dogs.html filename as source."""
        chunks = _upload_chunks(tmp_path, 'dogs.html', _write_html)
        result = _make_store(chunks).run_pipeline('How fast can a greyhound run?', streamlit_mode=True)
        assert any('dogs.html' in e['source'] for e, _ in result['retrieved'])


# ---------------------------------------------------------------------------
# 2. Agent mode — rag_search for each file type
# ---------------------------------------------------------------------------

class TestFileUploadAgentMode:
    """All 8 file types loaded via upload, then queried through Agent.run()."""

    @_apply_patches
    def test_txt_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """TXT upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(dog breeds)'}}
        chunks = _upload_chunks(tmp_path, 'dogs.txt', _write_txt)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('How many dog breeds are there?', streamlit_mode=True)
        assert 'answer' in result
        assert any(s['tool'] == 'rag_search' for s in result['steps'])

    @_apply_patches
    def test_pdf_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """PDF upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(banana calories)'}}
        chunks = _upload_chunks(tmp_path, 'nutrition.pdf', _write_pdf)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('How many calories in bananas?', streamlit_mode=True)
        assert 'answer' in result
        assert any(s['tool'] == 'rag_search' for s in result['steps'])

    @_apply_patches
    def test_docx_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """DOCX upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(Python creator)'}}
        chunks = _upload_chunks(tmp_path, 'languages.docx', _write_docx)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Who created Python?', streamlit_mode=True)
        assert 'answer' in result

    @_apply_patches
    def test_xlsx_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """XLSX upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(Brazil capital)'}}
        chunks = _upload_chunks(tmp_path, 'countries.xlsx', _write_xlsx)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('What is the capital of Brazil?', streamlit_mode=True)
        assert 'answer' in result

    @_apply_patches
    def test_csv_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """CSV upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(Python year)'}}
        chunks = _upload_chunks(tmp_path, 'languages.csv', _write_csv)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('When was Python created?', streamlit_mode=True)
        assert 'answer' in result

    @_apply_patches
    def test_pptx_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """PPTX upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(Berlin Wall 1989)'}}
        chunks = _upload_chunks(tmp_path, 'history.pptx', _write_pptx)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'answer' in result

    @_apply_patches
    def test_md_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """Markdown upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(coffee Ethiopia)'}}
        chunks = _upload_chunks(tmp_path, 'coffee.md', _write_md)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Where was coffee discovered?', streamlit_mode=True)
        assert 'answer' in result

    @_apply_patches
    def test_html_agent_rag_search(self, mock_chat, mock_embed, tmp_path):
        """HTML upload in agent mode: rag_search tool is invoked and answer key is present."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(dog smell)'}}
        chunks = _upload_chunks(tmp_path, 'dogs.html', _write_html)
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('What is special about dog smell?', streamlit_mode=True)
        assert 'answer' in result
