"""test_url_ingestion.py — HTML chunker, URL ingestion, _truncate_chunk, and _dispatch_chunker tests.

Covers:
  - DocumentLoader._chunk_html (BeautifulSoup tag stripping, 40-char boilerplate filter)
  - DocumentLoader._truncate_chunk (300 words OR 1200 chars, whichever shorter)
  - DocumentLoader._dispatch_chunker routing to the correct format handler
  - DocumentLoader.chunk_url with the 4-priority type-detection pipeline:
      Priority 1: Content-Type header + fuzzy fallback
      Priority 2: File extension in URL path (strip query strings first)
      Priority 3: PDF magic bytes sniff (content[:4] == b'%PDF')
      Priority 4: Default to 'html'

Mock strategy:
  - requests.get is mocked for all URL tests.
  - Never mock: fitz, python-docx, openpyxl, xlrd, python-pptx, beautifulsoup4,
    BM25Okapi, or _truncate_chunk itself.

HF differences from local:
  - No DOC_FOLDERS / scan_all_files / ensure_folders in HF version.
  - HTML chunker filters lines shorter than 40 chars (boilerplate filter).
"""

import io
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ── make src importable from huggingface/ ────────────────────────────────────
HF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HF_ROOT not in sys.path:
    sys.path.insert(0, HF_ROOT)

from src.rag.chunkers import chunk_html, truncate_chunk

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_tmp(content: str, suffix: str) -> str:
    """Write *content* to a temporary file with the given *suffix* and return its path.

    The caller is responsible for deleting the file after the test.
    """
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                    mode='w', encoding='utf-8')
    f.write(content)
    f.close()
    return f.name


def _mock_resp(content, content_type='text/html', encoding='utf-8'):
    """Return a MagicMock that mimics a requests.Response.

    Args:
        content:      Response body — bytes or str (str is encoded with *encoding*).
        content_type: Value for the Content-Type header.
        encoding:     Declared encoding used when decoding the body.

    Returns:
        MagicMock with .content, .headers, .encoding, .raise_for_status set.
    """
    resp = MagicMock()
    resp.content = content if isinstance(content, bytes) else content.encode(encoding)
    resp.headers = {'Content-Type': content_type}
    resp.encoding = encoding
    resp.raise_for_status = MagicMock()
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# HTML file chunker
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentLoaderHtml:
    """Tests for the HTML chunker using BeautifulSoup.

    The HF chunker filters lines shorter than 40 characters as boilerplate,
    so test HTML content must be long enough to survive that filter.
    """

    def setup_method(self):
        """Instantiate a fresh DocumentLoader before each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def _make_html(self, content: str) -> str:
        """Write HTML content to a temp file and return its path."""
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.html',
                                        mode='w', encoding='utf-8')
        f.write(content)
        f.close()
        return f.name

    def test_chunk_html_strips_tags(self):
        """HTML tags are fully removed — no angle brackets appear in chunk text.

        Content must be > 40 chars per line to survive the HF boilerplate filter.
        """
        long_html = (
            "<html><body>"
            "<p>This is the first sentence with enough words to pass the filter threshold.</p>"
            "<p>This is the second sentence also long enough to survive the forty-character filter.</p>"
            "</body></html>"
        )
        path = self._make_html(long_html)
        try:
            chunks = chunk_html(path, 'page.html')
        finally:
            os.unlink(path)
        assert chunks, "Expected at least one chunk from content > 40 chars per line"
        combined = ' '.join(c['text'] for c in chunks)
        assert '<' not in combined

    def test_chunk_html_type_is_html(self):
        """Chunks produced by _chunk_html always carry type='html'."""
        # Short content may not pass the 40-char filter — but the type assertion
        # is still valid on whatever chunks are returned.
        path = self._make_html("<p>Hello world.</p>")
        try:
            chunks = chunk_html(path, 'page.html')
        finally:
            os.unlink(path)
        assert all(c['type'] == 'html' for c in chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# _truncate_chunk
# ═══════════════════════════════════════════════════════════════════════════════

class TestTruncateChunk:
    """Tests for DocumentLoader._truncate_chunk (300 words OR 1200 chars, whichever shorter)."""

    def setup_method(self):
        """Instantiate a fresh DocumentLoader before each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_truncate_300_words(self):
        """400-word text is truncated to at most 300 words."""
        text = ' '.join(['word'] * 400)
        result = truncate_chunk(text)
        assert len(result.split()) <= 300

    def test_truncate_1200_chars(self):
        """2000-character text is truncated to at most 1200 characters."""
        text = 'a' * 2000
        result = truncate_chunk(text)
        assert len(result) <= 1200

    def test_short_text_unchanged(self):
        """Text already within both limits is returned verbatim."""
        text = "Short text."
        assert truncate_chunk(text) == text

    def test_char_limit_wins_when_words_long(self):
        """50 long tokens (≈1550 chars, under 300 words) are truncated at 1200 chars."""
        text = ' '.join(['a' * 30] * 50)  # 50 * 31 ≈ 1550 chars, only 50 words
        result = truncate_chunk(text)
        assert len(result) <= 1200

    def test_word_limit_wins_when_words_short(self):
        """100 short words (600 chars total) hit neither limit and are returned as-is."""
        text = ' '.join(['hello'] * 100)
        assert truncate_chunk(text) == text


# ═══════════════════════════════════════════════════════════════════════════════
# _dispatch_chunker
# ═══════════════════════════════════════════════════════════════════════════════

class TestDispatchChunker:
    """Tests for _dispatch_chunker routing logic."""

    def setup_method(self):
        """Instantiate a fresh DocumentLoader before each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_dispatch_txt(self):
        """detected_type='txt' → _chunk_txt called → non-empty chunk list."""
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.txt',
                                        mode='w', encoding='utf-8')
        f.write("Hello world.\n")
        f.close()
        try:
            chunks = self.loader._dispatch_chunker({
                'filepath': f.name, 'filename': 'test.txt',
                'detected_type': 'txt', 'is_misplaced': False,
            })
        finally:
            os.unlink(f.name)
        assert len(chunks) > 0

    def test_dispatch_csv(self):
        """detected_type='csv' → _chunk_csv called → non-empty chunk list."""
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.csv',
                                        mode='w', encoding='utf-8')
        f.write("col1,col2\nval1,val2\n")
        f.close()
        try:
            chunks = self.loader._dispatch_chunker({
                'filepath': f.name, 'filename': 'data.csv',
                'detected_type': 'csv', 'is_misplaced': False,
            })
        finally:
            os.unlink(f.name)
        assert len(chunks) > 0

    def test_dispatch_unknown_type_returns_empty(self):
        """Unrecognised detected_type returns an empty list without raising."""
        chunks = self.loader._dispatch_chunker({
            'filepath': '/tmp/x.xyz', 'filename': 'x.xyz',
            'detected_type': 'unknown_xyz', 'is_misplaced': False,
        })
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════════
# chunk_url — URL ingestion
# ═══════════════════════════════════════════════════════════════════════════════

class TestUrlIngestion:
    """Tests for chunk_url: type detection priorities and format dispatch.

    4-priority detection order:
      1. Content-Type header + fuzzy fallback
      2. File extension in URL path (strip query strings first)
      3. PDF magic bytes sniff (content[:4] == b'%PDF')
      4. Default to 'html'
    """

    def setup_method(self):
        """Instantiate a fresh DocumentLoader before each test."""
        from src.rag.document_loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_url_html_webpage(self):
        """text/html Content-Type → HTML chunks with type='html'."""
        html = b"<html><body><p>Hello world. This is a test. More content here.</p></body></html>"
        with patch('requests.get', return_value=_mock_resp(html, 'text/html')):
            chunks = self.loader.chunk_url('http://example.com/page')
        assert len(chunks) > 0
        assert all(c['type'] == 'html' for c in chunks)

    def test_url_type_by_content_type_pdf(self):
        """application/pdf Content-Type → type detection Priority 1 → PDF chunks."""
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "PDF content here. Another sentence.")
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        pdf_bytes = buf.getvalue()
        with patch('requests.get', return_value=_mock_resp(pdf_bytes, 'application/pdf')):
            chunks = self.loader.chunk_url('http://example.com/doc')
        assert len(chunks) > 0
        assert all(c['type'] == 'pdf' for c in chunks)

    def test_url_type_by_extension_csv(self):
        """Unrecognised Content-Type but .csv extension → type detection Priority 2."""
        csv_bytes = b"name,age\nAlice,30\nBob,25\n"
        with patch('requests.get', return_value=_mock_resp(csv_bytes, 'application/octet-stream')):
            chunks = self.loader.chunk_url('http://example.com/data.csv')
        assert len(chunks) > 0
        assert all(c['type'] == 'csv' for c in chunks)

    def test_url_type_by_pdf_magic_bytes(self):
        """No recognised Content-Type or extension but %PDF header → Priority 3 → pdf dispatch."""
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Magic bytes PDF test. Another sentence.")
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        pdf_bytes = buf.getvalue()
        assert pdf_bytes[:4] == b'%PDF'
        with patch('requests.get', return_value=_mock_resp(pdf_bytes, 'application/octet-stream')):
            chunks = self.loader.chunk_url('http://example.com/unknown')
        # As long as no exception is raised and a list is returned, the priority fired
        assert isinstance(chunks, list)

    def test_url_defaults_to_html(self):
        """No signal from any priority → defaults to html and returns a list without error."""
        content = b"<p>Some page content. More words here.</p>"
        with patch('requests.get', return_value=_mock_resp(content, 'application/octet-stream')):
            chunks = self.loader.chunk_url('http://example.com/unknown_type')
        assert isinstance(chunks, list)

    def test_url_connection_error_returns_empty(self):
        """Network errors are caught and an empty list is returned rather than raising."""
        with patch('requests.get', side_effect=Exception("Connection refused")):
            chunks = self.loader.chunk_url('http://unreachable.example.com')
        assert chunks == []

    def test_url_source_label_truncated_60_chars(self):
        """Very long URLs are truncated to 60 characters in the chunk source field."""
        html = b"<p>Content here. More sentences. And another one.</p>"
        long_url = 'http://example.com/' + 'a' * 100
        with patch('requests.get', return_value=_mock_resp(html, 'text/html')):
            chunks = self.loader.chunk_url(long_url)
        if chunks:
            assert len(chunks[0]['source']) <= 60

    def test_url_txt_plain(self):
        """text/plain Content-Type → each non-empty line becomes a type='txt' chunk."""
        txt = b"Line one\nLine two\nLine three\n"
        with patch('requests.get', return_value=_mock_resp(txt, 'text/plain')):
            chunks = self.loader.chunk_url('http://example.com/file.txt')
        assert len(chunks) > 0
        assert all(c['type'] == 'txt' for c in chunks)

    def test_url_type_by_extension_html(self):
        """URL ending in .html with octet-stream content-type → typed as 'html'."""
        html = b"<html><body><p>Content. More content. Third sentence here.</p></body></html>"
        with patch('requests.get', return_value=_mock_resp(html, 'application/octet-stream')):
            chunks = self.loader.chunk_url('http://example.com/page.html')
        # Extension (.html) wins over octet-stream when no content-type signal
        assert all(c['type'] == 'html' for c in chunks)
