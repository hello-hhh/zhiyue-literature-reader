"""document_loader.py — DocumentLoader class: owns all ingestion. HF Space version."""

import os
import re
import tempfile
from typing import Callable, List, Optional, Set
from urllib.parse import urlparse

from src.rag import binary_chunkers, chunkers
from src.rag.config import (
    DOCX_CHUNK_PARAS,
    EXT_TO_TYPE,
    HTML_CHUNK_SENTENCES,
    PDF_CHUNK_SENTENCES,
    PPTX_CHUNK_SLIDES,
    TXT_CHUNK_OVERLAP,
    TXT_CHUNK_SIZE,
)
from src.rag.url_utils import build_source_name, detect_url_type, extract_links, is_same_domain, url_matches_topic

__all__ = ['DocumentLoader']


class DocumentLoader:
    """Owns all document ingestion — chunkers, URL fetching.

    All format-specific chunking logic lives in chunkers.py as stateless
    functions. This class owns the state (extension map, chunk sizes) and
    the orchestration: detecting document type, dispatching to chunkers,
    and fetching URLs.

    State:
        ext_to_type (dict): Maps file extension strings to canonical type keys.
        chunk_sizes (dict): All chunk-size constants keyed by config name.

    Public API:
        chunk_url(url)          — Fetch a URL and route to the correct chunker.
        _dispatch_chunker(info) — Pick and call the right chunker function.
    """

    def __init__(self) -> None:
        """Load config constants as instance vars so they are easy to override in tests."""
        self.ext_to_type = EXT_TO_TYPE
        self.chunk_sizes = {
            'txt_chunk_size':        TXT_CHUNK_SIZE,
            'txt_chunk_overlap':     TXT_CHUNK_OVERLAP,
            'pdf_chunk_sentences':   PDF_CHUNK_SENTENCES,
            'docx_chunk_paras':      DOCX_CHUNK_PARAS,
            'pptx_chunk_slides':     PPTX_CHUNK_SLIDES,
            'html_chunk_sentences':  HTML_CHUNK_SENTENCES,
        }

    # ------------------------------------------------------------------ Public

    def chunk_url(self, url: str) -> List[dict]:
        """Fetch a URL and produce chunks using the appropriate format handler.

        Type detection uses four priorities in strict order:
            1. Content-Type response header (most reliable).
            2. File extension in the URL path (strips query strings first).
            3. PDF magic bytes sniff (content[:4] == b'%PDF').
            4. Default to 'html' when no other signal is present.

        Args:
            url: Public HTTP/HTTPS URL to fetch.

        Returns:
            List of chunk dicts, each with keys: text, source, start_line,
            end_line, type. Returns an empty list on network error.
        """
        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed.")
            return []

        url = url.strip()
        # Add https:// if the user pasted a URL without a scheme (e.g. en.wikipedia.org/...)
        # Without a scheme, urlparse returns an empty netloc and requests raises MissingSchema.
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        try:
            resp = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }, allow_redirects=True, stream=True)
            resp.raise_for_status()
            content = resp.content
            text    = None
        except Exception as e:
            print(f"  [ERROR] Could not fetch URL: {e}")
            return []

        # ── Priority 1: exact Content-Type header match ──────────────
        content_type = resp.headers.get('Content-Type', '').lower().split(';')[0].strip()
        dtype        = None

        CONTENT_TYPE_MAP = {
            'application/pdf':                                          'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword':                                       'docx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/vnd.ms-excel':                                 'xlsx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            'application/vnd.ms-powerpoint':                            'pptx',
            'text/csv':                                                 'csv',
            'text/plain':                                               'txt',
            'text/markdown':                                            'md',
            'text/html':                                                'html',
            'application/xhtml+xml':                                    'html',
        }
        dtype = CONTENT_TYPE_MAP.get(content_type)

        # Fuzzy fallback for unusual MIME subtypes (e.g. "application/x-pdf")
        if dtype is None:
            if 'pdf'          in content_type: dtype = 'pdf'
            elif 'word'       in content_type: dtype = 'docx'
            elif 'excel'      in content_type or 'spreadsheet' in content_type: dtype = 'xlsx'
            elif 'powerpoint' in content_type or 'presentation' in content_type: dtype = 'pptx'
            elif 'csv'        in content_type: dtype = 'csv'

        # ── Priority 2: file extension in URL path ───────────────────
        if dtype is None:
            parsed_path = urlparse(url).path.lower()
            clean_path  = re.sub(r'[?#].*$', '', parsed_path)
            ext         = os.path.splitext(clean_path)[1]
            dtype       = self.ext_to_type.get(ext)

        # ── Priority 3: PDF magic bytes ───────────────────────────────
        if dtype is None and content[:4] == b'%PDF':
            dtype = 'pdf'

        # ── Priority 4: default to HTML ───────────────────────────────
        if dtype is None:
            dtype = 'html'

        # ── Build source label from URL ───────────────────────────────
        parsed      = urlparse(url)
        source_name = (parsed.netloc + parsed.path).rstrip('/')
        if parsed.query:
            source_name += '?' + parsed.query[:20] + ('...' if len(parsed.query) > 20 else '')
        if len(source_name) > 60:
            source_name = source_name[:57] + '...'
        if not source_name:
            source_name = url[:60]

        # ── Binary formats: write to temp file, chunk, delete ────────
        if dtype in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            suffix = {'pdf': '.pdf', 'docx': '.docx', 'xlsx': '.xlsx',
                      'pptx': '.pptx', 'xls': '.xls'}[dtype]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                file_info = {
                    'filepath':      tmp_path,
                    'filename':      source_name,
                    'detected_type': dtype,
                    'is_misplaced':  False,
                }
                result = self._dispatch_chunker(file_info)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return result

        try:
            text = content.decode(resp.encoding or 'utf-8', errors='replace')
        except Exception:
            text = content.decode('utf-8', errors='replace')

        if dtype == 'csv':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w',
                                             encoding='utf-8', errors='replace') as tmp:
                tmp.write(text)
                tmp_path = tmp.name
            try:
                result = chunkers.chunk_csv(tmp_path, source_name)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return result

        if dtype == 'txt':
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return [{'text': line, 'source': source_name, 'start_line': i + 1,
                     'end_line': i + 1, 'type': 'txt'} for i, line in enumerate(lines)]

        if dtype == 'md':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.md', mode='w',
                                             encoding='utf-8', errors='replace') as tmp:
                tmp.write(text)
                tmp_path = tmp.name
            try:
                result = chunkers.chunk_md(tmp_path, source_name,
                                           self.chunk_sizes['txt_chunk_size'],
                                           self.chunk_sizes['txt_chunk_overlap'])
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return result

        # ── HTML / webpage ────────────────────────────────────────────
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(text, 'html.parser').get_text(separator=' ', strip=True)
        except ImportError:
            text = re.sub(r'<[^>]+>', ' ', text)

        sents  = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        result = []
        for i in range(0, len(sents), HTML_CHUNK_SENTENCES):
            window = sents[i: i + HTML_CHUNK_SENTENCES]
            if window:
                result.append({'text': ' '.join(window), 'source': source_name,
                                'start_line': i + 1, 'end_line': i + len(window),
                                'type': 'html'})
        return result

    def chunk_url_recursive(
        self,
        url: str,
        depth: int = 1,
        max_pages: int = 10,
        allowed_types: Optional[Set[str]] = None,
        topic_filter: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> List[dict]:
        """Crawl a seed URL and all linked pages up to a given depth.

        Depth and max_pages are intentionally small on the HF Space (default 1/10)
        because the free-tier CPU is shared and slow. Users can still increase them
        from the UI if they choose.

        Args:
            url:               Seed URL to start crawling from.
            depth:             How many link-levels deep to follow.
            max_pages:         Maximum total pages/documents to fetch and index.
            allowed_types:     Set of type strings to index, or None for all.
            topic_filter:      Optional keyword the URL path must contain to be crawled.
                               Empty string means no filter — crawl everything on the domain.
            progress_callback: Optional callback(url, dtype, chunk_count) after each page.

        Returns:
            Flat list of all chunk dicts from every crawled URL.
        """
        visited: Set[str] = set()
        all_chunks: List[dict] = []

        self._crawl_url(
            url.strip(), depth, max_pages,
            allowed_types, topic_filter, visited, all_chunks, progress_callback,
            is_seed=True,
        )

        print(f"\n  [CRAWL] Finished — {len(visited)} pages crawled, "
              f"{len(all_chunks)} chunks total")
        return all_chunks

    def chunk_topic_search(
        self,
        query: str,
        num_results: int = 10,
        depth: int = 1,
        max_pages_per_result: int = 3,
        allowed_types: Optional[Set[str]] = None,
        topic_filter: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> List[dict]:
        """Search the web for a topic and index the top results.

        Uses DuckDuckGo (no API key required) to find the top N pages for the
        query, then crawls each result URL with chunk_url_recursive() using the
        same depth and type controls as the manual URL crawl.

        Args:
            query:               The search query — e.g. "Elizabeth Olsen actress".
            num_results:         How many search result URLs to fetch and index (1–20).
            depth:               Crawl depth per result URL (1 = result page only,
                                 2 = result page + its links).
            max_pages_per_result: Hard cap on pages crawled per result URL.
            allowed_types:       Set of doc types to index (None = all types).
            topic_filter:        Keyword the URL path must contain to be crawled.
            progress_callback:   Optional callback(url, dtype, chunk_count).

        Returns:
            Flat list of all chunk dicts from every crawled result URL.
        """
        # Search DuckDuckGo via the HTML form endpoint.
        # This uses a plain POST request (requests + BeautifulSoup) rather than
        # the duckduckgo-search library, which is subject to aggressive IP-level
        # rate limiting on the JS API endpoints.  The HTML endpoint is the same
        # one used by the public website and is not rate-limited in the same way.
        print(f"\n  [SEARCH] Querying: {query!r}  (top {num_results} results)")
        urls = self._search_duckduckgo_html(query, num_results)

        all_chunks: List[dict] = []
        for i, url in enumerate(urls):
            print(f"  [SEARCH] [{i + 1}/{len(urls)}] {url[:80]}")
            try:
                chunks = self.chunk_url_recursive(
                    url,
                    depth=depth,
                    max_pages=max_pages_per_result,
                    allowed_types=allowed_types,
                    topic_filter=topic_filter,
                    progress_callback=progress_callback,
                )
                all_chunks.extend(chunks)
            except Exception as error:
                print(f"  [SEARCH] Error crawling {url}: {error}")

        print(f"\n  [SEARCH] Done — {len(urls)} URLs crawled, "
              f"{len(all_chunks)} chunks total")
        return all_chunks

    # ----------------------------------------------------------------- Private

    def _search_duckduckgo_html(self, query: str, num_results: int) -> List[str]:
        """Search DuckDuckGo via the public HTML form endpoint and return URLs.

        Uses a plain POST request instead of the duckduckgo-search library.
        The HTML endpoint (html.duckduckgo.com/html/) is not subject to the
        same aggressive IP-level rate limiting as the JS API endpoints that
        duckduckgo-search v6 uses internally.

        Args:
            query:       The search query string.
            num_results: Maximum number of result URLs to return.

        Returns:
            List of result URLs (ads and DuckDuckGo redirect links filtered out).
        """
        try:
            import requests as _requests
            from bs4 import BeautifulSoup
        except ImportError as import_error:
            print(f"  [SEARCH] Missing dependency: {import_error}")
            return []

        # Mimic a real browser so DuckDuckGo returns normal HTML results.
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/131.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://duckduckgo.com',
            'Referer': 'https://duckduckgo.com/',
        }

        try:
            response = _requests.post(
                'https://html.duckduckgo.com/html/',
                data={'q': query},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
        except Exception as error:
            print(f"  [SEARCH] DuckDuckGo HTML request failed: {error}")
            return []

        soup = BeautifulSoup(response.text, 'lxml')

        # Collect up to num_results clean result URLs.
        # Skip ad links — they go through duckduckgo.com/y.js rather than
        # pointing directly to the destination.
        urls: List[str] = []
        for anchor in soup.select('a.result__a'):
            href = anchor.get('href', '')
            if href.startswith('http') and '/y.js?' not in href:
                urls.append(href)
                if len(urls) >= num_results:
                    break

        print(f"  [SEARCH] Found {len(urls)} result URLs")
        return urls

    def _crawl_url(
        self,
        url: str,
        depth: int,
        max_pages: int,
        allowed_types: Optional[Set[str]],
        topic_filter: str,
        visited: Set[str],
        all_chunks: List[dict],
        progress_callback: Optional[Callable],
        is_seed: bool = False,
        seed_domain: str = '',
    ) -> None:
        """Recursively fetch a URL and follow links up to the given depth.

        See chunk_url_recursive() for full documentation. This internal helper
        mutates the shared visited set and all_chunks list rather than returning
        values, so progress is accumulated naturally across all recursive calls.

        Args:
            url:               Absolute URL to fetch.
            depth:             Remaining link levels to follow.
            max_pages:         Hard cap on total pages fetched.
            allowed_types:     If set, only index pages whose type is in this set.
            topic_filter:      Keyword the URL path must contain. Ignored for seed URL.
            visited:           Shared set of already-fetched URLs (mutated in place).
            all_chunks:        Shared accumulator for all chunks (mutated in place).
            progress_callback: Optional callback after each page.
            is_seed:           True only for the seed URL — always crawled regardless of filter.
        """
        # Add https:// if the URL has no scheme (e.g. en.wikipedia.org/wiki/...)
        # Without a scheme, urlparse returns an empty netloc and urljoin produces bare
        # paths (/wiki/SomeLink) that are rejected by the http:// check in extract_links.
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Normalise to prevent double-fetching the same page under different forms
        url = url.rstrip('/')    # /wiki/Article and /wiki/Article/ are the same page
        url = url.split('#')[0]  # fragments (#section) are client-side only

        if url in visited or len(visited) >= max_pages:
            return

        # Apply topic filter to all pages except the seed URL
        if not is_seed and not url_matches_topic(url, topic_filter):
            return

        # Stay on the same domain as the seed — blocks cross-domain links such
        # as subscription pages, paywalls, and magazine-sales sites.
        if seed_domain and not is_same_domain(url, seed_domain):
            return

        visited.add(url)

        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed.")
            return

        try:
            response = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }, allow_redirects=True, stream=True)
            response.raise_for_status()
            content = response.content
        except Exception as fetch_error:
            print(f"  [CRAWL] Could not fetch: {url} — {fetch_error}")
            return

        # Use the final URL after any HTTP redirects so domain checks and link
        # resolution work correctly (e.g. example.com → www.example.com).
        final_url    = response.url if hasattr(response, 'url') else url
        content_type = response.headers.get('Content-Type', '')
        dtype        = detect_url_type(final_url, content, content_type, self.ext_to_type)
        source_name  = build_source_name(final_url)

        print(f"  [CRAWL] [{dtype.upper()}] {source_name}")

        # Decode text content for text-based formats
        text = None
        if dtype not in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            try:
                text = content.decode(response.encoding or 'utf-8', errors='replace')
            except Exception:
                text = content.decode('utf-8', errors='replace')

        # Only index if type passes the filter
        chunks: List[dict] = []
        if allowed_types is None or dtype in allowed_types:
            # Reuse chunk_url for a single page — avoids duplicating chunking logic
            chunks = self.chunk_url(url)
            all_chunks.extend(chunks)

        if progress_callback is not None:
            progress_callback(url, dtype, len(chunks))

        # Only recurse from HTML pages — documents have no crawlable links
        if dtype != 'html' or depth <= 0 or text is None:
            return

        links = extract_links(text, final_url, self.ext_to_type)
        current_seed_domain = seed_domain if seed_domain else final_url
        for link in links:
            if len(visited) >= max_pages:
                break
            self._crawl_url(
                link, depth - 1, max_pages,
                allowed_types, topic_filter, visited, all_chunks, progress_callback,
                is_seed=False,
                seed_domain=current_seed_domain,
            )

    def _dispatch_chunker(self, file_info: dict) -> List[dict]:
        """Route a file_info dict to the correct chunker function.

        Text formats (txt, md, csv, html) use chunkers.py.
        Binary formats (pdf, docx, xlsx, xls, pptx) use binary_chunkers.py.

        Args:
            file_info: Dict with keys filepath, filename, detected_type, is_misplaced.

        Returns:
            List of chunk dicts. Empty list when the type is unrecognised.
        """
        fp  = file_info['filepath']
        fn  = file_info['filename']
        ext = os.path.splitext(fn)[1].lower()
        t   = file_info['detected_type']

        if t == 'txt':
            return chunkers.chunk_txt(fp, fn,
                                      self.chunk_sizes['txt_chunk_size'],
                                      self.chunk_sizes['txt_chunk_overlap'])
        elif t == 'md':
            return chunkers.chunk_md(fp, fn,
                                     self.chunk_sizes['txt_chunk_size'],
                                     self.chunk_sizes['txt_chunk_overlap'])
        elif t == 'pdf':
            return binary_chunkers.chunk_pdf(fp, fn,
                                             self.chunk_sizes['pdf_chunk_sentences'])
        elif t == 'docx':
            return binary_chunkers.chunk_docx(fp, fn,
                                              self.chunk_sizes['docx_chunk_paras'])
        elif t == 'xlsx':
            if ext == '.xls':
                return binary_chunkers.chunk_xls(fp, fn)
            elif ext == '.csv':
                return chunkers.chunk_csv(fp, fn)
            else:
                return binary_chunkers.chunk_xlsx(fp, fn)
        elif t == 'csv':
            return chunkers.chunk_csv(fp, fn)
        elif t == 'pptx':
            return binary_chunkers.chunk_pptx(fp, fn,
                                              self.chunk_sizes['pptx_chunk_slides'])
        elif t == 'html':
            return chunkers.chunk_html(fp, fn,
                                       self.chunk_sizes['html_chunk_sentences'])
        else:
            print(f"  [SKIP] No chunker for type '{t}' ({fn})")
            return []
