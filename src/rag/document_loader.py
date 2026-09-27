"""document_loader.py — DocumentLoader class: owns all ingestion."""

import os
import re
import sys
from typing import Callable, List, Optional, Set
from urllib.parse import urlparse

from src.rag import binary_chunkers, chunkers
from src.rag.config import (
    DOC_FOLDERS,
    DOCS_ROOT,
    DOCX_CHUNK_PARAS,
    EXT_TO_TYPE,
    HTML_CHUNK_SENTENCES,
    PDF_CHUNK_SENTENCES,
    PPTX_CHUNK_SLIDES,
    TXT_CHUNK_OVERLAP,
    TXT_CHUNK_SIZE,
)
from src.rag.url_crawl import chunk_content, crawl_url, search_duckduckgo_html
from src.rag.url_utils import build_source_name, detect_url_type

__all__ = ['DocumentLoader']


class DocumentLoader:
    """Owns all document ingestion: reading files, parsing 9 formats, chunking, URL fetching.

    All format-specific chunking logic lives in chunkers.py as stateless
    functions. This class owns the state (folder paths, chunk sizes) and
    the orchestration: scanning files, dispatching to the right chunker,
    and fetching URLs.

    State:
        doc_folders:  dict mapping type → folder path (from config)
        ext_to_type:  dict mapping file extension → document type
        chunk_sizes:  dict of all chunk-size constants (txt, pdf, docx, pptx, html)

    Public API:
        ensure_folders()                          — create ./docs subfolders if missing
        scan_all_files()                          — find every file under DOCS_ROOT, flag misplaced ones
        chunk_all_documents()                     — scan + dispatch all files → list of chunk dicts
        chunk_directory(directory)                — chunk all files in any given folder (used for benchmarking)
        chunk_url(url)                            — fetch a single URL, detect type, return chunks
        chunk_url_recursive(url, depth,           — crawl a seed URL and all linked pages
                            max_pages,
                            allowed_types,
                            progress_callback)
    """

    def __init__(self) -> None:
        """Bind config constants to instance state so they are easy to override in tests."""
        self.docs_root   = DOCS_ROOT
        self.doc_folders = DOC_FOLDERS
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

    def ensure_folders(self) -> None:
        """Create docs_root and all subfolders if they do not exist yet."""
        os.makedirs(self.docs_root, exist_ok=True)
        for folder in self.doc_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"  Created folder: {folder}/")

    def scan_all_files(self) -> List[dict]:
        """Recursively scan all files under DOCS_ROOT and return a list of file info dicts.

        Walks the entire DOCS_ROOT tree at any depth, so a folder dropped anywhere
        under ./docs/ — with any mix of file types and any number of sublevels —
        will have all its files detected and processed.

        The relative path from DOCS_ROOT is stored as 'filename' so that two files
        with the same name in different subfolders can be told apart in source citations
        (e.g. 'project/data/q1.xlsx' instead of just 'q1.xlsx').

        Files not in their canonical type subfolder are flagged as misplaced but
        are still processed with a notice.

        Returns:
            List of dicts with keys: filepath, filename (relative path), detected_type,
            found_in, canonical_dir, is_misplaced.
        """
        found = []
        if not os.path.isdir(self.docs_root):
            return found

        for root, _, files in os.walk(self.docs_root):
            for fname in files:
                ext           = os.path.splitext(fname)[1].lower()
                detected_type = self.ext_to_type.get(ext)
                if detected_type is None:
                    continue  # unsupported extension — skip silently

                filepath      = os.path.join(root, fname)
                # Store path relative to docs_root so source citations are unique
                # across deeply nested folders (e.g. 'project/data/q1.xlsx')
                relative_path = os.path.relpath(filepath, self.docs_root)
                canonical_dir = self.doc_folders[detected_type]
                is_misplaced  = (root != canonical_dir)

                found.append({
                    'filepath':      filepath,
                    'filename':      relative_path,
                    'detected_type': detected_type,
                    'found_in':      root,
                    'canonical_dir': canonical_dir,
                    'is_misplaced':  is_misplaced,
                })
                if is_misplaced:
                    print(f"  [MISPLACED] '{relative_path}' "
                          f"— detected as '{detected_type.upper()}', "
                          f"canonical folder is '{canonical_dir}/'. Processing anyway.")
        return found

    def chunk_all_documents(self) -> List[dict]:
        """Scan every file under ./docs/, route each to the correct chunker, and return all chunks.

        Files in the wrong folder are still processed (with a notice).
        Exits with code 1 if no supported documents are found.

        Returns:
            Flat list of all chunk dicts from every document.
        """
        print("\nLoading documents...")
        file_list = self.scan_all_files()

        if not file_list:
            print(f"\nNo supported documents found under '{self.docs_root}/'")
            print("Supported types: PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, MD, HTML")
            print("Place files in the matching subfolder (or any subfolder — smart detection handles the rest).")
            sys.exit(1)

        all_chunks  = []
        type_counts = {}

        for file_info in file_list:
            t      = file_info['detected_type']
            chunks = self._dispatch_chunker(file_info)
            all_chunks.extend(chunks)
            type_counts[t] = type_counts.get(t, 0) + len(chunks)
            flag = " [MISPLACED — processed anyway]" if file_info['is_misplaced'] else ""
            print(f"  [{t.upper()}] '{file_info['filename']}': {len(chunks)} chunks{flag}")

        print("\n  Chunk summary by type:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t.upper():<8} {count} chunks")
        print(f"  Total: {len(all_chunks)} chunks\n")

        return all_chunks

    def chunk_directory(self, directory: str) -> List[dict]:
        """Chunk all supported files inside a given directory (flat, one level deep).

        Unlike chunk_all_documents() which is tied to the configured ./docs/ tree,
        this method accepts any arbitrary folder path. This is used by the benchmark
        runner to load sample documents from benchmark_docs/ without mixing them into
        the main document index.

        Files that have no supported extension are silently skipped.

        Args:
            directory: Absolute or relative path to the folder to scan.

        Returns:
            Flat list of chunk dicts from every supported file in the directory.
            Returns an empty list if the directory does not exist or has no files.
        """
        all_chunks = []

        if not os.path.isdir(directory):
            print(f"  [benchmark] Directory not found: {directory}")
            return all_chunks

        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)

            # Skip sub-directories — only process files at the top level
            if not os.path.isfile(filepath):
                continue

            extension     = os.path.splitext(filename)[1].lower()
            detected_type = self.ext_to_type.get(extension)

            # Skip file types we do not support
            if detected_type is None:
                continue

            file_info = {
                'filepath':      filepath,
                'filename':      filename,
                'detected_type': detected_type,
                'found_in':      directory,
                'canonical_dir': directory,
                'is_misplaced':  False,
            }

            chunks = self._dispatch_chunker(file_info)
            all_chunks.extend(chunks)
            print(f"  [benchmark/{detected_type.upper()}] '{filename}': {len(chunks)} chunks")

        return all_chunks

    def chunk_url(self, url: str) -> List[dict]:
        """Fetch a single URL and return chunks using the appropriate format handler.

        Type detection uses four priorities in strict order (via url_utils):
            1. Content-Type response header (most reliable).
            2. File extension in the URL path (strips query strings first).
            3. PDF magic bytes sniff (content[:4] == b'%PDF').
            4. Default to 'html' when no other signal is present.

        Args:
            url: Public HTTP/HTTPS URL to fetch.

        Returns:
            List of chunk dicts. Empty list on network error.
        """
        try:
            import requests
        except ImportError:
            print("  [WARNING] requests not installed. pip install requests")
            return []

        url = url.strip()
        # Add https:// if the user pasted a URL without a scheme (e.g. en.wikipedia.org/...)
        # Without a scheme, urlparse returns an empty netloc and requests raises MissingSchema.
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        print(f"\n  Fetching URL: {url}")
        try:
            resp = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }, allow_redirects=True, stream=True)
            resp.raise_for_status()
            content = resp.content
        except Exception as e:
            print(f"  [ERROR] Could not fetch URL: {e}")
            return []

        content_type = resp.headers.get('Content-Type', '')
        dtype        = detect_url_type(url, content, content_type, self.ext_to_type)
        source_name  = build_source_name(url)

        print(f"  Detected type: {dtype.upper()} (content-type: {content_type.split(';')[0].strip()})")

        # Decode text content for text-based formats
        text = None
        if dtype not in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
            try:
                text = content.decode(resp.encoding or 'utf-8', errors='replace')
            except Exception:
                text = content.decode('utf-8', errors='replace')

        return self._chunk_content(content, text, dtype, source_name)

    def chunk_url_recursive(
        self,
        url: str,
        depth: int = 2,
        max_pages: int = 25,
        allowed_types: Optional[Set[str]] = None,
        topic_filter: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> List[dict]:
        """Crawl a seed URL and all linked pages up to a given depth.

        Follows links on HTML pages and chunks every discovered URL using the
        same type-detection and chunking pipeline as chunk_url(). Document URLs
        (PDF, DOCX, XLSX, etc.) are chunked but not recursed into — they do not
        contain links to follow.

        Args:
            url:               Seed URL to start crawling from.
            depth:             How many link-levels deep to follow (1 = direct
                               links only, 2 = links of links, etc.).
            max_pages:         Maximum total pages/documents to fetch and index.
            allowed_types:     Set of type strings to index, e.g. {'html', 'pdf'}.
                               If None, all detected types are indexed.
            topic_filter:      Optional keyword the URL path must contain to be
                               crawled (e.g. 'python', 'api'). Empty string means
                               no filter — all pages on the domain are crawled.
                               The seed URL itself is always crawled regardless.
            progress_callback: Optional function called after each page is fetched.
                               Signature: callback(url, detected_type, chunk_count).

        Returns:
            Flat list of all chunk dicts from every crawled URL.
            Returns an empty list if the seed URL is unreachable. Returns partial results if max_pages is reached before all links are followed.
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
                                 Leave empty to crawl all pages of each result.
            progress_callback:   Optional callback(url, dtype, chunk_count) called
                                 after each page is fetched.

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

        # Crawl each result URL and collect all chunks
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
                # Log but continue — one bad result URL should not abort the rest
                print(f"  [SEARCH] Error crawling {url}: {error}")

        print(f"\n  [SEARCH] Done — {len(urls)} URLs crawled, "
              f"{len(all_chunks)} chunks total")
        return all_chunks

    # ----------------------------------------------------------------- Private

    def _search_duckduckgo_html(self, query: str, num_results: int) -> List[str]:
        """Delegate to url_crawl.search_duckduckgo_html — no class state needed.

        Args:
            query:       The search query string.
            num_results: Maximum number of result URLs to return.

        Returns:
            List of result URLs (ads and DuckDuckGo redirect links filtered out).
        """
        return search_duckduckgo_html(query, num_results)

    def _chunk_content(
        self,
        content: bytes,
        text: Optional[str],
        dtype: str,
        source_name: str,
    ) -> List[dict]:
        """Delegate to url_crawl.chunk_content, passing this loader's settings.

        Args:
            content:     Raw response bytes from the HTTP request.
            text:        Decoded string for text-based formats; None for binary.
            dtype:       Document type string ('pdf', 'html', 'txt', etc.).
            source_name: Short citation label built from the URL.

        Returns:
            List of chunk dicts. Empty list if chunking fails or type is unknown.
        """
        return chunk_content(
            content, text, dtype, source_name,
            chunk_sizes=self.chunk_sizes,
            dispatch_chunker_fn=self._dispatch_chunker,
        )

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
        """Delegate to url_crawl.crawl_url, passing this loader's settings.

        Args:
            url:               Absolute URL to fetch.
            depth:             Remaining link levels to follow (decrements each call).
            max_pages:         Hard cap on total pages fetched across the whole crawl.
            allowed_types:     If set, only index pages whose detected type is in this set.
            topic_filter:      Keyword the URL path must contain. Ignored for the seed URL.
            visited:           Shared set of already-fetched URLs (mutated in place).
            all_chunks:        Shared list accumulating all chunk dicts (mutated in place).
            progress_callback: Optional callback — signature: callback(url, dtype, chunk_count).
            is_seed:           True only for the very first URL the user supplied.
            seed_domain:       Domain to stay on. Empty string means no restriction yet.
        """
        crawl_url(
            url, depth, max_pages,
            allowed_types, topic_filter, visited, all_chunks, progress_callback,
            ext_to_type=self.ext_to_type,
            chunk_content_fn=self._chunk_content,
            is_seed=is_seed,
            seed_domain=seed_domain,
        )

    def _dispatch_chunker(self, file_info: dict) -> List[dict]:
        """Route a single file to the correct chunker function based on its detected type.

        Args:
            file_info: Dict with keys filepath, filename, detected_type, is_misplaced
                       as produced by scan_all_files().

        Returns:
            List of chunk dicts from the appropriate chunker, or [] on error.
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
            # .xls uses xlrd (legacy binary format); .csv routed correctly if misplaced
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
