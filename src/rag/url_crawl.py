"""url_crawl.py — Stateless helpers for recursive URL crawling and content chunking.

WHY THIS FILE EXISTS:
    _crawl_url (152 lines) and _chunk_content (102 lines) were private methods on
    DocumentLoader that used no class state beyond chunk_sizes and ext_to_type.
    Extracting them here as module-level functions keeps document_loader.py under
    500 lines and makes each function independently testable without a full class.

USAGE:
    These functions are called exclusively by DocumentLoader. Do not call them
    directly from anywhere else — they are implementation details of ingestion.
"""

import os
import tempfile
from typing import Callable, Dict, List, Optional, Set

from src.rag import chunkers
from src.rag.url_utils import build_source_name, detect_url_type, extract_links, is_same_domain, url_matches_topic

__all__ = ['chunk_content', 'crawl_url', 'search_duckduckgo_html']


def search_duckduckgo_html(query: str, num_results: int) -> List[str]:
    """Search DuckDuckGo via the public HTML form endpoint and return result URLs.

    Uses the HTML form endpoint (html.duckduckgo.com) rather than a third-party
    library to avoid rate-limiting and dependency on the duckduckgo-search package,
    which breaks frequently across versions. The HTML endpoint is the same one used
    by the public website and is not subject to the aggressive IP-level rate limiting
    applied to the JS API endpoints that duckduckgo-search v6 uses internally.

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

    from bs4 import BeautifulSoup
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


def chunk_content(
    content: bytes,
    text: Optional[str],
    dtype: str,
    source_name: str,
    chunk_sizes: Dict,
    dispatch_chunker_fn: Callable,
) -> List[dict]:
    """Route fetched URL content to the correct chunker based on detected type.

    Binary formats (pdf, docx, xlsx, pptx, xls) are written to a temporary
    file on disk so the existing file-based chunkers can read them. The
    temp file is always deleted after chunking, even if an error occurs.

    Text formats (csv, md) are also written to a temp file because the
    chunkers expect a filepath. Plain text and HTML are chunked directly
    from the decoded string without touching disk.

    Args:
        content:            Raw response bytes from the HTTP request.
        text:               Decoded string for text-based formats; None for binary.
        dtype:              Document type string ('pdf', 'html', 'txt', etc.).
        source_name:        Short citation label built from the URL.
        chunk_sizes:        Dict of chunk-size constants from DocumentLoader.
        dispatch_chunker_fn: Callable that routes a file_info dict to the right chunker.

    Returns:
        List of chunk dicts. Empty list if chunking fails or type is unknown.
    """
    # Binary formats — write bytes to a temp file, chunk it, then delete
    binary_types = {'pdf', 'docx', 'xlsx', 'pptx', 'xls'}
    if dtype in binary_types:
        # Choose the right file extension so the chunker knows what format it is
        extension_map = {
            'pdf': '.pdf', 'docx': '.docx', 'xlsx': '.xlsx',
            'pptx': '.pptx', 'xls': '.xls',
        }
        suffix = extension_map[dtype]
        try:
            # Write to a named temp file that persists until we delete it
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Build a fake file_info dict so dispatch_chunker_fn can handle it
            file_info = {
                'filepath':      tmp_path,
                'filename':      source_name,
                'detected_type': dtype,
                'found_in':      '',
                'canonical_dir': '',
                'is_misplaced':  False,
            }
            return dispatch_chunker_fn(file_info)
        except Exception as error:
            print(f"  [ERROR] Could not chunk {dtype.upper()} from URL: {error}")
            return []
        finally:
            # Always clean up the temp file — even if an exception was raised
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # CSV and Markdown — write decoded text to a temp file
    if dtype in ('csv', 'md'):
        suffix = '.csv' if dtype == 'csv' else '.md'
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, mode='w', encoding='utf-8'
            ) as tmp:
                tmp.write(text or '')
                tmp_path = tmp.name

            file_info = {
                'filepath':      tmp_path,
                'filename':      source_name,
                'detected_type': dtype,
                'found_in':      '',
                'canonical_dir': '',
                'is_misplaced':  False,
            }
            return dispatch_chunker_fn(file_info)
        except Exception as error:
            print(f"  [ERROR] Could not chunk {dtype.upper()} from URL: {error}")
            return []
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Plain text — chunk line by line directly from the decoded string
    if dtype == 'txt':
        return chunkers.chunk_txt_from_string(
            text or '', source_name,
            chunk_sizes['txt_chunk_size'],
            chunk_sizes['txt_chunk_overlap'],
        )

    # HTML / webpage — strip tags and chunk sentences directly from the string
    if dtype == 'html':
        return chunkers.chunk_html_from_string(
            text or '', source_name,
            chunk_sizes['html_chunk_sentences'],
        )

    # Unknown type — skip with a notice
    print(f"  [SKIP] No URL chunker for type '{dtype}' ({source_name})")
    return []


def crawl_url(
    url: str,
    depth: int,
    max_pages: int,
    allowed_types: Optional[Set[str]],
    topic_filter: str,
    visited: Set[str],
    all_chunks: List[dict],
    progress_callback: Optional[Callable],
    ext_to_type: Dict,
    chunk_content_fn: Callable,
    is_seed: bool = False,
    seed_domain: str = '',
) -> None:
    """Recursively fetch a URL and follow links up to the given depth.

    This is the internal engine behind DocumentLoader.chunk_url_recursive().
    It mutates the shared `visited` set and `all_chunks` list so the caller
    can track progress across the full crawl without returning values at each level.

    Recursion stops when:
      - The URL has already been visited
      - The max_pages cap is reached
      - depth reaches 0
      - The URL does not match the topic filter (unless it is the seed URL)
      - The URL is on a different domain than the seed URL
      - The URL fails to fetch

    Document URLs (PDF, DOCX, etc.) are chunked but NOT recursed into —
    they do not contain HTML links to follow.

    Args:
        url:               Absolute URL to fetch.
        depth:             Remaining link levels to follow (decrements each call).
        max_pages:         Hard cap on total pages fetched across the whole crawl.
        allowed_types:     If set, only index pages whose detected type is in this set.
        topic_filter:      Keyword the URL path must contain. Ignored for the seed URL.
        visited:           Shared set of already-fetched URLs (mutated in place).
        all_chunks:        Shared list accumulating all chunk dicts (mutated in place).
        progress_callback: Optional callback called after each page is fetched.
                           Signature: callback(url, detected_type, chunk_count).
        ext_to_type:       Dict mapping file extension → document type (from DocumentLoader).
        chunk_content_fn:  Callable with signature chunk_content(content, text, dtype, source_name).
        is_seed:           True only for the very first (user-supplied) URL — the seed
                           is always fetched regardless of the topic filter.
        seed_domain:       The hostname of the first URL fetched in this crawl session.
                           All recursive calls must stay on this domain.
    """
    # Add https:// if the URL has no scheme (e.g. en.wikipedia.org/wiki/...)
    if url and not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Normalise the URL — strip trailing slash and fragment (#section)
    url = url.rstrip('/')
    url = url.split('#')[0]

    # Stop if we have already visited this URL or hit the page cap
    if url in visited or len(visited) >= max_pages:
        return

    # Apply topic filter to all pages except the seed URL
    if not is_seed and not url_matches_topic(url, topic_filter):
        return

    # Stay on the same domain as the seed URL
    if seed_domain and not is_same_domain(url, seed_domain):
        return

    # Mark as visited before fetching to prevent parallel loops
    visited.add(url)

    try:
        import requests
    except ImportError:
        print("  [WARNING] requests not installed. pip install requests")
        return

    # Fetch the page content
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        }, allow_redirects=True, stream=True)
        response.raise_for_status()
        content = response.content
    except Exception as fetch_error:
        print(f"  [CRAWL] Could not fetch: {url} — {fetch_error}")
        return

    # Detect the document type using the same 4-priority pipeline as chunk_url()
    final_url    = response.url if hasattr(response, 'url') else url
    content_type = response.headers.get('Content-Type', '')
    dtype        = detect_url_type(final_url, content, content_type, ext_to_type)
    source_name  = build_source_name(final_url)

    print(f"  [CRAWL] [{dtype.upper()}] {source_name}")

    # Decode text content for text-based formats
    text = None
    if dtype not in ('pdf', 'docx', 'xlsx', 'pptx', 'xls'):
        try:
            text = content.decode(response.encoding or 'utf-8', errors='replace')
        except Exception:
            text = content.decode('utf-8', errors='replace')

    # Only index this page if its type is in the allowed set (or no filter set)
    chunks: List[dict] = []
    if allowed_types is None or dtype in allowed_types:
        chunks = chunk_content_fn(content, text, dtype, source_name)
        all_chunks.extend(chunks)

    # Call the progress callback so the UI can update its progress bar
    if progress_callback is not None:
        progress_callback(url, dtype, len(chunks))

    # Only follow links from HTML pages — PDFs and DOCX files have no links to crawl
    if dtype not in {'html'} or depth <= 0 or text is None:
        return

    # Use final_url (post-redirect) as the base so relative links resolve correctly
    links = extract_links(text, final_url, ext_to_type)

    # Lock the crawl to this domain from the first resolved URL onward
    current_seed_domain = seed_domain if seed_domain else final_url

    # Recurse into each discovered link, reducing depth by one each level
    for link in links:
        if len(visited) >= max_pages:
            break
        crawl_url(
            link, depth - 1, max_pages,
            allowed_types, topic_filter, visited, all_chunks, progress_callback,
            ext_to_type=ext_to_type,
            chunk_content_fn=chunk_content_fn,
            is_seed=False,
            seed_domain=current_seed_domain,
        )
