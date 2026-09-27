"""url_utils.py — Stateless helper functions for URL type detection and crawling.

WHY THIS MODULE EXISTS:
    document_loader.py owns all ingestion state, but several URL-related
    operations are purely stateless — they take strings or bytes and return
    strings or lists without needing any class state. Extracting them here
    keeps document_loader.py within the 500-line limit and makes each
    function independently testable.

FUNCTIONS:
    detect_url_type   — 4-priority type detection (header → extension → bytes → html)
    build_source_name — build a short citation label from a URL
    is_utility_url    — check if a URL is a navigation/boilerplate page
    is_same_domain    — check if two URLs share the exact same domain
    extract_links     — extract crawlable links from an HTML page
"""

import os
import re
from typing import List
from urllib.parse import urljoin, urlparse

__all__ = [
    'detect_url_type',
    'build_source_name',
    'is_utility_url',
    'is_same_domain',
    'url_matches_topic',
    'extract_links',
]

# ── Content-Type header → document type mapping ───────────────────────────────

# Maps standard MIME types to the internal document type keys used throughout
_CONTENT_TYPE_MAP = {
    'application/pdf':                                                      'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/msword':                                                   'docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':   'xlsx',
    'application/vnd.ms-excel':                                             'xlsx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
    'application/vnd.ms-powerpoint':                                        'pptx',
    'text/csv':                                                             'csv',
    'text/plain':                                                           'txt',
    'text/markdown':                                                        'md',
    'text/html':                                                            'html',
    'application/xhtml+xml':                                                'html',
}

# ── Utility URL keywords ──────────────────────────────────────────────────────

# Pages with these words in their URL path OR hostname are navigation/boilerplate
# pages that rarely contain useful document content. Skipped during crawling.
_UTILITY_URL_KEYWORDS = {
    # Authentication and account management
    'login', 'signin', 'signup', 'register', 'logout', 'signout', 'auth',
    # E-commerce and subscriptions
    'cart', 'checkout', 'basket', 'payment', 'subscribe', 'unsubscribe',
    'newsletter', 'membership', 'premium',
    # Search and category index pages — not document content
    'search',
    # Legal / compliance
    'contact', 'privacy', 'terms', 'cookie', 'gdpr',
    # Admin / user pages
    'admin', 'dashboard', 'account', 'profile', 'settings',
    # Fundraising / infrastructure / geo-lookup tools
    'donate', 'geohack',
    # NOTE: 'main_page' is NOT here — the split regex uses '_' as a delimiter
    # so the compound slug would never match. It is handled via a direct
    # substring check in is_utility_url() instead.
}


def detect_url_type(url: str, content: bytes, content_type_header: str,
                    ext_to_type: dict) -> str:
    """Determine the document type of a fetched URL using 4 priority checks.

    The checks run in strict order and the first match wins:
        1. Content-Type response header (most reliable when present)
        2. File extension in the URL path (strips query strings first)
        3. PDF magic bytes — content[:4] == b'%PDF' (catches mislabelled PDFs)
        4. Default to 'html' when all other checks fail

    Args:
        url:                 The fetched URL string.
        content:             Raw response bytes.
        content_type_header: Value of the Content-Type response header.
        ext_to_type:         Extension → type mapping from config.EXT_TO_TYPE.

    Returns:
        Document type string: 'pdf', 'docx', 'xlsx', 'pptx', 'csv',
        'txt', 'md', or 'html'.
    """
    # Strip charset and other parameters from the content type
    clean_content_type = content_type_header.lower().split(';')[0].strip()
    detected_type      = _CONTENT_TYPE_MAP.get(clean_content_type)

    # Fuzzy fallback for non-standard content-type strings like
    # 'application/x-pdf' or 'application/octet-stream; type=word'
    if detected_type is None:
        if 'pdf'          in clean_content_type: detected_type = 'pdf'
        elif 'word'       in clean_content_type: detected_type = 'docx'
        elif 'excel'      in clean_content_type or 'spreadsheet' in clean_content_type:
            detected_type = 'xlsx'
        elif 'powerpoint' in clean_content_type or 'presentation' in clean_content_type:
            detected_type = 'pptx'
        elif 'csv'        in clean_content_type: detected_type = 'csv'

    # ── Priority 2: file extension in URL path ────────────────────────────
    if detected_type is None:
        # Strip query strings and fragments before extracting the extension
        parsed_path   = urlparse(url).path.lower()
        clean_path    = re.sub(r'[?#].*$', '', parsed_path)
        extension     = os.path.splitext(clean_path)[1]
        detected_type = ext_to_type.get(extension)

    # ── Priority 3: PDF magic bytes ───────────────────────────────────────
    # Every valid PDF starts with '%PDF' regardless of server headers
    if detected_type is None and content[:4] == b'%PDF':
        detected_type = 'pdf'

    # ── Priority 4: default to HTML ───────────────────────────────────────
    # The vast majority of bare URLs with no extension are webpages
    if detected_type is None:
        detected_type = 'html'

    return detected_type


def build_source_name(url: str) -> str:
    """Build a short, human-readable citation label from a URL.

    Combines the domain and path, strips trailing slashes, appends a
    truncated query string if present, and caps the total at 60 characters
    so citations stay readable inside answer text.

    Args:
        url: The full URL string.

    Returns:
        A citation label such as 'example.com/docs/api' or
        'example.com/search?q=python...'
    """
    parsed      = urlparse(url)
    source_name = (parsed.netloc + parsed.path).rstrip('/')

    # Append query string if present, but truncate long ones
    if parsed.query:
        source_name += '?' + parsed.query[:20] + ('...' if len(parsed.query) > 20 else '')

    # Cap total length so citations stay readable
    if len(source_name) > 60:
        source_name = source_name[:57] + '...'

    # Last resort: use the raw URL if nothing else produced a label
    if not source_name:
        source_name = url[:60]

    return source_name


def is_utility_url(url: str) -> bool:
    """Return True if the URL appears to be a navigation or utility page.

    Runs four checks in order:

        1. Utility keywords in path segments — login, donate, main_page, etc.
        2. Utility keywords in the hostname — catches donate.wikimedia.org,
           auth.wikimedia.org and similar subdomain-namespaced utility sites.
        3. MediaWiki action API endpoint — /w/index.php is the MediaWiki
           backend used for edit, history, print, and citation pages; these
           are never plain article content.
        4. MediaWiki namespace colons in path segments — Special:, Talk:,
           Help:, Wikipedia:, Portal:, etc.

    Args:
        url: The URL to check.

    Returns:
        True if the URL looks like a utility page, False otherwise.
    """
    parsed   = urlparse(url)
    path     = parsed.path.lower()
    hostname = parsed.netloc.lower()

    # Split path into individual segments for keyword matching
    path_parts = set(re.split(r'[/\-_.]', path))

    # ── Check 1: utility keywords in path segments ────────────────────────
    # NOTE: the split uses '_' as a delimiter, so compound slugs like main_page
    # become ['main', 'page'] and would never match 'main_page' as a token.
    # For that case we use the direct substring check in Check 1b below.
    if path_parts & _UTILITY_URL_KEYWORDS:
        return True

    # ── Check 1b: direct substring match for underscore-joined keywords ───
    # Handles /wiki/Main_Page, /wiki/Main_Page/subpage, etc.
    if 'main_page' in path:
        return True

    # ── Check 1c: disambiguation pages ────────────────────────────────────
    # /wiki/Elizabeth_Taylor_(disambiguation) is an index of name variants,
    # NOT article content. Following it causes exponential crawl growth because
    # each variant article then links to more pages.
    if 'disambiguation' in path:
        return True

    # ── Check 1d: bare domain URLs (no meaningful path) ───────────────────
    # https://wikimediafoundation.org, https://stats.wikimedia.org etc.
    # appear in Wikipedia's footer. They have no article path — just a domain.
    # These are organizational landing pages, not document content.
    if not path or path == '/':
        return True

    # ── Check 2: utility keywords in hostname ─────────────────────────────
    # Catches donate.wikimedia.org, auth.wikimedia.org, etc. where the
    # utility signal is in the subdomain, not the path.
    hostname_parts = set(re.split(r'[.\-]', hostname))
    if hostname_parts & _UTILITY_URL_KEYWORDS:
        return True

    # ── Check 3: MediaWiki action API endpoint ────────────────────────────
    # /w/index.php is used for ?action=edit, ?action=history, ?printable=yes,
    # ?title=Special:CiteThis, etc. None of these are article content pages.
    if '/w/index.php' in path:
        return True

    # ── Check 4: MediaWiki namespace colons in path segments ──────────────
    # Special:, Talk:, Help:, Wikipedia:, Portal: etc. all contain a colon.
    # Colons in URL path segments are extremely rare outside wiki namespaces.
    if any(':' in segment for segment in path_parts if segment):
        return True

    return False


def url_matches_topic(url: str, topic: str) -> bool:
    """Return True if the URL path contains the topic keyword.

    The check is case-insensitive and looks at the full URL path string —
    not just individual segments — so a topic like 'machine-learning' will
    match '/docs/machine-learning/intro' even though the hyphen splits it
    into two path segments.

    The seed URL (the one the user typed) is always considered a match so
    the crawl always starts. Only linked pages are filtered.

    Args:
        url:   The URL to check.
        topic: The keyword the URL path must contain (e.g. 'python', 'api').
               An empty string or whitespace means no filter — all URLs pass.

    Returns:
        True if the URL path contains the topic, or topic is empty.
    """
    # Empty topic means no filter — all URLs are allowed
    clean_topic = topic.strip().lower()
    if not clean_topic:
        return True

    # Check the path portion of the URL (not the domain or query string)
    path = urlparse(url).path.lower()
    return clean_topic in path


def is_same_domain(url: str, base_url: str) -> bool:
    """Return True if url is on the same base domain as base_url.

    Strips the 'www.' prefix before comparing so that example.com and
    www.example.com are treated as the same domain. This handles the very
    common case where a seed URL typed without 'www.' redirects to the
    www version and all page links point back to the www version.

    Blog subdomains (blog.example.com) are still treated as different
    domains — only 'www.' is normalised away.

    Args:
        url:      The URL to check.
        base_url: The seed URL whose domain is used as the reference.

    Returns:
        True if both URLs share the same base domain (after www. strip).
    """
    def _strip_www(netloc: str) -> str:
        """Remove leading 'www.' so www.example.com == example.com."""
        return netloc.lower().removeprefix('www.')

    return _strip_www(urlparse(url).netloc) == _strip_www(urlparse(base_url).netloc)


def extract_links(html_content: str, base_url: str, ext_to_type: dict) -> List[str]:
    """Extract crawlable links from an HTML page.

    Resolves relative URLs to absolute, then filters out:
        - Fragment-only links (#section)
        - mailto:, tel:, javascript: links
        - Utility/navigation pages (login, cart, privacy, etc.)
        - Duplicate URLs

    Links to other domains ARE included — the crawler follows whatever
    links appear on the page. The max_pages cap and topic_filter in
    chunk_url_recursive() control how far the crawl goes.

    Args:
        html_content: Decoded HTML string.
        base_url:     The URL the page was fetched from — used for
                      resolving relative links to absolute URLs.
        ext_to_type:  Extension → type mapping (reserved for future
                      type-specific filtering).

    Returns:
        Deduplicated list of absolute URLs that are candidates for crawling,
        in the order they appeared on the page.
    """
    # Parse the HTML and collect all href attribute values.
    # Skip links that carry a hreflang attribute — these are interlanguage links
    # (e.g. Wikipedia's sidebar "other languages" links: af.wikipedia.org, ar.wikipedia.org…).
    # They always point to the same content in a different language and are rarely
    # useful for a single-language RAG knowledge base.
    try:
        from bs4 import BeautifulSoup
        soup  = BeautifulSoup(html_content, 'html.parser')
        hrefs = [
            tag.get('href', '')
            for tag in soup.find_all('a', href=True)
            if not tag.get('hreflang')   # skip interlanguage / alternate-language links
        ]
    except ImportError:
        # Fallback if BeautifulSoup is not installed — hreflang filter not applied
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html_content)

    clean_links = []
    for href in hrefs:
        href = href.strip()

        # Skip empty, fragment-only, and non-http links
        if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue

        # Resolve relative paths to absolute URLs using the page's base URL
        absolute_url = urljoin(base_url, href)

        # Only follow standard web links
        if not absolute_url.startswith(('http://', 'https://')):
            continue

        # Remove the fragment portion (#section) — fragments are not separate pages
        absolute_url = absolute_url.split('#')[0]

        # Skip navigation and boilerplate pages (login, cart, privacy, etc.)
        if is_utility_url(absolute_url):
            continue

        clean_links.append(absolute_url)

    # Remove duplicate URLs while preserving the order they appeared
    seen         = set()
    unique_links = []
    for link in clean_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links
