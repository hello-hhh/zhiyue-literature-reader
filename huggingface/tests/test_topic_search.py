"""test_topic_search.py — Tests for DocumentLoader.chunk_topic_search() and
_search_duckduckgo_html().

Coverage:
    Unit:
        - _search_duckduckgo_html returns cleaned result URLs
        - ad links (y.js) are filtered out
        - non-http links are excluded
        - empty response body returns empty list
        - HTTP error returns empty list

    Functional:
        - chunk_topic_search calls search then crawls each result URL
        - chunk_topic_search returns chunks from all crawled pages
        - num_results controls how many URLs are searched
        - depth and max_pages_per_result are forwarded to chunk_url_recursive
        - topic_filter is forwarded to chunk_url_recursive
        - a failed crawl on one result does not abort the rest

    Boundary / negative:
        - zero results from search returns empty list
        - all result URLs fail to crawl → returns empty list
        - network error during DuckDuckGo request returns empty list

Mock strategy:
    requests.post is mocked for the DuckDuckGo HTML endpoint.
    requests.get  is mocked for the per-URL crawl fetches.
    BeautifulSoup, chunking logic, and BM25 are NOT mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader():
    """Return a DocumentLoader with a minimal config — no real folders needed."""
    from src.rag.document_loader import DocumentLoader
    loader = DocumentLoader()
    return loader


def _ddg_html_response(*urls):
    """Build a minimal DuckDuckGo HTML results page containing the given URLs.

    Each URL becomes a <a class="result__a" href="..."> element, which is
    exactly what _search_duckduckgo_html() looks for.
    """
    links = ''.join(
        f'<a class="result__a" href="{url}">Result</a>' for url in urls
    )
    return f'<html><body>{links}</body></html>'


def _mock_post(html_body, status_code=200):
    """Return a mock requests.post response with the given HTML body."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text        = html_body
    mock.raise_for_status = MagicMock(
        side_effect=None if status_code == 200
        else Exception(f"HTTP {status_code}")
    )
    return mock


def _mock_get(html_body, final_url=None):
    """Return a mock requests.get response for a crawled page."""
    mock            = MagicMock()
    mock.status_code = 200
    mock.content    = html_body.encode('utf-8')
    mock.text       = html_body
    mock.encoding   = 'utf-8'
    mock.url        = final_url or 'https://example.com/page'
    mock.headers    = {'Content-Type': 'text/html'}
    mock.raise_for_status = MagicMock()
    return mock


def _page_html(text):
    """Minimal HTML page with readable text content."""
    return f'<html><body><p>{text}</p></body></html>'


# ---------------------------------------------------------------------------
# Unit — _search_duckduckgo_html
# ---------------------------------------------------------------------------

class TestSearchDuckduckgoHtml:
    """Unit tests for _search_duckduckgo_html — the HTML-endpoint search helper."""

    def test_returns_clean_result_urls(self):
        """Organic result URLs are returned as a plain list."""
        html     = _ddg_html_response(
            'https://en.wikipedia.org/wiki/Python',
            'https://www.python.org',
        )
        post_mock = _mock_post(html)

        with patch('requests.post', return_value=post_mock):
            urls = _make_loader()._search_duckduckgo_html('python', num_results=10)

        assert 'https://en.wikipedia.org/wiki/Python' in urls
        assert 'https://www.python.org' in urls

    def test_ad_links_filtered_out(self):
        """Links containing /y.js? (DuckDuckGo ad redirects) are excluded."""
        html = (
            '<html><body>'
            '<a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=amazon.com">Ad</a>'
            '<a class="result__a" href="https://real-result.com/article">Real</a>'
            '</body></html>'
        )
        post_mock = _mock_post(html)

        with patch('requests.post', return_value=post_mock):
            urls = _make_loader()._search_duckduckgo_html('test', num_results=10)

        assert 'https://real-result.com/article' in urls
        assert not any('/y.js?' in u for u in urls)

    def test_num_results_caps_returned_urls(self):
        """Only up to num_results URLs are returned even if the page has more."""
        html = _ddg_html_response(
            'https://a.com/1', 'https://b.com/2', 'https://c.com/3',
            'https://d.com/4', 'https://e.com/5',
        )
        post_mock = _mock_post(html)

        with patch('requests.post', return_value=post_mock):
            urls = _make_loader()._search_duckduckgo_html('test', num_results=3)

        assert len(urls) == 3

    def test_empty_response_returns_empty_list(self):
        """A response with no result links returns an empty list (not an error)."""
        post_mock = _mock_post('<html><body>No results.</body></html>')

        with patch('requests.post', return_value=post_mock):
            urls = _make_loader()._search_duckduckgo_html('xyzzy_nothing_here', num_results=5)

        assert urls == []

    def test_http_error_returns_empty_list(self):
        """An HTTP error from DuckDuckGo returns an empty list (not a crash)."""
        error_mock = MagicMock()
        error_mock.raise_for_status.side_effect = Exception('HTTP 503')

        with patch('requests.post', side_effect=Exception('connection refused')):
            urls = _make_loader()._search_duckduckgo_html('test', num_results=5)

        assert urls == []

    def test_non_http_links_excluded(self):
        """Links that do not start with http:// or https:// are silently dropped."""
        html = (
            '<html><body>'
            '<a class="result__a" href="ftp://files.example.com/data">FTP</a>'
            '<a class="result__a" href="https://good.com/page">Good</a>'
            '</body></html>'
        )
        post_mock = _mock_post(html)

        with patch('requests.post', return_value=post_mock):
            urls = _make_loader()._search_duckduckgo_html('test', num_results=10)

        assert urls == ['https://good.com/page']


# ---------------------------------------------------------------------------
# Functional — chunk_topic_search end-to-end
# ---------------------------------------------------------------------------

class TestChunkTopicSearch:
    """Functional tests for chunk_topic_search — search → crawl → index."""

    def test_returns_chunks_from_crawled_pages(self):
        """Chunks are extracted from every successfully crawled result URL."""
        ddg_html  = _ddg_html_response('https://example.com/article')
        page_html = _page_html(
            'The quick brown fox jumps over the lazy dog. '
            'Python is a high-level programming language.'
        )

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get',  return_value=_mock_get(page_html, 'https://example.com/article')):
            chunks = _make_loader().chunk_topic_search('python', num_results=1, depth=0)

        assert len(chunks) > 0

    def test_num_results_limits_crawl_urls(self):
        """Only num_results URLs are crawled regardless of how many the search returns."""
        ddg_html  = _ddg_html_response(
            'https://example.com/a',
            'https://example.com/b',
            'https://example.com/c',
        )
        page_html = _page_html('Some content about the topic.')
        get_calls = []

        def _tracked_get(url, **kwargs):
            get_calls.append(url)
            return _mock_get(page_html, final_url=url)

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', side_effect=_tracked_get):
            _make_loader().chunk_topic_search('test', num_results=2, depth=0)

        # Only 2 of the 3 result URLs should have been crawled (unique URL count).
        assert len(set(get_calls)) == 2

    def test_zero_search_results_returns_empty_list(self):
        """When the search returns no URLs, chunk_topic_search returns []."""
        empty_ddg = _mock_post('<html><body></body></html>')

        with patch('requests.post', return_value=empty_ddg):
            chunks = _make_loader().chunk_topic_search('xyzzy', num_results=5, depth=0)

        assert chunks == []

    def test_failed_crawl_on_one_result_does_not_abort(self):
        """A network error crawling one result URL does not abort the others."""
        ddg_html  = _ddg_html_response(
            'https://example.com/good',
            'https://example.com/bad',
        )
        good_html = _page_html('Good content about machine learning models.')

        def _tracked_get(url, **kwargs):
            if 'bad' in url:
                raise Exception('Connection refused')
            return _mock_get(good_html, final_url=url)

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', side_effect=_tracked_get):
            chunks = _make_loader().chunk_topic_search('test', num_results=2, depth=0)

        # Good page should still have been indexed despite the bad one failing.
        assert len(chunks) > 0

    def test_all_crawls_fail_returns_empty_list(self):
        """If every result URL fails to fetch, an empty list is returned."""
        ddg_html = _ddg_html_response('https://example.com/a', 'https://example.com/b')

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', side_effect=Exception('All down')):
            chunks = _make_loader().chunk_topic_search('test', num_results=2, depth=0)

        assert chunks == []

    def test_depth_forwarded_to_crawl(self):
        """depth parameter controls how many link-levels are followed per result URL."""
        ddg_html   = _ddg_html_response('https://example.com')
        seed_html  = (
            '<html><body><p>Seed content.</p>'
            '<a href="https://example.com/child">child</a>'
            '</body></html>'
        )
        child_html = _page_html('Child page content about the search topic.')
        get_calls  = []

        def _tracked_get(url, **kwargs):
            get_calls.append(url)
            if 'child' in url:
                return _mock_get(child_html, final_url=url)
            return _mock_get(seed_html, final_url=url)

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', side_effect=_tracked_get):
            _make_loader().chunk_topic_search('test', num_results=1, depth=1, max_pages_per_result=5)

        # At depth=1 the child link should also have been fetched.
        child_fetched = any('child' in u for u in get_calls)
        assert child_fetched

    def test_max_pages_per_result_limits_crawl(self):
        """max_pages_per_result caps total pages fetched per result URL."""
        ddg_html  = _ddg_html_response('https://example.com')
        seed_html = (
            '<html><body><p>Seed.</p>'
            + ''.join(f'<a href="https://example.com/p{i}">p{i}</a>' for i in range(10))
            + '</body></html>'
        )
        page_html = _page_html('Linked page content.')
        get_calls = []

        def _tracked_get(url, **kwargs):
            get_calls.append(url)
            if 'example.com/p' in url:
                return _mock_get(page_html, final_url=url)
            return _mock_get(seed_html, final_url=url)

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', side_effect=_tracked_get):
            _make_loader().chunk_topic_search(
                'test', num_results=1, depth=1, max_pages_per_result=3,
            )

        # Seed + at most 2 linked pages = 3 total.
        assert len(set(get_calls)) <= 3


# ---------------------------------------------------------------------------
# Boundary / negative
# ---------------------------------------------------------------------------

class TestTopicSearchBoundary:
    """Boundary and negative tests for chunk_topic_search."""

    def test_single_result_url_single_page(self):
        """num_results=1, depth=0 — exactly one URL fetched, chunks returned."""
        ddg_html  = _ddg_html_response('https://example.com/only')
        page_html = _page_html('The only result page with some useful text content here.')

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', return_value=_mock_get(page_html, 'https://example.com/only')):
            chunks = _make_loader().chunk_topic_search('test', num_results=1, depth=0)

        assert len(chunks) > 0

    def test_topic_filter_forwarded(self):
        """topic_filter is passed through — pages whose path lacks the keyword are skipped."""
        ddg_html  = _ddg_html_response('https://example.com')
        seed_html = (
            '<html><body><p>Seed.</p>'
            '<a href="https://example.com/python/guide">python</a>'
            '<a href="https://example.com/java/guide">java</a>'
            '</body></html>'
        )
        page_html = _page_html('Language guide page.')
        get_calls = []

        def _tracked_get(url, **kwargs):
            get_calls.append(url)
            return _mock_get(page_html, final_url=url) if 'python' in url or 'java' in url \
                else _mock_get(seed_html, final_url=url)

        with patch('requests.post', return_value=_mock_post(ddg_html)), \
             patch('requests.get', side_effect=_tracked_get):
            _make_loader().chunk_topic_search(
                'python', num_results=1, depth=1,
                max_pages_per_result=10, topic_filter='python',
            )

        # Java page must NOT have been fetched.
        assert not any('java' in u for u in get_calls)
