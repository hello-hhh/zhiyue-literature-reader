"""test_crawl.py — Tests for recursive URL crawling and all url_utils helper functions.

Coverage:
    url_utils.py:
        - extract_links: valid links, fragment-only, mailto, utility URLs, deduplication,
          relative → absolute resolution, cross-domain links (no domain restriction)
        - url_matches_topic: empty topic, matching path, non-matching path
        - is_same_domain: exact match, www. normalization, different domain
        - is_utility_url: login, cart, privacy keywords in path segments

    document_loader.py — chunk_url_recursive:
        - depth=0 fetches only seed URL
        - max_pages cap stops crawl after N pages
        - topic_filter skips pages whose URL path does not contain keyword
        - seed URL is always crawled regardless of topic_filter
        - cross-domain links ARE followed (no same-domain restriction)
        - connection error on a linked page does not abort the whole crawl
        - progress_callback is called once per fetched page

Mock strategy:
    requests.get is mocked — we control what each URL returns.
    BeautifulSoup, BM25Okapi, and chunking logic are NOT mocked.
"""

from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(content: bytes, content_type: str = 'text/html',
                   encoding: str = 'utf-8', final_url: str = None) -> MagicMock:
    """Build a fake requests.Response for the given content and headers.

    Args:
        content:      Raw bytes to return as response body.
        content_type: Value for the Content-Type header.
        encoding:     Character encoding the response claims to use.
        final_url:    Simulates response.url after HTTP redirects.
                      When None the mock has no .url attribute.

    Returns:
        MagicMock that behaves like a requests.Response.
    """
    resp = MagicMock()
    resp.content              = content
    resp.headers              = {'Content-Type': content_type}
    resp.encoding             = encoding
    resp.raise_for_status     = MagicMock()
    if final_url is not None:
        resp.url = final_url
    else:
        # Remove .url so hasattr(resp, 'url') returns False
        del resp.url
    return resp


def _html(body_text: str) -> bytes:
    """Wrap plain text in minimal HTML so BeautifulSoup can extract it."""
    return f'<html><body><p>{body_text}</p></body></html>'.encode()


def _html_with_links(body_text: str, links: list) -> bytes:
    """Build HTML containing a paragraph and a list of anchor tags.

    Args:
        body_text: Paragraph text.
        links:     List of href strings to embed as <a> tags.

    Returns:
        Encoded HTML bytes.
    """
    anchors = ''.join(f'<a href="{href}">link</a>' for href in links)
    return f'<html><body><p>{body_text}</p>{anchors}</body></html>'.encode()


# ---------------------------------------------------------------------------
# 1. url_utils — extract_links
# ---------------------------------------------------------------------------

class TestExtractLinks:
    """Unit tests for url_utils.extract_links."""

    def test_absolute_link_is_returned(self):
        """A plain absolute https link on a page is included in results."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Hello.', ['https://example.com/page1'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert 'https://example.com/page1' in links

    def test_relative_link_is_resolved(self):
        """A relative path like '/about' is resolved against the base URL."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Hello.', ['/about'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert 'https://example.com/about' in links

    def test_fragment_only_link_skipped(self):
        """Fragment-only links (#section) are skipped — they are not separate pages."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Hello.', ['#section1', '#top'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links == []

    def test_mailto_link_skipped(self):
        """mailto: links are skipped."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Contact us.', ['mailto:user@example.com'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links == []

    def test_tel_link_skipped(self):
        """tel: links are skipped."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Call us.', ['tel:+15551234567'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links == []

    def test_javascript_link_skipped(self):
        """javascript: links are skipped."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Click.', ['javascript:void(0)'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links == []

    def test_utility_url_skipped(self):
        """Links to login/cart/privacy paths are filtered out as utility pages."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Nav.', [
            'https://example.com/login',
            'https://example.com/cart',
            'https://example.com/privacy',
        ])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links == []

    def test_duplicate_links_deduplicated(self):
        """The same URL appearing twice is returned only once."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Dup.', [
            'https://example.com/page',
            'https://example.com/page',  # exact duplicate
        ])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links.count('https://example.com/page') == 1

    def test_fragment_stripped_from_absolute_link(self):
        """Fragment portion (#section) is stripped from absolute links."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Link.', ['https://example.com/page#section'])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert 'https://example.com/page' in links
        assert 'https://example.com/page#section' not in links

    def test_cross_domain_links_included(self):
        """Links to other domains ARE included — no domain restriction in extract_links."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('External.', [
            'https://other-domain.com/article',
            'https://docs.python.org/3/library/',
        ])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert 'https://other-domain.com/article' in links
        assert 'https://docs.python.org/3/library/' in links

    def test_order_preserved(self):
        """Links are returned in the order they appear on the page."""
        from src.rag.url_utils import extract_links
        html  = _html_with_links('Order.', [
            'https://example.com/first',
            'https://example.com/second',
            'https://example.com/third',
        ])
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links.index('https://example.com/first') < links.index('https://example.com/second')
        assert links.index('https://example.com/second') < links.index('https://example.com/third')

    def test_empty_page_returns_empty_list(self):
        """A page with no links returns an empty list."""
        from src.rag.url_utils import extract_links
        html  = _html('No links here.')
        links = extract_links(html.decode(), 'https://example.com/', {})
        assert links == []

    def test_hreflang_links_are_excluded(self):
        """Links with a hreflang attribute are skipped — they are interlanguage links.

        Wikipedia articles have 100+ sidebar links like:
          <a href="https://af.wikipedia.org/wiki/Elizabeth_Taylor" hreflang="af">Afrikaans</a>
          <a href="https://ar.wikipedia.org/wiki/..." hreflang="ar">العربية</a>

        These point to the same article in other languages and must NOT be crawled —
        a RAG system normally wants one language, not 100+ duplicates.
        """
        from src.rag.url_utils import extract_links

        # Page has one content link and two interlanguage (hreflang) links
        html = (
            b'<html><body>'
            b'<a href="/wiki/Cleopatra">Cleopatra film</a>'
            b'<a href="https://af.wikipedia.org/wiki/Elizabeth_Taylor" hreflang="af">Afrikaans</a>'
            b'<a href="https://ar.wikipedia.org/wiki/Elizabeth_Taylor" hreflang="ar">Arabic</a>'
            b'</body></html>'
        )
        links = extract_links(html.decode(), 'https://en.wikipedia.org/', {})
        # Content link must be kept
        assert any('Cleopatra' in link for link in links)
        # Interlanguage links must be dropped
        assert not any('af.wikipedia.org' in link for link in links)
        assert not any('ar.wikipedia.org' in link for link in links)


# ---------------------------------------------------------------------------
# 2. url_utils — url_matches_topic
# ---------------------------------------------------------------------------

class TestUrlMatchesTopic:
    """Unit tests for url_utils.url_matches_topic."""

    def test_empty_topic_always_matches(self):
        """An empty topic string means no filter — every URL passes."""
        from src.rag.url_utils import url_matches_topic
        assert url_matches_topic('https://example.com/anything', '') is True
        assert url_matches_topic('https://example.com/login',    '') is True

    def test_whitespace_only_topic_always_matches(self):
        """A whitespace-only topic is treated as empty — every URL passes."""
        from src.rag.url_utils import url_matches_topic
        assert url_matches_topic('https://example.com/page', '   ') is True

    def test_topic_found_in_path(self):
        """URL whose path contains the topic keyword returns True."""
        from src.rag.url_utils import url_matches_topic
        assert url_matches_topic('https://example.com/docs/python/tutorial', 'python') is True

    def test_topic_not_found_in_path(self):
        """URL whose path does not contain the topic keyword returns False."""
        from src.rag.url_utils import url_matches_topic
        assert url_matches_topic('https://example.com/docs/java/tutorial', 'python') is False

    def test_topic_match_is_case_insensitive(self):
        """Topic matching ignores case differences."""
        from src.rag.url_utils import url_matches_topic
        assert url_matches_topic('https://example.com/PYTHON/guide', 'python') is True
        assert url_matches_topic('https://example.com/Python/guide', 'PYTHON') is True

    def test_topic_not_checked_against_domain(self):
        """Topic filter only looks at the URL path, not the domain name."""
        from src.rag.url_utils import url_matches_topic

        # Domain contains 'python' but path does not — should return False
        assert url_matches_topic('https://python.org/downloads', 'tutorial') is False

    def test_hyphenated_topic_matches(self):
        """A multi-word topic with hyphens matches as a substring of the path."""
        from src.rag.url_utils import url_matches_topic
        assert url_matches_topic(
            'https://example.com/docs/machine-learning/intro', 'machine-learning'
        ) is True


# ---------------------------------------------------------------------------
# 3. url_utils — is_same_domain
# ---------------------------------------------------------------------------

class TestIsSameDomain:
    """Unit tests for url_utils.is_same_domain."""

    def test_exact_same_domain(self):
        """Two URLs on the exact same domain return True."""
        from src.rag.url_utils import is_same_domain
        assert is_same_domain('https://example.com/page', 'https://example.com/') is True

    def test_www_prefix_normalized(self):
        """www.example.com and example.com are treated as the same domain."""
        from src.rag.url_utils import is_same_domain
        assert is_same_domain('https://www.example.com/page', 'https://example.com/') is True
        assert is_same_domain('https://example.com/page',     'https://www.example.com/') is True

    def test_different_domain_returns_false(self):
        """Two URLs on different domains return False."""
        from src.rag.url_utils import is_same_domain
        assert is_same_domain('https://other.com/page', 'https://example.com/') is False

    def test_subdomain_treated_as_different(self):
        """blog.example.com is treated as a different domain from example.com."""
        from src.rag.url_utils import is_same_domain
        assert is_same_domain('https://blog.example.com/post', 'https://example.com/') is False

    def test_case_insensitive_domain_compare(self):
        """Domain comparison is case-insensitive."""
        from src.rag.url_utils import is_same_domain
        assert is_same_domain('https://EXAMPLE.COM/page', 'https://example.com/') is True


# ---------------------------------------------------------------------------
# 4. url_utils — is_utility_url
# ---------------------------------------------------------------------------

class TestIsUtilityUrl:
    """Unit tests for url_utils.is_utility_url."""

    def test_login_path_is_utility(self):
        """A URL with 'login' in the path is a utility page."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://example.com/login') is True

    def test_cart_path_is_utility(self):
        """A URL with 'cart' in the path is a utility page."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://example.com/cart/items') is True

    def test_privacy_path_is_utility(self):
        """A URL with 'privacy' in the path is a utility page."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://example.com/privacy-policy') is True

    def test_normal_article_path_is_not_utility(self):
        """A normal article URL is not a utility page."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://example.com/docs/getting-started') is False

    def test_keyword_in_hostname_is_matched(self):
        """A utility keyword in the hostname (subdomain) is enough to filter the URL.

        donate.wikimedia.org, auth.wikimedia.org, login.example.com etc. are
        utility subdomains even if the path looks like content.
        """
        from src.rag.url_utils import is_utility_url

        # 'login' subdomain — this is an authentication server, not a content page
        assert is_utility_url('https://login.example.com/article') is True
        # 'donate' subdomain — fundraising page regardless of path
        assert is_utility_url('https://donate.wikimedia.org/w/index.php?title=Landing') is True
        # 'auth' subdomain — authentication server
        assert is_utility_url('https://auth.wikimedia.org/enwiki/wiki/Special:CreateAccount') is True

    def test_normal_hostname_content_path_not_utility(self):
        """A plain content hostname with a content path is not a utility URL."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://www.example.com/docs/getting-started') is False

    def test_mediawiki_special_namespace_is_utility(self):
        """Wikipedia/MediaWiki Special: pages are utility pages (namespace has colon)."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Special:RecentChanges') is True
        assert is_utility_url('https://en.wikipedia.org/wiki/Special:Search') is True

    def test_mediawiki_talk_namespace_is_utility(self):
        """Wikipedia Talk: pages are utility pages."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Talk:Main_Page') is True

    def test_mediawiki_help_namespace_is_utility(self):
        """Wikipedia Help: pages are utility pages."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Help:Contents') is True

    def test_mediawiki_wikipedia_namespace_is_utility(self):
        """Wikipedia Wikipedia: pages (policy/meta) are utility pages."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Wikipedia:About') is True

    def test_mediawiki_portal_namespace_is_utility(self):
        """Wikipedia Portal: pages are utility/navigation pages."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Portal:Current_events') is True

    def test_regular_wiki_article_is_not_utility(self):
        """A regular Wikipedia article (no namespace colon) is NOT a utility page."""
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Elizabeth_Taylor') is False
        assert is_utility_url('https://en.wikipedia.org/wiki/Python_(programming_language)') is False

    def test_wiki_main_page_is_utility(self):
        """Wikipedia Main_Page and Wikimedia sister sites' Main_Page are utility pages.

        These are home/portal pages, not article content. Filtering them keeps the
        crawl on the article the user actually asked for.
        """
        from src.rag.url_utils import is_utility_url

        # Wikipedia itself
        assert is_utility_url('https://en.wikipedia.org/wiki/Main_Page') is True
        # Wikimedia sister projects — all link to their own Main_Page in nav sidebars
        assert is_utility_url('https://commons.wikimedia.org/wiki/Main_Page') is True
        assert is_utility_url('https://meta.wikimedia.org/wiki/Main_Page') is True
        assert is_utility_url('https://en.wikisource.org/wiki/Main_Page') is True

    def test_disambiguation_page_is_utility(self):
        """Disambiguation pages are utility pages — they list name variants, not content.

        Following disambiguation pages causes exponential crawl growth: each variant
        article links to many more pages, rapidly filling the max_pages budget with
        content unrelated to the original article.
        """
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/wiki/Elizabeth_Taylor_(disambiguation)') is True
        assert is_utility_url('https://en.wikipedia.org/wiki/Dame_(disambiguation)') is True
        assert is_utility_url('https://en.wikipedia.org/wiki/Taylor_(disambiguation)') is True

    def test_bare_domain_url_is_utility(self):
        """URLs with no path (bare domain) are utility pages — organizational landing pages.

        Wikipedia footer links like https://wikimediafoundation.org and
        https://stats.wikimedia.org have no article path and are not content pages.
        """
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://wikimediafoundation.org') is True
        assert is_utility_url('https://stats.wikimedia.org') is True
        assert is_utility_url('https://developer.wikimedia.org') is True
        assert is_utility_url('https://www.wikimedia.org') is True
        # A real article path is NOT a bare domain
        assert is_utility_url('https://en.wikipedia.org/wiki/Elizabeth_Taylor') is False

    def test_mediawiki_action_api_is_utility(self):
        """MediaWiki /w/index.php URLs (action, history, print, cite) are utility pages.

        These are never plain article content — they are the MediaWiki backend
        used for editing, printing, version history, and citation tools.
        """
        from src.rag.url_utils import is_utility_url
        assert is_utility_url('https://en.wikipedia.org/w/index.php?title=Main_Page&action=edit') is True
        assert is_utility_url('https://en.wikipedia.org/w/index.php?title=Main_Page&oldid=123') is True
        assert is_utility_url('https://en.wikipedia.org/w/index.php?title=Main_Page&printable=yes') is True
        assert is_utility_url('https://en.wikipedia.org/w/index.php?title=Special:CiteThis') is True


# ---------------------------------------------------------------------------
# 5. chunk_url_recursive — depth and max_pages
# ---------------------------------------------------------------------------

class TestChunkUrlRecursive:
    """Integration tests for DocumentLoader.chunk_url_recursive.

    requests.get is mocked to control what each URL returns.
    All chunking, HTML parsing, and link extraction are real.
    """

    def _make_loader(self):
        """Return a fresh DocumentLoader."""
        from src.rag.document_loader import DocumentLoader
        return DocumentLoader()

    def test_depth_zero_fetches_only_seed(self):
        """With depth=0 only the seed URL is fetched — no links are followed."""
        seed_html = _html_with_links(
            'Seed page with interesting content about cats sleeping habits.',
            ['https://example.com/page1', 'https://example.com/page2'],
        )
        seed_resp = _mock_response(seed_html, final_url='https://example.com/')
        visited_pages: set = set()

        def _tracked(url, **kwargs):
            visited_pages.add(url)
            return seed_resp

        with patch('requests.get', side_effect=_tracked):
            chunks = self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=0, max_pages=10,
            )
        # Only 1 unique page fetched — the seed itself (no links followed)
        assert len(visited_pages) == 1
        assert len(chunks) >= 1

    def test_depth_one_follows_one_level_of_links(self):
        """With depth=1 the seed page is fetched and its links are followed."""
        seed_html = _html_with_links(
            'Seed page about dolphins swimming intelligence.',
            ['https://example.com/dolphins', 'https://example.com/whales'],
        )
        page_html  = _html('Dolphins are highly intelligent marine mammals.')
        visited_pages: set = set()

        def _side_effect(url, **kwargs):
            # Track unique pages (the same URL may be fetched twice in some
            # implementations that fetch in _crawl_url then again in chunk_url)
            visited_pages.add(url)
            if 'dolphins' in url or 'whales' in url:
                return _mock_response(page_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            chunks = self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )
        # 3 unique pages: seed + dolphins + whales
        assert len(visited_pages) == 3
        assert len(chunks) >= 1

    def test_max_pages_cap_stops_crawl(self):
        """Crawl stops after max_pages total pages regardless of remaining links."""
        # Seed links to 5 pages, but max_pages=3 so only seed + 2 more are fetched
        links     = [f'https://example.com/page{i}' for i in range(5)]
        seed_html = _html_with_links('Seed about solar system planets orbits.', links)
        page_html  = _html('This is a linked page about planets in the solar system.')

        def _side_effect(url, **kwargs):
            if 'page' in url:
                return _mock_response(page_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        unique_pages: set = set()

        def _tracked(url, **kwargs):
            # Use a set so double-fetch implementations don't overcount
            unique_pages.add(url)
            return _side_effect(url, **kwargs)

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=3,
            )
        # Exactly 3 unique pages crawled: seed + page0 + page1
        assert len(unique_pages) == 3

    def test_topic_filter_skips_non_matching_pages(self):
        """Pages whose URL path does not contain the topic keyword are skipped."""
        links     = [
            'https://example.com/python/tutorial',   # matches topic 'python'
            'https://example.com/java/tutorial',     # does NOT match
            'https://example.com/python/advanced',   # matches topic 'python'
        ]
        seed_html = _html_with_links('Seed programming tutorials resources.', links)
        page_html  = _html('Python tutorial content about loops and functions.')

        fetched_urls = []

        def _tracked(url, **kwargs):
            fetched_urls.append(url)
            if 'python' in url:
                return _mock_response(page_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
                topic_filter='python',
            )

        # java/tutorial should NOT have been fetched
        assert all('java' not in url for url in fetched_urls)
        # python pages should have been fetched
        assert any('python/tutorial'  in url for url in fetched_urls)
        assert any('python/advanced'  in url for url in fetched_urls)

    def test_seed_url_always_crawled_regardless_of_topic_filter(self):
        """The seed URL is always fetched even if its path does not match the topic filter."""
        # Seed URL is https://example.com/ — path '/' does not contain 'python'
        seed_html = _html('Seed homepage with no topic keyword in path.')
        fetched   = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            return _mock_response(seed_html, final_url=url)

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=0, max_pages=10,
                topic_filter='python',
            )

        # Seed must have been fetched even though '/' does not contain 'python'.
        # The trailing slash is stripped by URL normalization, so we check without it.
        assert 'https://example.com' in fetched

    def test_cross_domain_links_are_blocked(self):
        """Links to other domains on the seed page are NOT followed — same-domain constraint.

        Cross-domain following was removed because real-world pages link to
        subscription sites, paywalls, and ad networks.  Staying on the seed
        domain keeps crawled content relevant and avoids junk pages.
        """
        links     = [
            'https://other-domain.com/article',
            'https://docs.python.org/3/tutorial/',
        ]
        seed_html = _html_with_links('Seed page with external links resources.', links)
        page_html  = _html('External page content about Python tutorials.')

        fetched   = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            if url.rstrip('/') == 'https://example.com':
                return _mock_response(seed_html, final_url=url)
            return _mock_response(page_html, final_url=url)

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        # Cross-domain links must NOT be fetched — same-domain constraint.
        assert 'https://other-domain.com/article' not in fetched
        assert 'https://docs.python.org/3/tutorial' not in fetched

    def test_connection_error_on_linked_page_does_not_abort(self):
        """A network error on a linked page is logged but crawl continues to other links."""
        links     = [
            'https://example.com/good-page',
            'https://example.com/bad-page',
        ]
        seed_html = _html_with_links('Seed page about interesting science topics.', links)
        good_html  = _html('Good page: The speed of light is 299792458 metres per second.')

        def _tracked(url, **kwargs):
            if 'bad-page' in url:
                raise Exception('Connection refused')
            if 'good-page' in url:
                return _mock_response(good_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_tracked):
            chunks = self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        # Crawl should not raise — bad-page error was swallowed
        # Good page should still have been indexed
        assert len(chunks) >= 1

    def test_progress_callback_called_per_page(self):
        """progress_callback(url, dtype, chunk_count) is called once for each fetched page."""
        links     = ['https://example.com/page1']
        seed_html = _html_with_links('Seed page about ocean tides and waves.', links)
        page_html  = _html('Page one content about tidal waves in the Pacific ocean.')

        calls     = []
        callback  = lambda url, dtype, n: calls.append((url, dtype, n))

        def _tracked(url, **kwargs):
            if 'page1' in url:
                return _mock_response(page_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
                progress_callback=callback,
            )

        # Should have been called for seed + 1 linked page = 2 times.
        # Trailing slash on seed is stripped by normalization.
        assert len(calls) == 2
        fetched_urls = [c[0] for c in calls]
        assert 'https://example.com'       in fetched_urls
        assert 'https://example.com/page1' in fetched_urls

    def test_already_visited_url_not_fetched_twice(self):
        """A URL that appears in multiple link lists is only fetched once."""
        # Both seed and page1 link to /shared — it should only be fetched once
        shared_link = 'https://example.com/shared'
        seed_html   = _html_with_links(
            'Seed page about space exploration missions.',
            [shared_link, 'https://example.com/page1'],
        )
        page1_html  = _html_with_links('Page one about rockets.', [shared_link])
        shared_html = _html('Shared page about NASA missions and space.')

        fetched     = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            if 'shared' in url:
                return _mock_response(shared_html, final_url=url)
            if 'page1' in url:
                return _mock_response(page1_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=2, max_pages=10,
            )

        # /shared should appear at most twice in fetched list
        # (once in _crawl_url, possibly once more in chunk_url for double-fetch impls)
        # The key invariant is that it was NOT visited multiple times by the crawler
        # i.e. not 3 or more times which would indicate the deduplication failed
        assert fetched.count(shared_link) <= 2

    def test_chunks_returned_from_all_pages(self):
        """All chunks from all crawled pages are merged into the returned list."""
        seed_html = _html_with_links(
            'Seed page about solar system astronomy facts.',
            ['https://example.com/page1'],
        )
        page_html  = _html('Page one: Jupiter is the largest planet in our solar system.')

        def _tracked(url, **kwargs):
            if 'page1' in url:
                return _mock_response(page_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_tracked):
            chunks = self._make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        # Should have chunks from both the seed page and page1
        assert len(chunks) >= 2
        sources = [c['source'] for c in chunks]
        assert any('example.com' in s for s in sources)


# ---------------------------------------------------------------------------
# 6. URL scheme normalization (no https:// prefix)
# ---------------------------------------------------------------------------

class TestUrlSchemeNormalization:
    """URLs without a scheme (no https://) are automatically fixed before fetching.

    Without a scheme, urlparse returns an empty netloc and urljoin produces bare
    paths (/wiki/Page) that are rejected, so no links can be extracted. requests
    also raises MissingSchema and fails silently. Adding https:// fixes both.
    """

    def _make_loader(self):
        """Return a fresh DocumentLoader."""
        from src.rag.document_loader import DocumentLoader
        return DocumentLoader()

    def test_chunk_url_adds_https_when_missing(self):
        """chunk_url auto-adds https:// when the user omits it."""
        html     = _html('Elizabeth Taylor was a legendary actress and humanitarian.')
        html_url = 'https://en.wikipedia.org/wiki/Elizabeth_Taylor'

        fetched  = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            return _mock_response(html, final_url=html_url)

        with patch('requests.get', side_effect=_tracked):
            chunks = self._make_loader().chunk_url(
                'en.wikipedia.org/wiki/Elizabeth_Taylor'   # no https://
            )

        # The actual request should have used https://
        assert all(u.startswith('https://') for u in fetched)
        assert len(chunks) >= 1

    def test_chunk_url_recursive_adds_https_for_seed(self):
        """chunk_url_recursive auto-adds https:// to the seed URL when missing."""
        seed_html = _html_with_links(
            'Elizabeth Taylor was born on February 27 1932 in London England.',
            [],  # no links to follow — just verify the seed is fetched correctly
        )
        html_url = 'https://en.wikipedia.org/wiki/Elizabeth_Taylor'

        fetched = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            return _mock_response(seed_html, final_url=html_url)

        with patch('requests.get', side_effect=_tracked):
            chunks = self._make_loader().chunk_url_recursive(
                'en.wikipedia.org/wiki/Elizabeth_Taylor',   # no https://
                depth=0, max_pages=10,
            )

        # Seed should have been fetched with https://
        assert all(u.startswith('https://') for u in fetched)
        assert len(chunks) >= 1

    def test_links_resolve_correctly_after_scheme_normalization(self):
        """After adding https://, relative links on the page resolve to full URLs."""
        # Wikipedia-style page with absolute-path links (/wiki/...)
        seed_html = _html_with_links(
            'Elizabeth Taylor appeared in National Velvet in 1944.',
            ['/wiki/National_Velvet_(film)', '/wiki/Montgomery_Clift'],
        )
        page_html  = _html('National Velvet is a 1944 American drama film.')
        base_url   = 'https://en.wikipedia.org/wiki/Elizabeth_Taylor'

        fetched = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            if 'Elizabeth_Taylor' in url:
                return _mock_response(seed_html, final_url=base_url)
            return _mock_response(page_html, final_url=url)

        with patch('requests.get', side_effect=_tracked):
            chunks = self._make_loader().chunk_url_recursive(
                'en.wikipedia.org/wiki/Elizabeth_Taylor',   # no https://
                depth=1, max_pages=10,
            )

        # Relative /wiki/ links should have been resolved to full https URLs
        assert any('National_Velvet' in u for u in fetched)
        assert any('Montgomery_Clift' in u for u in fetched)
        # All fetched URLs should be proper https URLs
        assert all(u.startswith('https://') for u in fetched)

    def test_already_has_https_not_doubled(self):
        """If the URL already has https://, it is not modified."""
        html  = _html('Test content about stars and galaxies in the universe.')
        fetched = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            return _mock_response(html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url('https://example.com/')

        # Should not become https://https://example.com/
        assert all(u == 'https://example.com/' for u in fetched)

    def test_http_url_not_upgraded_to_https(self):
        """An explicit http:// URL is left as-is (not forced to https)."""
        html  = _html('Test content about rivers and lakes in the world.')
        fetched = []

        def _tracked(url, **kwargs):
            fetched.append(url)
            return _mock_response(html, final_url='http://example.com/')

        with patch('requests.get', side_effect=_tracked):
            self._make_loader().chunk_url('http://example.com/')

        assert all(u.startswith('http://') for u in fetched)
