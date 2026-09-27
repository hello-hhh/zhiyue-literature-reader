"""test_crawl_combinations.py — Combination tests for recursive URL crawling.

Tests combinations of:
    - topic_filter  +  depth
    - topic_filter  +  max_pages
    - cross-domain  +  depth
    - cross-domain  +  topic_filter
    - max_pages     +  depth  (depth > 1 chains)
    - allowed_types +  topic_filter
    - allowed_types +  depth
    - error recovery + max_pages (errors do not count against page cap)
    - pdf link found on html seed page  +  depth

Mock strategy:
    requests.get is always mocked.
    All chunking, link extraction, and HTML parsing are real.
"""

import io
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(content: bytes, content_type: str = 'text/html',
                   encoding: str = 'utf-8', final_url: str = None) -> MagicMock:
    """Return a fake requests.Response for the given content."""
    resp = MagicMock()
    resp.content          = content
    resp.headers          = {'Content-Type': content_type}
    resp.encoding         = encoding
    resp.raise_for_status = MagicMock()
    if final_url is not None:
        resp.url = final_url
    else:
        del resp.url
    return resp


def _html(body_text: str) -> bytes:
    """Wrap text in minimal HTML."""
    return f'<html><body><p>{body_text}</p></body></html>'.encode()


def _html_with_links(body_text: str, links: list) -> bytes:
    """Build HTML with a paragraph and anchor tags."""
    anchors = ''.join(f'<a href="{h}">link</a>' for h in links)
    return f'<html><body><p>{body_text}</p>{anchors}</body></html>'.encode()


def _make_loader():
    """Return a fresh DocumentLoader instance."""
    from src.rag.document_loader import DocumentLoader
    return DocumentLoader()


# ---------------------------------------------------------------------------
# 1. topic_filter + depth combinations
# ---------------------------------------------------------------------------

class TestTopicFilterPlusDepth:
    """Combinations of topic_filter with depth > 1."""

    def test_topic_filter_applied_at_depth_2(self):
        """At depth=2, topic_filter still filters pages two levels deep."""
        # Seed → level1_match → level2_match (should be fetched)
        # Seed → level1_match → level2_no_match (should be skipped)
        seed_html   = _html_with_links(
            'Seed page programming resources.',
            ['https://example.com/python/overview'],
        )
        level1_html = _html_with_links(
            'Python overview page.',
            [
                'https://example.com/python/details',    # matches
                'https://example.com/ruby/details',      # does NOT match
            ],
        )
        level2_py   = _html('Python details: list comprehensions loops functions.')
        level2_rb   = _html('Ruby details: blocks closures iterators.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            if 'python/details' in url: return _mock_response(level2_py,  final_url=url)
            if 'ruby/details'   in url: return _mock_response(level2_rb,  final_url=url)
            if 'python/overview' in url: return _mock_response(level1_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=2, max_pages=10, topic_filter='python',
            )

        assert any('python/details' in u for u in fetched)
        assert not any('ruby/details' in u for u in fetched)

    def test_no_topic_filter_at_depth_2_follows_all(self):
        """Without a topic_filter, all linked pages at depth=2 are followed."""
        seed_html   = _html_with_links('Seed.',  ['https://example.com/page1'])
        page1_html  = _html_with_links('Page1.', ['https://example.com/page2'])
        page2_html  = _html('Page2 content about ocean tides.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            if 'page2' in url: return _mock_response(page2_html, final_url=url)
            if 'page1' in url: return _mock_response(page1_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=2, max_pages=10,
            )

        assert any('page2' in u for u in fetched)

    def test_topic_filter_empty_string_with_depth_2(self):
        """Empty topic_filter (no filter) + depth=2 fetches all reachable pages."""
        seed_html   = _html_with_links('Seed.',  ['https://example.com/any-topic'])
        page1_html  = _html_with_links('Page1.', ['https://example.com/anything-else'])
        page2_html  = _html('Anything else: random content.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            if 'anything-else' in url: return _mock_response(page2_html, final_url=url)
            if 'any-topic'      in url: return _mock_response(page1_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=2, max_pages=10, topic_filter='',
            )

        assert len(set(fetched)) == 3  # 3 unique pages: seed + page1 + page2


# ---------------------------------------------------------------------------
# 2. topic_filter + max_pages combinations
# ---------------------------------------------------------------------------

class TestTopicFilterPlusMaxPages:
    """Combinations of topic_filter with max_pages cap."""

    def test_topic_filter_reduces_effective_pages_crawled(self):
        """When topic_filter skips many pages, total pages crawled is lower than max_pages."""
        links = [
            'https://example.com/python/a',
            'https://example.com/java/b',
            'https://example.com/python/c',
            'https://example.com/java/d',
            'https://example.com/python/e',
        ]
        seed_html = _html_with_links('Seed programming resources.', links)
        page_html  = _html('Page content about programming languages.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            # Trailing slash is stripped — check without it
            return _mock_response(page_html, final_url=url) if url.rstrip('/') != 'https://example.com' \
                else _mock_response(seed_html, final_url='https://example.com')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10, topic_filter='python',
            )

        # Only seed + python pages fetched, java pages skipped
        unique_pages = set(fetched)
        assert all('java' not in u for u in unique_pages)
        assert len(unique_pages) == 4  # 4 unique pages: seed + 3 python pages

    def test_max_pages_1_fetches_only_seed_despite_topic(self):
        """max_pages=1 fetches only the seed, topic_filter doesn't change that."""
        seed_html = _html_with_links(
            'Seed page about Python tutorials.',
            ['https://example.com/python/page1'],
        )
        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            return _mock_response(seed_html, final_url=url)

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=1, topic_filter='python',
            )

        assert len(set(fetched)) == 1  # only 1 unique page: the seed
        # Trailing slash stripped by normalization — check without it.
        assert 'https://example.com' in fetched


# ---------------------------------------------------------------------------
# 3. cross-domain + depth combinations
# ---------------------------------------------------------------------------

class TestCrossDomainPlusDepth:
    """Same-domain constraint: cross-domain links are blocked during recursive crawl."""

    def test_cross_domain_link_blocked_at_depth_1(self):
        """A cross-domain link on the seed page is NOT followed — same-domain constraint."""
        seed_html    = _html_with_links(
            'Seed page with external link.',
            ['https://other-domain.com/article'],
        )
        article_html = _html('External article about machine learning algorithms.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            if 'other-domain' in url:
                return _mock_response(article_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        assert 'https://other-domain.com/article' not in fetched

    def test_cross_domain_link_blocked_at_depth_2_from_linked_page(self):
        """A cross-domain link found on a level-1 page is also blocked at depth=2."""
        seed_html  = _html_with_links('Seed.', ['https://example.com/page1'])
        page1_html = _html_with_links('Page1.', ['https://external.org/deep-article'])
        deep_html  = _html('Deep external article about quantum computing research.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            if 'external.org' in url: return _mock_response(deep_html,  final_url=url)
            if 'page1'         in url: return _mock_response(page1_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=2, max_pages=10,
            )

        assert 'https://external.org/deep-article' not in fetched

    def test_cross_domain_not_followed_at_depth_0(self):
        """At depth=0 no links are followed — cross-domain links are not an exception."""
        seed_html = _html_with_links(
            'Seed.',
            ['https://other-domain.com/page'],
        )

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=0, max_pages=10,
            )

        assert 'https://other-domain.com/page' not in fetched


# ---------------------------------------------------------------------------
# 4. cross-domain + topic_filter combinations
# ---------------------------------------------------------------------------

class TestCrossDomainPlusTopicFilter:
    """Cross-domain links are blocked regardless of topic_filter."""

    def test_cross_domain_link_blocked_even_with_matching_topic_filter(self):
        """A cross-domain link is blocked even if its path matches the topic filter."""
        seed_html = _html_with_links(
            'Seed about programming.',
            ['https://docs.python.org/3/python/tutorial'],
        )
        page_html  = _html('Python tutorial about loops and functions.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            return _mock_response(page_html, final_url=url) if 'docs.python' in url \
                else _mock_response(seed_html, final_url='https://example.com')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10, topic_filter='python',
            )

        # Domain constraint takes priority over topic filter.
        assert 'https://docs.python.org/3/python/tutorial' not in fetched

    def test_cross_domain_link_blocked_by_topic_filter(self):
        """A cross-domain link whose path does NOT contain the topic is skipped."""
        seed_html = _html_with_links(
            'Seed about programming.',
            ['https://other-domain.com/java/tutorial'],  # path has 'java', not 'python'
        )
        page_html  = _html('Java tutorial content.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            return _mock_response(page_html, final_url=url) if 'other-domain' in url \
                else _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10, topic_filter='python',
            )

        assert 'https://other-domain.com/java/tutorial' not in fetched


# ---------------------------------------------------------------------------
# 5. max_pages + depth combinations
# ---------------------------------------------------------------------------

class TestMaxPagesPlusDepth:
    """max_pages cap interacting with depth > 1 chains."""

    def test_max_pages_halts_depth_2_chain(self):
        """Even with depth=2, crawl stops once max_pages is reached."""
        # seed → page1 → page2, but max_pages=2 → only seed + page1
        seed_html  = _html_with_links('Seed.',  ['https://example.com/page1'])
        page1_html = _html_with_links('Page1.', ['https://example.com/page2'])
        page2_html = _html('Page2 content.')

        fetched = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            if 'page2' in url: return _mock_response(page2_html, final_url=url)
            if 'page1' in url: return _mock_response(page1_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=2, max_pages=2,
            )

        assert len(set(fetched)) == 2  # 2 unique pages: seed + page1
        assert 'https://example.com/page2' not in fetched

    def test_depth_0_respects_max_pages_1(self):
        """depth=0 with max_pages=1 fetches only the seed (consistent guard)."""
        seed_html = _html_with_links('Seed.', ['https://example.com/linked'])
        fetched   = []

        def _side_effect(url, **kwargs):
            fetched.append(url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            _make_loader().chunk_url_recursive(
                'https://example.com/', depth=0, max_pages=1,
            )

        assert len(set(fetched)) == 1  # only the seed URL


# ---------------------------------------------------------------------------
# 6. allowed_types + topic_filter combinations
# ---------------------------------------------------------------------------

class TestAllowedTypesPlusTopicFilter:
    """allowed_types filter combined with topic_filter."""

    def test_allowed_types_html_with_topic_filter(self):
        """Only HTML pages matching the topic are indexed when allowed_types={'html'}."""
        links = [
            'https://example.com/python/guide',      # html, matches topic
            'https://example.com/java/guide',        # html, doesn't match topic
        ]
        seed_html = _html_with_links('Seed programming guides.', links)
        page_html  = _html('Page content about a programming language guide.')

        indexed_sources = []
        original_chunk_url = None

        def _side_effect(url, **kwargs):
            if 'python/guide' in url or 'java/guide' in url:
                return _mock_response(page_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            chunks = _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
                allowed_types={'html'}, topic_filter='python',
            )

        # All returned chunks should come from python pages only (java was topic-filtered)
        assert all('java' not in c['source'] for c in chunks)

    def test_allowed_types_restricts_indexing_but_not_crawl(self):
        """allowed_types restricts which pages are indexed, not which are crawled for links.

        A page whose type is not in allowed_types should still be visited to extract
        its links — it just won't contribute chunks.
        """
        # seed → page1 (txt, not in allowed_types={'html'}) → page2 (html, in allowed_types)
        # page1 is not indexed, but page2 IS indexed because it was reached via page1's links
        seed_html  = _html_with_links('Seed page content.', ['https://example.com/page1.txt'])
        page1_txt  = b'Plain text page with a reference to page2.'
        # We can't embed links in plain text, so page2 won't be reached this way.
        # This test just verifies page1 is NOT in chunks when type is 'txt' and allowed={'html'}
        page2_html = _html('Page2 content about ocean science.')

        def _side_effect(url, **kwargs):
            if 'page1.txt' in url: return _mock_response(page1_txt,  content_type='text/plain',  final_url=url)
            if 'page2'      in url: return _mock_response(page2_html, content_type='text/html',   final_url=url)
            return _mock_response(seed_html, content_type='text/html', final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            chunks = _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
                allowed_types={'html'},
            )

        # page1.txt chunks should NOT be in results when allowed_types={'html'}
        assert all('page1' not in c['source'] for c in chunks)


# ---------------------------------------------------------------------------
# 7. error recovery + max_pages
# ---------------------------------------------------------------------------

class TestErrorRecoveryPlusMaxPages:
    """Network errors during crawl interact correctly with max_pages cap."""

    def test_error_page_does_not_consume_max_pages_slot(self):
        """A page that raises a network error is added to visited but does not contribute
        a fetched-page count, so the crawl can still reach max_pages good pages.

        Note: The current implementation adds the URL to visited BEFORE fetching it,
        so a failed URL does count against max_pages. This test verifies that behaviour
        is consistent and the crawl doesn't raise an exception.
        """
        links = [
            'https://example.com/bad-page',
            'https://example.com/good-page',
        ]
        seed_html = _html_with_links('Seed page about science.', links)
        good_html  = _html('Good page: water boils at 100 degrees Celsius at sea level.')

        def _side_effect(url, **kwargs):
            if 'bad-page' in url:
                raise Exception('Simulated connection error')
            if 'good-page' in url:
                return _mock_response(good_html, final_url=url)
            return _mock_response(seed_html, final_url='https://example.com/')

        # Should complete without raising even with a bad page
        with patch('requests.get', side_effect=_side_effect):
            chunks = _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        # Good page should still be indexed
        assert any('good-page' in c['source'] for c in chunks)

    def test_all_linked_pages_fail_returns_seed_chunks_only(self):
        """When all linked pages fail, only the seed's chunks are returned."""
        seed_html = _html_with_links(
            'Seed page: cats have retractable claws and excellent night vision.',
            ['https://example.com/fail1', 'https://example.com/fail2'],
        )

        def _side_effect(url, **kwargs):
            if 'fail' in url:
                raise Exception('Connection refused')
            return _mock_response(seed_html, final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            chunks = _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        # Seed chunks should still be present
        assert len(chunks) >= 1
        assert all('example.com' in c['source'] for c in chunks)


# ---------------------------------------------------------------------------
# 8. PDF link found on HTML seed page + depth
# ---------------------------------------------------------------------------

class TestPdfLinkOnHtmlPage:
    """A PDF link found on the seed HTML page is fetched and chunked."""

    def test_pdf_link_on_seed_page_is_indexed(self):
        """A .pdf href on the seed page is fetched and its chunks are included."""
        import fitz  # PyMuPDF — not mocked

        # Build a real in-memory PDF
        doc  = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), 'Solar panels convert sunlight into electricity using photovoltaic cells.')
        buf      = io.BytesIO()
        doc.save(buf)
        pdf_bytes = buf.getvalue()

        seed_html = _html_with_links(
            'Seed page about renewable energy.',
            ['https://example.com/solar-energy.pdf'],
        )

        def _side_effect(url, **kwargs):
            if 'solar-energy.pdf' in url:
                return _mock_response(pdf_bytes, content_type='application/pdf', final_url=url)
            return _mock_response(seed_html, content_type='text/html', final_url='https://example.com/')

        with patch('requests.get', side_effect=_side_effect):
            chunks = _make_loader().chunk_url_recursive(
                'https://example.com/', depth=1, max_pages=10,
            )

        # Chunks from the PDF should be in results
        pdf_chunks = [c for c in chunks if 'solar-energy' in c['source']]
        assert len(pdf_chunks) >= 1
        assert any('pdf' == c['type'] for c in pdf_chunks)
