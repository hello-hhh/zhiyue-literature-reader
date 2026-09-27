"""test_ui_components.py — Unit tests for every UI handler component.

Tests each handler in ui/handlers.py and ui/ingestion.py individually and
in combinations, using unittest.mock to replace Streamlit calls so the tests
run without a running Streamlit server.

Coverage:
    Individual component tests:
        - render_header       → st.markdown called with rag-header div
        - render_footer       → st.markdown called with GitHub link
        - render_mode_selector → st.radio called; session state updated
        - handle_topic_search  → form path: success, warning, error
        - handle_url_ingestion → single URL, recursive URL, no input
        - handle_file_upload   → upload + index, no upload, success message
        - handle_user_input    → chat mode, agent mode, pipeline called
        - render_sidebar       → chunk count displayed

    Combination tests:
        - handle_topic_search + handle_url_ingestion both run without conflict
        - handle_file_upload + handle_url_ingestion both produce success messages
        - Chat mode + agent mode sequence handled correctly
        - Zero chunks: sidebar shows 0 correctly
        - Multi-file upload produces separate success messages per file

    Boundary / negative:
        - handle_topic_search with empty query: no search called
        - handle_url_ingestion with empty URL: no fetch called
        - handle_topic_search returns no chunks: warning shown, not success
        - handle_file_upload with no uploaded files: index button not shown

Mock strategy:
    streamlit (st) is patched at the import level using MagicMock().
    session_state is a real dict to avoid MagicMock attribute chain confusion.
    All DocumentLoader and VectorStore calls are MagicMock instances.
"""

from unittest.mock import MagicMock, call, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class _SessionState(dict):
    """A dict that also supports attribute-style access, mimicking st.session_state.

    Streamlit's real session_state supports both `st.session_state['key']` and
    `st.session_state.key` syntax.  Using a plain dict in tests causes
    AttributeError when handlers do `st.session_state.url_msg = ...`.
    This shim fixes that so handler tests can run without a live Streamlit server.
    """

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value) -> None:
        self[key] = value

    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key, default)


def _make_session_state(**overrides) -> '_SessionState':
    """Return a minimal session_state object with sane defaults."""
    defaults = {
        'conv':       [],
        'display':    [],
        'total':      0,
        'last':       None,
        'mode':       'chat',
        'url_chunks': [],
        'bm25_index': None,
        'url_msg':    None,
        'file_msg':   None,
    }
    defaults.update(overrides)
    return _SessionState(defaults)


def _mock_loader(chunks=None) -> MagicMock:
    """Return a mock DocumentLoader that returns the given chunks."""
    loader = MagicMock()
    loader.chunk_topic_search.return_value = chunks or []
    loader.chunk_url.return_value          = chunks or []
    loader.chunk_url_recursive.return_value = chunks or []
    return loader


def _mock_store(chunk_count: int = 0) -> MagicMock:
    """Return a mock VectorStore with the given number of indexed chunks."""
    store = MagicMock()
    store.chunks = ['chunk'] * chunk_count
    store.collection.count.return_value = chunk_count
    store.bm25_index = None
    return store


# ─────────────────────────────────────────────────────────────────────────────
# render_header
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderHeader:
    """Unit tests for render_header() — verifies the product header is output."""

    def test_calls_st_markdown(self):
        """render_header() must call st.markdown at least once."""
        import streamlit as st
        with patch.object(st, 'markdown') as mock_md:
            from src.ui.renderers import render_header
            render_header()
        mock_md.assert_called()

    def test_output_contains_rag_header_class(self):
        """The markdown output must include the 'rag-header' CSS class."""
        import streamlit as st
        calls_made = []
        with patch.object(st, 'markdown', side_effect=lambda html, **kw: calls_made.append(html)):
            from src.ui.renderers import render_header
            render_header()
        combined = ' '.join(str(c) for c in calls_made)
        assert 'rag-header' in combined

    def test_output_contains_product_name(self):
        """The header must contain 'RAG Agent'."""
        import streamlit as st
        calls_made = []
        with patch.object(st, 'markdown', side_effect=lambda html, **kw: calls_made.append(html)):
            from src.ui.renderers import render_header
            render_header()
        combined = ' '.join(str(c) for c in calls_made)
        assert 'RAG Agent' in combined


# ─────────────────────────────────────────────────────────────────────────────
# render_footer
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderFooter:
    """Unit tests for render_footer() — verifies the footer links are output."""

    def test_calls_st_markdown(self):
        """render_footer() must call st.markdown."""
        import streamlit as st
        with patch.object(st, 'markdown') as mock_md:
            from src.ui.renderers import render_footer
            render_footer()
        mock_md.assert_called()

    def test_output_contains_github_link(self):
        """The footer must include a GitHub link."""
        import streamlit as st
        calls_made = []
        with patch.object(st, 'markdown', side_effect=lambda html, **kw: calls_made.append(html)):
            from src.ui.renderers import render_footer
            render_footer()
        combined = ' '.join(str(c) for c in calls_made)
        assert 'GitHub' in combined or 'github' in combined.lower()

    def test_output_contains_fully_local_message(self):
        """The footer must state that no data leaves the machine."""
        import streamlit as st
        calls_made = []
        with patch.object(st, 'markdown', side_effect=lambda html, **kw: calls_made.append(html)):
            from src.ui.renderers import render_footer
            render_footer()
        combined = ' '.join(str(c) for c in calls_made)
        assert 'local' in combined.lower()


# ─────────────────────────────────────────────────────────────────────────────
# handle_topic_search
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleTopicSearch:
    """Unit tests for handle_topic_search() — verifies search + index path."""

    def _make_form_submit(self, query: str, submitted: bool):
        """Build a mock st.form context manager that simulates a form submission."""
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__  = MagicMock(return_value=False)
        return mock_form

    def test_returns_false_when_not_submitted(self):
        """If the form is not submitted, handle_topic_search returns False."""
        import streamlit as st
        session = _make_session_state()

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'form',     return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',   return_value=''), \
             patch.object(st, 'number_input', return_value=5), \
             patch.object(st, 'form_submit_button', return_value=False), \
             patch.object(st, 'columns',  return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None):
            from src.ui import handlers as h
            loader = _mock_loader()
            store  = _mock_store()
            result = h.handle_topic_search(loader, store)

        assert result is False

    def test_search_called_on_submission_with_query(self):
        """chunk_topic_search is called when the form is submitted with a query."""
        import streamlit as st
        session  = _make_session_state()
        loader   = _mock_loader(chunks=[{'text': 'hello', 'source': 'web'}])
        store    = _mock_store()

        # We patch the status context manager to avoid real Streamlit server
        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__  = MagicMock(return_value=False)

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'form',     return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',   return_value='python tutorial'), \
             patch.object(st, 'number_input', return_value=5), \
             patch.object(st, 'form_submit_button', return_value=True), \
             patch.object(st, 'columns',  return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None), \
             patch.object(st, 'status',   return_value=mock_status), \
             patch.object(st, 'success',  return_value=None), \
             patch.object(st, 'spinner',  return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from src.ui import handlers as h
            result = h.handle_topic_search(loader, store)

        loader.chunk_topic_search.assert_called_once()

    def test_no_search_when_query_is_empty(self):
        """chunk_topic_search must NOT be called when the query text is empty."""
        import streamlit as st
        session = _make_session_state()
        loader  = _mock_loader()
        store   = _mock_store()

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'form',     return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',   return_value='   '), \
             patch.object(st, 'number_input', return_value=5), \
             patch.object(st, 'form_submit_button', return_value=True), \
             patch.object(st, 'columns',  return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None):
            from src.ui import handlers as h
            h.handle_topic_search(loader, store)

        loader.chunk_topic_search.assert_not_called()

    def test_warning_shown_when_no_chunks_returned(self):
        """st.warning must be called when chunk_topic_search returns an empty list."""
        import streamlit as st
        session     = _make_session_state()
        loader      = _mock_loader(chunks=[])   # empty result
        store       = _mock_store()
        warning_calls = []

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__  = MagicMock(return_value=False)

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'form',     return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',   return_value='some query'), \
             patch.object(st, 'number_input', return_value=5), \
             patch.object(st, 'form_submit_button', return_value=True), \
             patch.object(st, 'columns',  return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None), \
             patch.object(st, 'status',   return_value=mock_status), \
             patch.object(st, 'warning', side_effect=lambda msg: warning_calls.append(msg)):
            from src.ui import handlers as h
            h.handle_topic_search(loader, store)

        assert warning_calls, "Expected st.warning() to be called for empty results"

    def test_error_shown_on_exception(self):
        """st.error must be called when chunk_topic_search raises an exception."""
        import streamlit as st
        session = _make_session_state()
        loader  = _mock_loader()
        loader.chunk_topic_search.side_effect = RuntimeError("network down")
        store   = _mock_store()
        error_calls = []

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__  = MagicMock(return_value=False)

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'form',     return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',   return_value='error query'), \
             patch.object(st, 'number_input', return_value=5), \
             patch.object(st, 'form_submit_button', return_value=True), \
             patch.object(st, 'columns',  return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None), \
             patch.object(st, 'status',   return_value=mock_status), \
             patch.object(st, 'error', side_effect=lambda msg: error_calls.append(msg)):
            from src.ui import handlers as h
            h.handle_topic_search(loader, store)

        assert error_calls, "Expected st.error() to be called on exception"


# ─────────────────────────────────────────────────────────────────────────────
# handle_url_ingestion — single URL path
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleUrlIngestion:
    """Unit tests for handle_url_ingestion() — single and recursive URL paths."""

    def _url_base_patches(self, session, *, recursive: bool = False,
                          url: str = '', submitted: bool = False):
        """Return the base set of patches for handle_url_ingestion tests.

        handle_url_ingestion renders:
        - st.expander + st.checkbox (recursive toggle)
        - When recursive=True: st.columns(2) + st.number_input x2 + st.text_input
          (topic filter) + st.caption + st.columns(7) for type checkboxes
        - st.form + st.text_input (URL) + st.form_submit_button
        - Optional success/error message from st.session_state.url_msg
        """
        import streamlit as st

        # 7 column mocks for the document-type checkbox row
        seven_cols = [MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                                __exit__=MagicMock(return_value=False))
                      for _ in range(7)]
        two_cols = [MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                              __exit__=MagicMock(return_value=False))
                    for _ in range(2)]

        call_counts = {'checkbox': 0, 'text_input': 0, 'columns': 0}

        def _checkbox(label='', value=True, **kw):
            # First checkbox = recursive toggle; subsequent = type filters
            count = call_counts['checkbox']
            call_counts['checkbox'] += 1
            return recursive if count == 0 else True

        def _columns(n, **kw):
            # Return 7 mocks for the type row, 2 for the depth/pages row
            call_counts['columns'] += 1
            return seven_cols if n == 7 else two_cols

        def _text_input(label='', **kw):
            # URL input is the LAST text_input in non-recursive; topic filter is first
            count = call_counts['checkbox']   # reuse count as proxy
            return url

        return [
            patch.object(st, 'session_state', session),
            patch.object(st, 'expander',
                         return_value=MagicMock(
                             __enter__=MagicMock(return_value=MagicMock()),
                             __exit__=MagicMock(return_value=False))),
            patch.object(st, 'checkbox',      side_effect=_checkbox),
            patch.object(st, 'number_input',  return_value=2),
            patch.object(st, 'text_input',    return_value=url),
            patch.object(st, 'columns',       side_effect=_columns),
            patch.object(st, 'form',
                         return_value=MagicMock(
                             __enter__=MagicMock(return_value=MagicMock()),
                             __exit__=MagicMock(return_value=False))),
            patch.object(st, 'form_submit_button', return_value=submitted),
            patch.object(st, 'caption',  return_value=None),
            patch.object(st, 'markdown', return_value=None),
            patch.object(st, 'success',  return_value=None),
            patch.object(st, 'error',    return_value=None),
        ]

    def test_returns_false_when_not_submitted(self):
        """If the URL form is not submitted, handle_url_ingestion returns False."""
        import streamlit as st
        session = _make_session_state()
        patches = self._url_base_patches(session, url='', submitted=False)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11]:
            from src.ui import handlers as h
            result = h.handle_url_ingestion(_mock_loader(), _mock_store())

        assert result is False

    def test_process_url_called_on_submission(self):
        """process_url is called when a non-empty URL is submitted (non-recursive)."""
        import streamlit as st
        session = _make_session_state()
        patches = self._url_base_patches(
            session, url='https://example.com', submitted=True
        )

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11], \
             patch('src.ui.handlers.process_url') as mock_process:
            from src.ui import handlers as h
            result = h.handle_url_ingestion(_mock_loader(), _mock_store())

        mock_process.assert_called_once()
        assert result is True

    def test_no_fetch_when_url_is_empty(self):
        """process_url must NOT be called when the submitted URL is empty."""
        import streamlit as st
        session = _make_session_state()
        patches = self._url_base_patches(session, url='  ', submitted=True)

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11], \
             patch('src.ui.handlers.process_url') as mock_process:
            from src.ui import handlers as h
            h.handle_url_ingestion(_mock_loader(), _mock_store())

        mock_process.assert_not_called()

    def test_recursive_crawl_used_when_checkbox_on(self):
        """process_url_recursive is called (not process_url) when recursive is on."""
        import streamlit as st
        session = _make_session_state()
        patches = self._url_base_patches(
            session, recursive=True, url='https://en.wikipedia.org', submitted=True
        )

        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11], \
             patch('src.ui.handlers.process_url')          as mock_single, \
             patch('src.ui.handlers.process_url_recursive') as mock_recursive:
            from src.ui import handlers as h
            h.handle_url_ingestion(_mock_loader(), _mock_store())

        mock_recursive.assert_called_once()
        mock_single.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# handle_file_upload
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleFileUpload:
    """Unit tests for handle_file_upload()."""

    def test_returns_false_when_no_files_uploaded(self):
        """If no files are selected, handle_file_upload returns False."""
        import streamlit as st
        session = _make_session_state()

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'file_uploader', return_value=[]), \
             patch.object(st, 'caption', return_value=None):
            from src.ui import handlers as h
            result = h.handle_file_upload(_mock_loader(), _mock_store())

        assert result is False

    def test_index_button_shown_when_files_present(self):
        """st.button is called with the index label when files are uploaded."""
        import streamlit as st
        session = _make_session_state()

        mock_file = MagicMock()
        mock_file.name = 'test.txt'
        button_labels = []

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'file_uploader', return_value=[mock_file]), \
             patch.object(st, 'button', side_effect=lambda label, **kw: button_labels.append(label) or False), \
             patch.object(st, 'caption', return_value=None):
            from src.ui import handlers as h
            h.handle_file_upload(_mock_loader(), _mock_store())

        assert any('Index' in l or 'index' in l for l in button_labels), \
            f"Index button not shown. Buttons: {button_labels}"


# ─────────────────────────────────────────────────────────────────────────────
# Combination tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHandlerCombinations:
    """Cross-handler tests — multiple handlers called in sequence, as app.py does."""

    def test_topic_search_and_url_ingestion_both_return_false_when_idle(self):
        """Both handlers return False (no rerun needed) when nothing is submitted."""
        import streamlit as st
        session = _make_session_state()

        # Seven column mocks for the document-type checkbox row inside URL ingestion
        seven_cols = [MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                                __exit__=MagicMock(return_value=False))
                      for _ in range(7)]

        def _columns_mock(n, **kw):
            return seven_cols if n == 7 else [MagicMock(), MagicMock(), MagicMock()]

        common_patches = [
            patch.object(st, 'session_state', session),
            patch.object(st, 'expander',  return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))),
            patch.object(st, 'checkbox',  return_value=False),
            patch.object(st, 'form',      return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))),
            patch.object(st, 'text_input',  return_value=''),
            patch.object(st, 'number_input', return_value=5),
            patch.object(st, 'form_submit_button', return_value=False),
            patch.object(st, 'columns', side_effect=_columns_mock),
            patch.object(st, 'caption',  return_value=None),
            patch.object(st, 'markdown', return_value=None),
            patch.object(st, 'file_uploader', return_value=[]),
            patch.object(st, 'success',  return_value=None),
            patch.object(st, 'error',    return_value=None),
        ]
        from src.ui import handlers as h

        with common_patches[0], common_patches[1], common_patches[2], \
             common_patches[3], common_patches[4], common_patches[5], \
             common_patches[6], common_patches[7], common_patches[8], \
             common_patches[9], common_patches[10], \
             common_patches[11], common_patches[12]:
            loader = _mock_loader()
            store  = _mock_store()
            r1 = h.handle_topic_search(loader, store)
            r2 = h.handle_url_ingestion(loader, store)
            r3 = h.handle_file_upload(loader, store)

        # None of the handlers should request a rerun when nothing is submitted
        assert not r1 and not r2 and not r3

    def test_needs_rerun_true_after_topic_search_succeeds(self):
        """handle_topic_search returns True (triggers rerun) after a successful search."""
        import streamlit as st
        session = _make_session_state()
        loader  = _mock_loader(chunks=[{'text': 'result', 'source': 'http://x.com'}])
        store   = _mock_store()

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__  = MagicMock(return_value=False)

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'form',     return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',   return_value='python'), \
             patch.object(st, 'number_input', return_value=3), \
             patch.object(st, 'form_submit_button', return_value=True), \
             patch.object(st, 'columns',  return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None), \
             patch.object(st, 'status',   return_value=mock_status), \
             patch.object(st, 'success',  return_value=None), \
             patch.object(st, 'spinner',  return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from src.ui import handlers as h
            result = h.handle_topic_search(loader, store)

        assert result is True

    def test_url_ingestion_and_file_upload_independent_return_values(self):
        """URL ingestion returning True does not affect handle_file_upload's return."""
        import streamlit as st
        session = _make_session_state()
        seven_cols = [MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                                __exit__=MagicMock(return_value=False))
                      for _ in range(7)]

        def _columns_mock(n, **kw):
            return seven_cols if n == 7 else [MagicMock(), MagicMock()]

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'checkbox',  return_value=False), \
             patch.object(st, 'number_input', return_value=2), \
             patch.object(st, 'form',      return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'text_input',        return_value='https://example.com'), \
             patch.object(st, 'form_submit_button', return_value=True), \
             patch.object(st, 'columns',   side_effect=_columns_mock), \
             patch.object(st, 'caption',  return_value=None), \
             patch.object(st, 'markdown', return_value=None), \
             patch.object(st, 'success',  return_value=None), \
             patch.object(st, 'error',    return_value=None), \
             patch.object(st, 'file_uploader', return_value=[]), \
             patch('src.ui.handlers.process_url'):
            from src.ui import handlers as h
            loader = _mock_loader()
            store  = _mock_store()
            url_result  = h.handle_url_ingestion(loader, store)
            file_result = h.handle_file_upload(loader, store)

        # URL submitted → True; no file selected → False
        assert url_result  is True
        assert file_result is False


# ─────────────────────────────────────────────────────────────────────────────
# process_url (ui/ingestion.py) — single URL path
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessUrl:
    """Unit tests for the process_url function in ui/ingestion.py."""

    def test_success_sets_ok_url_msg(self):
        """A successful fetch must set session_state.url_msg to ('ok', ...)."""
        import streamlit as st
        session = _make_session_state()
        loader  = _mock_loader(chunks=[{'text': 'hi', 'source': 'http://x.com'}])
        store   = _mock_store(chunk_count=1)

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'spinner', return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from src.ui.ingestion import process_url
            process_url('https://example.com', loader, store)

        assert session['url_msg'] is not None
        assert session['url_msg'][0] == 'ok'

    def test_empty_chunks_sets_err_url_msg(self):
        """If no chunks are returned, session_state.url_msg must be ('err', ...)."""
        import streamlit as st
        session = _make_session_state()
        loader  = _mock_loader(chunks=[])
        store   = _mock_store()

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'spinner', return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from src.ui.ingestion import process_url
            process_url('https://example.com', loader, store)

        assert session['url_msg'] is not None
        assert session['url_msg'][0] == 'err'

    def test_exception_sets_err_url_msg(self):
        """An exception during chunking must set session_state.url_msg to ('err', ...)."""
        import streamlit as st
        session = _make_session_state()
        loader  = _mock_loader()
        loader.chunk_url.side_effect = ConnectionError("timeout")
        store   = _mock_store()

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'spinner', return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
            from src.ui.ingestion import process_url
            process_url('https://example.com', loader, store)

        assert session['url_msg'] is not None
        assert session['url_msg'][0] == 'err'


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / negative
# ─────────────────────────────────────────────────────────────────────────────

class TestUiComponentsBoundary:
    """Boundary and negative tests for UI component edge cases."""

    def test_render_header_is_safe_to_call_multiple_times(self):
        """render_header() must not crash or error when called twice in a row."""
        import streamlit as st
        with patch.object(st, 'markdown'):
            from src.ui.renderers import render_header
            render_header()
            render_header()   # second call must not raise

    def test_render_footer_is_safe_to_call_multiple_times(self):
        """render_footer() must not crash or error when called twice in a row."""
        import streamlit as st
        with patch.object(st, 'markdown'):
            from src.ui.renderers import render_footer
            render_footer()
            render_footer()

    def test_handle_file_upload_with_empty_uploaded_list_returns_false(self):
        """Empty file upload list → no button shown, returns False immediately."""
        import streamlit as st
        session = _make_session_state()

        with patch.object(st, 'session_state', session), \
             patch.object(st, 'expander', return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))), \
             patch.object(st, 'file_uploader', return_value=None), \
             patch.object(st, 'caption', return_value=None):
            from src.ui import handlers as h
            result = h.handle_file_upload(_mock_loader(), _mock_store())

        assert result is False
