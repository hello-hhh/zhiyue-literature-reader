"""Unit tests for ui/theme.py and ui/session.py."""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ui/theme.py
# ---------------------------------------------------------------------------

class TestTheme:
    def test_css_is_string(self):
        from ui.theme import CSS
        assert isinstance(CSS, str)

    def test_css_contains_ibm_plex(self):
        from ui.theme import CSS
        assert 'IBM Plex' in CSS

    def test_css_contains_all_badge_classes(self):
        from ui.theme import CSS
        for cls in ('b-fact', 'b-comp', 'b-gen', 'b-ok', 'b-warn'):
            assert cls in CSS, f"Missing CSS class: {cls}"

    def test_css_contains_rag_title(self):
        from ui.theme import CSS
        assert 'rag-title' in CSS

    def test_css_contains_chat_message_selector(self):
        from ui.theme import CSS
        assert 'stChatMessage' in CSS

    def test_badge_classes_keys(self):
        from ui.theme import BADGE_CLASSES
        assert set(BADGE_CLASSES.keys()) == {'factual', 'comparison', 'general', 'summarise'}

    def test_badge_classes_values(self):
        from ui.theme import BADGE_CLASSES
        assert BADGE_CLASSES['factual']    == 'b-fact'
        assert BADGE_CLASSES['comparison'] == 'b-comp'
        assert BADGE_CLASSES['general']    == 'b-gen'
        assert BADGE_CLASSES['summarise']  == 'b-gen'

    def test_confidence_badge_keys(self):
        from ui.theme import CONFIDENCE_BADGE
        assert True  in CONFIDENCE_BADGE
        assert False in CONFIDENCE_BADGE

    def test_confidence_badge_values(self):
        from ui.theme import CONFIDENCE_BADGE
        assert CONFIDENCE_BADGE[True]  == 'b-ok'
        assert CONFIDENCE_BADGE[False] == 'b-warn'

    def test_avatar_keys(self):
        from ui.theme import AVATAR
        assert set(AVATAR.keys()) == {'user', 'assistant', 'agent'}

    def test_avatar_values_are_strings(self):
        from ui.theme import AVATAR
        for v in AVATAR.values():
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# ui/session.py
# ---------------------------------------------------------------------------

class TestSessionDefaults:
    def test_session_defaults_has_all_keys(self):
        from ui.session import SESSION_DEFAULTS
        expected = {'conv', 'display', 'total', 'last', 'mode',
                    'url_chunks', 'bm25_index', 'url_msg', 'file_msg'}
        assert set(SESSION_DEFAULTS.keys()) == expected

    def test_session_defaults_mode_is_chat(self):
        from ui.session import SESSION_DEFAULTS
        assert SESSION_DEFAULTS['mode'] == 'chat'

    def test_session_defaults_total_is_zero(self):
        from ui.session import SESSION_DEFAULTS
        assert SESSION_DEFAULTS['total'] == 0

    def test_session_defaults_lists_are_empty(self):
        from ui.session import SESSION_DEFAULTS
        for k in ('conv', 'display', 'url_chunks'):
            assert SESSION_DEFAULTS[k] == []

    def test_session_defaults_nones(self):
        from ui.session import SESSION_DEFAULTS
        for k in ('last', 'bm25_index', 'url_msg', 'file_msg'):
            assert SESSION_DEFAULTS[k] is None


class TestInitSessionState:
    def _make_mock_st(self, existing_keys=()):
        """Return a mock st with session_state backed by a plain dict."""
        mock_st = MagicMock()
        state = {}
        for k in existing_keys:
            state[k] = 'existing_value'
        # Assign a plain dict — supports `in`, `[]`, naturally
        mock_st.session_state = state
        return mock_st, state

    def test_sets_all_defaults_when_empty(self):
        from ui.session import SESSION_DEFAULTS
        mock_st, state = self._make_mock_st()
        with patch('ui.session.st', mock_st):
            from ui.session import init_session_state
            init_session_state()
        for k, v in SESSION_DEFAULTS.items():
            assert k in state

    def test_does_not_overwrite_existing_key(self):
        mock_st, state = self._make_mock_st(existing_keys=['mode'])
        state['mode'] = 'agent'
        with patch('ui.session.st', mock_st):
            from ui.session import init_session_state
            init_session_state()
        assert state['mode'] == 'agent'


class TestGetActiveBm25:
    def test_returns_session_bm25_when_set(self):
        mock_st = MagicMock()
        mock_st.session_state.bm25_index = 'session_bm25'
        with patch('ui.session.st', mock_st):
            from ui.session import get_active_bm25
            result = get_active_bm25('base_bm25')
        assert result == 'session_bm25'

    def test_returns_base_bm25_when_session_is_none(self):
        mock_st = MagicMock()
        mock_st.session_state.bm25_index = None
        with patch('ui.session.st', mock_st):
            from ui.session import get_active_bm25
            result = get_active_bm25('base_bm25')
        assert result == 'base_bm25'
