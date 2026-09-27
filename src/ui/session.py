"""ui/session.py — Streamlit session state helpers."""

import streamlit as st

__all__ = ['SESSION_DEFAULTS', 'init_session_state', 'get_active_bm25']

# Session state keys and their initial values.
# conv:        full conversation history shown in the chat view (list of dicts)
# display:     rendered HTML or text for each turn (list of strings)
# total:       number of queries answered in this session (int)
# last:        pipeline result dict from the most recent query (dict or None)
# mode:        'chat' or 'agent' — controls which pipeline to invoke (str)
# url_chunks:  chunks added via URL ingestion (list of chunk dicts)
# bm25_index:  rebuilt BM25 index after URL/file upload (BM25Okapi or None)
# url_msg:     status message for the URL ingestion panel (str or None)
# file_msg:    status message for the file upload panel (str or None)
SESSION_DEFAULTS: dict = {
    'conv':        [],
    'display':     [],
    'total':       0,
    'last':        None,
    'mode':        'chat',
    'url_chunks':  [],
    'bm25_index':  None,
    'url_msg':     None,
    'file_msg':    None,
}


def init_session_state() -> None:
    """Set defaults for missing session state keys only."""
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_active_bm25(base_bm25):
    """Return session BM25 if updated after upload, else the base index."""
    return st.session_state.bm25_index if st.session_state.bm25_index is not None else base_bm25
