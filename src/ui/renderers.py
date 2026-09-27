"""renderers.py — Pure Streamlit rendering functions for the RAG Agent UI.

WHY THIS FILE EXISTS:
    Rendering functions (header, footer, chat history, agent tools panel)
    have no backend logic — they only call st.markdown / st.chat_message.
    Keeping them separate from event handlers (handle_url_ingestion, etc.)
    makes both files shorter and easier to navigate.

RULE:
    Functions in this file must not trigger any backend computation.
    If a function calls store, loader, or Agent — it belongs in handlers.py.
"""

import streamlit as st

from src.rag.vector_store import VectorStore

__all__ = [
    'render_header',
    'render_footer',
    'render_mode_selector',
    'render_chat_history',
    'render_clear_button',
]


def render_header() -> None:
    """Show the app title, tagline, and capability chips at the top of the page."""
    st.markdown(
        """
        <div class="rag-header">
          <div class="rag-title">🧠 RAG Agent</div>
          <div class="rag-sub">AI-powered document intelligence — fully local, on-device</div>
          <div class="rag-chips">
            <span class="rag-chip">Hybrid Search</span>
            <span class="rag-chip">Reranking</span>
            <span class="rag-chip">Agent Mode</span>
            <span class="rag-chip">PDF</span>
            <span class="rag-chip">DOCX</span>
            <span class="rag-chip">XLSX</span>
            <span class="rag-chip">PPTX</span>
            <span class="rag-chip">CSV</span>
            <span class="rag-chip">MD</span>
            <span class="rag-chip">HTML</span>
            <span class="rag-chip">URL Crawl</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render a minimal footer with a GitHub link and a friendly call-to-action."""
    st.markdown(
        '<div style="text-align:center;font-size:.72rem;color:#9ca3af;margin-top:1.5rem;'
        'padding-top:.65rem;border-top:1px solid #e5e7eb;font-family:\'IBM Plex Mono\',monospace;">'
        'Fully local · no data leaves your machine<br/>'
        '<a href="https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent" '
        'target="_blank" rel="noopener noreferrer" '
        'style="color:#1a56db;font-weight:600;text-decoration:none;">⭐ Star on GitHub</a>'
        ' &nbsp;·&nbsp; '
        '<a href="https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent/issues" '
        'target="_blank" rel="noopener noreferrer" '
        'style="color:#1a56db;text-decoration:none;">Report an issue ↗</a>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_mode_selector() -> None:
    """Show the Chat / Agent radio and save the user's choice in session state.

    When Agent mode is selected, also shows a panel listing the available tools
    so the user knows what they can ask the agent to do.
    """
    chosen_mode = st.radio(
        "Mode:",
        ["Chat", "Agent"],
        horizontal=True,
        index=0 if st.session_state.mode == 'chat' else 1,
    )
    # Store as lowercase so the rest of the code can compare with 'chat' / 'agent'
    st.session_state.mode = chosen_mode.lower()

    if st.session_state.mode == 'agent':
        _render_agent_tools_panel()


def render_chat_history() -> None:
    """Display all previous messages in the conversation as chat bubbles."""
    for message in st.session_state.display:
        avatar = _pick_avatar(message['role'])
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'], unsafe_allow_html=True)


def render_clear_button(store: VectorStore) -> None:
    """Show a Clear button below the chat when there are messages to clear.

    Clicking it wipes the on-screen conversation and the store's memory so
    the next question starts with a clean slate.

    Args:
        store: VectorStore whose conversation history should also be cleared.
    """
    if not st.session_state.display:
        return  # Nothing to clear yet — hide the button

    _, button_column = st.columns([6, 1])
    with button_column:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.conv    = []
            st.session_state.display = []
            st.session_state.last    = None
            st.session_state.total   = 0
            store.clear_conversation()
            st.rerun()


# ── Private helpers ────────────────────────────────────────────────────────────


def _render_agent_tools_panel() -> None:
    """Show the agent tools reference card when agent mode is active."""
    st.markdown(
        """
        <div class="tools-panel">
        <b>🤖 Agent Tools</b> &nbsp;—&nbsp; the agent picks the right tool automatically<br><br>
        <b>🔍 rag_search</b> — search the knowledge base<br>
        <span style="color:#6b7280;font-size:.82rem">e.g. "what skills does the resume mention?"</span><br><br>
        <b>🧮 calculator</b> — evaluate any maths expression<br>
        <span style="color:#6b7280;font-size:.82rem">e.g. "what is 15% of 85 000?"</span><br><br>
        <b>📝 summarise</b> — summarise a document or section<br>
        <span style="color:#6b7280;font-size:.82rem">e.g. "summarise the resume"</span><br><br>
        <b>💬 sentiment</b> — analyse tone &amp; sentiment<br>
        <span style="color:#6b7280;font-size:.82rem">e.g. "what is the tone of the cover letter?"</span><br><br>
        <b>🌐 translate</b> — translate to any language<br>
        <span style="color:#6b7280;font-size:.82rem">e.g. "translate the summary to Spanish"</span><br><br>
        <b>✅ finish</b> — return the final answer to the user
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_agent_steps_html(steps: list) -> str:
    """Turn a list of agent step dicts into an HTML string for the chat bubble.

    Args:
        steps: List of dicts with keys: 'step', 'tool', 'arg', 'result'.

    Returns:
        HTML string with one <div class="step"> per step.
    """
    html_parts = []
    for step in steps:
        # Truncate long arguments so they don't overflow the chat bubble
        short_arg    = step["arg"][:50]    + "..." if len(step["arg"]) > 50    else step["arg"]
        short_result = step["result"][:80] + "..." if len(step["result"]) > 80 else step["result"]
        html_parts.append(
            f'<div class="step">'
            f'Step {step["step"]}: {step["tool"]}({short_arg}) → {short_result}'
            f'</div>'
        )
    return "".join(html_parts)


def _pick_avatar(role: str) -> str:
    """Return the emoji avatar for a given message role.

    Args:
        role: 'user', 'agent', or 'assistant'.

    Returns:
        An emoji string.
    """
    avatars = {'user': '🧑', 'agent': '🤖', 'assistant': '💬'}
    return avatars.get(role, '💬')
