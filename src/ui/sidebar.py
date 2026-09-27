"""ui/sidebar.py — Sidebar rendering functions for the Streamlit UI.

WHY THIS FILE EXISTS:
    handlers.py was approaching the 500-line limit. The sidebar rendering
    functions (pipeline info, session stats, document type counts) are a
    natural, self-contained unit that can live in their own module.

ADDING A NEW SIDEBAR PANEL:
    1. Write a new _render_* function here.
    2. Call it from render_sidebar() below.
    You never need to change handlers.py for sidebar-only changes.
"""

import streamlit as st

from src.rag.vector_store import VectorStore
from src.ui.theme import BADGE_CLASSES, CONFIDENCE_BADGE

__all__ = ['render_sidebar', 'render_clear_added_chunks_button']


def render_sidebar(store: VectorStore, local_chunks: list) -> None:
    """Render the right-hand sidebar: pipeline info, session stats, chunk counts.

    Args:
        store:        VectorStore — used to generate source labels for chunks.
        local_chunks: All chunks loaded from the local ./docs folder.
    """
    # Section heading — styled with the .sidebar-section CSS class (uppercase, muted)
    st.markdown('<div class="sidebar-section">Pipeline</div>', unsafe_allow_html=True)

    if st.session_state.last:
        pipeline_data = st.session_state.last['data']
        if st.session_state.last['type'] == 'chat':
            _render_pipeline_chat_info(pipeline_data, store)
        else:
            _render_pipeline_agent_info(pipeline_data)
        st.markdown("---")

    render_clear_added_chunks_button(store)
    _render_session_stats(local_chunks)
    _render_document_type_counts(local_chunks)


def _render_pipeline_chat_info(data: dict, store: VectorStore) -> None:
    """Show query type badge, confidence badge, and pre/post rerank chunks.

    Args:
        data:  The result dict returned by store.run_pipeline().
        store: VectorStore — used to generate human-readable source labels.
    """
    query_type    = data['query_type']
    badge_class   = BADGE_CLASSES.get(query_type, 'b-gen')
    st.markdown(f'<span class="badge {badge_class}">{query_type}</span>', unsafe_allow_html=True)

    confidence_class = CONFIDENCE_BADGE[data['is_confident']]
    confidence_label = (
        f"conf:{data['best_score']:.2f}" if data['is_confident']
        else f"low:{data['best_score']:.2f}"
    )
    st.markdown(f'<span class="badge {confidence_class}">{confidence_label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    # "Before rerank" heading using the muted uppercase sidebar-section style
    st.markdown('<div class="sidebar-section">Before rerank</div>', unsafe_allow_html=True)
    for chunk_entry, similarity_score in data['retrieved'][:4]:
        source_label = store._source_label(chunk_entry)
        st.markdown(
            f'<div class="chunk"><span class="cs">{similarity_score:.3f}</span> '
            f'<span class="src">[{chunk_entry["source"]} {source_label}]</span><br/>'
            f'{chunk_entry["text"][:55]}...</div>',
            unsafe_allow_html=True,
        )

    # "After rerank" heading — shows both similarity score and LLM rerank score
    st.markdown('<div class="sidebar-section">After rerank</div>', unsafe_allow_html=True)
    for chunk_entry, similarity_score, rerank_score in data['reranked']:
        source_label = store._source_label(chunk_entry)
        st.markdown(
            f'<div class="chunk">'
            f'<span class="cs">sim:{similarity_score:.2f} re:{rerank_score:.0f}</span> '
            f'<span class="src">[{chunk_entry["source"]} {source_label}]</span><br/>'
            f'{chunk_entry["text"][:55]}...</div>',
            unsafe_allow_html=True,
        )


def _render_pipeline_agent_info(data: dict) -> None:
    """Show a compact list of tool calls the agent made.

    Args:
        data: The result dict returned by agent.run().
    """
    st.markdown('<div class="sidebar-section">Agent Steps</div>', unsafe_allow_html=True)
    for step in data['steps']:
        st.markdown(
            f'<div class="step">{step["step"]}. {step["tool"]}</div>',
            unsafe_allow_html=True,
        )


def render_clear_added_chunks_button(store: VectorStore) -> None:
    """Show a button to remove all URL and file-upload chunks from the knowledge base.

    Only visible when there are runtime chunks to remove. Clicking it calls
    store.clear_added_chunks(), resets session state, and reruns the page.
    Local document chunks (loaded from ./docs/ at startup) are never removed.

    Args:
        store: VectorStore — used to remove chunks and rebuild the index.
    """
    url_chunk_count = len(st.session_state.url_chunks)
    if url_chunk_count == 0:
        return   # Nothing to clear — hide the button

    st.markdown("---")
    if st.button(
        f"🗑 Clear added content ({url_chunk_count} chunks)",
        use_container_width=True,
        help="Removes all URL and file-upload chunks. Local docs are kept.",
    ):
        removed = store.clear_added_chunks()
        st.session_state.url_chunks  = []
        st.session_state.bm25_index  = None
        st.session_state.url_msg     = None
        st.session_state.file_msg    = None
        st.toast(f"Removed {removed} chunks. Knowledge base reset to local docs.")
        st.rerun()


def _render_session_stats(local_chunks: list) -> None:
    """Show query count, memory turns, total chunks, URL chunks, and mode.

    Args:
        local_chunks: Chunks loaded from the local ./docs folder (not uploads).
    """
    url_chunk_count = len(st.session_state.url_chunks)
    total_chunks    = len(local_chunks) + url_chunk_count

    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    stats = [
        ("Queries",    st.session_state.total),
        ("Memory",     f"{len(st.session_state.conv) // 2} turns"),
        ("Chunks",     total_chunks),
        ("URL chunks", url_chunk_count),
        ("Mode",       st.session_state.mode),
    ]
    for label, value in stats:
        st.markdown(
            f'<div class="stat">{label} <span class="sv">{value}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")


def _render_document_type_counts(local_chunks: list) -> None:
    """Show how many chunks belong to each document type (PDF, DOCX, etc.).

    Args:
        local_chunks: Chunks loaded from the local ./docs folder.
    """
    # Count how many chunks are from each document type
    type_counts: dict = {}
    for chunk in local_chunks:
        document_type = chunk.get('type', '?')
        type_counts[document_type] = type_counts.get(document_type, 0) + 1

    st.markdown('<div class="sidebar-section">Document Types</div>', unsafe_allow_html=True)
    for document_type, count in sorted(type_counts.items()):
        st.markdown(
            f'<div class="stat">{document_type.upper()} '
            f'<span class="sv">{count}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
