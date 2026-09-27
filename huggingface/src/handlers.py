"""handlers.py — Gradio event handlers and singleton initialisation for the HF Space.

Owns: singletons (_loader, _store), formatting helpers, all event handler functions,
and private pipeline runners. UI layout lives in ui_builder.py. CSS lives in theme.py.

To add a feature: write a handler here, wire it in ui_builder.py. Never touch app.py.
"""

import logging
import os
import re

import gradio as gr

from src.rag.agent import Agent
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore

# Module-level logger — errors go to the logging system, not the terminal
logger = logging.getLogger(__name__)

__all__ = [
    'chat', 'upload_file', 'fetch_url', 'fetch_url_recursive',
    'search_topic', 'clear_chat', 'clear_added_chunks',
    '_initialize', '_chunk_count',
]

# ── Singletons ─────────────────────────────────────────────────────────────────
# These are created once on first use and reused for the lifetime of the Space.

_loader: DocumentLoader = None   # type: ignore[assignment]
_store:  VectorStore    = None   # type: ignore[assignment]


def _initialize():
    """Create the DocumentLoader and VectorStore on first call, then reuse them.

    Returns:
        Tuple of (DocumentLoader, VectorStore).
    """
    global _loader, _store
    if _loader is None:
        _loader = DocumentLoader()
    if _store is None:
        _store = VectorStore()
        _store.build_or_load([])   # start with an empty knowledge base
    return _loader, _store


def _chunk_count() -> int:
    """Return the number of chunks currently in the VectorStore collection."""
    return _store.collection.count() if _store and _store.collection else 0


# ── Formatting helpers ─────────────────────────────────────────────────────────


def _pipeline_summary(data: dict) -> str:
    """Format run_pipeline output as a markdown string for the pipeline info panel.

    Args:
        data: The dict returned by store.run_pipeline(streamlit_mode=True).

    Returns:
        A markdown string with query type, confidence score, and chunk previews.
    """
    if not data:
        return ""

    query_type   = data.get('query_type', '')
    best_score   = data.get('best_score', 0.0)
    is_confident = data.get('is_confident', False)

    lines = [
        f"**Query type:** `{query_type}`",
        f"**Confidence:** {'✅' if is_confident else '⚠️'} `{best_score:.3f}`",
        "",
        "**Before rerank (top 4)**",
    ]

    for chunk_entry, similarity_score in data.get('retrieved', [])[:4]:
        source_label = _store._source_label(chunk_entry)
        lines.append(
            f"- `{similarity_score:.3f}` "
            f"[{chunk_entry['source']} {source_label}] "
            f"{chunk_entry['text'][:60]}..."
        )

    lines += ["", "**After rerank**"]
    for chunk_entry, similarity_score, rerank_score in data.get('reranked', []):
        source_label = _store._source_label(chunk_entry)
        lines.append(
            f"- sim:`{similarity_score:.2f}` re:`{rerank_score:.0f}` "
            f"[{chunk_entry['source']} {source_label}] "
            f"{chunk_entry['text'][:60]}..."
        )

    return '\n'.join(lines)


def _agent_steps_md(steps: list) -> str:
    """Format agent steps as a markdown string for display in the chat.

    Args:
        steps: List of step dicts with keys: 'step', 'tool', 'arg', 'result'.

    Returns:
        Markdown string with one bold heading and one blockquote per step.
    """
    lines = []
    for step in steps:
        short_arg    = step['arg'][:60]    + '...' if len(step['arg']) > 60    else step['arg']
        short_result = step['result'][:80] + '...' if len(step['result']) > 80 else step['result']
        lines.append(f"**Step {step['step']}** `{step['tool']}({short_arg})`")
        lines.append(f"*{short_result}*")
        lines.append("")
    return '\n'.join(lines)


# ── Core event handlers ────────────────────────────────────────────────────────


def chat(message: str, history: list, mode: str):
    """Handle one chat turn — route to Agent or pipeline and return the result.

    Args:
        message: The question or task the user typed.
        history: Gradio message history (list of role/content dicts).
        mode:    'Chat' or 'Agent'.

    Returns:
        Tuple of (updated_history, pipeline_info_markdown).
    """
    loader, store = _initialize()

    if not message or not message.strip():
        return history, ""

    # Allow pure-math questions even when the knowledge base is empty
    is_math_question = bool(re.search(r'[\d].*[\+\-\*\/\%]|[\+\-\*\/\%].*[\d]', message))
    knowledge_base_empty = (store.collection is None or store.collection.count() == 0)

    if not is_math_question and knowledge_base_empty:
        empty_notice = "⚠️ No documents in the knowledge base yet. Please upload a file or add a URL first."
        updated_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": empty_notice},
        ]
        return updated_history, ""

    if mode == "Agent":
        response, pipeline_info = _run_agent_mode(message, store)
    else:
        response, pipeline_info = _run_chat_mode(message, store)

    updated_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response},
    ]
    return updated_history, pipeline_info


def upload_file(file_objs, progress=None):
    """Chunk and index one or more uploaded files into the live knowledge base.

    Accepts a single file or a list of files so users can select multiple files
    (or all files from a folder) in one pick.

    Args:
        file_objs: A single Gradio filepath string / file-like object, or a list of them.
        progress:  Optional gr.Progress() for showing a progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if file_objs is None:
        return "No file selected.", f"Chunks in knowledge base: {_chunk_count()}"

    # Normalise to a list so single-file and multi-file uploads use the same code path
    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    total_chunks  = 0
    indexed_names = []
    errors        = []

    for i, file_obj in enumerate(file_objs):
        # Gradio 5 passes a plain filepath string; older versions pass a file-like object
        filepath  = file_obj if isinstance(file_obj, str) else file_obj.name
        filename  = os.path.basename(filepath)
        extension = os.path.splitext(filename)[1].lower()
        doc_type  = loader.ext_to_type.get(extension, 'txt')

        step = (i + 1) / len(file_objs)
        progress(step * 0.5, desc=f"Reading {filename}...")

        try:
            file_info = {
                'filepath':      filepath,
                'filename':      filename,
                'detected_type': doc_type,
                'is_misplaced':  False,
            }
            new_chunks = loader._dispatch_chunker(file_info)

            if new_chunks:
                progress(step * 0.5 + 0.4, desc=f"Embedding {len(new_chunks)} chunks from {filename}...")
                store.add_chunks(new_chunks, id_prefix='file')
                total_chunks += len(new_chunks)
                indexed_names.append(filename)
            else:
                errors.append(f"⚠️ No text extracted from **{filename}**.")

        except Exception as error:
            logger.error("File upload failed for '%s': %s", filename, error, exc_info=True)
            errors.append(f"❌ Error indexing **{filename}**: {error}")

    if total_chunks > 0:
        # Rebuild BM25 once after all files are indexed (more efficient than per-file)
        progress(0.95, desc="Rebuilding search index...")
        store.rebuild_bm25(store.chunks)
        progress(1.0, desc="Done")

    # Build a readable status message summarising the whole batch
    parts = []
    if indexed_names:
        names_str = ", ".join(f"**{n}**" for n in indexed_names)
        parts.append(f"✅ Indexed {names_str} — {total_chunks} chunks added.")
    parts.extend(errors)

    status = "\n".join(parts) if parts else "No files processed."
    return status, f"Chunks in knowledge base: **{_chunk_count()}**"


def fetch_url(url: str, progress=None):
    """Fetch a public URL, chunk its content, and add it to the knowledge base.

    Args:
        url:      The public HTTP/HTTPS URL to fetch.
        progress: Optional gr.Progress() for showing a progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if not url or not url.strip():
        return "No URL provided.", f"Chunks in knowledge base: {_chunk_count()}"

    clean_url = url.strip()
    try:
        progress(0.2, desc="Fetching URL...")
        new_chunks = loader.chunk_url(clean_url)

        if new_chunks:
            progress(0.5, desc=f"Embedding {len(new_chunks)} chunks on CPU — please wait...")
            store.add_chunks(new_chunks, id_prefix='url')
            progress(0.9, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (
                f"✅ Indexed **{clean_url[:60]}** — {len(new_chunks)} chunks added.",
                f"Chunks in knowledge base: **{_chunk_count()}**",
            )
        else:
            return (
                "⚠️ No content extracted from URL.",
                f"Chunks in knowledge base: {_chunk_count()}",
            )

    except Exception as error:
        logger.error("URL fetch failed for '%s': %s", clean_url, error, exc_info=True)
        return (
            f"❌ Error fetching URL: {error}",
            f"Chunks in knowledge base: {_chunk_count()}",
        )


def fetch_url_recursive(url: str, depth: int, max_pages: int,
                        topic_filter: str,
                        use_html: bool, use_pdf: bool, use_docx: bool,
                        use_xlsx: bool, use_csv: bool, use_pptx: bool,
                        use_md: bool, progress=None):
    """Crawl a seed URL recursively and index all discovered pages.

    Args:
        url:          The seed URL to crawl from.
        depth:        How many link-levels deep to follow (1–3).
        max_pages:    Maximum total pages to fetch (1–50).
        topic_filter: Optional keyword the URL path must contain to be crawled.
                      Empty string means no filter — crawl everything on the domain.
        use_html, use_pdf, use_docx, use_xlsx, use_csv, use_pptx, use_md:
                      Which document types to include in the crawl.
        progress:     Optional gr.Progress() for a progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if not url or not url.strip():
        return "No URL provided.", f"Chunks in knowledge base: {_chunk_count()}"

    # Build the set of allowed document types from the checkboxes
    type_map    = {'html': use_html, 'pdf': use_pdf, 'docx': use_docx,
                   'xlsx': use_xlsx, 'csv': use_csv, 'pptx': use_pptx, 'md': use_md}
    allowed     = {t for t, checked in type_map.items() if checked}
    allowed_set = allowed if allowed else None   # None means "all types"

    clean_url  = url.strip()
    pages_done = [0]   # mutable list so the callback can update it

    def _progress_callback(page_url: str, dtype: str, chunk_count: int) -> None:
        """Update the Gradio progress bar after each page is crawled."""
        pages_done[0] += 1
        # Progress is approximate — we don't know the total in advance
        fraction = min(0.9, pages_done[0] / max(max_pages, 1))
        short    = page_url[:60] + '...' if len(page_url) > 60 else page_url
        progress(fraction, desc=f"[{dtype.upper()}] {short}")

    try:
        progress(0.05, desc="Starting crawl...")
        new_chunks = loader.chunk_url_recursive(
            clean_url,
            depth=int(depth),
            max_pages=int(max_pages),
            allowed_types=allowed_set,
            topic_filter=topic_filter,
            progress_callback=_progress_callback,
        )

        if new_chunks:
            progress(0.92, desc=f"Embedding {len(new_chunks)} chunks on CPU — please wait...")
            store.add_chunks(new_chunks, id_prefix='url')
            progress(0.97, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (
                f"✅ Crawled {pages_done[0]} pages — {len(new_chunks)} chunks added.",
                f"Chunks in knowledge base: **{_chunk_count()}**",
            )
        else:
            return (
                "⚠️ No content extracted from crawl.",
                f"Chunks in knowledge base: {_chunk_count()}",
            )

    except Exception as error:
        logger.error("Recursive crawl failed for '%s': %s", clean_url, error, exc_info=True)
        return (
            f"❌ Crawl error: {error}",
            f"Chunks in knowledge base: {_chunk_count()}",
        )


def search_topic(query: str, num_results: int, depth: int,
                 pages_per: int, progress=None):
    """Search DuckDuckGo for a topic and index the top result pages.

    Args:
        query:       The search query.
        num_results: Number of search result URLs to fetch (1–20).
        depth:       Crawl depth per result URL (1–3).
        pages_per:   Max pages crawled per result URL.
        progress:    Optional gr.Progress for the progress bar.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if progress is None:
        progress = gr.Progress()

    loader, store = _initialize()

    if not query or not query.strip():
        return "No query provided.", f"Chunks in knowledge base: {_chunk_count()}"

    pages_done = [0]

    def _progress_callback(page_url: str, dtype: str, chunk_count: int) -> None:
        """Update the Gradio progress bar after each page is fetched."""
        pages_done[0] += 1
        fraction = min(0.9, pages_done[0] / max(int(num_results) * int(pages_per), 1))
        short    = page_url[:60] + '...' if len(page_url) > 60 else page_url
        progress(fraction, desc=f"[{dtype.upper()}] {short}")

    try:
        progress(0.05, desc=f"Searching for '{query.strip()}'...")
        new_chunks = loader.chunk_topic_search(
            query.strip(),
            num_results=int(num_results),
            depth=int(depth),
            max_pages_per_result=int(pages_per),
            progress_callback=_progress_callback,
        )

        if new_chunks:
            progress(0.92, desc=f"Embedding {len(new_chunks)} chunks...")
            store.add_chunks(new_chunks, id_prefix='search')
            progress(0.97, desc="Rebuilding search index...")
            store.rebuild_bm25(store.chunks)
            progress(1.0, desc="Done")
            return (
                f"✅ Indexed {len(new_chunks)} chunks from "
                f"{int(num_results)} search results for \"{query.strip()}\".",
                f"Chunks in knowledge base: **{_chunk_count()}**",
            )
        else:
            return (
                "⚠️ No content extracted. Try a different query.",
                f"Chunks in knowledge base: {_chunk_count()}",
            )
    except Exception as error:
        logger.error("Topic search failed for '%s': %s", query, error, exc_info=True)
        return (
            f"❌ Search error: {error}",
            f"Chunks in knowledge base: {_chunk_count()}",
        )


def clear_chat():
    """Reset the chat history and the store's conversation memory.

    Returns:
        Tuple of (empty_history, empty_pipeline_info).
    """
    _store.clear_conversation()
    return [], ""


def clear_added_chunks():
    """Remove all URL and file-upload chunks from the knowledge base.

    Keeps local document chunks (loaded at startup). Rebuilds BM25 from
    the remaining local chunks only.

    Returns:
        Tuple of (status_message, chunk_counter_markdown).
    """
    if _store is None:
        return "⚠️ Not initialised yet.", f"Chunks in knowledge base: 0"

    removed = _store.clear_added_chunks()
    if removed > 0:
        return (
            f"✅ Removed {removed} chunks. Knowledge base reset to local documents.",
            f"Chunks in knowledge base: **{_chunk_count()}**",
        )
    return (
        "ℹ️ No URL or file-upload chunks to remove.",
        f"Chunks in knowledge base: **{_chunk_count()}**",
    )


# ── Private pipeline runners ───────────────────────────────────────────────────


def _run_agent_mode(message: str, store: VectorStore):
    """Run the agent loop and return (response_markdown, pipeline_info).

    Args:
        message: The user's task or question.
        store:   VectorStore the agent uses for rag_search.

    Returns:
        Tuple of (formatted_response_string, pipeline_info_string).
    """
    agent         = Agent(store)
    result        = agent.run(message, streamlit_mode=True)
    steps_md      = _agent_steps_md(result['steps'])
    response      = f"{steps_md}\n**Answer:** {result['answer']}"
    pipeline_info = f"**Agent mode** — {len(result['steps'])} steps"
    return response, pipeline_info


def _run_chat_mode(message: str, store: VectorStore):
    """Run the RAG pipeline and return (response_text, pipeline_info_markdown).

    Args:
        message: The user's question.
        store:   VectorStore that runs the full retrieval + generation pipeline.

    Returns:
        Tuple of (response_text, pipeline_info_markdown).
    """
    result        = store.run_pipeline(message, streamlit_mode=True)
    pipeline_info = _pipeline_summary(result)
    return result['response'], pipeline_info
