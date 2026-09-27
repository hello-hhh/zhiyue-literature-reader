"""ui/handlers.py — Streamlit event handlers for ingestion and chat.

WHY THIS FILE EXISTS:
    app.py must stay under 50 lines so it is easy to read at a glance.
    Event handlers (URL ingestion, file upload, topic search, user input)
    live here. Pure rendering functions live in renderers.py.

ADDING A NEW PANEL OR FEATURE:
    1. Write a new handler function here with a clear docstring.
    2. Import it in app.py and call it in the right place.
"""

import logging
import os
import tempfile

import streamlit as st

from src.rag.agent import Agent
from src.rag.config import URL_CRAWL_MAX_DEPTH, URL_CRAWL_MAX_PAGES
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.ui.ingestion import process_url, process_url_recursive
from src.ui.renderers import _format_agent_steps_html
from src.ui.sidebar import render_sidebar

logger = logging.getLogger(__name__)

__all__ = [
    'initialize',
    'handle_url_ingestion',
    'handle_file_upload',
    'handle_topic_search',
    'handle_user_input',
    'render_sidebar',
]


@st.cache_resource
def initialize() -> tuple:
    """Load documents and build the vector index once, then cache the result.

    Called once at app startup. The @st.cache_resource decorator ensures
    Streamlit does not re-run this on every page interaction — only on the
    first load or after a server restart.

    Returns:
        Tuple of (DocumentLoader, VectorStore) ready for use.
    """
    loader = DocumentLoader()
    loader.ensure_folders()
    # The original CLI intentionally exits when ./docs is empty. A personal
    # library, however, must have a useful first-run empty state so users can
    # reach the import screen before any documents exist.
    chunks = loader.chunk_all_documents() if loader.scan_all_files() else []
    store  = VectorStore()
    try:
        store.build_or_load(chunks)
        store.index_error = ""
    except Exception as error:
        # Reading, notes, and highlighting must remain available even when the
        # optional local Ollama service is not running. Initialise an empty
        # collection; importing/indexing and AI questions can be retried after
        # Ollama is started.
        logger.warning("Ollama is unavailable; starting in reading-only mode: %s", error)
        store.build_or_load([])
        store.index_error = str(error)
    return loader, store


def handle_url_ingestion(loader, store: VectorStore) -> bool:
    """Show the URL input form and process any URL the user submits.

    The recursive crawl toggle is placed OUTSIDE the form so that checking
    it immediately shows or hides the depth, max-pages, and type controls
    without requiring a form submit first. The URL text box and the submit
    button stay inside the form so the URL is cleared after each submission.

    Args:
        loader: DocumentLoader — handles fetching and chunking the URL.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if a URL was successfully submitted and the page should refresh.
        False if nothing happened (form not submitted or already shown result).
    """
    needs_rerun = False

    with st.expander("🌐 Add a URL", expanded=False):

        st.caption("Paste any public URL — webpage, PDF, DOCX, XLSX, CSV, PPTX, or Markdown.")

        # ── Toggle outside the form so it triggers an immediate rerun ──────
        use_recursive = st.checkbox(
            "🕷️ Recursive crawl — follow links and index multiple pages",
            value=False,
            key='use_recursive',
        )

        # ── Crawl settings — visible only when recursive mode is on ────────
        crawl_depth     = URL_CRAWL_MAX_DEPTH
        crawl_max_pages = URL_CRAWL_MAX_PAGES
        crawl_topic     = ''
        allowed_types   = None

        if use_recursive:
            st.markdown(
                '<div class="note">💡 <b>Topic filter is strongly recommended for Wikipedia.</b><br/>'
                'Without it, depth ≥ 2 follows every inline link — TIFF → Toronto → Ontario → …<br/>'
                'Set topic = article name (e.g. <code>Elizabeth_Olsen</code>) to stay on one subject.</div>',
                unsafe_allow_html=True,
            )

            col_depth, col_pages = st.columns(2)
            with col_depth:
                crawl_depth = st.number_input(
                    "Depth",
                    min_value=1, max_value=3, value=URL_CRAWL_MAX_DEPTH,
                    help="1 = only links on the seed page.  2 = links of links.  "
                         "Higher values may be slow on large sites.",
                    key='crawl_depth',
                )
            with col_pages:
                crawl_max_pages = st.number_input(
                    "Max pages",
                    min_value=1, max_value=50, value=URL_CRAWL_MAX_PAGES,
                    help="Hard cap on total pages fetched. Increase for larger sites.",
                    key='crawl_max_pages',
                )

            crawl_topic = st.text_input(
                "Topic filter — strongly recommended",
                placeholder="e.g. Elizabeth_Olsen  ·  python  ·  machine-learning",
                help="Only follow links whose URL path contains this word. "
                     "Without it, depth ≥ 2 follows every inline link and explodes the page budget.",
                key='crawl_topic',
            )

            st.caption("Index these document types during the crawl:")
            type_cols   = st.columns(7)
            type_labels = ['HTML', 'PDF', 'DOCX', 'XLSX', 'CSV', 'PPTX', 'MD']
            type_keys   = ['html', 'pdf', 'docx', 'xlsx', 'csv', 'pptx', 'md']
            checked_types = []
            for col, label, key in zip(type_cols, type_labels, type_keys):
                with col:
                    if st.checkbox(label, value=True, key=f'crawl_type_{key}'):
                        checked_types.append(key)
            # None means "accept all types" — only filter if at least one is unchecked
            allowed_types = set(checked_types) if len(checked_types) < 7 else None

        # ── URL input + submit button inside the form (cleared on submit) ──
        with st.form('url_form', clear_on_submit=True):
            url_input = st.text_input(
                "URL",
                placeholder="https://example.com/page  ·  en.wikipedia.org/wiki/Elizabeth_Taylor  ·  https://site.com/file.pdf",
                help="You can omit https:// — it will be added automatically.",
            )
            submitted = st.form_submit_button("⬆ Fetch & index")

        if submitted and url_input.strip():
            if use_recursive:
                process_url_recursive(
                    url_input.strip(), loader, store,
                    depth=int(crawl_depth),
                    max_pages=int(crawl_max_pages),
                    allowed_types=allowed_types,
                    topic_filter=crawl_topic,
                )
            else:
                process_url(url_input.strip(), loader, store)
            needs_rerun = True

        # Show the result message from the most recent URL submission
        if st.session_state.url_msg:
            kind, message = st.session_state.url_msg
            st.success(message) if kind == 'ok' else st.error(message)

    return needs_rerun


def handle_file_upload(loader, store: VectorStore) -> bool:
    """Show the file upload panel and index any files the user uploads.

    Accepts one or many files at once. Each file is saved to a temporary
    path, dispatched through the correct chunker, embedded, and added to
    the knowledge base. BM25 is rebuilt after every indexed file.

    Args:
        loader: DocumentLoader — handles chunking each file by its type.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if a file was indexed and the page should refresh. False otherwise.
    """
    needs_rerun = False

    with st.expander("📎 Upload files", expanded=False):
        st.caption("PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, Markdown, HTML — one or many at once.")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=[
                "pdf", "txt", "docx", "doc", "xlsx", "xls",
                "pptx", "ppt", "csv", "md", "markdown", "html", "htm",
            ],
            key="file_uploader",
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("⬆ Index files", key="file_index_btn"):
            for uploaded_file in uploaded_files:
                _process_uploaded_file(uploaded_file, loader, store)
            needs_rerun = True

        # Show the result message from the most recent upload batch
        if st.session_state.get('file_msg'):
            kind, message = st.session_state.file_msg
            st.success(message) if kind == 'ok' else st.error(message)

    return needs_rerun


def handle_topic_search(loader, store: VectorStore) -> bool:
    """Show the topic search panel and index the top web results for a query.

    Searches DuckDuckGo (no API key needed) for the query, then crawls
    each result URL using the same recursive pipeline as the URL crawl.

    Args:
        loader: DocumentLoader — runs the search and chunking.
        store:  VectorStore   — stores and indexes the resulting chunks.

    Returns:
        True if results were indexed and the page should refresh. False otherwise.
    """
    needs_rerun = False

    with st.expander("🔍 Search & index a topic", expanded=False):
        st.caption("Search the web for a topic and index the top results automatically.")
        st.markdown(
            '<div class="note">Uses DuckDuckGo — no API key needed.<br/>'
            'Each result URL is crawled to the chosen depth, same as the URL crawl.</div>',
            unsafe_allow_html=True,
        )

        with st.form('topic_search_form', clear_on_submit=True):
            query_input = st.text_input(
                "Search query",
                placeholder="e.g. Elizabeth Olsen actress  ·  Python asyncio tutorial",
                help="Enter a topic or question. Top results will be fetched and indexed.",
            )

            col_results, col_depth, col_pages = st.columns(3)
            with col_results:
                num_results = st.number_input(
                    "Results", min_value=1, max_value=20, value=5,
                    help="How many search result URLs to fetch.",
                )
            with col_depth:
                search_depth = st.number_input(
                    "Depth", min_value=1, max_value=3, value=1,
                    help="1 = result page only.  2 = also follow links on each result.",
                )
            with col_pages:
                pages_per = st.number_input(
                    "Pages per result", min_value=1, max_value=10, value=3,
                    help="Max pages crawled per result URL (applies when depth > 1).",
                )

            submitted = st.form_submit_button("🔍 Search & index")

        if submitted and query_input.strip():
            try:
                with st.status(
                    f"Searching for '{query_input.strip()}'…", expanded=True
                ) as status_box:
                    crawl_log: list = []

                    def progress_callback(page_url: str, dtype: str, chunk_count: int) -> None:
                        """Called after each page is fetched — updates the live crawl log."""
                        short_url = page_url[:70] + '...' if len(page_url) > 70 else page_url
                        crawl_log.append(f"[{dtype.upper()}] {short_url} — {chunk_count} chunks")
                        status_box.write('\n'.join(crawl_log[-8:]))

                    new_chunks = loader.chunk_topic_search(
                        query_input.strip(),
                        num_results=int(num_results),
                        depth=int(search_depth),
                        max_pages_per_result=int(pages_per),
                        progress_callback=progress_callback,
                    )

                    status_box.update(
                        label=(
                            f"Done — {len(crawl_log)} pages crawled, "
                            f"{len(new_chunks)} chunks indexed"
                        ),
                        state="complete",
                        expanded=False,
                    )

                if new_chunks:
                    store.add_chunks(new_chunks, id_prefix='search')
                    store.rebuild_bm25(store.chunks)
                    st.success(
                        f"Indexed {len(new_chunks)} chunks from "
                        f"{int(num_results)} search results for '{query_input.strip()}'."
                    )
                    needs_rerun = True
                else:
                    st.warning(
                        "Search returned no results. Try a more specific query."
                    )
            except Exception as error:
                logger.error("Topic search failed: %s", error, exc_info=True)
                st.error(f"Search error: {error}")

    return needs_rerun


def handle_user_input(user_input: str, store: VectorStore) -> None:
    """Process a question or task from the chat input and store the result.

    Routes to agent mode or chat (pipeline) mode depending on session state.
    Shows a progress bar while running. Stores the result so app.py can call
    st.rerun() and display it.

    Args:
        user_input: The text the user typed into the chat box.
        store:      VectorStore used for retrieval and response generation.
    """
    st.session_state.url_msg = None
    st.session_state.display.append({'role': 'user', 'content': user_input})

    progress_slot = st.empty()

    if st.session_state.mode == 'agent':
        result = _run_agent(user_input, store, progress_slot)
        answer_html = _format_agent_steps_html(result['steps'])
        content = f"{answer_html}<br/><strong>Answer:</strong> {result['answer']}"
        st.session_state.display.append({'role': 'agent', 'content': content})
        st.session_state.last = {'type': 'agent', 'data': result}
    else:
        result = _run_pipeline(user_input, store, progress_slot)
        st.session_state.display.append({'role': 'assistant', 'content': result['response']})
        st.session_state.last = {'type': 'chat', 'data': result}

    st.session_state.total += 1


# ── Private helpers ────────────────────────────────────────────────────────────


def _process_uploaded_file(uploaded_file, loader, store: VectorStore) -> None:
    """Write an uploaded file to a temp path, chunk it, and index the chunks.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        loader:        DocumentLoader — dispatches to the right chunker.
        store:         VectorStore   — stores and indexes the chunks.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    document_type  = loader.ext_to_type.get(file_extension, 'txt')

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            file_info = {
                'filepath':      tmp_path,
                'filename':      uploaded_file.name,
                'detected_type': document_type,
                'is_misplaced':  False,
            }
            new_chunks = loader._dispatch_chunker(file_info)

        if new_chunks:
            with st.spinner(f"Embedding {len(new_chunks)} chunks from {uploaded_file.name}..."):
                store.add_chunks(new_chunks, id_prefix='file')

            st.session_state.url_chunks.extend(new_chunks)
            store.rebuild_bm25(store.chunks)
            st.session_state.bm25_index = store.bm25_index

            total_chunks = store.collection.count()
            st.session_state.file_msg = (
                'ok',
                f"Indexed '{uploaded_file.name}' — {len(new_chunks)} chunks added. "
                f"Total: {total_chunks}",
            )
        else:
            st.session_state.file_msg = (
                'err',
                f"Could not extract text from '{uploaded_file.name}'.",
            )

    except Exception as error:
        logger.error("File indexing failed for '%s': %s", uploaded_file.name, error, exc_info=True)
        st.session_state.file_msg = ('err', f"Error indexing '{uploaded_file.name}': {error}")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_error:
            logger.warning("Could not delete temp file '%s': %s", tmp_path, cleanup_error)


def _run_agent(user_input: str, store: VectorStore, progress_slot) -> dict:
    """Run the agent loop and return the result dict.

    Args:
        user_input:    The task the user typed.
        store:         VectorStore the agent uses for rag_search.
        progress_slot: A st.empty() slot for showing a progress bar.

    Returns:
        Agent result dict with keys: 'answer', 'steps'.
    """
    progress_bar = progress_slot.progress(0, text="Agent starting...")
    progress_bar.progress(30, text="Agent: searching knowledge base...")
    result = Agent(store).run(user_input, streamlit_mode=True)
    progress_bar.progress(100, text="Agent: done!")
    progress_slot.empty()
    return result


def _run_pipeline(user_input: str, store: VectorStore, progress_slot) -> dict:
    """Run the RAG pipeline with tokens streamed directly into the chat UI.

    Shows a progress bar during retrieval and reranking, clears it before
    the LLM call, then streams tokens into a chat message bubble in real time
    using st.write_stream(). The response appears word by word — no waiting.

    Args:
        user_input:    The question the user typed.
        store:         VectorStore that runs the pipeline.
        progress_slot: A st.empty() slot for the progress bar.

    Returns:
        Pipeline result dict with keys: response, retrieved, reranked,
        is_confident, best_score, query_type.
    """
    # Show progress bar while retrieval and reranking complete (these take time)
    progress_bar = progress_slot.progress(0, text="Classifying query...")
    progress_bar.progress(25, text="Retrieving documents...")
    progress_bar.progress(55, text="Reranking results...")
    progress_bar.progress(90, text="Ready...")

    # Run all pipeline steps up to the LLM call
    result = store.prepare_pipeline(user_input)

    # Clear the progress bar — the streaming response is the visual feedback now
    progress_slot.empty()

    # Low-confidence path — fixed response, nothing to stream
    if 'stream' not in result:
        return result

    # Stream tokens directly into a chat message bubble as they arrive
    with st.chat_message('assistant', avatar='💬'):
        raw_response = st.write_stream(result['stream'])

    # Apply hallucination filter, update conversation history, and log
    filtered_response = store.finalize_pipeline(
        user_input, result['query_type'], result['reranked'], raw_response
    )
    result['response'] = filtered_response
    return result
