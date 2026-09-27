"""ui/ingestion.py — URL ingestion processing functions.

WHY THIS FILE EXISTS:
    handlers.py handles all Streamlit rendering and event routing.
    The URL ingestion functions (_process_url, _process_url_recursive)
    are extracted here to keep handlers.py under the 500-line limit.
    They are private implementation details of the URL ingestion flow.

WHEN TO USE:
    Import process_url and process_url_recursive into handlers.py only.
    Do not call these functions from app.py or any other file.
"""

import logging
from typing import Optional

import streamlit as st

from src.rag.vector_store import VectorStore

# Module-level logger — errors go to the logging system, not the terminal
logger = logging.getLogger(__name__)

__all__ = ['process_url', 'process_url_recursive']


def process_url(url: str, loader, store: VectorStore) -> None:
    """Fetch a single URL, chunk it, and add the chunks to the knowledge base.

    Updates st.session_state.url_msg with ('ok', message) on success or
    ('err', message) on failure so the caller can display the right banner.

    Args:
        url:    The public URL to fetch.
        loader: DocumentLoader — does the fetching and chunking.
        store:  VectorStore   — stores the chunks and rebuilds BM25.
    """
    try:
        with st.spinner(f"Fetching {url}..."):
            new_chunks = loader.chunk_url(url)

        if new_chunks:
            with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                store.add_chunks(new_chunks, id_prefix='url')

            # BM25 must be rebuilt so the new chunks are included in keyword search
            st.session_state.url_chunks.extend(new_chunks)
            store.rebuild_bm25(store.chunks)
            st.session_state.bm25_index = store.bm25_index

            total_chunks = store.collection.count()
            st.session_state.url_msg = (
                'ok',
                f"Added {len(new_chunks)} chunks. Total in knowledge base: {total_chunks}",
            )
        else:
            st.session_state.url_msg = (
                'err',
                "Could not fetch or parse the URL. Check it is publicly accessible.",
            )

    except Exception as error:
        logger.error("URL ingestion failed for '%s': %s", url, error, exc_info=True)
        st.session_state.url_msg = ('err', f"Error fetching URL: {error}")


def process_url_recursive(
    url: str,
    loader,
    store: VectorStore,
    depth: int,
    max_pages: int,
    allowed_types: Optional[set],
    topic_filter: str = '',
) -> None:
    """Crawl a seed URL recursively and index all discovered pages.

    Shows a live progress log using st.status() so the user can watch
    each page being discovered in real time. Updates st.session_state.url_msg
    with the final result after the crawl completes.

    Args:
        url:           The seed URL to start crawling from.
        loader:        DocumentLoader — does the crawling and chunking.
        store:         VectorStore   — stores the chunks and rebuilds BM25.
        depth:         How many link-levels deep to follow.
        max_pages:     Maximum total pages to fetch.
        allowed_types: Set of type strings to index, or None for all types.
        topic_filter:  Optional keyword the URL path must contain to be crawled.
                       Empty string means no filter — crawl everything on the domain.
    """
    try:
        # st.status() shows a live collapsible log during the crawl
        with st.status(f"Crawling {url}...", expanded=True) as status_box:
            crawl_log: list = []

            def progress_callback(page_url: str, dtype: str, chunk_count: int) -> None:
                """Called after each page is fetched — updates the live crawl log."""
                short_url = page_url[:70] + '...' if len(page_url) > 70 else page_url
                crawl_log.append(f"[{dtype.upper()}] {short_url} — {chunk_count} chunks")
                # Show the last 8 lines so the log doesn't grow too long on screen
                status_box.write('\n'.join(crawl_log[-8:]))

            new_chunks = loader.chunk_url_recursive(
                url,
                depth=depth,
                max_pages=max_pages,
                allowed_types=allowed_types,
                topic_filter=topic_filter,
                progress_callback=progress_callback,
            )

            status_box.update(
                label=f"Crawl complete — {len(crawl_log)} pages, {len(new_chunks)} chunks",
                state="complete",
                expanded=False,
            )

        if new_chunks:
            with st.spinner(f"Embedding {len(new_chunks)} chunks..."):
                store.add_chunks(new_chunks, id_prefix='url')

            # Rebuild BM25 once after all crawled chunks are added
            st.session_state.url_chunks.extend(new_chunks)
            store.rebuild_bm25(store.chunks)
            st.session_state.bm25_index = store.bm25_index

            total_chunks = store.collection.count()
            st.session_state.url_msg = (
                'ok',
                f"Crawled {len(crawl_log)} pages — added {len(new_chunks)} chunks. "
                f"Total in knowledge base: {total_chunks}",
            )
        else:
            st.session_state.url_msg = (
                'err',
                "No content extracted from the crawl. Check the URL is publicly accessible.",
            )

    except Exception as error:
        logger.error("Recursive crawl failed for '%s': %s", url, error, exc_info=True)
        st.session_state.url_msg = ('err', f"Error during crawl: {error}")
