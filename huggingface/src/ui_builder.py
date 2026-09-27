"""ui_builder.py — Assembles the complete Gradio Blocks UI for the HF Space.

WHY THIS FILE EXISTS:
    build_demo() wires ~30 Gradio components together. Keeping that in
    handlers.py would push it past 500 lines and mix layout concerns with
    event-handler logic. This module owns layout and wiring only — no
    business logic lives here.

USAGE:
    from src.ui_builder import build_demo
    demo = build_demo()
    demo.launch()
"""

import gradio as gr
from src.handlers import chat, clear_added_chunks, clear_chat, fetch_url, fetch_url_recursive, search_topic, upload_file
from src.theme import CSS

from src.rag.config import URL_CRAWL_MAX_DEPTH, URL_CRAWL_MAX_PAGES

__all__ = ['build_demo']


def build_demo():
    """Assemble and return the complete Gradio Blocks app.

    All UI components, layout, and event wiring are defined here.
    app.py calls this once and launches the returned demo object.

    Returns:
        A configured gradio.Blocks instance ready to launch.
    """
    with gr.Blocks(css=CSS, title="RAG Agent — Ask Your Documents") as demo:

        gr.HTML("""
        <div class="rag-header">
            <div class="rag-title">🧠 RAG Agent</div>
            <div class="rag-sub">AI-powered document intelligence — fully local, on-device</div>
            <div class="rag-chips">
                <span class="rag-chip">Hybrid Search</span>
                <span class="rag-chip">BM25 + Dense</span>
                <span class="rag-chip">LLM Reranker</span>
                <span class="rag-chip">ReAct Agent</span>
                <span class="rag-chip">PDF · DOCX · XLSX · PPTX · CSV · MD · HTML · TXT</span>
            </div>
        </div>
        """)

        with gr.Row():

            # ── Left column: chat + ingestion ──────────────────────────────
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    type='messages',
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="",
                        show_label=False,
                        scale=5,
                    )
                    mode_radio = gr.Radio(
                        choices=["Chat", "Agent"],
                        value="Chat",
                        label="Mode",
                        info="Chat: direct RAG answer. Agent: multi-step reasoning with tools.",
                        scale=1,
                    )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=4)
                    clear_btn  = gr.Button("🗑 Clear chat", scale=1)

                gr.Markdown("---")

                with gr.Accordion("📎 Upload files", open=False):
                    gr.Markdown(
                        '<div class="note">Select one or more files. '
                        'Supported: PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, MD, HTML.</div>',
                        sanitize_html=False,
                    )
                    file_upload = gr.File(
                        label="Choose files to index",
                        file_types=[
                            ".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls",
                            ".pptx", ".ppt", ".csv", ".md", ".markdown", ".html", ".htm",
                        ],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("⬆ Index files", variant="secondary")
                    upload_msg = gr.Markdown("")

                with gr.Accordion("🌐 Add a URL", open=False):
                    gr.Markdown(
                        '<div class="note">Paste any public URL — webpage, PDF, DOCX, etc. '
                        'The content will be chunked and added to the knowledge base.</div>',
                        sanitize_html=False,
                    )
                    url_input = gr.Textbox(
                        placeholder="https://example.com/article  or  https://example.com/report.pdf",
                        label="Public URL",
                    )
                    url_btn = gr.Button("⬆ Fetch & index", variant="secondary")
                    url_msg = gr.Markdown("")

                with gr.Accordion("🔍 Search & index a topic", open=False):
                    gr.Markdown(
                        '<div class="note">Search the web for a topic and index the top results. '
                        'Uses DuckDuckGo — no API key needed.<br/>'
                        'Each result URL is crawled to the chosen depth, same as the URL crawl.</div>',
                        sanitize_html=False,
                    )
                    search_query_input = gr.Textbox(
                        placeholder="e.g. Elizabeth Olsen actress  ·  Python asyncio tutorial",
                        label="Search query",
                    )
                    with gr.Row():
                        search_num_results = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Results",
                            info="Number of search result URLs to fetch and index.",
                        )
                        search_depth = gr.Slider(
                            minimum=1, maximum=3, value=1, step=1,
                            label="Depth",
                            info="1 = result page only. 2 = also follow links on each result.",
                        )
                        search_pages_per = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Pages per result",
                            info="Max pages crawled per result URL (applies when depth > 1).",
                        )
                    search_btn = gr.Button("🔍 Search & index", variant="secondary")
                    search_msg = gr.Markdown("")

                with gr.Accordion("🕷️ Recursive web crawl", open=False):
                    gr.Markdown(
                        '<div class="note">'
                        '⚠️ <b>Topic filter is required for Wikipedia</b> — without it, depth ≥ 2 '
                        'follows every inline link and fills your page budget in seconds.<br/>'
                        'Set topic = article name, e.g. <code>Elizabeth_Olsen</code>.<br/>'
                        'On HF free CPU: keep depth=1, max 10 pages to avoid timeout.'
                        '</div>',
                        sanitize_html=False,
                    )
                    crawl_url_input = gr.Textbox(
                        placeholder="https://en.wikipedia.org/wiki/Elizabeth_Taylor",
                        label="Seed URL — crawl starts here",
                    )
                    with gr.Row():
                        crawl_depth = gr.Slider(
                            minimum=1, maximum=3, value=URL_CRAWL_MAX_DEPTH, step=1,
                            label="Depth",
                            info="1 = seed page only + its direct links. 2 = follow those links too.",
                        )
                        crawl_max_pages = gr.Slider(
                            minimum=1, maximum=50, value=URL_CRAWL_MAX_PAGES, step=1,
                            label="Max pages",
                            info="Hard cap on total pages fetched.",
                        )
                    crawl_topic_input = gr.Textbox(
                        placeholder="e.g. Elizabeth_Olsen",
                        label="Topic filter — strongly recommended",
                        info="Only follow links whose URL path contains this word. "
                             "Without it, depth ≥ 2 follows every inline link (TIFF → Toronto → Ontario…).",
                    )
                    gr.Markdown("**Index these file types:**")
                    with gr.Row():
                        crawl_html  = gr.Checkbox(value=True,  label="HTML")
                        crawl_pdf   = gr.Checkbox(value=True,  label="PDF")
                        crawl_docx  = gr.Checkbox(value=True,  label="DOCX")
                        crawl_xlsx  = gr.Checkbox(value=True,  label="XLSX")
                        crawl_csv   = gr.Checkbox(value=True,  label="CSV")
                        crawl_pptx  = gr.Checkbox(value=True,  label="PPTX")
                        crawl_md    = gr.Checkbox(value=True,  label="MD")
                    crawl_btn = gr.Button("🕷️ Start crawl", variant="secondary")
                    crawl_msg = gr.Markdown("")

            # ── Right column: status, agent tools, pipeline info ──────────
            with gr.Column(scale=1):

                startup_status    = gr.Markdown(value="⏳ Initializing...", label="")
                chunk_counter     = gr.Markdown(value="Chunks in knowledge base: **0**", label="")
                clear_chunks_btn  = gr.Button("🗑 Clear added content", variant="secondary", size="sm")
                clear_chunks_msg  = gr.Markdown("")

                gr.Markdown("### Agent Tools")
                gr.Markdown(
                    "Use **Agent** mode to activate these tools.\n\n"
                    "**🔍 rag_search** — search your documents\n\n"
                    "**🧮 calculator** — evaluate math expressions\n\n"
                    "**📝 summarise** — summarise a document section\n\n"
                    "**💬 sentiment** — analyse tone & sentiment\n\n"
                    "**🌐 translate** — translate to any language\n\n"
                    "**✅ finish** — return the final answer"
                )

                gr.Markdown("---")
                gr.Markdown("### Pipeline")
                pipeline_box = gr.Markdown(
                    value="*Query type, confidence score, and retrieved chunks appear here after each query.*",
                    label="",
                )

        # ── Event wiring ──────────────────────────────────────────────────

        def _submit(message, history, mode):
            """Handle both the Send button and pressing Enter in the text box."""
            new_history, info = chat(message, history, mode)
            return new_history, info, ""   # the "" clears the message box

        submit_btn.click(
            fn=_submit,
            inputs=[msg_box, chatbot, mode_radio],
            outputs=[chatbot, pipeline_box, msg_box],
        )
        msg_box.submit(
            fn=_submit,
            inputs=[msg_box, chatbot, mode_radio],
            outputs=[chatbot, pipeline_box, msg_box],
        )
        clear_btn.click(fn=clear_chat, outputs=[chatbot, pipeline_box])
        clear_chunks_btn.click(
            fn=clear_added_chunks,
            outputs=[clear_chunks_msg, chunk_counter],
        )

        def _upload(file_objs):
            """Thin wrapper to map upload_file outputs to Gradio components."""
            status_message, counter_text = upload_file(file_objs)
            return status_message, counter_text

        upload_btn.click(
            fn=_upload,
            inputs=[file_upload],
            outputs=[upload_msg, chunk_counter],
        )

        def _fetch(url):
            """Wrapper that also clears the URL input box after fetching."""
            status_message, counter_text = fetch_url(url)
            return status_message, counter_text, ""  # "" clears the URL box

        url_btn.click(
            fn=_fetch,
            inputs=[url_input],
            outputs=[url_msg, chunk_counter, url_input],
        )

        def _search(query, num_results, depth, pages_per):
            """Wrapper that maps Gradio inputs to search_topic and clears the query box."""
            status, counter = search_topic(query, num_results, depth, pages_per)
            return status, counter, ""   # "" clears the query box

        search_btn.click(
            fn=_search,
            inputs=[search_query_input, search_num_results, search_depth, search_pages_per],
            outputs=[search_msg, chunk_counter, search_query_input],
        )

        def _crawl(url, depth, max_pages, topic,
                   use_html, use_pdf, use_docx, use_xlsx, use_csv, use_pptx, use_md):
            """Wrapper that maps Gradio inputs to fetch_url_recursive and clears the URL box."""
            status, counter = fetch_url_recursive(
                url, depth, max_pages, topic,
                use_html, use_pdf, use_docx, use_xlsx, use_csv, use_pptx, use_md,
            )
            return status, counter, ""   # "" clears the crawl URL input

        crawl_btn.click(
            fn=_crawl,
            inputs=[
                crawl_url_input, crawl_depth, crawl_max_pages, crawl_topic_input,
                crawl_html, crawl_pdf, crawl_docx,
                crawl_xlsx, crawl_csv, crawl_pptx, crawl_md,
            ],
            outputs=[crawl_msg, chunk_counter, crawl_url_input],
        )

        def _on_load(progress=None):
            """Eagerly initialise both singletons on page load so the first query is fast."""
            import src.handlers as _h

            from src.rag.document_loader import DocumentLoader
            from src.rag.vector_store import VectorStore

            if progress is None:
                progress = gr.Progress()
            progress(0.1, desc="Starting up...")
            if _h._loader is None:
                progress(0.3, desc="Loading document processor...")
                _h._loader = DocumentLoader()
            if _h._store is None:
                progress(0.5, desc="Loading embedding model (first run may take 1-2 min)...")
                _h._store = VectorStore()
                progress(0.8, desc="Initializing vector store...")
                _h._store.build_or_load([])
            progress(1.0, desc="Ready!")
            return "✅ Ready", f"Chunks in knowledge base: **{_h._chunk_count()}**"

        demo.load(fn=_on_load, outputs=[startup_status, chunk_counter])

        # Custom footer with GitHub links — replaces the hidden default Gradio footer
        gr.HTML("""
        <div class="rag-footer">
            Fully local · no data leaves your machine<br/>
            <a href="https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent"
               target="_blank" rel="noopener noreferrer">⭐ Star on GitHub</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent/issues"
               target="_blank" rel="noopener noreferrer">Report an issue ↗</a>
        </div>
        """)

    return demo
