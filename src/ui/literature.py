"""Chinese personal literature-reader interface."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.library.preview import (
    extract_pdf_page_text,
    get_pdf_page_count,
    load_preview_text,
    text_to_html,
)
from src.library.repository import LibraryRepository, READING_STATUSES
from src.rag.document_loader import DocumentLoader


def get_library_repository() -> LibraryRepository:
    return LibraryRepository()


@st.cache_resource
def get_document_loader() -> DocumentLoader:
    """Create the parser used by the file and web import screens."""
    loader = DocumentLoader()
    loader.ensure_folders()
    return loader


def render_navigation(repository: LibraryRepository) -> str:
    documents = repository.list_documents()
    st.sidebar.markdown("<div class='brand'>知阅</div>", unsafe_allow_html=True)
    st.sidebar.caption("个人文献阅读器")
    page = st.sidebar.radio(
        "导航",
        ["文献库", "阅读空间", "导入文献"],
        key="nav_page",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    reading = sum(doc["status"] == "阅读中" for doc in documents)
    finished = sum(doc["status"] == "已读" for doc in documents)
    col_a, col_b = st.sidebar.columns(2)
    col_a.metric("文献", len(documents))
    col_b.metric("已读", finished)
    if reading:
        st.sidebar.caption(f"正在阅读 {reading} 篇")
    st.sidebar.markdown(
        "<div class='source-note'>基于开源 RAG Agent 二次开发</div>",
        unsafe_allow_html=True,
    )
    return page


def render_header() -> None:
    st.markdown(
        """
        <div class="reader-header">
          <div>
            <div class="reader-title">知阅</div>
            <div class="reader-subtitle">整理、阅读并理解你的个人文献</div>
          </div>
          <div class="reader-formats">PDF · Word · 网页</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _set_selected(document_id: str) -> None:
    st.session_state.selected_document_id = document_id
    st.session_state.preview_page = 1
    st.session_state.preview_excerpt = ""
    st.session_state.nav_page = "阅读空间"


def render_library(repository: LibraryRepository) -> None:
    st.subheader("我的文献库")
    st.caption("用阅读状态、标签和收藏整理自己的资料。")
    filter_a, filter_b, filter_c = st.columns([2, 1, 1])
    with filter_a:
        query = st.text_input(
            "搜索", placeholder="搜索标题、作者或标签", label_visibility="collapsed"
        )
    with filter_b:
        status = st.selectbox(
            "状态", ["全部", *READING_STATUSES], label_visibility="collapsed"
        )
    with filter_c:
        favorites_only = st.toggle("只看收藏", value=False)

    documents = repository.list_documents(
        search=query, status=status, favorites_only=favorites_only
    )
    if not documents:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-title">文献库还是空的</div>
              <div>从左侧进入“导入文献”，添加 PDF、Word 或网页链接。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for document in documents:
        with st.container(border=True):
            info, progress_col, action = st.columns([5, 2, 1.4], vertical_alignment="center")
            with info:
                favorite = "★ " if document["favorite"] else ""
                st.markdown(f"#### {favorite}{html.escape(document['title'])}")
                type_labels = {"pdf": "PDF", "docx": "Word", "url": "网页"}
                metadata = [type_labels.get(document["source_type"], "文献"), document["status"]]
                if document["author"]:
                    metadata.append(document["author"])
                if document["tags"]:
                    metadata.append(document["tags"])
                st.caption(" · ".join(metadata))
            with progress_col:
                st.progress(document["progress"] / 100, text=f"阅读 {document['progress']}%")
            with action:
                st.button(
                    "打开阅读",
                    key=f"open_{document['id']}",
                    use_container_width=True,
                    on_click=_set_selected,
                    args=(document["id"],),
                )


def _import_uploaded_file(
    uploaded_file,
    *,
    loader,
    repository: LibraryRepository,
    title: str,
    author: str,
    tags: str,
) -> tuple[bool, str]:
    data = uploaded_file.getvalue()
    extension = Path(uploaded_file.name).suffix.lower()
    source_type = loader.ext_to_type.get(extension)
    if source_type not in {"pdf", "docx"}:
        return False, "目前导入入口仅接受 PDF 和 DOCX。"

    document, created = repository.add_file(
        filename=uploaded_file.name,
        data=data,
        source_type=source_type,
        title=title,
        author=author,
        tags=tags,
        rag_source=uploaded_file.name,
    )
    if not created:
        st.session_state.selected_document_id = document["id"]
        return False, "这份文献已经在文献库中。"

    file_info = {
        "filepath": document["stored_path"],
        "filename": uploaded_file.name,
        "detected_type": source_type,
        "is_misplaced": False,
    }
    try:
        chunks = loader._dispatch_chunker(file_info)
        if not chunks:
            repository.delete(document["id"])
            return False, "没有从文件中提取到可阅读文字。"
        st.session_state.selected_document_id = document["id"]
        return True, f"已导入《{document['title']}》，共解析 {len(chunks)} 个正文片段。"
    except Exception:
        repository.delete(document["id"])
        raise


def _import_url(
    url: str,
    *,
    loader,
    repository: LibraryRepository,
    title: str,
    author: str,
    tags: str,
) -> tuple[bool, str]:
    chunks = loader.chunk_url(url)
    if not chunks:
        return False, "没有解析到网页正文，请确认链接公开可访问。"
    seen: set[str] = set()
    paragraphs: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            paragraphs.append(text)
    preview_text = "\n\n".join(paragraphs)
    rag_source = str(chunks[0].get("source", url))
    document, created = repository.add_url(
        url=url,
        preview_text=preview_text,
        title=title,
        author=author,
        tags=tags,
        rag_source=rag_source,
    )
    if not created:
        st.session_state.selected_document_id = document["id"]
        return False, "这个网页已经在文献库中。"
    st.session_state.selected_document_id = document["id"]
    return True, f"已保存网页《{document['title']}》，共解析 {len(chunks)} 个正文片段。"


def render_import_page(loader, repository: LibraryRepository) -> None:
    st.subheader("导入文献")
    st.caption("文件只保存在本机；网页会保存一份清洗后的正文快照。")
    file_tab, url_tab = st.tabs(["上传 PDF / Word", "解析网页链接"])

    with file_tab:
        uploaded = st.file_uploader(
            "选择文件", type=["pdf", "docx"], accept_multiple_files=False
        )
        col_title, col_author = st.columns(2)
        title = col_title.text_input("标题（可选）", key="file_title")
        author = col_author.text_input("作者（可选）", key="file_author")
        tags = st.text_input("标签（用逗号分隔）", key="file_tags")
        if st.button("导入文献", type="primary", disabled=uploaded is None):
            try:
                with st.spinner("正在解析并保存文献…"):
                    ok, message = _import_uploaded_file(
                        uploaded, loader=loader, repository=repository,
                        title=title, author=author, tags=tags,
                    )
                st.success(message) if ok else st.warning(message)
            except Exception as error:
                st.error(f"导入失败：{error}")

    with url_tab:
        url = st.text_input("网页链接", placeholder="https://example.com/article")
        col_title, col_author = st.columns(2)
        title = col_title.text_input("标题（可选）", key="url_title")
        author = col_author.text_input("作者（可选）", key="url_author")
        tags = st.text_input("标签（用逗号分隔）", key="url_tags")
        if st.button("解析并保存网页", type="primary", disabled=not url.strip()):
            try:
                with st.spinner("正在抓取并保存网页正文…"):
                    ok, message = _import_url(
                        url.strip(), loader=loader, repository=repository,
                        title=title, author=author, tags=tags,
                    )
                st.success(message) if ok else st.warning(message)
            except Exception as error:
                st.error(f"网页解析失败：{error}")


def _render_pdf(path: Path, page: int) -> None:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    components.html(
        f"""
        <iframe title="PDF preview"
          src="data:application/pdf;base64,{encoded}#page={max(1, page)}&view=FitH"
          style="width:100%;height:720px;border:0;border-radius:8px;background:#f4f4f2">
        </iframe>
        """,
        height=730,
        scrolling=False,
    )


def _render_text_document(
    document: dict,
    excerpt: str,
    *,
    text: str | None = None,
    storage_suffix: str = "全文",
) -> None:
    body = text_to_html(text if text is not None else load_preview_text(document), excerpt)
    storage_key = f"zhiyue-highlights-{document['id']}-{storage_suffix}"
    components.html(
        f"""
        <style>
          body {{ margin:0; color:#252525; background:#fafaf8; font:16px/1.8 system-ui,sans-serif; }}
          .reader-tools {{ position:sticky; top:0; z-index:2; display:flex; align-items:center;
            justify-content:space-between; gap:12px; padding:9px 14px; color:#59636d;
            background:#f7f7f4; border:1px solid #e6e6e1; border-bottom:0;
            border-radius:8px 8px 0 0; font-size:13px; }}
          .highlight-actions {{ display:flex; gap:7px; flex-shrink:0; }}
          button {{ border:1px solid #ccd2d7; border-radius:5px; padding:5px 10px;
            background:white; color:#315b78; cursor:pointer; font:inherit; }}
          button:hover {{ background:#eef3f6; }}
          article {{ padding:28px 34px; border:1px solid #e6e6e1; border-radius:0 0 8px 8px;
            background:white; min-height:610px; }}
          p {{ margin:0 0 1em; }}
          mark {{ background:#fff0a8; padding:2px 0; }}
          mark.user-highlight {{ background:#ffe082; border-radius:2px; }}
          ::selection {{ background:#ffd45c; color:#171717; }}
        </style>
        <div class="reader-tools">
          <span>划选文字即可高亮，高亮仅保存在当前浏览器</span>
          <div class="highlight-actions">
            <button id="undo-highlight" type="button">撤销上一次</button>
            <button id="clear-highlights" type="button">全部清除</button>
          </div>
        </div>
        <article id="reader-article">{body}</article>
        <script>
          const article = document.getElementById("reader-article");
          const storageKey = {storage_key!r};
          const saved = localStorage.getItem(storageKey);
          if (saved) article.innerHTML = saved;

          function saveHighlights() {{
            localStorage.setItem(storageKey, article.innerHTML);
          }}

          article.addEventListener("mouseup", () => {{
            const selection = window.getSelection();
            if (!selection || selection.isCollapsed || !selection.rangeCount) return;
            const range = selection.getRangeAt(0);
            if (!article.contains(range.commonAncestorContainer)) return;
            const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT);
            const selectedNodes = [];
            const highlightId = `${{Date.now()}}-${{Math.random().toString(36).slice(2)}}`;
            let node;
            while ((node = walker.nextNode())) {{
              if (range.intersectsNode(node) && node.textContent.length) selectedNodes.push(node);
            }}
            selectedNodes.reverse().forEach((textNode) => {{
              if (textNode.parentElement.closest("mark.user-highlight")) return;
              const start = textNode === range.startContainer ? range.startOffset : 0;
              const end = textNode === range.endContainer ? range.endOffset : textNode.length;
              if (start >= end) return;
              const selectedPart = textNode.splitText(start);
              selectedPart.splitText(end - start);
              const mark = document.createElement("mark");
              mark.className = "user-highlight";
              mark.dataset.highlightId = highlightId;
              selectedPart.replaceWith(mark);
              mark.appendChild(selectedPart);
            }});
            selection.removeAllRanges();
            saveHighlights();
          }});

          function unwrap(mark) {{
            mark.replaceWith(document.createTextNode(mark.textContent));
          }}

          document.getElementById("undo-highlight").addEventListener("click", () => {{
            const marks = [...article.querySelectorAll("mark.user-highlight")];
            if (!marks.length) return;
            const last = marks[marks.length - 1];
            const highlightId = last.dataset.highlightId;
            const targets = highlightId
              ? marks.filter((mark) => mark.dataset.highlightId === highlightId)
              : [last];
            targets.forEach(unwrap);
            article.normalize();
            if (article.querySelector("mark.user-highlight")) saveHighlights();
            else localStorage.removeItem(storageKey);
          }});

          document.getElementById("clear-highlights").addEventListener("click", () => {{
            localStorage.removeItem(storageKey);
            article.querySelectorAll("mark.user-highlight").forEach(unwrap);
            article.normalize();
          }});
        </script>
        """,
        height=720,
        scrolling=True,
    )


def _change_pdf_page(delta: int, page_count: int) -> None:
    current = int(st.session_state.get("preview_page", 1))
    st.session_state.preview_page = min(max(current + delta, 1), page_count)


def _render_highlightable_pdf(document: dict, path: Path, excerpt: str) -> None:
    page_count = get_pdf_page_count(path)
    current_page = min(max(int(st.session_state.get("preview_page", 1)), 1), page_count)
    st.session_state.preview_page = current_page

    previous, page_label, following = st.columns([1, 2, 1], vertical_alignment="center")
    previous.button(
        "上一页",
        disabled=current_page <= 1,
        use_container_width=True,
        on_click=_change_pdf_page,
        args=(-1, page_count),
    )
    page_label.markdown(
        f"<div class='pdf-page-label'>第 {current_page} / {page_count} 页</div>",
        unsafe_allow_html=True,
    )
    following.button(
        "下一页",
        disabled=current_page >= page_count,
        use_container_width=True,
        on_click=_change_pdf_page,
        args=(1, page_count),
    )
    page_text = extract_pdf_page_text(path, current_page)
    if not page_text:
        st.warning("这一页没有可提取的文字，可能是扫描图片。请切换到 PDF 原版查看。")
        return
    _render_text_document(
        document,
        excerpt,
        text=page_text,
        storage_suffix=f"第{current_page}页",
    )


def _save_reading_record(repository: LibraryRepository, document_id: str) -> None:
    status = st.session_state[f"record_status_{document_id}"]
    progress = st.session_state[f"record_progress_{document_id}"]
    if progress == 100:
        status = "已读"
    elif progress > 0 and status == "未读":
        status = "阅读中"
    repository.update_reading(
        document_id,
        status=status,
        progress=progress,
        notes=st.session_state[f"record_notes_{document_id}"],
        tags=st.session_state[f"record_tags_{document_id}"],
        favorite=st.session_state[f"record_favorite_{document_id}"],
    )


def _delete_document(repository: LibraryRepository, document: dict) -> None:
    repository.delete(document["id"])
    st.session_state.selected_document_id = None
    st.session_state.preview_excerpt = ""


def render_reading_workspace(repository: LibraryRepository) -> None:
    documents = repository.list_documents()
    if not documents:
        st.info("请先进入“导入文献”，添加一份 PDF、Word 或网页资料。")
        return

    document_ids = [doc["id"] for doc in documents]
    selected_id = st.session_state.get("selected_document_id")
    if selected_id not in document_ids:
        selected_id = document_ids[0]
        st.session_state.selected_document_id = selected_id
    selected_id = st.selectbox(
        "当前文献",
        document_ids,
        index=document_ids.index(selected_id),
        format_func=lambda doc_id: next(doc["title"] for doc in documents if doc["id"] == doc_id),
    )
    st.session_state.selected_document_id = selected_id
    document = repository.get(selected_id)

    st.markdown(f"### {html.escape(document['title'])}")
    st.caption(document["author"] or document["source_ref"])
    path = Path(document["stored_path"])
    if document["source_type"] == "pdf" and path.exists():
        view_mode = st.radio(
            "阅读模式",
            ["文本阅读（支持划选高亮）", "PDF 原版"],
            horizontal=True,
            key=f"pdf_view_mode_{document['id']}",
            label_visibility="collapsed",
        )
        if view_mode == "PDF 原版":
            _render_pdf(path, st.session_state.get("preview_page", 1))
        else:
            _render_highlightable_pdf(
                document, path, st.session_state.get("preview_excerpt", "")
            )
    else:
        _render_text_document(document, st.session_state.get("preview_excerpt", ""))

    with st.expander("阅读记录与个人笔记", expanded=False):
        record_a, record_b = st.columns([1, 2])
        with record_a:
            st.selectbox(
                "阅读状态", READING_STATUSES,
                index=READING_STATUSES.index(document["status"]),
                key=f"record_status_{document['id']}",
            )
            st.slider(
                "阅读进度", 0, 100, int(document["progress"]), format="%d%%",
                key=f"record_progress_{document['id']}",
            )
            st.checkbox(
                "收藏这篇文献", value=bool(document["favorite"]),
                key=f"record_favorite_{document['id']}",
            )
            st.text_input("标签", value=document["tags"], key=f"record_tags_{document['id']}")
        with record_b:
            st.text_area(
                "个人笔记", value=document["notes"], height=180,
                placeholder="记录观点、疑问或待办事项…",
                key=f"record_notes_{document['id']}",
            )
        save_col, delete_col = st.columns([4, 1])
        save_col.button(
            "保存阅读记录", type="primary", on_click=_save_reading_record,
            args=(repository, document["id"]),
        )
        with delete_col.popover("删除"):
            confirmed = st.checkbox("同时从本机文献库移除原文件")
            st.button(
                "确认删除", disabled=not confirmed, on_click=_delete_document,
                args=(repository, document), use_container_width=True,
            )
