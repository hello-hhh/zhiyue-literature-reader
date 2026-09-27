"""知阅 — local personal literature reader."""

import streamlit as st

from src.ui.literature import (
    get_document_loader,
    get_library_repository,
    render_header,
    render_import_page,
    render_library,
    render_navigation,
    render_reading_workspace,
)
from src.ui.theme import CSS


st.set_page_config(page_title="知阅 · 个人文献阅读器", page_icon="📖", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

loader = get_document_loader()
repository = get_library_repository()

page = render_navigation(repository)
render_header()

if page == "文献库":
    render_library(repository)
elif page == "导入文献":
    render_import_page(loader, repository)
else:
    render_reading_workspace(repository)
