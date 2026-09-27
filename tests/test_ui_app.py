"""AppTest coverage for the redesigned Chinese literature-reader shell."""

from pathlib import Path
from unittest.mock import MagicMock, patch


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_app():
    from streamlit.testing.v1 import AppTest

    mock_repository = MagicMock()
    mock_repository.list_documents.return_value = []

    with patch("src.ui.literature.get_document_loader", return_value=MagicMock()), \
         patch("src.ui.literature.get_library_repository", return_value=mock_repository):
        app = AppTest.from_file(APP_PATH, default_timeout=30)
        app.run()
    return app


def test_app_loads_without_exception():
    app = _run_app()
    assert not app.exception


def test_app_renders_minimal_reader_header():
    app = _run_app()
    markdown = " ".join(item.value for item in app.markdown)
    assert "reader-title" in markdown
    assert "知阅" in markdown
    assert "PDF · Word · 网页" in markdown


def test_sidebar_navigation_has_three_requested_sections():
    app = _run_app()
    assert app.radio
    assert list(app.radio[0].options) == ["文献库", "阅读空间", "导入文献"]
    assert app.radio[0].value == "文献库"


def test_empty_library_has_clear_import_guidance():
    app = _run_app()
    markdown = " ".join(item.value for item in app.markdown)
    assert "文献库还是空的" in markdown
    assert "导入文献" in markdown


def test_navigation_opens_unified_import_page():
    app = _run_app()
    app.radio[0].set_value("导入文献").run()
    assert len(app.tabs) == 2
    assert [tab.label for tab in app.tabs] == ["上传 PDF / Word", "解析网页链接"]
    assert any("网页链接" in item.label for item in app.text_input)


def test_css_contains_reader_palette():
    app = _run_app()
    css = " ".join(item.value for item in app.markdown)
    assert "--reader-accent" in css
    assert "--reader-canvas" in css
    assert '[data-testid="stToolbar"]' in css
    assert "拖放 PDF 或 Word 文件到这里" in css


def test_reader_app_does_not_load_ai_or_vector_search():
    app_source = APP_PATH.read_text(encoding="utf-8")
    literature_source = (
        APP_PATH.parent / "src" / "ui" / "literature.py"
    ).read_text(encoding="utf-8")

    assert "src.ui.handlers" not in app_source
    assert "src.ui.session" not in app_source
    assert "VectorStore" not in literature_source
    assert "_run_pipeline" not in literature_source
    assert "AI 阅读助手" not in literature_source


def test_reader_highlights_can_undo_once_or_clear_all():
    from src.ui.literature import _render_text_document

    document = {"id": "law-paper", "preview_text": "第一处内容\n第二处内容"}
    with patch("src.ui.literature.components.html") as render_html:
        _render_text_document(document, "")

    component = render_html.call_args.args[0]
    assert "撤销上一次" in component
    assert "全部清除" in component
    assert "data-highlight-id" not in component
    assert "mark.dataset.highlightId = highlightId" in component
    assert 'getElementById("undo-highlight")' in component
