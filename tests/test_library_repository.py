"""Tests for persistent personal-library metadata and reading state."""

from pathlib import Path

from src.library.repository import LibraryRepository


def _repository(tmp_path: Path) -> LibraryRepository:
    return LibraryRepository(tmp_path / "library.db", tmp_path / "docs")


def test_add_file_persists_metadata_and_source(tmp_path):
    repository = _repository(tmp_path)
    document, created = repository.add_file(
        filename="paper.pdf",
        data=b"paper bytes",
        source_type="pdf",
        title="课程论文",
        author="张同学",
        tags="课程,人工智能",
        rag_source="paper.pdf",
    )

    assert created is True
    assert document["title"] == "课程论文"
    assert document["rag_source"] == "paper.pdf"
    assert Path(document["stored_path"]).exists()


def test_duplicate_file_is_not_created_twice(tmp_path):
    repository = _repository(tmp_path)
    first, _ = repository.add_file(
        filename="paper.pdf", data=b"same", source_type="pdf"
    )
    second, created = repository.add_file(
        filename="renamed.pdf", data=b"same", source_type="pdf"
    )

    assert created is False
    assert second["id"] == first["id"]
    assert len(repository.list_documents()) == 1


def test_update_reading_and_filter_library(tmp_path):
    repository = _repository(tmp_path)
    document, _ = repository.add_file(
        filename="paper.pdf", data=b"paper", source_type="pdf", title="检索研究"
    )
    repository.update_reading(
        document["id"],
        status="阅读中",
        progress=42,
        notes="继续阅读第三节",
        tags="检索,RAG",
        favorite=True,
    )

    filtered = repository.list_documents(
        search="RAG", status="阅读中", favorites_only=True
    )
    assert len(filtered) == 1
    assert filtered[0]["progress"] == 42
    assert filtered[0]["notes"] == "继续阅读第三节"


def test_add_url_creates_local_snapshot(tmp_path):
    repository = _repository(tmp_path)
    document, created = repository.add_url(
        url="https://example.com/article",
        title="网页文章",
        preview_text="第一段\n\n第二段",
        rag_source="example.com/article",
    )

    assert created is True
    snapshot = Path(document["stored_path"]).read_text(encoding="utf-8")
    assert "第一段" in snapshot
    assert document["source_type"] == "url"


def test_delete_removes_metadata_and_local_file(tmp_path):
    repository = _repository(tmp_path)
    document, _ = repository.add_file(
        filename="paper.pdf", data=b"paper", source_type="pdf"
    )
    stored_path = Path(document["stored_path"])

    assert repository.delete(document["id"]) is True
    assert repository.get(document["id"]) is None
    assert not stored_path.exists()
