"""SQLite-backed personal literature library.

The vector database answers questions; this repository owns the reader-facing
metadata that a vector store should not: reading state, tags, notes, progress,
favourites, original URL, and the locally stored source file.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


READING_STATUSES = ("未读", "阅读中", "已读")


class LibraryRepository:
    """Persist documents and personal reading data in a local SQLite file."""

    def __init__(
        self,
        db_path: str | Path = "data/literature_library.db",
        storage_root: str | Path = "docs",
    ) -> None:
        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    rag_source TEXT NOT NULL DEFAULT '',
                    stored_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    tags TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '未读',
                    progress INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    favorite INTEGER NOT NULL DEFAULT 0,
                    preview_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "rag_source" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN rag_source TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_updated "
                "ON documents(updated_at DESC)"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _safe_stem(value: str) -> str:
        stem = Path(value).stem.strip() or "document"
        stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", stem, flags=re.UNICODE)
        return stem.strip("-")[:60] or "document"

    @staticmethod
    def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def add_file(
        self,
        *,
        filename: str,
        data: bytes,
        source_type: str,
        title: str = "",
        author: str = "",
        tags: str = "",
        preview_text: str = "",
        rag_source: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Store an uploaded file and return ``(document, created)``."""
        digest = hashlib.sha256(data).hexdigest()
        existing = self.find_by_hash(digest)
        if existing:
            return existing, False

        document_id = uuid.uuid4().hex
        extension = Path(filename).suffix.lower()
        type_folder = {
            "pdf": "pdfs",
            "docx": "docx",
            "html": "html",
            "txt": "txts",
        }.get(source_type, "txts")
        stored_path = self.storage_root / type_folder / (
            f"{self._safe_stem(filename)}-{document_id[:8]}{extension}"
        )
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(data)

        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, title, author, source_type, source_ref, rag_source, stored_path,
                    content_hash, tags, preview_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    title.strip() or Path(filename).stem,
                    author.strip(),
                    source_type,
                    filename,
                    rag_source or filename,
                    str(stored_path),
                    digest,
                    tags.strip(),
                    preview_text,
                    now,
                    now,
                ),
            )
        return self.get(document_id), True

    def add_url(
        self,
        *,
        url: str,
        preview_text: str,
        title: str = "",
        author: str = "",
        tags: str = "",
        rag_source: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Save a parsed webpage as a local HTML reading snapshot."""
        normalized_url = url.strip()
        digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
        existing = self.find_by_hash(digest)
        if existing:
            return existing, False

        parsed = urlparse(normalized_url)
        fallback_title = parsed.netloc + (parsed.path.rstrip("/").split("/")[-1] or "")
        document_id = uuid.uuid4().hex
        filename = f"{self._safe_stem(title or fallback_title)}-{document_id[:8]}.html"
        stored_path = self.storage_root / "html" / filename
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        escaped = (
            preview_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>\n")
        )
        snapshot = (
            "<!doctype html><html><head><meta charset='utf-8'></head>"
            f"<body><article>{escaped}</article></body></html>"
        )
        stored_path.write_text(snapshot, encoding="utf-8")

        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, title, author, source_type, source_ref, rag_source, stored_path,
                    content_hash, tags, preview_text, created_at, updated_at
                ) VALUES (?, ?, ?, 'url', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    title.strip() or fallback_title or normalized_url,
                    author.strip(),
                    normalized_url,
                    rag_source,
                    str(stored_path),
                    digest,
                    tags.strip(),
                    preview_text,
                    now,
                    now,
                ),
            )
        return self.get(document_id), True

    def find_by_hash(self, digest: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (digest,)
            ).fetchone()
        return self._as_dict(row)

    def get(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._as_dict(row)

    def list_documents(
        self,
        *,
        search: str = "",
        status: str = "全部",
        favorites_only: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if search.strip():
            needle = f"%{search.strip()}%"
            clauses.append("(title LIKE ? OR author LIKE ? OR tags LIKE ?)")
            parameters.extend([needle, needle, needle])
        if status != "全部":
            clauses.append("status = ?")
            parameters.append(status)
        if favorites_only:
            clauses.append("favorite = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM documents {where} ORDER BY favorite DESC, updated_at DESC",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def update_reading(
        self,
        document_id: str,
        *,
        status: str,
        progress: int,
        notes: str,
        tags: str,
        favorite: bool,
    ) -> dict[str, Any] | None:
        if status not in READING_STATUSES:
            raise ValueError(f"Unsupported reading status: {status}")
        progress = max(0, min(100, int(progress)))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                   SET status = ?, progress = ?, notes = ?, tags = ?,
                       favorite = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    status,
                    progress,
                    notes.strip(),
                    tags.strip(),
                    int(favorite),
                    self._now(),
                    document_id,
                ),
            )
        return self.get(document_id)

    def delete(self, document_id: str, *, remove_file: bool = True) -> bool:
        document = self.get(document_id)
        if not document:
            return False
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        if remove_file:
            path = Path(document["stored_path"])
            try:
                path.resolve().relative_to(self.storage_root.resolve())
                path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
        return True
