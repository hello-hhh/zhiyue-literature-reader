"""Unit tests for src/rag/logger.py."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tmp_log():
    """Return a path to a temp file that does NOT yet exist."""
    f = tempfile.NamedTemporaryFile(suffix='.json', delete=True)
    path = f.name
    f.close()
    return path


# ---------------------------------------------------------------------------
# _read_log
# ---------------------------------------------------------------------------

class TestReadLog:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        path = str(tmp_path / 'missing.json')
        with patch('src.rag.logger.LOG_FILE', path):
            from src.rag.logger import _read_log
            assert _read_log() == []

    def test_returns_empty_list_on_corrupt_json(self, tmp_path):
        path = tmp_path / 'bad.json'
        path.write_text('NOT JSON')
        with patch('src.rag.logger.LOG_FILE', str(path)):
            from src.rag.logger import _read_log
            assert _read_log() == []

    def test_returns_list_when_valid(self, tmp_path):
        path = tmp_path / 'log.json'
        path.write_text(json.dumps([{'a': 1}]))
        with patch('src.rag.logger.LOG_FILE', str(path)):
            from src.rag.logger import _read_log
            assert _read_log() == [{'a': 1}]


# ---------------------------------------------------------------------------
# _write_log
# ---------------------------------------------------------------------------

class TestWriteLog:
    def test_writes_indented_json(self, tmp_path):
        path = tmp_path / 'log.json'
        with patch('src.rag.logger.LOG_FILE', str(path)):
            from src.rag.logger import _write_log
            _write_log([{'x': 42}])
        content = path.read_text()
        assert json.loads(content) == [{'x': 42}]
        assert '\n' in content  # indented

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / 'log.json'
        path.write_text(json.dumps([{'old': True}]))
        with patch('src.rag.logger.LOG_FILE', str(path)):
            from src.rag.logger import _write_log
            _write_log([{'new': True}])
        assert json.loads(path.read_text()) == [{'new': True}]


# ---------------------------------------------------------------------------
# log_interaction
# ---------------------------------------------------------------------------

class TestLogInteraction:
    def test_creates_file_if_missing(self, tmp_path):
        path = str(tmp_path / 'log.json')
        with patch('src.rag.logger.LOG_FILE', path):
            from src.rag.logger import log_interaction
            log_interaction('q', 'factual', 3, [0.8, 0.7], 'answer')
        assert os.path.exists(path)

    def test_entry_has_all_fields(self, tmp_path):
        path = str(tmp_path / 'log.json')
        with patch('src.rag.logger.LOG_FILE', path):
            from src.rag.logger import log_interaction
            log_interaction('hello', 'factual', 2, [0.9, 0.8], 'world')
        with open(path) as f:
            entries = json.loads(f.read())
        assert len(entries) == 1
        e = entries[0]
        assert e['query'] == 'hello'
        assert e['query_type'] == 'factual'
        assert e['chunks_used'] == 2
        assert e['top_similarity'] == 0.9
        assert e['avg_similarity'] == round((0.9 + 0.8) / 2, 4)
        assert e['response_length'] == len('world')
        assert 'timestamp' in e

    def test_top_similarity_zero_when_empty_scores(self, tmp_path):
        path = str(tmp_path / 'log.json')
        with patch('src.rag.logger.LOG_FILE', path):
            from src.rag.logger import log_interaction
            log_interaction('q', 'general', 0, [], 'resp')
        with open(path) as f:
            e = json.loads(f.read())[0]
        assert e['top_similarity'] == 0
        assert e['avg_similarity'] == 0

    def test_appends_multiple_entries(self, tmp_path):
        path = str(tmp_path / 'log.json')
        with patch('src.rag.logger.LOG_FILE', path):
            from src.rag.logger import log_interaction
            log_interaction('q1', 'factual', 1, [0.9], 'a1')
            log_interaction('q2', 'general', 2, [0.7], 'a2')
        with open(path) as f:
            entries = json.loads(f.read())
        assert len(entries) == 2
        assert entries[0]['query'] == 'q1'
        assert entries[1]['query'] == 'q2'

    def test_scores_rounded_to_4_places(self, tmp_path):
        path = str(tmp_path / 'log.json')
        with patch('src.rag.logger.LOG_FILE', path):
            from src.rag.logger import log_interaction
            log_interaction('q', 'factual', 1, [0.123456789], 'a')
        with open(path) as f:
            e = json.loads(f.read())[0]
        assert e['top_similarity'] == 0.1235
