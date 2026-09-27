"""Unit tests for the Benchmarker class.

Covers:
    DEFAULT_TEST_CASES      — backward-compatibility checks (5 cat facts, required keys)
    __init__                — store and file path binding
    _compute_summary        — mean/std/min/max computation
    _compare_runs           — delta indicators (▲/▼/─)
    _export_csv             — CSV file creation and column structure
    _read_results / _save_results — JSON persistence round-trip
    run()                   — full pipeline with mocked ollama and VectorStore
"""

import csv
import json
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Provide a MagicMock VectorStore pre-configured with canned retrieve and rerank returns.

    The entry text contains 'sleep' and '16' so keyword-based metrics
    (precision_at_k, mrr) return non-zero scores in run() tests.
    """
    from src.rag.vector_store import VectorStore
    store = MagicMock(spec=VectorStore)
    entry = {
        'text':       'Cats sleep 16 hours a day.',
        'source':     'cats.txt',
        'start_line': 1,
        'end_line':   1,
        'type':       'txt',
    }
    store._expand_query.return_value   = ['cats sleep']
    store._hybrid_retrieve.return_value = [(entry, 0.9)]
    store._rerank.return_value          = [(entry, 0.9, 0.9)]
    return store


@pytest.fixture
def benchmarker(mock_store):
    """Provide a Benchmarker bound to the mock_store fixture."""
    from src.rag.benchmarker import Benchmarker
    return Benchmarker(mock_store)


def _mock_chat(**kw):
    """Mock for ollama.chat that handles both streaming and non-streaming calls.

    Streaming calls (stream=True) simulate answer generation.
    Non-streaming calls simulate LLM-as-judge scoring — always return '5'
    so all LLM-judge metrics normalise to 1.0 in tests.
    """
    if kw.get('stream'):
        return [{'message': {'content': 'Cats sleep 16 hours a day.'}}]
    # LLM-as-judge calls: return the highest rating digit
    return {'message': {'content': '5'}}


# ---------------------------------------------------------------------------
# DEFAULT_TEST_CASES
# ---------------------------------------------------------------------------

class TestDefaultTestCases:
    """Checks for DEFAULT_TEST_CASES: 15 questions, 4 domains, backward-compatible first 5."""

    def test_has_15_cases(self):
        """DEFAULT_TEST_CASES contains exactly 15 test case dicts (5 cat + 4 Python + 3 CSV + 3 ML)."""
        from src.rag.benchmarker import Benchmarker
        assert len(Benchmarker.DEFAULT_TEST_CASES) == 15

    def test_all_have_question(self):
        """Every test case dict has a 'question' key."""
        from src.rag.benchmarker import Benchmarker
        for tc in Benchmarker.DEFAULT_TEST_CASES:
            assert 'question' in tc

    def test_all_have_expected_keywords(self):
        """Every test case dict has an 'expected_keywords' key."""
        from src.rag.benchmarker import Benchmarker
        for tc in Benchmarker.DEFAULT_TEST_CASES:
            assert 'expected_keywords' in tc

    def test_all_have_ground_truth(self):
        """Every test case dict has a 'ground_truth' key for the full metric suite."""
        from src.rag.benchmarker import Benchmarker
        for tc in Benchmarker.DEFAULT_TEST_CASES:
            assert 'ground_truth' in tc
            assert tc['ground_truth']   # must be a non-empty string

    def test_all_have_query_type(self):
        """Every test case dict has a 'query_type' key for per-type breakdown."""
        from src.rag.benchmarker import Benchmarker
        for tc in Benchmarker.DEFAULT_TEST_CASES:
            assert 'query_type' in tc

    def test_first_question_about_cats_sleep(self):
        """First test case question mentions 'sleep' (backward compatibility)."""
        from src.rag.benchmarker import Benchmarker
        assert 'sleep' in Benchmarker.DEFAULT_TEST_CASES[0]['question'].lower()

    def test_first_keywords_contain_16(self):
        """First test case expected_keywords includes '16' (backward compatibility)."""
        from src.rag.benchmarker import Benchmarker
        assert '16' in Benchmarker.DEFAULT_TEST_CASES[0]['expected_keywords']


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests that Benchmarker.__init__ sets store and file paths from config."""

    def test_has_store(self, benchmarker, mock_store):
        """Benchmarker.store is the same object as the injected mock_store."""
        assert benchmarker.store is mock_store

    def test_has_results_file(self, benchmarker):
        """Benchmarker.results_file matches BENCHMARK_FILE from config."""
        from src.rag.config import BENCHMARK_FILE
        assert benchmarker.results_file == BENCHMARK_FILE

    def test_has_csv_file(self, benchmarker):
        """Benchmarker.csv_file matches BENCHMARK_CSV from config."""
        from src.rag.config import BENCHMARK_CSV
        assert benchmarker.csv_file == BENCHMARK_CSV


# ---------------------------------------------------------------------------
# _compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    """Tests for the summary statistics computation."""

    def _make_results(self):
        """Build two minimal result dicts that cover all metric keys."""
        return [
            {
                'question': 'q1', 'query_type': 'factual',
                'ground_truth': 'gt', 'response': 'r',
                'faithfulness_llm': 0.8, 'answer_relevancy_llm': 0.9,
                'ground_truth_match': 0.7, 'keyword_recall': 1.0,
                'context_relevance': 0.6, 'precision_at_5': 0.8,
                'mrr': 1.0, 'latency_ms': 1000.0, 'overall': 0.84,
            },
            {
                'question': 'q2', 'query_type': 'factual',
                'ground_truth': 'gt', 'response': 'r',
                'faithfulness_llm': 0.6, 'answer_relevancy_llm': 0.7,
                'ground_truth_match': 0.5, 'keyword_recall': 0.5,
                'context_relevance': 0.4, 'precision_at_5': 0.6,
                'mrr': 0.5, 'latency_ms': 800.0, 'overall': 0.57,
            },
        ]

    def test_returns_dict(self, benchmarker):
        """_compute_summary returns a dict."""
        results = self._make_results()
        summary = benchmarker._compute_summary(results)
        assert isinstance(summary, dict)

    def test_all_scored_metrics_present(self, benchmarker):
        """Summary contains all 7 scored metrics as keys."""
        results = self._make_results()
        summary = benchmarker._compute_summary(results)
        for key in ['faithfulness_llm', 'answer_relevancy_llm', 'ground_truth_match',
                    'keyword_recall', 'context_relevance', 'precision_at_5', 'mrr']:
            assert key in summary

    def test_each_metric_has_mean_std_min_max(self, benchmarker):
        """Each metric sub-dict contains 'mean', 'std', 'min', 'max'."""
        results = self._make_results()
        summary = benchmarker._compute_summary(results)
        for key in ['faithfulness_llm', 'overall']:
            assert 'mean' in summary[key]
            assert 'std'  in summary[key]
            assert 'min'  in summary[key]
            assert 'max'  in summary[key]

    def test_mean_is_correct(self, benchmarker):
        """faithfulness_llm mean of [0.8, 0.6] is 0.7."""
        results = self._make_results()
        summary = benchmarker._compute_summary(results)
        assert abs(summary['faithfulness_llm']['mean'] - 0.7) < 0.01

    def test_overall_mean_is_present(self, benchmarker):
        """summary['overall_mean'] is a scalar float."""
        results = self._make_results()
        summary = benchmarker._compute_summary(results)
        assert 'overall_mean' in summary
        assert isinstance(summary['overall_mean'], float)

    def test_single_result_std_is_zero(self, benchmarker):
        """Single result: std is 0.0 (cannot compute standard deviation from one value)."""
        single = [self._make_results()[0]]
        summary = benchmarker._compute_summary(single)
        assert summary['faithfulness_llm']['std'] == 0.0

    def test_none_ground_truth_skipped(self, benchmarker):
        """Results with ground_truth_match=None are skipped in the summary."""
        results = [
            {
                'question': 'q', 'query_type': 'factual',
                'ground_truth': '', 'response': 'r',
                'faithfulness_llm': 0.8, 'answer_relevancy_llm': 0.9,
                'ground_truth_match': None,       # no ground truth provided
                'keyword_recall': 1.0, 'context_relevance': 0.6,
                'precision_at_5': 0.8, 'mrr': 1.0,
                'latency_ms': 1000.0, 'overall': 0.85,
            }
        ]
        summary = benchmarker._compute_summary(results)
        # ground_truth_match should be absent (all values were None)
        assert 'ground_truth_match' not in summary


# ---------------------------------------------------------------------------
# _compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    """Tests that format_run_comparison produces a formatted delta string with ▲/▼/─ indicators.

    The formatting logic was moved from Benchmarker._compare_runs() to
    benchmark_report.format_run_comparison() when benchmarker.py hit the 500-line limit.
    Tests now call the module function directly.
    """

    def _make_summary(self, faith, relev):
        """Build a minimal summary dict in the nested {'mean': ...} format."""
        return {
            'faithfulness_llm':     {'mean': faith, 'std': 0.0, 'min': faith, 'max': faith},
            'answer_relevancy_llm': {'mean': relev, 'std': 0.0, 'min': relev, 'max': relev},
            'overall':              {'mean': (faith + relev) / 2, 'std': 0.0,
                                     'min': (faith + relev) / 2, 'max': (faith + relev) / 2},
            'overall_mean': (faith + relev) / 2,
        }

    def test_returns_string(self):
        """format_run_comparison returns a string."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(
            self._make_summary(0.8, 0.7),
            self._make_summary(0.7, 0.6),
        )
        assert isinstance(result, str)

    def test_shows_improvement_indicator(self):
        """Current scores higher than previous: '▲' is present in the output."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(
            self._make_summary(0.9, 0.9),
            self._make_summary(0.5, 0.5),
        )
        assert '▲' in result

    def test_shows_decline_indicator(self):
        """Current scores lower than previous: '▼' is present in the output."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(
            self._make_summary(0.4, 0.4),
            self._make_summary(0.9, 0.9),
        )
        assert '▼' in result

    def test_shows_unchanged_indicator(self):
        """Current and previous scores identical: '─' is present in the output."""
        from src.rag.benchmark_report import format_run_comparison
        summary = self._make_summary(0.7, 0.7)
        result  = format_run_comparison(summary, summary)
        assert '─' in result


# ---------------------------------------------------------------------------
# _read_results / _save_results
# ---------------------------------------------------------------------------

class TestReadSaveResults:
    """Tests the JSON persistence helpers."""

    def test_read_returns_empty_list_when_no_file(self, benchmarker):
        """Non-existent file: _read_results returns an empty list."""
        benchmarker.results_file = '/tmp/nonexistent_benchmark_xyz.json'
        assert benchmarker._read_results() == []

    def test_save_and_read_round_trip(self, benchmarker, tmp_path):
        """_save_results then _read_results: data survives the round trip."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        data = [{'timestamp': '2024-01-01', 'summary': {'overall_mean': 0.75}, 'results': []}]
        benchmarker._save_results(data)
        loaded = benchmarker._read_results()
        assert len(loaded) == 1
        assert loaded[0]['summary']['overall_mean'] == 0.75

    def test_corrupt_file_returns_empty_list(self, benchmarker, tmp_path):
        """Corrupt JSON file: _read_results returns an empty list instead of raising."""
        path = tmp_path / 'bench.json'
        path.write_text('{ not valid json }')
        benchmarker.results_file = str(path)
        assert benchmarker._read_results() == []


# ---------------------------------------------------------------------------
# _export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    """Tests that _export_csv creates a valid CSV file with the correct columns."""

    def _make_result(self):
        """Build a single complete result dict matching _run_single output format."""
        return {
            'question':             'How many hours do cats sleep?',
            'query_type':           'factual',
            'ground_truth':         'Cats sleep 16 hours.',
            'response':             'Cats sleep 16 hours a day.',
            'faithfulness_llm':     0.8,
            'answer_relevancy_llm': 0.9,
            'ground_truth_match':   0.75,
            'keyword_recall':       1.0,
            'context_relevance':    0.7,
            'precision_at_5':       0.8,
            'mrr':                  1.0,
            'latency_ms':           1100.0,
            'overall':              0.86,
        }

    def test_csv_file_is_created(self, benchmarker, tmp_path):
        """_export_csv creates a file at csv_file path."""
        benchmarker.csv_file = str(tmp_path / 'bench.csv')
        benchmarker._export_csv([self._make_result()], '2024-01-01 00:00:00')
        assert os.path.exists(benchmarker.csv_file)

    def test_csv_has_header_row(self, benchmarker, tmp_path):
        """CSV file contains a header row with 'question' and 'faithfulness_llm'."""
        benchmarker.csv_file = str(tmp_path / 'bench.csv')
        benchmarker._export_csv([self._make_result()], '2024-01-01 00:00:00')
        with open(benchmarker.csv_file, newline='') as f:
            header = next(csv.reader(f))
        assert 'question'         in header
        assert 'faithfulness_llm' in header
        assert 'mrr'              in header
        assert 'latency_ms'       in header

    def test_csv_has_data_row(self, benchmarker, tmp_path):
        """CSV file contains one data row for the single result provided."""
        benchmarker.csv_file = str(tmp_path / 'bench.csv')
        benchmarker._export_csv([self._make_result()], '2024-01-01 00:00:00')
        with open(benchmarker.csv_file, newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]['question'] == 'How many hours do cats sleep?'

    def test_csv_includes_timestamp(self, benchmarker, tmp_path):
        """Each CSV row includes the run timestamp."""
        benchmarker.csv_file = str(tmp_path / 'bench.csv')
        benchmarker._export_csv([self._make_result()], '2024-01-01 12:00:00')
        with open(benchmarker.csv_file, newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['timestamp'] == '2024-01-01 12:00:00'

    def test_empty_results_creates_no_file(self, benchmarker, tmp_path):
        """Empty results list: no CSV file is written."""
        benchmarker.csv_file = str(tmp_path / 'bench.csv')
        benchmarker._export_csv([], '2024-01-01 00:00:00')
        assert not os.path.exists(benchmarker.csv_file)


# ---------------------------------------------------------------------------
# run — integration with mocks
# ---------------------------------------------------------------------------

class TestRun:
    """Tests the full run() evaluation loop with mocked Ollama and VectorStore."""

    def test_run_returns_summary_dict(self, benchmarker, tmp_path):
        """run() with one custom test case: returns a dict with 'overall' key."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        benchmarker.csv_file     = str(tmp_path / 'bench.csv')
        test_cases = [{
            'question':          'How many hours do cats sleep?',
            'ground_truth':      'Cats sleep 16 hours.',
            'expected_keywords': ['sleep', '16'],
            'query_type':        'factual',
        }]
        with patch('ollama.chat', side_effect=_mock_chat):
            summary = benchmarker.run(test_cases)
        assert isinstance(summary, dict)
        assert 'overall' in summary

    def test_run_saves_json_file(self, benchmarker, tmp_path):
        """run() creates the JSON results file at results_file path."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        benchmarker.csv_file     = str(tmp_path / 'bench.csv')
        test_cases = [{
            'question':          'Do cats sleep a lot?',
            'ground_truth':      'Yes.',
            'expected_keywords': ['sleep'],
            'query_type':        'factual',
        }]
        with patch('ollama.chat', side_effect=_mock_chat):
            benchmarker.run(test_cases)
        assert os.path.exists(benchmarker.results_file)

    def test_run_exports_csv_file(self, benchmarker, tmp_path):
        """run() creates the CSV export file at csv_file path."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        benchmarker.csv_file     = str(tmp_path / 'bench.csv')
        test_cases = [{
            'question':          'Do cats sleep a lot?',
            'ground_truth':      'Yes.',
            'expected_keywords': ['sleep'],
            'query_type':        'factual',
        }]
        with patch('ollama.chat', side_effect=_mock_chat):
            benchmarker.run(test_cases)
        assert os.path.exists(benchmarker.csv_file)

    def test_run_uses_default_test_cases(self, benchmarker, tmp_path):
        """run() called with no arguments uses DEFAULT_TEST_CASES and returns a summary."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        benchmarker.csv_file     = str(tmp_path / 'bench.csv')
        with patch('ollama.chat', side_effect=_mock_chat):
            summary = benchmarker.run()
        assert isinstance(summary, dict)

    def test_run_summary_has_all_metrics(self, benchmarker, tmp_path):
        """run() summary contains all 7 scored metrics plus latency and overall."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        benchmarker.csv_file     = str(tmp_path / 'bench.csv')
        test_cases = [{
            'question':          'Do cats sleep a lot?',
            'ground_truth':      'Yes, cats sleep about 16 hours.',
            'expected_keywords': ['sleep'],
            'query_type':        'factual',
        }]
        with patch('ollama.chat', side_effect=_mock_chat):
            summary = benchmarker.run(test_cases)
        for key in ['faithfulness_llm', 'answer_relevancy_llm', 'keyword_recall',
                    'context_relevance', 'precision_at_5', 'mrr', 'overall']:
            assert key in summary

    def test_run_appends_to_existing_history(self, benchmarker, tmp_path):
        """run() called twice: JSON history file contains two run entries."""
        benchmarker.results_file = str(tmp_path / 'bench.json')
        benchmarker.csv_file     = str(tmp_path / 'bench.csv')
        test_cases = [{
            'question':          'Do cats sleep?',
            'ground_truth':      'Yes.',
            'expected_keywords': ['sleep'],
            'query_type':        'factual',
        }]
        with patch('ollama.chat', side_effect=_mock_chat):
            benchmarker.run(test_cases)
            benchmarker.run(test_cases)
        with open(benchmarker.results_file) as f:
            history = json.load(f)
        assert len(history) == 2
