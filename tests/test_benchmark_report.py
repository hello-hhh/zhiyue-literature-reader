"""Unit tests for src/rag/benchmark_report.py.

Covers all 4 public functions:
    print_per_query_table   — terminal table output (captured via capsys)
    print_summary_table     — terminal summary with bar charts (captured via capsys)
    print_by_query_type     — per-type breakdown (captured via capsys)
    format_run_comparison   — delta string with ▲/▼/─ indicators (returns str)

No mocking needed — all functions are pure output formatters that take plain dicts.
"""

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(question='What is X?', faith=0.8, relev=0.7, gt=0.6, kw=1.0,
                 ctx=0.7, p5=0.8, mrr=1.0, latency=1200.0, qtype='factual'):
    """Build a minimal per-question result dict with all required metric keys."""
    return {
        'question':             question,
        'query_type':           qtype,
        'ground_truth':         'The answer is X.',
        'response':             'X is the answer.',
        'faithfulness_llm':     faith,
        'answer_relevancy_llm': relev,
        'ground_truth_match':   gt,
        'keyword_recall':       kw,
        'context_relevance':    ctx,
        'precision_at_5':       p5,
        'mrr':                  mrr,
        'latency_ms':           latency,
        'overall':              round((faith + relev + gt + kw + ctx + p5 + mrr) / 7, 3),
    }


def _make_summary(faith=0.8, relev=0.7):
    """Build a minimal summary dict in the nested {'mean': ...} format."""
    return {
        'faithfulness_llm':     {'mean': faith, 'std': 0.05, 'min': faith - 0.1, 'max': faith + 0.1},
        'answer_relevancy_llm': {'mean': relev, 'std': 0.04, 'min': relev - 0.1, 'max': relev + 0.1},
        'keyword_recall':       {'mean': 0.9, 'std': 0.1, 'min': 0.8, 'max': 1.0},
        'context_relevance':    {'mean': 0.7, 'std': 0.05, 'min': 0.6, 'max': 0.8},
        'precision_at_5':       {'mean': 0.8, 'std': 0.1, 'min': 0.6, 'max': 1.0},
        'mrr':                  {'mean': 0.9, 'std': 0.2, 'min': 0.5, 'max': 1.0},
        'latency_ms':           {'mean': 1200.0, 'std': 100.0, 'min': 1000.0, 'max': 1400.0},
        'overall':              {'mean': 0.8, 'std': 0.03, 'min': 0.75, 'max': 0.85},
        'overall_mean': 0.8,
    }


# ── TestPrintPerQueryTable ─────────────────────────────────────────────────────

class TestPrintPerQueryTable:
    """Tests for print_per_query_table — verifies terminal output content."""

    def test_prints_per_question_results_header(self, capsys):
        """Output must contain the 'PER-QUESTION RESULTS' section header."""
        from src.rag.benchmark_report import print_per_query_table
        print_per_query_table([_make_result()])
        captured = capsys.readouterr().out
        assert 'PER-QUESTION RESULTS' in captured

    def test_prints_question_number(self, capsys):
        """The question row must contain the 1-based row number."""
        from src.rag.benchmark_report import print_per_query_table
        print_per_query_table([_make_result(question='Test question one?')])
        captured = capsys.readouterr().out
        assert '1' in captured

    def test_long_question_is_truncated(self, capsys):
        """Questions longer than 40 chars are truncated with '..' in the output."""
        from src.rag.benchmark_report import print_per_query_table
        long_q = 'This is a very long question that exceeds the 40 character limit easily'
        print_per_query_table([_make_result(question=long_q)])
        captured = capsys.readouterr().out
        assert '..' in captured

    def test_none_ground_truth_shows_na(self, capsys):
        """Results with ground_truth_match=None must show 'N/A' in the table."""
        from src.rag.benchmark_report import print_per_query_table
        result = _make_result()
        result['ground_truth_match'] = None
        print_per_query_table([result])
        captured = capsys.readouterr().out
        assert 'N/A' in captured

    def test_multiple_rows_printed(self, capsys):
        """Two results: output contains two row numbers (1 and 2)."""
        from src.rag.benchmark_report import print_per_query_table
        print_per_query_table([_make_result('Q1?'), _make_result('Q2?')])
        captured = capsys.readouterr().out
        # Both row numbers must appear in the output
        assert '  1   ' in captured or '  1 ' in captured
        assert '  2   ' in captured or '  2 ' in captured

    def test_empty_results_prints_header_only(self, capsys):
        """Empty results list: only the section header is printed, no crash."""
        from src.rag.benchmark_report import print_per_query_table
        print_per_query_table([])
        captured = capsys.readouterr().out
        assert 'PER-QUESTION RESULTS' in captured


# ── TestPrintSummaryTable ──────────────────────────────────────────────────────

class TestPrintSummaryTable:
    """Tests for print_summary_table — verifies the SUMMARY section output."""

    def test_prints_summary_header(self, capsys):
        """Output must contain the 'SUMMARY' section header."""
        from src.rag.benchmark_report import print_summary_table
        print_summary_table(_make_summary())
        captured = capsys.readouterr().out
        assert 'SUMMARY' in captured

    def test_prints_faithfulness_row(self, capsys):
        """Output must contain the faithfulness metric label."""
        from src.rag.benchmark_report import print_summary_table
        print_summary_table(_make_summary())
        captured = capsys.readouterr().out
        assert 'faithfulness' in captured

    def test_prints_bar_chart_characters(self, capsys):
        """Output must contain the bar chart block characters (█ and ░)."""
        from src.rag.benchmark_report import print_summary_table
        print_summary_table(_make_summary())
        captured = capsys.readouterr().out
        assert '█' in captured or '░' in captured

    def test_latency_row_shows_ms_unit(self, capsys):
        """Latency row must contain 'ms' to indicate it is in milliseconds."""
        from src.rag.benchmark_report import print_summary_table
        print_summary_table(_make_summary())
        captured = capsys.readouterr().out
        assert 'ms' in captured

    def test_overall_row_present(self, capsys):
        """Output must contain the 'overall' row at the end of the table."""
        from src.rag.benchmark_report import print_summary_table
        print_summary_table(_make_summary())
        captured = capsys.readouterr().out
        assert 'overall' in captured

    def test_missing_metric_skipped_gracefully(self, capsys):
        """Summary dict without ground_truth_match: no crash, other rows still printed."""
        from src.rag.benchmark_report import print_summary_table
        summary = _make_summary()
        del summary['faithfulness_llm']
        print_summary_table(summary)
        captured = capsys.readouterr().out
        # At least the remaining metrics must appear
        assert 'answer_relevancy' in captured


# ── TestPrintByQueryType ───────────────────────────────────────────────────────

class TestPrintByQueryType:
    """Tests for print_by_query_type — verifies the BY QUERY TYPE section."""

    def test_prints_by_query_type_header(self, capsys):
        """Output must contain the 'BY QUERY TYPE' section header."""
        from src.rag.benchmark_report import print_by_query_type
        print_by_query_type([_make_result(qtype='factual')])
        captured = capsys.readouterr().out
        assert 'BY QUERY TYPE' in captured

    def test_shows_query_type_label(self, capsys):
        """Output must contain the query type name ('factual')."""
        from src.rag.benchmark_report import print_by_query_type
        print_by_query_type([_make_result(qtype='factual')])
        captured = capsys.readouterr().out
        assert 'factual' in captured

    def test_multiple_types_both_shown(self, capsys):
        """Two different query types: both labels appear in the output."""
        from src.rag.benchmark_report import print_by_query_type
        results = [_make_result(qtype='factual'), _make_result(qtype='comparison')]
        print_by_query_type(results)
        captured = capsys.readouterr().out
        assert 'factual' in captured
        assert 'comparison' in captured

    def test_question_count_shown(self, capsys):
        """Output must mention the number of questions per type."""
        from src.rag.benchmark_report import print_by_query_type
        results = [_make_result(qtype='factual'), _make_result(qtype='factual')]
        print_by_query_type(results)
        captured = capsys.readouterr().out
        # 2 factual questions — plural "questions" must appear
        assert 'questions' in captured

    def test_empty_results_no_output(self, capsys):
        """Empty results list: no output printed (nothing to show)."""
        from src.rag.benchmark_report import print_by_query_type
        print_by_query_type([])
        captured = capsys.readouterr().out
        assert captured == ''


# ── TestFormatRunComparison ────────────────────────────────────────────────────

class TestFormatRunComparison:
    """Tests for format_run_comparison — verifies the delta comparison string."""

    def test_returns_string(self):
        """format_run_comparison must return a string."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(_make_summary(0.8, 0.7), _make_summary(0.7, 0.6))
        assert isinstance(result, str)

    def test_shows_improvement_indicator(self):
        """Current > previous: '▲' must appear in the output."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(_make_summary(0.9, 0.9), _make_summary(0.5, 0.5))
        assert '▲' in result

    def test_shows_decline_indicator(self):
        """Current < previous: '▼' must appear in the output."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(_make_summary(0.4, 0.4), _make_summary(0.9, 0.9))
        assert '▼' in result

    def test_shows_unchanged_indicator(self):
        """Current == previous: '─' must appear in the output."""
        from src.rag.benchmark_report import format_run_comparison
        summary = _make_summary(0.7, 0.7)
        result  = format_run_comparison(summary, summary)
        assert '─' in result

    def test_shows_vs_previous_run_header(self):
        """Output must contain the 'vs PREVIOUS RUN' section header."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(_make_summary(), _make_summary())
        assert 'vs PREVIOUS RUN' in result

    def test_metric_name_in_output(self):
        """Output must contain the metric name (e.g. 'faithfulness_llm')."""
        from src.rag.benchmark_report import format_run_comparison
        result = format_run_comparison(_make_summary(0.8, 0.7), _make_summary(0.7, 0.6))
        assert 'faithfulness_llm' in result

    def test_old_flat_format_previous_run(self):
        """Previous summary with flat float values (old format) is handled without crashing."""
        from src.rag.benchmark_report import format_run_comparison

        # Old format: metrics were plain floats, not nested dicts
        old_format_summary = {
            'faithfulness_llm':     0.75,
            'answer_relevancy_llm': 0.65,
        }
        result = format_run_comparison(_make_summary(), old_format_summary)
        # Should not raise — old float format is supported
        assert isinstance(result, str)
