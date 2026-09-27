"""benchmark_report.py — Stateless terminal report formatting for the benchmark.

WHY THIS FILE EXISTS:
    benchmarker.py would exceed 500 lines if the report-printing functions were
    kept inside the Benchmarker class. Because the print functions have no state
    of their own (they operate entirely on the dicts passed in), they belong here
    as module-level functions — consistent with the rule: classes own state,
    modules own stateless functions.

PUBLIC API:
    print_per_query_table(results)       — one row per question
    print_summary_table(summary)         — mean/std/min/max + bar chart
    print_by_query_type(results)         — overall score grouped by type
    format_run_comparison(curr, prev)    — delta vs previous run (returns str)
"""

from typing import List

__all__ = [
    'print_per_query_table',
    'print_summary_table',
    'print_by_query_type',
    'format_run_comparison',
]

# Width of the separator lines used throughout the terminal report.
# Must match _LINE_WIDTH in benchmarker.py.
_LINE_WIDTH = 72


def print_per_query_table(results: List[dict]) -> None:
    """Print a table showing all metric scores for each individual question.

    Columns: question number, question text (truncated to 40 chars), then
    Faith, Relev, GT, KwRec, Ctx, P@5, MRR, and latency in milliseconds.

    Args:
        results: List of per-question result dicts from Benchmarker._run_single().
    """
    header = (
        f"\n  {'#':<4} {'Question':<40}  "
        f"{'Faith':>5} {'Relev':>5} {'GT':>5} "
        f"{'KwRec':>5} {'Ctx':>5} {'P@5':>5} {'MRR':>5} {'ms':>7}"
    )
    divider = '  ' + '─' * 88

    print('\n' + '═' * _LINE_WIDTH)
    print('  PER-QUESTION RESULTS')
    print('═' * _LINE_WIDTH)
    print(header)
    print(divider)

    for i, r in enumerate(results, start=1):
        # Truncate long questions so the table stays within a readable width
        question_display = r['question']
        if len(question_display) > 40:
            question_display = question_display[:38] + '..'

        gt_str = f"{r['ground_truth_match']:>5.2f}" if r['ground_truth_match'] is not None else '  N/A'
        print(
            f"  {i:<4} {question_display:<40}  "
            f"{r['faithfulness_llm']:>5.2f} "
            f"{r['answer_relevancy_llm']:>5.2f} "
            f"{gt_str} "
            f"{r['keyword_recall']:>5.2f} "
            f"{r['context_relevance']:>5.2f} "
            f"{r['precision_at_5']:>5.2f} "
            f"{r['mrr']:>5.2f} "
            f"{r['latency_ms']:>7.0f}"
        )


def print_summary_table(summary: dict) -> None:
    """Print a summary table with mean, std, min, max, and a bar chart for each metric.

    Latency (milliseconds) is shown on its own line because it is not a 0–1 score.
    The overall mean is printed last, separated by a divider, as the headline number.

    Args:
        summary: Dict from Benchmarker._compute_summary() — each metric maps to
                 a sub-dict with 'mean', 'std', 'min', 'max'.
    """
    # Display order and human-readable labels — order matches the 3 quality dimensions
    metric_labels = [
        ('faithfulness_llm',     'faithfulness (LLM)    '),
        ('answer_relevancy_llm', 'answer_relevancy (LLM)'),
        ('ground_truth_match',   'ground_truth_match    '),
        ('keyword_recall',       'keyword_recall        '),
        ('context_relevance',    'context_relevance     '),
        ('precision_at_5',       'precision_at_5        '),
        ('mrr',                  'mrr                   '),
    ]

    print('\n' + '═' * _LINE_WIDTH)
    print('  SUMMARY')
    print('═' * _LINE_WIDTH)
    print(f"  {'Metric':<24} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}  Bar")
    print('  ' + '─' * 66)

    for key, label in metric_labels:
        if key not in summary:
            continue
        s   = summary[key]
        bar = '[' + '█' * int(s['mean'] * 20) + '░' * (20 - int(s['mean'] * 20)) + ']'
        print(f"  {label} {s['mean']:>6.3f} {s['std']:>6.3f} {s['min']:>6.3f} {s['max']:>6.3f}  {bar}")

    # Latency is shown on its own line — it is milliseconds, not a 0-1 score
    if 'latency_ms' in summary:
        s = summary['latency_ms']
        print(
            f"\n  {'latency_ms':<24} {s['mean']:>6.0f} {s['std']:>6.0f} "
            f"{s['min']:>6.0f} {s['max']:>6.0f}  ms"
        )

    # Overall score at the bottom, separated by a divider
    if 'overall' in summary:
        s   = summary['overall']
        bar = '[' + '█' * int(s['mean'] * 20) + '░' * (20 - int(s['mean'] * 20)) + ']'
        print('  ' + '─' * 66)
        print(f"  {'overall':<24} {s['mean']:>6.3f} {s['std']:>6.3f} {s['min']:>6.3f} {s['max']:>6.3f}  {bar}")


def print_by_query_type(results: List[dict]) -> None:
    """Print a breakdown of overall scores grouped by query type.

    Shows how the pipeline performs on factual vs comparison vs summarise queries.
    Useful for spotting whether a particular query type is systematically weaker.

    Args:
        results: List of per-question result dicts from Benchmarker._run_single().
    """
    # Group each result by its query_type label
    by_type: dict = {}
    for r in results:
        qtype = r.get('query_type', 'unspecified')
        by_type.setdefault(qtype, []).append(r['overall'])

    if not by_type:
        return

    print('\n' + '═' * _LINE_WIDTH)
    print('  BY QUERY TYPE')
    print('═' * _LINE_WIDTH)

    for qtype, scores in sorted(by_type.items()):
        mean        = sum(scores) / len(scores)
        count_label = f"({len(scores)} question{'s' if len(scores) != 1 else ''})"
        bar         = '[' + '█' * int(mean * 20) + '░' * (20 - int(mean * 20)) + ']'
        print(f"  {qtype:<20} {count_label:<15}  overall {mean:.3f}  {bar}")


def format_run_comparison(current: dict, previous: dict) -> str:
    """Build a formatted delta string comparing the current run to the previous one.

    Shows ▲ (improved), ▼ (declined), or ─ (unchanged) for each metric.

    Args:
        current:  Summary dict from the current run (Benchmarker._compute_summary output).
        previous: Summary dict from the previous run (loaded from JSON history).

    Returns:
        Multi-line string ready to print to the terminal.
    """
    lines = ['\n' + '═' * _LINE_WIDTH, '  vs PREVIOUS RUN', '═' * _LINE_WIDTH]

    for key in current:
        # Skip the scalar overall_mean and non-metric keys
        if key == 'overall_mean' or not isinstance(current[key], dict):
            continue

        curr_mean = current[key].get('mean')
        # Support both new format (nested dict) and old flat format (plain float)
        prev_entry = previous.get(key)
        if isinstance(prev_entry, dict):
            prev_mean = prev_entry.get('mean')
        else:
            prev_mean = prev_entry

        if curr_mean is None or prev_mean is None:
            continue

        delta     = curr_mean - prev_mean
        indicator = '▲' if delta > 0.001 else '▼' if delta < -0.001 else '─'
        lines.append(
            f"  {key:<25} {prev_mean:>6.3f} → {curr_mean:>6.3f}  "
            f"{indicator}{abs(delta):.3f}"
        )

    return '\n'.join(lines)
