"""benchmarker.py — Benchmarker class.

Owns all evaluation, run comparison, result persistence, and report output.

METRICS COMPUTED PER QUESTION:
    faithfulness_llm      — LLM judges whether the answer stays grounded in context (0–1)
    answer_relevancy_llm  — LLM judges whether the answer addresses the question (0–1)
    ground_truth_match    — F1 word overlap between the response and a known correct answer (0–1)
    keyword_recall        — fraction of expected keywords found in the response (0–1)
    context_relevance     — mean cosine similarity of top retrieved chunks (0–1)
    precision_at_5        — fraction of top-5 chunks that contain an expected keyword (0–1)
    mrr                   — reciprocal rank of the first relevant retrieved chunk (0–1)
    latency_ms            — end-to-end query time in milliseconds (lower is better)

All scoring functions live in metrics.py and are imported here.
"""

import csv
import json
import os
import statistics
import time
from datetime import datetime
from typing import List, Optional

import ollama

from src.rag.benchmark_report import (
    format_run_comparison,
    print_by_query_type,
    print_per_query_table,
    print_summary_table,
)
from src.rag.config import BENCHMARK_CSV, BENCHMARK_FILE, LANGUAGE_MODEL, TOP_RERANK, TOP_RETRIEVE
from src.rag.metrics import (
    score_answer_relevancy_llm,
    score_context_relevance,
    score_faithfulness_llm,
    score_ground_truth_match,
    score_keyword_recall,
    score_mrr,
    score_precision_at_k,
)
from src.rag.vector_store import VectorStore

# Width of the separator lines used throughout the terminal report
_LINE_WIDTH = 72


class Benchmarker:
    """Owns all evaluation logic: scoring, run comparison, and result persistence.

    Takes a VectorStore as a dependency so it runs the exact same pipeline the
    user runs in production — benchmark results reflect real system behaviour,
    not isolated mock scores.

    State:
        store:        VectorStore reference (used for expand_query, hybrid_retrieve, rerank)
        results_file: path to the JSON file where all benchmark runs are appended
        csv_file:     path to the CSV file written after each run (latest run only)

    Public API:
        run(test_cases) — run all test cases, print a full report, save to disk
    """

    # 15 test cases covering 3 domains and 3 file types.
    # The original 5 cat-fact questions are preserved unchanged for backward compatibility.
    # Added: 4 Python-language questions (txt), 3 CSV/team questions, 3 ML questions.
    DEFAULT_TEST_CASES = [
        # ── Domain 1: Cat facts (original 5, from docs/txts/cat_facts.txt) ──────
        {
            'question':          'How many hours do cats sleep per day?',
            'ground_truth':      'Cats sleep between 12 and 16 hours per day.',
            'expected_keywords': ['sleep', '16'],
            'query_type':        'factual',
        },
        {
            'question':          'Can cats see in dim light?',
            'ground_truth':      'Yes, cats can see in dim light due to their large pupils and many rod cells.',
            'expected_keywords': ['dim', 'light', 'see'],
            'query_type':        'factual',
        },
        {
            'question':          'How many toes do cats have on their front paws?',
            'ground_truth':      'Cats have five toes on each front paw.',
            'expected_keywords': ['five', 'toes', 'front'],
            'query_type':        'factual',
        },
        {
            'question':          'How many whiskers does a cat have?',
            'ground_truth':      'A cat typically has 12 whiskers on each side of its face.',
            'expected_keywords': ['whiskers', '12'],
            'query_type':        'factual',
        },
        {
            'question':          'Can cats taste sweet food?',
            'ground_truth':      'No, cats cannot taste sweetness because they lack sweet taste receptors.',
            'expected_keywords': ['sweet', 'taste'],
            'query_type':        'factual',
        },
        # ── Domain 2: Python language facts (from benchmark_docs/python-language.txt) ──
        {
            'question':          'Who created Python and when was it first released?',
            'ground_truth':      'Python was created by Guido van Rossum and first released in 1991.',
            'expected_keywords': ['Guido', '1991'],
            'query_type':        'factual',
        },
        {
            'question':          'What keyword does Python use to produce values from a generator?',
            'ground_truth':      'Python generators use the yield keyword to produce values lazily.',
            'expected_keywords': ['yield', 'generator'],
            'query_type':        'factual',
        },
        {
            'question':          'What is the GIL in Python?',
            'ground_truth':      'The GIL is the Global Interpreter Lock which prevents true parallel execution of Python threads.',
            'expected_keywords': ['GIL', 'parallel', 'threads'],
            'query_type':        'factual',
        },
        {
            'question':          'What version of Python introduced f-strings?',
            'ground_truth':      'Python f-strings were introduced in Python 3.6.',
            'expected_keywords': ['3.6', 'f-string'],
            'query_type':        'factual',
        },
        # ── Domain 3: Team members (from benchmark_docs/team-members.csv) ──────
        {
            'question':          'What is Carol Davis role and department?',
            'ground_truth':      'Carol Davis is an NLP Research Scientist in the AI Research department.',
            'expected_keywords': ['Carol', 'NLP', 'Research'],
            'query_type':        'factual',
        },
        {
            'question':          'How many years of experience does Alice Chen have?',
            'ground_truth':      'Alice Chen has 8 years of experience.',
            'expected_keywords': ['Alice', '8'],
            'query_type':        'factual',
        },
        {
            'question':          'What programming languages does David Kim know?',
            'ground_truth':      'David Kim knows Go, Python, and Rust.',
            'expected_keywords': ['David', 'Go', 'Python'],
            'query_type':        'factual',
        },
        # ── Domain 4: Machine learning concepts (from benchmark_docs/machine-learning.md) ──
        {
            'question':          'What is overfitting in machine learning?',
            'ground_truth':      'Overfitting occurs when a model memorises the training data and performs poorly on new data.',
            'expected_keywords': ['overfitting', 'training', 'data'],
            'query_type':        'factual',
        },
        {
            'question':          'What is the F1 score and when is it useful?',
            'ground_truth':      'F1 score is the harmonic mean of precision and recall, useful when classes are imbalanced.',
            'expected_keywords': ['F1', 'precision', 'recall'],
            'query_type':        'factual',
        },
        {
            'question':          'What does dropout do in neural network training?',
            'ground_truth':      'Dropout randomly deactivates neurons during training to prevent co-adaptation.',
            'expected_keywords': ['dropout', 'neurons', 'training'],
            'query_type':        'factual',
        },
    ]

    def __init__(self, store: VectorStore):
        """Bind an initialised VectorStore and set the output file paths.

        Args:
            store: VectorStore with build_or_load() already called.
        """
        self.store        = store
        self.results_file = BENCHMARK_FILE
        self.csv_file     = BENCHMARK_CSV

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, test_cases: Optional[List[dict]] = None) -> List[dict]:
        """Run evaluation on a list of questions and print a full scored report.

        For each question the pipeline:
        1. Expands the query into 3 variants using the language model.
        2. Retrieves the top chunks using hybrid BM25 + dense vector search.
        3. Reranks the chunks using the type-aware LLM reranker.
        4. Generates an answer from the top chunks.
        5. Scores the answer on 7 metrics (2 via LLM judge, 5 lexical/numeric).

        The report printed to the terminal includes a per-question table,
        a summary table with mean/std/min/max, a per-query-type breakdown,
        and a run-over-run delta comparison.

        Args:
            test_cases: List of dicts. Required keys: 'question', 'expected_keywords'.
                        Optional keys: 'ground_truth' (str), 'query_type' (str).
                        Defaults to DEFAULT_TEST_CASES (5 cat-fact questions).

        Returns:
            Summary dict mapping each metric to its {'mean', 'std', 'min', 'max'}.
            Also contains 'overall_mean' (float) for quick comparison.
        """
        test_cases = test_cases or self.DEFAULT_TEST_CASES
        timestamp  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print('\n' + '═' * _LINE_WIDTH)
        print(f"  RAG PIPELINE BENCHMARK  ·  {timestamp}  ·  {len(test_cases)} questions")
        print('═' * _LINE_WIDTH)

        # Run every test case and collect the per-question metric results
        results = []
        for index, tc in enumerate(test_cases, start=1):
            print(f"\n  [{index}/{len(test_cases)}] {tc['question']}")
            result = self._run_single(
                question     = tc['question'],
                keywords     = tc.get('expected_keywords', []),
                ground_truth = tc.get('ground_truth', ''),
                query_type   = tc.get('query_type', 'unspecified'),
            )
            results.append(result)
            # Print a compact one-line summary while the run is in progress
            gt_part = f"gt={result['ground_truth_match']:.2f}  " if result['ground_truth_match'] is not None else ''
            print(
                f"         faith={result['faithfulness_llm']:.2f}  "
                f"relev={result['answer_relevancy_llm']:.2f}  "
                f"{gt_part}"
                f"kw={result['keyword_recall']:.2f}  "
                f"ctx={result['context_relevance']:.2f}  "
                f"p@5={result['precision_at_5']:.2f}  "
                f"mrr={result['mrr']:.2f}  "
                f"{result['latency_ms']:.0f}ms"
            )

        # Compute averages, std dev, min, max for every metric
        summary = self._compute_summary(results)

        # Print the full formatted report sections (stateless functions in benchmark_report.py)
        print_per_query_table(results)
        print_summary_table(summary)
        print_by_query_type(results)

        # Compare this run against the previous one if a history file exists
        runs = self._read_results()
        if runs:
            print(format_run_comparison(summary, runs[-1]['summary']))

        # Persist results to JSON history and export latest run to CSV
        runs.append({'timestamp': timestamp, 'summary': summary, 'results': results})
        self._save_results(runs)
        self._export_csv(results, timestamp)

        print(f"\n  Saved  → {self.results_file}")
        print(f"  Export → {self.csv_file}")
        print('═' * _LINE_WIDTH + '\n')

        return summary

    # ── Private — evaluation ─────────────────────────────────────────────────

    def _run_single(
        self,
        question:     str,
        keywords:     List[str],
        ground_truth: str,
        query_type:   str,
    ) -> dict:
        """Run the full retrieval + generation + scoring pipeline for one question.

        Measures end-to-end latency from query expansion through answer generation.
        LLM-as-judge scoring happens after the timer stops (it is evaluation overhead,
        not pipeline latency).

        Args:
            question:     The question to evaluate.
            keywords:     Expected keywords a correct answer should contain.
            ground_truth: The known correct answer (empty string if unavailable).
            query_type:   Category label e.g. 'factual', 'comparison', 'summarise'.

        Returns:
            Dict with all metric scores, the generated response, and latency.
        """
        # ── Retrieval and generation (timed) ──────────────────────────────────
        start_ms = time.time() * 1000

        queries   = self.store._expand_query(question)
        retrieved = self.store._hybrid_retrieve(queries, top_n=TOP_RETRIEVE)
        reranked  = self.store._rerank(question, retrieved, top_n=TOP_RERANK)

        context_text = '\n'.join(f"- {entry['text']}" for entry, _, _ in reranked)
        stream = ollama.chat(
            model    = LANGUAGE_MODEL,
            messages = [
                {
                    'role':    'system',
                    'content': (
                        "You are a factual assistant. Answer in 1–2 sentences "
                        f"using ONLY the facts below.\n\nFacts:\n{context_text}"
                    ),
                },
                {'role': 'user', 'content': question},
            ],
            stream=True,
        )
        response   = ''.join(chunk['message']['content'] for chunk in stream)
        latency_ms = time.time() * 1000 - start_ms

        # ── Scoring (after timer stops) ───────────────────────────────────────
        faith = score_faithfulness_llm(question, response, context_text, LANGUAGE_MODEL)
        relev = score_answer_relevancy_llm(question, response, LANGUAGE_MODEL)
        kw    = score_keyword_recall(response, keywords)
        ctx   = score_context_relevance(reranked, TOP_RERANK)
        p5    = score_precision_at_k(reranked, keywords, k=5)
        mrr   = score_mrr(reranked, keywords)

        # Ground truth match only computed when a ground truth answer is provided
        gt = score_ground_truth_match(response, ground_truth) if ground_truth else None

        # Overall: mean of all scored metrics (ground_truth_match included when available)
        scored_values = [faith, relev, kw, ctx, p5, mrr]
        if gt is not None:
            scored_values.append(gt)
        overall = sum(scored_values) / len(scored_values)

        return {
            'question':             question,
            'query_type':           query_type,
            'ground_truth':         ground_truth,
            'response':             response,
            'faithfulness_llm':     round(faith, 3),
            'answer_relevancy_llm': round(relev, 3),
            'ground_truth_match':   round(gt, 3) if gt is not None else None,
            'keyword_recall':       round(kw, 3),
            'context_relevance':    round(ctx, 3),
            'precision_at_5':       round(p5, 3),
            'mrr':                  round(mrr, 3),
            'latency_ms':           round(latency_ms, 1),
            'overall':              round(overall, 3),
        }

    def _compute_summary(self, results: List[dict]) -> dict:
        """Compute mean, std, min, and max for every metric across all questions.

        Args:
            results: List of per-question result dicts from _run_single().

        Returns:
            Dict mapping each metric name to a sub-dict with 'mean', 'std', 'min', 'max'.
            Also includes 'overall_mean' (float) for quick backward-compatible comparison.
        """
        # Scored metrics that count toward the overall score
        scored_metrics = [
            'faithfulness_llm', 'answer_relevancy_llm', 'ground_truth_match',
            'keyword_recall', 'context_relevance', 'precision_at_5', 'mrr',
        ]
        # Extra metrics shown in the report but not counted in the overall average
        extra_metrics = ['latency_ms', 'overall']

        summary: dict = {}
        for metric in scored_metrics + extra_metrics:
            # Skip None values — ground_truth_match is None when no ground truth was given
            values = [r[metric] for r in results if r.get(metric) is not None]
            if not values:
                continue
            summary[metric] = {
                'mean': round(sum(values) / len(values), 3),
                # stdev requires at least 2 values; use 0.0 for a single test case
                'std':  round(statistics.stdev(values) if len(values) > 1 else 0.0, 3),
                'min':  round(min(values), 3),
                'max':  round(max(values), 3),
            }

        # Top-level overall mean kept for easy access and backward compatibility
        overall_values = [r['overall'] for r in results]
        summary['overall_mean'] = round(sum(overall_values) / len(overall_values), 3)
        return summary

    # ── Private — persistence ────────────────────────────────────────────────
    # Output formatting moved to src/rag/benchmark_report.py (stateless module functions)

    def _export_csv(self, results: List[dict], timestamp: str) -> None:
        """Write the current run's per-question results to a CSV file.

        The CSV is overwritten on each run and contains the latest results only.
        For full run history, use the JSON file (benchmark_results.json).

        Columns: timestamp, question, query_type, ground_truth,
                 all 7 metric scores, latency_ms, overall, response.

        Args:
            results:   List of per-question result dicts.
            timestamp: Formatted timestamp string for this run.
        """
        if not results:
            return

        columns = [
            'timestamp', 'question', 'query_type', 'ground_truth',
            'faithfulness_llm', 'answer_relevancy_llm', 'ground_truth_match',
            'keyword_recall', 'context_relevance', 'precision_at_5',
            'mrr', 'latency_ms', 'overall', 'response',
        ]
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                # Add the run timestamp to every row
                writer.writerow({**r, 'timestamp': timestamp})

    def _save_results(self, runs: list) -> None:
        """Overwrite the JSON results file with the full run history.

        Args:
            runs: List of all benchmark run dicts (timestamp + summary + results).
        """
        with open(self.results_file, 'w') as f:
            json.dump(runs, f, indent=2)

    def _read_results(self) -> list:
        """Load all previous benchmark runs from the JSON history file.

        Returns:
            List of run dicts, or an empty list if the file does not exist or is corrupt.
        """
        if not os.path.exists(self.results_file):
            return []
        try:
            with open(self.results_file) as f:
                return json.load(f)
        except Exception:
            return []
