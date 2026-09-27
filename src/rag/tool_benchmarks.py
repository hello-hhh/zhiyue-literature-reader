"""tool_benchmarks.py — Stateless functions for benchmarking all 6 agent tools.

WHY THIS FILE EXISTS:
    The Benchmarker class tests the RAG pipeline (retrieval + generation quality).
    Agent tools have different evaluation criteria:
        - calculator:   exact numeric correctness
        - sentiment:    format compliance + valid label
        - summarise:    keyword coverage in output
        - translate:    non-empty output containing translation
        - topic_search: chunks returned from DocumentLoader.chunk_topic_search()

    These concerns belong in a module (no class state needed) so they do not
    push benchmarker.py over the 500-line file-size limit.

USAGE:
    from src.rag.tool_benchmarks import run_tool_benchmarks
    run_tool_benchmarks(store)   # prints results to the terminal and saves to JSON

OUTPUT FILES:
    tool_benchmark_results.json  — full per-test results; all runs are appended
                                   (same format as benchmark_results.json)

OUTPUT COLUMNS:
    Tool         — which tool was tested
    Input        — the input given to the tool
    Pass/Fail    — whether the result met the success criterion
    Note         — what a passing result should look like
"""

import json
import os
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

from src.rag.agent import Agent
from src.rag.config import TOOL_BENCHMARK_FILE
from src.rag.vector_store import VectorStore

__all__ = ['TOOL_TEST_CASES', 'run_tool_benchmarks']

# Width of separator lines used in the terminal report
_LINE_WIDTH = 72


# ── Private check helpers ──────────────────────────────────────────────────────
# Defined before TOOL_TEST_CASES so they can be referenced directly in the list.

def _check_sentiment_format(result: str) -> bool:
    """Return True if the sentiment output contains all 4 required fields.

    The Agent's _tool_sentiment() must return a result with exactly these labels:
        Sentiment:   (Positive / Negative / Neutral / Mixed)
        Tone:        (one short phrase)
        Key phrases: (2-4 phrases)
        Explanation: (1-2 sentences)

    We check for the field labels (not the values) so the test works regardless
    of what the LLM says about a particular piece of text.
    """
    required_fields = ['Sentiment:', 'Tone:', 'Key phrases:', 'Explanation:']
    return all(field in result for field in required_fields)


def _check_valid_sentiment_label(result: str) -> bool:
    """Return True if the result contains at least one valid sentiment category label."""
    valid_labels = ['Positive', 'Negative', 'Neutral', 'Mixed']
    return any(label in result for label in valid_labels)


def _check_calculator_approx(expected: float, tolerance: float = 0.01):
    """Return a check function that verifies numeric output is within tolerance of expected.

    Args:
        expected:  The correct numeric answer.
        tolerance: How close the actual output must be to the expected value.

    Returns:
        A callable(result_str) → bool.
    """
    def check(result: str) -> bool:
        try:
            return abs(float(result.strip()) - expected) < tolerance
        except (ValueError, TypeError):
            return False
    return check


# ── Test case definitions ──────────────────────────────────────────────────────
# Each entry describes one test for one tool.
# Required keys:
#   tool      — which Agent tool to invoke ('calculator', 'sentiment', 'summarise')
#   input     — argument string passed to the tool
#   check     — callable(result_str) → bool; True means the test passed
#   note      — short human-readable description of the success criterion

TOOL_TEST_CASES: List[dict] = [

    # ── Calculator: deterministic arithmetic ────────────────────────────────
    # Results must match exactly because eval() is deterministic.
    {
        'tool':  'calculator',
        'input': '6 * 7',
        'check': lambda r: r.strip() == '42',
        'note':  '6 * 7 = 42',
    },
    {
        'tool':  'calculator',
        'input': '(100 + 50) / 3',
        'check': _check_calculator_approx(50.0),
        'note':  '(100 + 50) / 3 ≈ 50.0',
    },
    {
        'tool':  'calculator',
        'input': 'sqrt(4)',
        # Letters like 's','q','r','t' are not in the allowed character set
        # so the calculator must reject this and return an error message.
        'check': lambda r: 'error' in r.lower() or 'unsafe' in r.lower(),
        'note':  "letter chars not allowed — must return an error message",
    },
    {
        'tool':  'calculator',
        'input': '365 * 24',
        'check': lambda r: r.strip() == '8760',
        'note':  '365 * 24 = 8760',
    },
    {
        'tool':  'calculator',
        'input': '15% of 85000',
        # % of is normalised to (15/100*85000) = 12750.0 before eval
        'check': _check_calculator_approx(12750.0, tolerance=0.5),
        'note':  '15% of 85000 = 12750.0',
    },

    # ── Sentiment: output must have all 4 required fields and a valid label ──
    {
        'tool':  'sentiment',
        'input': 'I absolutely love this product. It works perfectly every time.',
        'check': _check_sentiment_format,
        'note':  'must contain Sentiment, Tone, Key phrases, Explanation fields',
    },
    {
        'tool':  'sentiment',
        'input': 'This is a terrible experience. Nothing works and support is useless.',
        'check': _check_sentiment_format,
        'note':  'must contain Sentiment, Tone, Key phrases, Explanation fields',
    },
    {
        'tool':  'sentiment',
        'input': 'Water boils at 100 degrees Celsius at sea level.',
        'check': _check_sentiment_format,
        'note':  'must contain Sentiment, Tone, Key phrases, Explanation fields',
    },
    {
        'tool':  'sentiment',
        'input': 'I absolutely love this product. It works perfectly every time.',
        'check': _check_valid_sentiment_label,
        'note':  'must contain a valid label (Positive/Negative/Neutral/Mixed)',
    },

    # ── Summarise: output must contain key terms from the input ─────────────
    {
        'tool':  'summarise',
        'input': (
            'Python was created by Guido van Rossum. '
            'It was first released in 1991. '
            'Python emphasizes readability and uses significant indentation. '
            'It supports object-oriented, procedural, and functional programming.'
        ),
        'check': lambda r: 'python' in r.lower() or 'guido' in r.lower(),
        'note':  'summary must mention Python or Guido',
    },
    {
        'tool':  'summarise',
        'input': (
            'Machine learning is a subset of artificial intelligence. '
            'It enables computers to learn from data without being explicitly programmed. '
            'Common algorithms include decision trees, neural networks, and support vector machines. '
            'Supervised learning uses labelled data. Unsupervised learning finds hidden patterns.'
        ),
        'check': lambda r: 'learn' in r.lower() or 'machine' in r.lower(),
        'note':  'summary must mention machine learning or learning',
    },
    {
        'tool':  'summarise',
        # Very short input — must still produce a non-empty response
        'input': 'The sky is blue. The sun is yellow.',
        'check': lambda r: len(r.strip()) > 0,
        'note':  'must return a non-empty summary for short input',
    },

    # ── Translate: output must be non-empty and differ from the input ────────
    # Long input (≥15 words) → translated directly without RAG search.
    {
        'tool':  'translate',
        'input': (
            'Spanish: The sky is blue and the sun is shining brightly '
            'over the mountains today in the afternoon.'
        ),
        'check': lambda r: len(r.strip()) > 0,
        'note':  'must return a non-empty Spanish translation',
    },
    {
        'tool':  'translate',
        'input': (
            'French: Python is a high-level programming language that emphasizes '
            'readability and supports multiple programming paradigms including '
            'object-oriented and functional styles.'
        ),
        'check': lambda r: len(r.strip()) > 0,
        'note':  'must return a non-empty French translation',
    },
    {
        'tool':  'translate',
        # Input with no colon — should default to English with non-empty output
        'input': (
            'El cielo es azul y el sol brilla intensamente sobre las montanas '
            'en esta hermosa tarde de verano en el campo.'
        ),
        'check': lambda r: len(r.strip()) > 0,
        'note':  'no language prefix — should default to English, non-empty output',
    },

    # ── Topic search: full pipeline — search → fetch → chunk → add to store ──
    # Mirrors the URL ingestion test approach: network layer mocked offline,
    # real DocumentLoader + real chunkers + real VectorStore.add_chunks().
    # check() receives "<fetched_chunks>/<added_chunks>" so we can verify both.
    {
        'tool':  'topic_search',
        'input': 'Python programming language',
        # Verify search returned chunks AND they were added to the store
        'check': lambda r: all(int(n) > 0 for n in r.split('/')),
        'note':  'fetch>0 chunks / add>0 chunks to store (mocked network)',
    },
    {
        'tool':  'topic_search',
        'input': 'machine learning algorithms',
        'check': lambda r: all(int(n) > 0 for n in r.split('/')),
        'note':  'fetch>0 chunks / add>0 chunks to store (mocked network)',
    },
    {
        'tool':  'topic_search',
        # depth=1, 1 result — same options as the default UI setting
        'input': 'space exploration NASA',
        'check': lambda r: all(int(n) > 0 for n in r.split('/')),
        'note':  'fetch>0 chunks / add>0 chunks to store (mocked network)',
    },
]


# ── Public ────────────────────────────────────────────────────────────────────

def run_tool_benchmarks(store: VectorStore) -> dict:
    """Run all agent tool test cases and print a scored report to the terminal.

    Creates a fresh Agent for each test so tool state does not carry over
    between tests. Calculator tests do not call the LLM. Sentiment and
    summarise tests each call the language model once.

    Args:
        store: An already-initialised VectorStore. Needed by the sentiment
               tool (it calls _expand_query and _hybrid_retrieve internally).

    Returns:
        Dict with keys: total (int), passed (int), failed (int),
        pass_rate (float 0–1), results (list of per-test dicts).
    """
    print('\n' + '═' * _LINE_WIDTH)
    print('  AGENT TOOL BENCHMARK')
    print('═' * _LINE_WIDTH)
    print(f"  {'#':<4} {'Tool':<14} {'Status':<7} {'Input (truncated)':<40} Note")
    print('  ' + '─' * 68)

    results = []

    for index, tc in enumerate(TOOL_TEST_CASES, start=1):
        tool_name  = tc['tool']
        tool_input = tc['input']
        check_fn   = tc['check']
        note       = tc['note']

        # A fresh Agent per test ensures each test starts with clean state
        agent = Agent(store)

        try:
            result_text = _invoke_tool(agent, tool_name, tool_input)
            passed      = bool(check_fn(result_text))
            error_note  = ''
        except Exception as exc:
            result_text = ''
            passed      = False
            error_note  = f"Exception: {exc}"

        status = 'PASS' if passed else 'FAIL'
        # Truncate long inputs so the table stays within a readable width
        display_input = tool_input[:37] + '..' if len(tool_input) > 39 else tool_input
        display_note  = error_note if error_note else note

        print(
            f"  {index:<4} {tool_name:<14} {status:<7} "
            f"{display_input!r:<40}  {display_note}"
        )

        results.append({
            'tool':   tool_name,
            'input':  tool_input,
            'result': result_text,
            'passed': passed,
            'note':   display_note,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total     = len(results)
    n_passed  = sum(1 for r in results if r['passed'])
    n_failed  = total - n_passed
    pass_rate = n_passed / total if total > 0 else 0.0

    print('\n' + '─' * _LINE_WIDTH)
    print(f"  Total: {n_passed}/{total} passed  ({pass_rate:.0%})")

    # Per-tool breakdown so we can see which tool is weakest
    tool_names_seen = sorted({r['tool'] for r in results})
    for name in tool_names_seen:
        tool_results = [r for r in results if r['tool'] == name]
        tool_passed  = sum(1 for r in tool_results if r['passed'])
        print(f"    {name:<14} {tool_passed}/{len(tool_results)}")

    summary = {
        'total':     total,
        'passed':    n_passed,
        'failed':    n_failed,
        'pass_rate': round(pass_rate, 3),
        'results':   results,
    }

    # Append this run to the JSON history file so we can track trends over time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _save_tool_results(summary, timestamp)
    print(f"\n  Saved  → {TOOL_BENCHMARK_FILE}")
    print('═' * _LINE_WIDTH + '\n')

    return summary


# ── Private ───────────────────────────────────────────────────────────────────

def _save_tool_results(summary: dict, timestamp: str) -> None:
    """Append one benchmark run to the JSON history file.

    Each run is stored as a dict with 'timestamp', 'pass_rate', and 'results'.
    Older runs are never deleted so we can compare across time.
    If the file does not exist or is corrupt, starts a fresh list.

    Args:
        summary:   The return value of run_tool_benchmarks (total/passed/results etc.)
        timestamp: Human-readable timestamp string (e.g. '2024-01-15 14:30:00').
    """
    # Read existing history — start fresh if file is missing or unreadable
    existing_runs: list = []
    if os.path.isfile(TOOL_BENCHMARK_FILE):
        try:
            with open(TOOL_BENCHMARK_FILE, 'r', encoding='utf-8') as file_handle:
                existing_runs = json.load(file_handle)
        except (json.JSONDecodeError, OSError):
            existing_runs = []

    # Append the current run to the history list
    run_entry = {
        'timestamp': timestamp,
        'pass_rate': summary['pass_rate'],
        'passed':    summary['passed'],
        'total':     summary['total'],
        'results':   summary['results'],
    }
    existing_runs.append(run_entry)

    # Write the updated list back to disk
    with open(TOOL_BENCHMARK_FILE, 'w', encoding='utf-8') as file_handle:
        json.dump(existing_runs, file_handle, indent=2)


def _invoke_tool(agent: Agent, tool_name: str, tool_input: str) -> str:
    """Call one Agent private tool method directly and return its string output.

    Why call private methods here?  These benchmarks test individual tools in
    isolation — not the full ReAct loop. Calling the loop would add LLM round-
    trips to decide which tool to use, making the benchmark slow and non-
    deterministic. The private-method call is deliberate and analogous to how
    tests call private methods to verify internal behaviour.

    For topic_search, the network is mocked so the test runs offline.
    The return value is the number of chunks as a string (checked with int(r) > 0).

    Args:
        agent:      A freshly created Agent instance.
        tool_name:  One of 'calculator', 'sentiment', 'summarise',
                    'translate', 'topic_search'.
        tool_input: The argument string to pass to the tool.

    Returns:
        The string output of the tool. For topic_search, returns the chunk
        count as a string so the check lambda can cast it with int().

    Raises:
        ValueError: if tool_name is not a recognised tool.
    """
    if tool_name == 'calculator':
        return agent._tool_calculator(tool_input)
    if tool_name == 'sentiment':
        return agent._tool_sentiment(tool_input)
    if tool_name == 'summarise':
        return agent._tool_summarise(tool_input)
    if tool_name == 'translate':
        return agent._tool_translate(tool_input)
    if tool_name == 'topic_search':
        return _invoke_topic_search(agent.store, tool_input)
    raise ValueError(
        f"_invoke_tool: unsupported tool '{tool_name}'. "
        "Supported tools: calculator, sentiment, summarise, translate, topic_search."
    )


def _invoke_topic_search(store: VectorStore, query: str) -> str:
    """Run the full topic search pipeline with a mocked network layer.

    Mirrors the URL ingestion test approach:
      1. Mock DuckDuckGo to return one fake result URL
      2. Mock requests.get to return a fake HTML page
      3. Run real DocumentLoader.chunk_topic_search() — real chunkers
      4. Run real store.add_chunks() + rebuild_bm25() — real VectorStore
      5. Return "<fetched>/<added>" so the check can verify both steps

    Network is mocked so the test runs offline and deterministically.

    Args:
        store: The live VectorStore — add_chunks() is called for real.
        query: The topic search query string.

    Returns:
        "<fetched_count>/<added_count>" e.g. "3/3".
        Returns "0/0" on any exception so the check fails gracefully.
    """
    from src.rag.document_loader import DocumentLoader

    # Fake HTML page — long enough to produce at least one chunk after tag stripping.
    # Five sentences ensures HTML_CHUNK_SENTENCES=5 yields at least one window.
    fake_html_body = (
        '<html><body>'
        '<p>This benchmark page covers the search query topic in detail. '
        'It contains enough sentences for the chunker to create multiple chunks. '
        'The topic search pipeline fetches URLs returned by the search engine. '
        'Each URL is crawled and the content is split into overlapping windows. '
        'The resulting chunks are embedded and added to the vector store index.</p>'
        '</body></html>'
    )

    fake_url = 'https://benchmark-mock.example.com/page'

    # Mock response object — mirrors what requests.get returns for a real URL.
    # Every attribute that crawl_url accesses must be set explicitly so
    # MagicMock does not return another MagicMock where a string or int is expected.
    mock_response                  = MagicMock()
    mock_response.content          = fake_html_body.encode('utf-8')
    mock_response.headers          = {'Content-Type': 'text/html; charset=utf-8'}
    mock_response.encoding         = 'utf-8'
    mock_response.url              = fake_url   # crawl_url reads response.url for final_url
    mock_response.raise_for_status = MagicMock()

    try:
        with patch.object(
            DocumentLoader,
            '_search_duckduckgo_html',
            return_value=[fake_url],
        ), patch('requests.get', return_value=mock_response):
            loader       = DocumentLoader()
            # Step 1 — fetch and chunk (real chunkers, mocked network)
            new_chunks   = loader.chunk_topic_search(
                query,
                num_results=1,
                depth=1,
                max_pages_per_result=1,
            )
            fetched = len(new_chunks)

            # Step 2 — add to store (real VectorStore, same as UI does it)
            if new_chunks:
                store.add_chunks(new_chunks, id_prefix='bench_search')
                store.rebuild_bm25(store.chunks)
            added = fetched  # add_chunks adds all or raises, so added == fetched

        return f"{fetched}/{added}"

    except Exception as exc:
        # "0/0" causes check lambda (int(n) > 0 for both) to fail clearly
        print(f"    [topic_search bench error] {exc}")
        return "0/0"
