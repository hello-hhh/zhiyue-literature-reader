# BENCHMARK.md — Benchmark Methodology and Metric Reference

This document covers the full benchmarking system — why the metrics were chosen, how each one is calculated, how to interpret results, and how to add your own test cases.

**Quick navigation:**
- [Overview](#overview)
- [Why 7 metrics](#why-7-metrics)
- [The three quality dimensions](#the-three-quality-dimensions)
- [How each metric is calculated](#how-each-metric-is-calculated)
- [How to interpret the summary statistics](#how-to-interpret-the-summary-statistics)
- [How to interpret drops in the run comparison](#how-to-interpret-drops-in-the-run-comparison)
- [Agent tool benchmark](#agent-tool-benchmark)
- [RAGAS evaluation](#ragas-evaluation)
- [Sample terminal output](#sample-terminal-output)
- [How to add your own test cases](#how-to-add-your-own-test-cases)
- [Test domains and sample files](#test-domains-and-sample-files)

---

## Overview

The benchmark system has three independent evaluation modes:

```bash
python3 main.py --benchmark   # custom 7-metric pipeline + agent tool benchmark
python3 main.py --ragas       # RAGAS LLM-as-judge evaluation (see below)
```

**`--benchmark` Phase 1 — RAG pipeline quality:**
- 15 questions across 4 domains (cat facts, Python language, team members CSV, machine learning)
- Sample files are committed to `benchmark_docs/` — no extra setup needed
- 7 metrics per question: 2 LLM-as-judge + 5 numeric
- Prints per-question table, summary statistics, by-query-type breakdown, and run-over-run delta
- Results saved to `benchmark_results.json` (full history) and `benchmark_results.csv` (latest run)

**`--benchmark` Phase 2 — Agent tool correctness:**
- 18 tests: 5 calculator, 4 sentiment, 3 summarise, 3 translate, 3 topic_search
- Results saved to `tool_benchmark_results.json`

**`--ragas` — RAGAS industry-standard metrics:**
- 4 LLM-as-judge metrics: Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall
- Runs on the same DEFAULT_TEST_CASES as Phase 1
- Requires optional dependencies: `pip install -e ".[eval]"`
- See [RAGAS evaluation](#ragas-evaluation) below

Every `--benchmark` run is compared against the previous one with delta indicators (▲ improved / ▼ regressed / ─ unchanged).

---

## Why 7 Metrics

A RAG system can fail in three independent ways. Using only one or two metrics hides which part is broken.

**Example of why this matters:**

A high faithfulness score with a low MRR means the LLM is faithfully quoting whatever it retrieved — but it retrieved the wrong chunks. You cannot see this failure with faithfulness alone. You need both faithfulness (generation quality) and MRR (retrieval ranking quality) to pinpoint it.

---

## The Three Quality Dimensions

| Dimension | Metrics | What a low score here means |
|-----------|---------|----------------------------|
| **Retrieval quality** | Context Relevance, Precision@5, MRR | The system is fetching the wrong chunks — the answer exists in the documents but was not retrieved |
| **Generation quality** | Faithfulness, Answer Relevancy | The retrieved context was good but the LLM ignored it, hallucinated, or went off-topic |
| **Factual accuracy** | Ground Truth Match, Keyword Recall | The answer is roughly correct but missing specific facts, numbers, or expected phrasing |

**Diagnostic patterns:**

| Pattern | What it usually means |
|---------|----------------------|
| Low faithfulness + high context relevance | Model is hallucinating despite good chunks. Check the system prompt and SIMILARITY_THRESHOLD. |
| Low context relevance + low precision@5 | Retrieval is broken upstream. Check document indexing and embedding quality. |
| Low MRR + adequate context relevance | Relevant chunks exist but are ranking low. Reranking quality may have degraded. |
| Low ground truth match + high faithfulness | Answer is correct but phrased differently from the expected answer. May not be a real problem. |

---

## How Each Metric is Calculated

### Faithfulness (LLM-as-judge)

Asks the language model to grade whether every claim in the response is grounded in the retrieved context:

```
LLM rates 1–5 → normalised to 0.0–1.0

1 = answer invents facts not in the context (hallucination)
5 = every claim comes directly from the retrieved chunks
```

More reliable than word-overlap alone because it catches faithful paraphrases — which word-overlap would penalise as hallucination (*"Felines rest 12–16 hours daily"* faithfully captures *"cats sleep 16 hours"* but shares few words).

---

### Answer Relevancy (LLM-as-judge)

Asks the language model to grade whether the answer directly addresses the question:

```
LLM rates 1–5 → normalised to 0.0–1.0

1 = answer ignores the question entirely
5 = answer completely and directly addresses the question
```

---

### Ground Truth Match

F1 word overlap between the response and the known correct answer:

```
precision = words in common / words in response
recall    = words in common / words in ground truth
F1        = 2 × precision × recall / (precision + recall)
```

Requires a `ground_truth` field in each test case. Measures lexical similarity to the expected answer — not semantic similarity. A fluent paraphrase scores low here even if correct.

Common stopwords are excluded from the comparison (a, the, is, are, was, etc.).

---

### Keyword Recall

Fraction of expected answer keywords found in the response:

```
keyword_recall = keywords found in response / total expected keywords
```

Each test case supplies an `expected_keywords` list. For *"How many hours do cats sleep?"* the list is `['sleep', '16']`. If both appear in the response, recall = 1.0.

---

### Context Relevance

Mean cosine similarity of the top reranked chunks to the query:

```
context_relevance = mean(cosine_similarity(chunk_embedding, query_embedding))
                    over the top TOP_RERANK chunks
```

Measures retrieval quality independently of the LLM. A low score here means the problem is upstream in retrieval, not in generation.

---

### Precision@5

Fraction of the top-5 retrieved chunks that contain at least one expected keyword:

```
precision@5 = relevant chunks in top 5 / 5
```

A chunk is "relevant" if it contains at least one expected keyword from the test case.

**Why both Precision@5 and MRR:**

- **Precision@5:** *"Of the 5 chunks I retrieved, how many were useful?"* — measures coverage
- **MRR:** *"How far down did I have to look for the first useful chunk?"* — measures ranking quality

A system can have high Precision@5 but low MRR if it retrieves many relevant chunks but buries them behind irrelevant ones at the top. The reranker is specifically designed to fix this — a reranking improvement shows up as an MRR increase before Precision@5.

---

### MRR (Mean Reciprocal Rank)

How high up the list the first relevant chunk appeared:

```
MRR = 1 / rank_of_first_relevant_chunk

MRR = 1.00 → the very first chunk was relevant
MRR = 0.50 → the second chunk was the first relevant one
MRR = 0.33 → the third chunk was the first relevant one
MRR = 0.00 → no relevant chunk found
```

---

### Overall

Mean of all 7 scored metrics (latency is excluded):

```
overall = mean(faithfulness, answer_relevancy, ground_truth_match,
               keyword_recall, context_relevance, precision_at_5, mrr)
```

---

## How to Interpret the Summary Statistics

| Statistic | What it means |
|-----------|--------------|
| **Mean** | Average across all test questions — the headline score |
| **Std** | Standard deviation — how consistent the score is. High Std = pipeline performs very differently on different questions. Low Std with high Mean is ideal. |
| **Min / Max** | Worst and best individual question scores. A large gap between Min and Max confirms the inconsistency the Std hints at. |
| **latency_ms** | End-to-end wall-clock time per question, from query expansion to last generated token. Does not include LLM-as-judge scoring (evaluation overhead, not pipeline latency). |

---

## How to Interpret Drops in the Run Comparison

| Metric drops | What it usually means |
|---|---|
| **faithfulness_llm** | LLM generating content not grounded in documents. Try raising SIMILARITY_THRESHOLD (0.40 → 0.50) |
| **answer_relevancy_llm** | LLM is digressing. Often caused by ambiguous test questions or a recently changed system prompt |
| **ground_truth_match** | Response wording drifted from expected. The answer may still be correct but phrased differently |
| **keyword_recall** | Model is omitting expected facts. Check whether the keywords are present in the indexed documents |
| **context_relevance** | Retrieval is degraded. New documents may have diluted the BM25 index or the embedding model is struggling with a new content type |
| **precision_at_5** | Retrieved chunks no longer contain the answer. Check whether the right document is indexed |
| **mrr** | The first relevant chunk is ranking lower. Reranking quality may have dropped — check the reranker model |

---

## Agent Tool Benchmark

**Why separate from the RAG pipeline benchmark:**

The RAG pipeline benchmark tests retrieval and generation quality — it uses the LLM as both the system under test and the judge. The tool benchmark tests deterministic correctness of the agent tools — no LLM-as-judge, direct pass/fail checks.

**The 18 tests:**

| Tool | # Tests | What passes |
|------|---------|-------------|
| **calculator** | 5 | Arithmetic results match exactly (with tolerance for floats); unsafe characters (letters) are rejected with an error message |
| **sentiment** | 4 | Output contains all 4 required fields (`Sentiment:`, `Tone:`, `Key phrases:`, `Explanation:`) and a valid label (Positive/Negative/Neutral/Mixed) |
| **summarise** | 3 | Output is non-empty and contains at least one key term from the input |
| **translate** | 3 | Output is non-empty; tests cover direct translation (long input ≥15 words) and no-language-prefix default to English |
| **topic_search** | 3 | Network mocked offline; real chunkers + real `store.add_chunks()` called; verified both fetched > 0 and added > 0 chunks |

**Calculator tests are fully deterministic** — no LLM call, direct eval. Sentiment, summarise, and translate tests call the language model once per test. Topic_search tests mock the network layer and run real chunkers offline.

**Calculator allowed characters:** `0123456789+-*/(). ` — note that `**` (power) passes (two asterisks are both in the allowed set) but `sqrt(4)` fails (letters `s`, `q`, `r`, `t` are not in the allowed set).

---

## RAGAS Evaluation

RAGAS adds a second layer of evaluation using metrics that are widely recognised in the RAG research community. Where the custom benchmark uses hand-crafted scoring functions, RAGAS uses the LLM itself as the judge for each metric.

### Why Both

| | Custom benchmark (`--benchmark`) | RAGAS (`--ragas`) |
|-|----------------------------------|-------------------|
| Speed | Faster — some metrics are numeric | Slower — every metric calls the LLM |
| Determinism | Numeric metrics are exact; LLM scores vary slightly | All scores vary with temperature |
| Interpretability | Each metric formula is transparent | Scores follow academic RAGAS definitions |
| Dependencies | No extras needed | `pip install -e ".[eval]"` |
| Signal | 7 metrics across retrieval + generation | 4 metrics with academic alignment |

Having both gives a complete picture: the custom benchmark catches regressions quickly, and RAGAS provides scores that can be compared against published benchmarks.

### The 4 RAGAS Metrics

| Metric | What it measures | How it is computed |
|--------|-----------------|-------------------|
| **Faithfulness** | Every claim in the answer is grounded in the retrieved context | LLM decomposes the answer into claims; checks each claim against context |
| **ResponseRelevancy** | The answer directly addresses the question asked | Embedding cosine similarity between generated question and original question |
| **ContextPrecision** | The most relevant chunks are ranked highest | LLM judges whether each retrieved chunk was needed to answer the question |
| **ContextRecall** | The retrieved context covers the ground truth | LLM checks how much of the ground truth is covered by the retrieved chunks |

### Installation

```bash
pip install -e ".[eval]"
```

This installs `ragas>=0.2.0`, `langchain-ollama`, and `datasets` as an optional dependency group. The rest of the application runs normally if these are not installed.

### Running

```bash
python3 main.py --ragas
```

The same `DEFAULT_TEST_CASES` from `Benchmarker` are used. To pass custom test cases, call `run_ragas_evaluation()` directly:

```python
from src.rag.ragas_eval import run_ragas_evaluation, print_ragas_results

result = run_ragas_evaluation(store, test_cases=[
    {'question': 'What is Alice's role?', 'ground_truth': 'Senior Engineer'},
])
print_ragas_results(result)
```

Each test case requires `question` and `ground_truth`. The `ground_truth` is used by ContextRecall to verify how much of the expected answer is present in the retrieved context.

---

## Sample Terminal Output

```
════════════════════════════════════════════════════════════════════════
  RAG PIPELINE BENCHMARK  ·  2026-03-28 00:55:24  ·  15 questions
════════════════════════════════════════════════════════════════════════

  [1/15] How many hours do cats sleep per day?
         faith=1.00  relev=0.75  gt=0.62  kw=1.00  ctx=0.78  p@5=0.40  mrr=1.00  11588ms
  [2/15] Can cats see in dim light?
         faith=1.00  relev=1.00  gt=0.45  kw=1.00  ctx=0.78  p@5=0.60  mrr=1.00  12395ms
  ...

════════════════════════════════════════════════════════════════════════
  SUMMARY
════════════════════════════════════════════════════════════════════════

  Metric                     Mean    Std    Min    Max  Bar
  ──────────────────────────────────────────────────────────────────
  faithfulness (LLM)      0.967  0.088  0.750  1.000  [███████████████████░]
  answer_relevancy (LLM)  0.867  0.160  0.500  1.000  [█████████████████░░░]
  ground_truth_match      0.748  0.218  0.421  1.000  [██████████████░░░░░░]
  keyword_recall          0.967  0.129  0.500  1.000  [███████████████████░]
  context_relevance       0.656  0.144  0.276  0.779  [█████████████░░░░░░░]
  precision_at_5          0.453  0.207  0.200  0.800  [█████████░░░░░░░░░░░]
  mrr                     1.000  0.000  1.000  1.000  [████████████████████]

  latency_ms              11665   1085   9760  13030  ms
  ──────────────────────────────────────────────────────────────────
  overall                  0.808  0.067  0.635  0.901  [████████████████░░░░]

════════════════════════════════════════════════════════════════════════
  vs PREVIOUS RUN
════════════════════════════════════════════════════════════════════════
  keyword_recall             1.000 →  0.967  ▼0.033
  context_relevance          0.719 →  0.656  ▼0.063
  overall                    0.721 →  0.808  ▲0.087

════════════════════════════════════════════════════════════════════════
  AGENT TOOL BENCHMARK
════════════════════════════════════════════════════════════════════════
  #    Tool           Status  Input                                     Note
  ────────────────────────────────────────────────────────────────────────
  1    calculator     PASS    '6 * 7'                                   6 * 7 = 42
  2    calculator     PASS    '(100 + 50) / 3'                          (100 + 50) / 3 ≈ 50.0
  3    calculator     PASS    'sqrt(4)'                                 letter chars not allowed
  4    calculator     PASS    '365 * 24'                                365 * 24 = 8760
  5    calculator     PASS    '15% of 85000'                            15% of 85000 = 12750.0
  6    sentiment      PASS    'I absolutely love this product...'       4 fields present
  7    sentiment      PASS    'This is a terrible experience...'        4 fields present
  8    sentiment      PASS    'Water boils at 100 degrees Celsius...'   4 fields present
  9    sentiment      PASS    'I absolutely love this product...'       valid label
  10   summarise      PASS    'Python was created by Guido...'          mentions Python or Guido
  11   summarise      PASS    'Machine learning is a subset...'         mentions machine learning
  12   summarise      PASS    'The sky is blue. The sun is yellow.'     non-empty summary
  13   translate      PASS    'Spanish: The sky is blue and the sun..'  non-empty Spanish translation
  14   translate      PASS    'French: Python is a high-level progra..'  non-empty French translation
  15   translate      PASS    'El cielo es azul y el sol brilla...'     default to English, non-empty
  16   topic_search   PASS    'Python programming language'             fetch>0 chunks / add>0 chunks
  17   topic_search   PASS    'machine learning algorithms'             fetch>0 chunks / add>0 chunks
  18   topic_search   PASS    'space exploration NASA'                  fetch>0 chunks / add>0 chunks

────────────────────────────────────────────────────────────────────────
  Total: 18/18 passed  (100%)
    calculator      5/5
    sentiment       4/4
    summarise       3/3
    translate       3/3
    topic_search    3/3

  Saved  → tool_benchmark_results.json
════════════════════════════════════════════════════════════════════════
```

---

## How to Add Your Own Test Cases

```python
from src.rag.benchmarker import Benchmarker
from src.rag.vector_store import VectorStore

store = VectorStore()
bench = Benchmarker(store)

my_test_cases = [
    {
        'question':          "What is the candidate's most recent job title?",
        'ground_truth':      'The candidate is a Senior Machine Learning Engineer.',
        'expected_keywords': ['engineer', 'machine learning', 'senior'],
        'query_type':        'factual',
    },
    {
        'question':          'What year did the company reach $1M revenue?',
        'ground_truth':      'The company reached $1 million in revenue in 2022.',
        'expected_keywords': ['2022', 'million', 'revenue'],
        'query_type':        'factual',
    },
]

bench.run(test_cases=my_test_cases)
```

Or add them permanently to `DEFAULT_TEST_CASES` in `src/rag/benchmarker.py`.

**Tips for writing good test cases:**
- `ground_truth` — write the ideal one-sentence answer. Used by ground_truth_match and LLM-as-judge.
- `expected_keywords` — include the exact words or numbers you expect (2–4 is ideal). Too many keywords penalise natural phrasing.
- `query_type` — label as `'factual'`, `'comparison'`, or `'summarise'` to see per-type breakdowns.
- Mix question types to get a balanced overall score that reflects real usage.

---

## Test Domains and Sample Files

The 15 default test questions are spread across 4 domains. Sample files are committed to `benchmark_docs/` — no extra setup needed.

| Domain | File | Format | Questions |
|--------|------|--------|-----------|
| Cat facts | (indexed separately in `docs/`) | TXT | 5 — basic factual questions |
| Python language | `benchmark_docs/python-language.txt` | TXT | 4 — Python history and features |
| Team members | `benchmark_docs/team-members.csv` | CSV | 3 — structured data retrieval |
| Machine learning | `benchmark_docs/machine-learning.md` | MD | 3 — ML concept questions |

**Why these 4 domains:**

- **Cat facts** — simple factual retrieval baseline (backward compatible with all previous runs)
- **Python language** — tests TXT format retrieval with factual questions about a well-known domain
- **Team members CSV** — tests structured data retrieval where most RAG systems fail (spreadsheet rows as key=value pairs)
- **Machine learning MD** — tests Markdown format retrieval with conceptual questions

**Evaluating the benchmark with a real domain:**

The infrastructure is already in place — `ground_truth`, `query_type`, and `chunk_directory()`. Only the domain-specific documents and questions need to be added. A production evaluation dataset with 50–100 domain-specific questions, human-verified ground truth answers, and a mix of factual/comparison/summarise query types would give much stronger signal.
