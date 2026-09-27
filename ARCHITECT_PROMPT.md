# Architect Prompt — RAG Agent Refactor

Paste everything below this line into Claude Code.

---

IMPORTANT WORKFLOW:
- Work ONE step at a time
- STOP after each step and wait for my approval
- Do NOT proceed until I say "continue" or "next"
- Show exactly what changed before asking to move on
- Never commit without my explicit "UI looks good, commit"

---

I have a working 1600-line Python RAG application in `rag_app.py`.
Read `CLAUDE.md` fully before doing anything — it is the authoritative
specification written from a complete reading of `rag_app.py`, `README.md`,
and `requirements.txt`. Follow every instruction exactly.

---

## Commercial Standard — Always Apply

> **Every file, every feature, every commit must meet the highest commercial software standard.**
> This applies to code, documentation, repository structure, UI, and tests — always.
> If something exists, keep it at commercial standard. If something is missing, add it to that standard.
> Make the minimum change needed to reach and maintain that standard — no more, no less.

- Code is structured, readable, tested, and documented to production grade
- Repository has SECURITY.md, CHANGELOG.md, issue templates, PR template, and a wiki
- All MD files (README, CONTRIBUTING, DESIGN, ARCHITECTURE, BENCHMARK, CHANGELOG, wiki) are accurate and up to date
- UI meets SaaS-quality standard — see UI Standards section below
- Test suite covers all test types with counts and results visible in README and wiki

---

## Design philosophy

This refactor uses **4 classes and 4 modules**. This is deliberate.

Components with no meaningful state of their own (query classification,
response generation, conversation memory, individual agent tools) become
**private methods** (`_`) on the class that owns the state they operate on.
This reduces indirection and keeps related logic together without
sacrificing clarity.

Do NOT create extra classes. Do NOT create a BaseTool abstract class.
Do NOT create separate classes for Calculator, Summariser, or SentimentAnalyser.
They are private methods on Agent.

---

## Project structure (current)

```
rag/
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── config.py              ← MODULE: constants only
│   │   ├── logger.py              ← MODULE: stateless log functions
│   │   ├── chunkers.py            ← MODULE: text-based chunkers (txt, md, csv, html)
│   │   ├── binary_chunkers.py     ← MODULE: binary format chunkers (pdf, docx, xlsx, xls, pptx)
│   │   ├── url_crawl.py           ← MODULE: URL crawl and DuckDuckGo search
│   │   ├── url_utils.py           ← MODULE: URL type detection, link extraction, topic filter
│   │   ├── document_loader.py     ← CLASS: DocumentLoader
│   │   ├── vector_store.py        ← CLASS: VectorStore
│   │   ├── agent.py               ← CLASS: Agent (6 tools)
│   │   ├── benchmarker.py         ← CLASS: Benchmarker
│   │   ├── benchmark_report.py    ← MODULE: stateless terminal report formatting
│   │   ├── metrics.py             ← MODULE: 7 stateless scoring functions
│   │   └── tool_benchmarks.py     ← MODULE: tool benchmark suite
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── handlers.py            ← MODULE: event handlers (url, file, topic, user input)
│   │   ├── renderers.py           ← MODULE: pure Streamlit render functions
│   │   ├── sidebar.py             ← MODULE: sidebar rendering
│   │   ├── theme.py               ← MODULE: CSS + style constants
│   │   └── session.py             ← MODULE: session state helpers
│   └── cli/
│       ├── __init__.py
│       └── runner.py              ← MODULE: terminal chat, agent, benchmark entry points
├── tests/                         ← 848 local tests (31+ files)
├── huggingface/                   ← HF Space deployment (385 tests)
│   └── src/
│       ├── handlers.py
│       ├── theme.py
│       ├── ui_builder.py          ← build_demo()
│       └── rag/                   ← same structure as src/rag/
├── app.py                         ← Streamlit thin wrapper (<50 lines)
├── main.py                        ← CLI thin wrapper (<50 lines)
├── requirements.txt               ← add lxml only
├── pyproject.toml
├── DESIGN.md
├── CLAUDE.md
└── README.md
```

---

## The 4 classes

### CLASS 1: `DocumentLoader`
Owns all ingestion — chunkers, URL fetching, misplaced detection.

```python
class DocumentLoader:
    def __init__(self)              # loads config constants as instance vars
    # Public
    def ensure_folders(self)
    def scan_all_files(self)        # misplaced file detection
    def chunk_all_documents(self)
    def chunk_url(self, url)        # 4-priority type detection
    # Private
    def _dispatch_chunker(self, file_info)
    def _chunk_txt(self, filepath, filename)
    def _chunk_md(self, filepath, filename)
    def _chunk_pdf(self, filepath, filename)
    def _chunk_docx(self, filepath, filename)   # + table rows + dedup merged cells
    def _chunk_xlsx(self, filepath, filename)
    def _chunk_xls(self, filepath, filename)    # xlrd fallback
    def _chunk_csv(self, filepath, filename)
    def _chunk_pptx(self, filepath, filename)
    def _chunk_html(self, filepath, filename)
    def _truncate_chunk(self, text)             # 300 words OR 1200 chars
```

### CLASS 2: `VectorStore`
Owns ChromaDB, BM25, hybrid search, reranking, query pipeline,
response generation, and conversation history.

```python
class VectorStore:
    def __init__(self)
    # Public
    def build_or_load(self, chunks)
    def add_chunks(self, chunks, id_prefix)
    def rebuild_bm25(self, all_chunks)
    def run_pipeline(self, query, streamlit_mode=False)
    def stream_response(self, stream)
    def clear_conversation(self)
    # Private — vector/search
    def _embed(self, text)
    def _truncate_for_embedding(self, text)     # 200 words AND 1200 chars
    def _cosine_similarity(self, a, b)
    def _hybrid_retrieve(self, queries, top_n, alpha=0.5)
    def _rerank(self, query, candidates, top_n)
    def _rerank_prompt(self, query, entry)      # 7 variants — preserve exactly
    # Private — query
    def _classify_query(self, query)            # summarise→comparison→factual→general
    def _expand_query(self, query)              # LLM 2 rewrites + original
    def _check_confidence(self, results)
    def _smart_top_n(self, query_type)
    # Private — response
    def _build_instruction_prompt(self, context)
    def _source_label(self, entry)
    def _synthesize(self, question, context)
    def _filter_hallucination(self, response)
```

### CLASS 3: `Agent`
Owns ReAct loop and all 6 tools as private methods.

```python
class Agent:
    AGENT_SYSTEM_PROMPT: str        # class constant — preserve every word
    def __init__(self, store: VectorStore)
    # Public
    def run(self, user_query, streamlit_mode=False)
    # Private — loop
    def _parse_tool_call(self, response_text)   # two regex patterns
    def _dispatch_tool(self, tool_name, tool_arg)
    def _synthesize_final_answer(self, query, context)
    def _fast_path_summarise(self, query, streamlit_mode)
    def _fast_path_sentiment(self, query, streamlit_mode)
    # Private — tools (NOT separate classes)
    def _tool_rag_search(self, query)
    def _tool_calculator(self, expression)      # safe eval
    def _tool_summarise(self, text)             # adaptive length
    def _tool_sentiment(self, text_or_query)    # optional RAG search
    def _tool_translate(self, language_and_text)  # any→any language; short queries search first
```

### CLASS 4: `Benchmarker`
Owns all evaluation, scoring, and results management.

```python
class Benchmarker:
    DEFAULT_TEST_CASES: list        # 5 cat facts — preserve exactly
    def __init__(self, store: VectorStore)
    # Public
    def run(self, test_cases=None)
    # Private
    def _score_faithfulness(self, response, reranked)
    def _score_relevancy(self, question, response)
    def _score_keyword_recall(self, response, keywords)
    def _score_context_relevance(self, reranked)
    def _save_results(self, results)
    def _compare_runs(self, current, previous)
    def _read_results(self)
```

---

## The modules

```python
# config.py — constants only, no functions
EMBEDDING_MODEL, LANGUAGE_MODEL, DOCS_ROOT, DOC_FOLDERS,
EXT_TO_TYPE, CHROMA_DIR, CHROMA_COLLECTION, LOG_FILE,
BENCHMARK_FILE, SIMILARITY_THRESHOLD, TOP_RETRIEVE, TOP_RERANK,
TXT_CHUNK_SIZE, TXT_CHUNK_OVERLAP, PDF_CHUNK_SENTENCES,
DOCX_CHUNK_PARAS, PPTX_CHUNK_SLIDES, HTML_CHUNK_SENTENCES

# logger.py — stateless functions
def log_interaction(query, qtype, chunks_used, sim_scores, response)
def _read_log() -> list
def _write_log(entries: list)

# chunkers.py — text-based chunker functions (txt, md, csv, html)
# binary_chunkers.py — binary format chunker functions (pdf, docx, xlsx, xls, pptx)
# url_crawl.py — stateless URL crawl and DuckDuckGo search functions
# url_utils.py — URL type detection, link extraction, topic filtering helpers

# ui/theme.py — constants
CSS: str            # IBM Plex Mono stylesheet — preserve every rule
BADGE_CLASSES: dict
CONFIDENCE_BADGE: dict
AVATAR: dict

# ui/session.py — functions
def init_session_state()
def get_active_bm25(base_bm25)

# ui/renderers.py — pure Streamlit render functions
def render_header()
def render_footer()
def render_chat_history(display)
def render_clear_button()
def render_mode_selector()
```

---

## Entry points

### `main.py` (~40 lines)
```python
from src.rag.config import *
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.rag.agent import Agent
from src.rag.benchmarker import Benchmarker
import argparse

def initialize():
    loader = DocumentLoader()
    loader.ensure_folders()
    chunks = loader.chunk_all_documents()
    store = VectorStore()
    store.build_or_load(chunks)
    return loader, store

if __name__ == '__main__':
    loader, store = initialize()
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--agent', action='store_true')
    args = parser.parse_args()
    if args.benchmark:
        Benchmarker(store).run()
    elif args.agent:
        agent = Agent(store)
        # terminal agent loop — preserve exact output format from original
    else:
        # terminal chat loop — preserve exact output format from original
```

### `app.py` (~45 lines)
```python
# @st.cache_resource initialize()
# ui/session.init_session_state()
# ui/theme.CSS applied
# run_streamlit() — full UI logic
# Preserve: deferred _needs_rerun pattern, active_bm25 logic
```

---

## pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "rag-agent"
version = "1.0.0"
description = "A fully local, production-grade RAG system"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "ollama", "rank-bm25", "chromadb", "streamlit", "requests",
    "pymupdf", "python-docx", "openpyxl", "xlrd", "python-pptx",
    "beautifulsoup4", "lxml",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "pytest-mock"]

[project.urls]
Homepage = "https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent"
Demo = "https://huggingface.co/spaces/anjanatiha2024/ragdoll"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.setuptools.packages.find]
where = ["src"]
```

---

## DESIGN.md — write this at Step 9

Contents (write based on the actual code):
- Why 4 classes instead of more granular decomposition
- What each class owns and why those responsibilities belong together
- Why tools are private methods on Agent, not separate classes
- Tradeoffs: ChromaDB vs Pinecone, local vs API, BM25 vs pure dense
- How you'd scale this to production (async, batching, caching, distributed)
- What the type-aware reranker solves that a generic reranker doesn't
- Benchmark metrics — what they measure and what the scores mean
- What you'd do differently with more time / larger team

---

## Critical facts — get these right

```python
# Embed call — mock ollama.embed, NOT ollama.embeddings
ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]

# URL type detection — 4 priorities
# 1. Content-Type header
# 2. File extension in URL path (strip ?query first)
# 3. PDF magic bytes: content[:4] == b'%PDF'
# 4. Default to 'html'

# VectorStore rebuild
if existing >= len(chunks): return  # skip
if existing > 0: delete all then rebuild

# Agent parse_tool_call — preserve both patterns
re.search(r'(?i)TOOL:\s*(\w+)\s*\(\s*(.+?)\s*\)', text, re.DOTALL)  # with parens
re.search(r'(?i)TOOL:\s*(\w+)\s+(.+)', text)                         # without parens

# Hallucination filter — preserve both lists
_no_info_phrases = ["there is no information", "i couldn't find", ...]
_hallucination_pivots = ["however,", "but i can", "but,", ...]

# Deferred rerun — preserve
_needs_rerun = False
# ... process ...
_needs_rerun = True
if _needs_rerun: st.rerun()
```

---

## TDD cycle — mandatory for every file

```
1. Stubs only
2. Tests → RED
3. Implement → GREEN
4. Integration tests → GREEN
5. Streamlit tests → GREEN
6. STOP: "Please test UI"
7. Wait for "UI looks good, commit"
8. Commit → next step
```

## Test types — use all of these for every feature

Every feature must have tests from each of the following categories:

| Type | What it checks |
|------|---------------|
| **Unit** | One function in isolation — correct output for known input |
| **Functional** | End-to-end flow through a feature — inputs → expected outputs |
| **Integration** | Multiple components working together (e.g. loader → store → pipeline) |
| **Regression** | Edge cases that broke in the past — prevents them coming back |
| **Boundary** | Minimum, maximum, and zero-value inputs |
| **Negative** | Invalid input, missing data, network errors — must not crash |
| **Parametrized combination** | Cross-product of modes × doc types × query types |

Tests live in `tests/` (local) AND `huggingface/tests/` (HF space). Both must pass before any commit.

## Always keep MD and requirements files up to date

After every feature, fix, or refactor:
- Update `README.md` if the public interface, feature list, or test counts changed
- Update `CLAUDE.md` / `ARCHITECT_PROMPT.md` if a new rule or pattern was established
- Update `requirements.txt` / `huggingface/requirements.txt` if dependencies changed
- Do this as part of the same task — not as a separate step, not as a follow-up

---

## Mock strategy

```python
ollama.embed  → {'embeddings': [[0.1, 0.2, ...]]}
ollama.chat   → {'message': {'content': 'mock'}}
chromadb      → chromadb.EphemeralClient()
requests.get  → Mock(.content, .headers, .encoding, .raise_for_status)

# Never mock:
fitz, python-docx, openpyxl, xlrd, python-pptx, beautifulsoup4
BM25Okapi, truncation, misplaced detection, calculator eval
```

---

## 11-step refactor order

```
Step 0:  structure + pyproject.toml + __init__.py + lxml → STOP
Step 1:  config.py → STOP
Step 2:  logger.py → STOP
Step 3:  ui/theme.py + ui/session.py → browser check → commit
Step 4:  DocumentLoader → integration tests → browser check → commit
Step 5:  VectorStore → integration tests → browser check → commit
Step 6:  Agent → integration tests → browser check → commit
Step 7:  Benchmarker → run --benchmark → STOP → commit
Step 8:  slim app.py + main.py → full UI test → commit
Step 9:  DESIGN.md → STOP → commit
Step 10: full integration tests → show results → STOP → commit
Step 11: final verification → final UI test → final commit
```

---

## Coding Standards

### File size
- **Maximum 500 lines per file.** Split along a clear conceptual boundary if exceeded.
- Entry points (`main.py`, `app.py`) must stay under 50 lines — they wire classes together and nothing else.
- Test files are split by concern: mock-chunk tests, file upload tests, URL ingestion tests.

### Modularity
- **One responsibility per file.** A file that does two things should be two files.
- Classes own state and the operations that depend on that state.
- Stateless operations belong in modules (plain functions, not classes).
- Private methods (`_prefix`) are implementation details — never called from outside the class.
- Do not create helper or utility classes for one-time use. A plain function in a module is simpler.
- Do not write more than ~30 lines in a single method. Extract a private helper if longer.
- Do not nest more than 3 levels of indentation. Flatten with early returns.

### Commenting — PEP 257 / Google style

Every public class and every public method must have a docstring. Private methods get a
one-line comment only when the logic is non-obvious.

**Class docstring — what state the class owns and its public API:**
```python
class VectorStore:
    """Owns all retrieval, search, and response generation.

    State:
        client:       chromadb.PersistentClient
        collection:   ChromaDB collection
        chunks:       list of all indexed chunks
        bm25_index:   BM25Okapi for keyword search
        conversation: multi-turn conversation history
    """
```

**Public method — one-line summary, Args, Returns:**
```python
def run_pipeline(self, query: str, streamlit_mode: bool = False) -> dict:
    """Run the full RAG pipeline for a user query.

    Args:
        query: The user's question.
        streamlit_mode: If True, return pipeline metadata dict.
            If False, stream response to the terminal.

    Returns:
        dict with keys: response, retrieved, reranked,
        is_confident, best_score, query_type.
    """
```

**Private method — one-line comment only when non-obvious:**
```python
def _cosine_similarity(self, a: list, b: list) -> float:
    # Manual dot product — avoids numpy import for a single call.
    dot = sum(x * y for x, y in zip(a, b))
    ...
```

**Inline comment — explains WHY, not what:**
```python
# Greedy (.+) without DOTALL: captures up to the last ')' on the line.
# Non-greedy (.+?) would stop at the first ')' and truncate nested parens.
match = re.search(r'(?i)TOOL:\s*(\w+)\s*\(\s*(.+)\s*\)', response_text)
```

Do not restate the code in comments:
```python
# BAD
i += 1  # increment i by 1

# GOOD
i += 1  # skip the header row
```

### Naming
| Kind | Convention | Example |
|------|------------|---------|
| Class | `PascalCase` | `DocumentLoader` |
| Public method / function | `snake_case` | `run_pipeline` |
| Private method | `_snake_case` | `_embed` |
| Constant | `UPPER_SNAKE_CASE` | `TOP_RETRIEVE` |
| Local variable | `snake_case` | `qt`, `bs`, `vs` |

### What to avoid
- Do not add error handling for things that cannot fail in normal operation.
  Only validate at system boundaries: user input, external APIs, file I/O.
- Do not repeat the same import inside multiple functions. Import once at the top.
- Do not add docstrings or comments to code you did not write or change.
- Do not design for hypothetical future requirements. Write the minimum needed now.

---

## UI Standards — Always Apply

Every UI screen, panel, or component must meet **the highest commercial product standard**.

- **Pleasant color palette** — sea blue + teal green gradient (`--blue-600: #0891b2`, `--teal-600: #0d9488`). No raw Streamlit or Gradio defaults.
- **Clean visual hierarchy** — clear headings, labels, consistent spacing, intentional whitespace
- **Helpful notes on every input** — placeholder text with real examples, captions, and `help=` tooltips so the user never has to guess what a control does
- **Convenient UX** — smart defaults, inline guidance, success/error feedback after every action
- **Elegant proportions** — group related things, separate unrelated things; never cram controls together
- **SaaS-quality standard** — aim for the level of polish seen in Hugging Face Spaces showcases and commercial tools
- **Concise UI text** — all labels, headings, captions, and help text must be short and informative. No padding, no repetition. If it takes more than one line to read, it is too long.

Any feature that affects the user's workflow must have a short note or caption explaining its purpose and how to use it effectively.

---

## Pre-commit MD update rule

Before every final commit, update all relevant MD files — one at a time, in this order:

1. `README.md`
2. `CLAUDE.md`
3. `ARCHITECT_PROMPT.md`
4. `DESIGN.md`
5. `CONTRIBUTING.md`
6. `docs_technical/ARCHITECTURE.md`
7. `docs_technical/BENCHMARK.md`
8. `huggingface/README.md`

For each file: read it, show the proposed changes and why, then wait for approval before writing. Do not batch updates. Do not skip files that may be affected. Do not commit until all relevant files have been reviewed and approved.

---

## Absolute rules

1. `rag_app.py` is READ-ONLY — never modify it, only extract from it
2. Copy, do not rewrite — copy logic from `rag_app.py` verbatim.
   Only change what is strictly necessary to make it work in its new
   location (e.g. adding self., updating import paths, passing deps
   via constructor). No simplifying, improving, renaming, reformatting.
3. Change structure, not behavior
2. One file per commit
3. TDD for every file — no exceptions
4. GREEN before asking me to review
5. Browser UI check before every commit
6. Commit only after "UI looks good, commit"
7. Mock `ollama.embed` — NOT `ollama.embeddings`
8. No extra classes beyond the 4
9. No BaseTool, no Calculator class, no Summariser class, no SentimentAnalyser class
10. No packages added to requirements.txt (lxml only)
11. No proceed without my "continue"

---

## Start here — Step 0

Create the full folder structure, all empty `__init__.py` files,
`pyproject.toml`, and add `lxml` to `requirements.txt`.
List every file and folder created.
STOP — do not touch `rag_app.py`.
Wait for "continue".
