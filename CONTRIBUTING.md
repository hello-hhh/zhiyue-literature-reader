# CONTRIBUTING.md — Developer Guide

This guide covers everything needed to contribute to the RAG Agent project.

**Quick navigation:**
- [Development environment setup](#development-environment-setup)
- [Running the tests](#running-the-tests)
- [Test categories](#test-categories)
- [Test files — full breakdown](#test-files--full-breakdown)
- [Mock strategy](#mock-strategy)
- [Code structure](#code-structure)
- [How to add a new file format](#how-to-add-a-new-file-format)
- [How to add a new agent tool](#how-to-add-a-new-agent-tool)
- [Code standards](#code-standards)
- [Reporting security issues](#reporting-security-issues)
- [Pull request guidelines](#pull-request-guidelines)

---

## Development Environment Setup

```bash
git clone https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent.git
cd Retrieval-Augmented-Generation-RAG-Agent

# Python 3.11 required
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate       # Windows: rag_env_311\Scripts\activate

# Runtime dependencies
pip install -r requirements.txt

# Dev dependencies (pytest, pytest-cov, pytest-mock)
pip install -e ".[dev]"

# Pull the models (Ollama must be installed — https://ollama.com/download)
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
```

---

## Running the Tests

```bash
# All 859 local tests (run this before every pull request)
pytest

# With a line-by-line coverage report
pytest --cov=src --cov-report=term-missing

# Just one test file (faster for focused debugging)
pytest tests/test_agent.py

# Just one specific test
pytest tests/test_agent.py::TestToolCalculator::test_tool_calculator_basic

# All tests matching a keyword
pytest -k "calculator"

# Verbose — print each test name as it runs
pytest -v

# HF Space tests (run from inside the huggingface/ folder)
cd huggingface && pytest
```

All tests must be green before submitting a pull request.

---

## Test Categories

| Type | What it checks | Example |
|------|---------------|---------|
| **Unit** | One method at a time, all dependencies mocked | `test_expand_query_returns_3` |
| **Integration** | Two or more real components, no mocks for file libraries | `test_load_pdf`, `test_url_html_webpage` |
| **Contract** | Output shape — correct keys, types, structure | `test_chunk_has_5_keys` |
| **Regression** | Exact prompt text and phrase lists locked down | `test_hallucination_phrases_unchanged` |
| **Boundary** | Empty files, single-item inputs, at-limit sizes | `test_xlsx_header_only_returns_empty` |
| **Negative** | Wrong or missing input handled gracefully | `test_http_404_returns_empty_list` |
| **Functional** | End-to-end flow through one feature path, limited mocking | `test_chunk_topic_search_returns_chunks` |
| **Combination** | Parametrized matrix — all modes × all doc types × all URL types | `test_text_doc_type_pipeline[pdf]` |
| **UI (AppTest)** | Streamlit app renders correctly using `streamlit.testing.v1.AppTest` | `test_mode_radio_has_chat_and_agent_options` |
| **UI (mocked st)** | Handler functions called with mocked Streamlit — no server needed | `test_search_called_on_submission_with_query` |

---

## Test Files — Full Breakdown

### Local (`tests/`) — 859 tests across 31+ files

| File | Covers |
|------|--------|
| `test_document_loader.py` | Chunkers, scan, misplaced detection (unit) |
| `test_vector_store.py` | BM25, dense retrieval, build logic (unit) |
| `test_vector_store_pipeline.py` | Full pipeline, rerank, classify (unit) |
| `test_agent.py` | ReAct loop, tools, fast paths (unit) |
| `test_benchmarker.py` | Benchmarker run, summary, CSV export, 15-case defaults (unit) |
| `test_metrics.py` | All 7 scoring functions (unit) |
| `test_tool_benchmarks.py` | Calculator/sentiment/summarise check helpers, invoke_tool (unit) |
| `test_benchmark_report.py` | Terminal report formatting functions (unit) |
| `test_integration_loader.py` | All 9 formats + URL ingestion + chunk_directory (integration) |
| `test_integration_pipeline.py` | Full RAG pipeline end-to-end (integration) |
| `test_combinations.py` | Chat/Agent × all 8 doc types (parametrized) |
| `test_combinations_url.py` | URL × all content types (parametrized) |
| `test_combinations_analysis.py` | Query classification × all types |
| `test_contracts.py` | Output shape contracts |
| `test_contracts_pipeline.py` | Pipeline output contracts |
| `test_regression.py` | Prompts and phrase lists locked down |
| `test_boundary_negative.py` | Empty files, wrong input, edge cases |
| `test_doc_types_and_modes.py` | All 8 formats in chat mode |
| `test_doc_types_agent.py` | All 8 formats in agent mode |
| `test_url_ingestion.py` | URL fetch + type detection |
| `test_url_pipeline.py` | URL → pipeline end-to-end |
| `test_file_upload.py` | File upload pipeline |
| `test_file_upload_tools.py` | Upload edge cases |
| `test_crawl.py` | Recursive URL crawl — depth, same-domain constraint, utility URL filtering, progress callback |
| `test_crawl_combinations.py` | Crawl × depth × topic filter × max pages parametrized matrix |
| `test_topic_search.py` | DuckDuckGo HTML search + chunk_topic_search end-to-end |
| `test_ui_app.py` | Streamlit app structure via AppTest — header, mode selector, session state |
| `test_ui_components.py` | Each UI handler individually with mocked Streamlit |
| `test_theme_session.py` | UI session state helpers |
| `test_handlers.py` | Streamlit handlers (pure helpers) and CLI runner |
| `test_logger.py` | Interaction logging |

### HF Space (`huggingface/tests/`) — 385 tests across 15+ files

Same categories adapted for the HF deployment — InferenceClient instead of Ollama, EphemeralClient instead of persistent ChromaDB. Includes `test_crawl.py`, `test_crawl_combinations.py`, and `test_topic_search.py`.

---

## Mock Strategy

```python
# Always mock — these make network or model calls:
ollama.embed    → {'embeddings': [[0.1, 0.2, ...]]}   # NOT ollama.embeddings
ollama.chat     → {'message': {'content': 'mock response'}}
requests.get    → Mock with .content, .headers, .encoding, .raise_for_status()
requests.post   → Mock with .text (HTML body), .raise_for_status()
                  # used for DuckDuckGo HTML endpoint in topic search tests
chromadb        → chromadb.EphemeralClient()            # in integration tests
streamlit (st)  → MagicMock() per-method                # in UI component tests

# Never mock — use real libraries on real temporary files:
fitz (PyMuPDF), python-docx, openpyxl, xlrd, python-pptx, beautifulsoup4
BM25Okapi, chunk truncation, misplaced detection, calculator eval
```

**UI test rule:** Use `streamlit.testing.v1.AppTest` for app-level structure tests (header, mode selector, session state). Use `unittest.mock.patch.object(st, 'method')` for individual handler tests that involve forms or widgets — this avoids the AppTest "nested forms" limitation in some Streamlit versions.

The rule: mock anything that makes a network call or loads a model. Never mock the file parsing libraries — those are tested with real temporary files so bugs in parsing are caught.

---

## Code Structure

```
src/rag/
  config.py             ← all constants — edit here to change models, thresholds, chunk sizes
  chunkers.py           ← text-based chunker functions (txt, md, csv, html)
  binary_chunkers.py    ← binary format chunker functions (pdf, docx, xlsx, xls, pptx)
  url_crawl.py          ← stateless URL crawl and DuckDuckGo search functions
  url_utils.py          ← URL type detection, link extraction, topic filtering helpers
  document_loader.py    ← ingestion orchestration and URL fetching
  vector_store.py       ← retrieval pipeline and response generation
  agent.py              ← ReAct loop and 6 tools
  benchmarker.py        ← evaluation orchestration and output
  benchmark_report.py   ← stateless terminal report formatting functions
  metrics.py            ← 7 stateless scoring functions
  tool_benchmarks.py    ← calculator/sentiment/summarise benchmark module
src/ui/
  handlers.py           ← event handlers (url, file, topic, user input)
  renderers.py          ← pure Streamlit render functions
  sidebar.py            ← sidebar rendering
  theme.py              ← CSS and style constants
  session.py            ← session state helpers
src/cli/
  runner.py             ← terminal chat, agent, and benchmark entry points
```

**The core rule:** Classes own state. Modules own stateless functions. This distinction is strict.

- `DocumentLoader`, `VectorStore`, `Agent`, `Benchmarker` are classes — they own state
- `config`, `logger`, `chunkers`, `binary_chunkers`, `url_crawl`, `url_utils`, `metrics`, `benchmark_report`, `tool_benchmarks` are modules — they own stateless functions

If you find yourself adding state to a module function, it should probably be a method on the relevant class instead.

---

## How to Add a New File Format

1. Add the file extension to `EXT_TO_TYPE` in `src/rag/config.py`
2. Add a folder entry to `DOC_FOLDERS` in `src/rag/config.py`
3. Write a `chunk_yourformat(filepath, filename)` function in `src/rag/chunkers.py` (text formats) or `src/rag/binary_chunkers.py` (binary formats)
4. Add the routing case to `_dispatch_chunker()` in `src/rag/document_loader.py`
5. Write unit tests in `tests/test_document_loader.py`
6. Write integration tests in `tests/test_integration_loader.py` (use a real temp file — do not mock the parser)
7. Add a reranker prompt variant for the new type in `VectorStore._rerank_prompt()` in `src/rag/vector_store.py`
8. Add a source citation label case in `VectorStore._source_label()` in `src/rag/vector_store.py`

---

## How to Add a New Agent Tool

1. Add a `_tool_yourname(self, arg: str) -> str` private method to `Agent` in `src/rag/agent.py`
2. Add the tool name and description to `AGENT_SYSTEM_PROMPT` in the same file
3. Add the routing case to `_dispatch_tool()` in `src/rag/agent.py`
4. Write unit tests in `tests/test_agent.py`
5. Add a benchmark test case to `TOOL_TEST_CASES` in `src/rag/tool_benchmarks.py`

---

## Code Standards

### The core principle

> Write structured code that is easy to understand, change, and read. Code must be readable by someone who is not a software engineer.

### File size limits

- **Maximum 500 lines per file.** If a file exceeds 500 lines, split it along a clear conceptual boundary.
- `app.py` and `main.py` must stay under 50 lines — they only wire classes together, no logic.

### Naming conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | `UpperCamelCase` | `DocumentLoader`, `VectorStore` |
| Public methods | `lower_case_with_underscores` | `run_pipeline`, `chunk_url` |
| Private methods | `_lower_case_with_underscores` | `_embed`, `_hybrid_retrieve` |
| Module-level constants | `ALL_CAPS_WITH_UNDERSCORES` | `EMBEDDING_MODEL`, `TOP_RETRIEVE` |
| Local variables | `lower_case_with_underscores`, full plain English names | `chunk_size` not `cs` |
| Modules / files | `lower_case_with_underscores` | `document_loader.py`, `vector_store.py` |

### Docstring format (Google style)

Every public class and public method must have a docstring:

```python
class VectorStore:
    """Owns all retrieval, search, and response generation.

    State:
        client:       chromadb.PersistentClient
        collection:   ChromaDB collection
        chunks:       list of all indexed chunks
        bm25_index:   BM25Okapi index for keyword search
        conversation: conversation history for multi-turn context
    """

def run_pipeline(self, query: str, streamlit_mode: bool = False) -> dict:
    """Run the full RAG pipeline for a user query.

    Args:
        query: The user's question.
        streamlit_mode: If True, return a dict with pipeline metadata.

    Returns:
        dict with keys: response, retrieved, reranked, is_confident,
        best_score, query_type.
    """
```

Private methods get a one-line comment only when the logic is non-obvious.

### Inline comments — explain WHY, not what

```python
i += 1  # skip the header row          ← GOOD — explains intent
i += 1  # increment i by 1             ← BAD  — restates the code
```

### Readability

```python
# BAD — saves two lines, hard to follow
result = [c for c in chunks if c.get('score', 0) >= t][:n]

# GOOD — takes three lines, anyone can follow it
passing_chunks = [chunk for chunk in chunks if chunk.get('score', 0) >= threshold]
top_chunks = passing_chunks[:max_results]
```

### What to avoid

- Do not write more than ~30 lines in a single method. Extract a private helper.
- Do not nest more than 3 levels of indentation. Flatten with early returns.
- Do not add error handling for things that cannot fail in normal operation. Only validate at system boundaries (user input, external APIs, file I/O).
- Do not add packages to `requirements.txt` unless genuinely necessary. Dev-only packages go in `pyproject.toml`.

---

## Reporting Security Issues

Do **not** open a public GitHub issue for security vulnerabilities.
Email **anjanatiha@gmail.com** instead — see [SECURITY.md](SECURITY.md) for details.

---

## Pull Request Guidelines

- One focused change per PR — don't mix features and refactors
- All tests must pass: `pytest` green before opening a PR
- Follow the existing code style: plain English names, docstrings on every public method, type hints on all signatures
- No new packages in `requirements.txt` unless genuinely necessary
- See [DESIGN.md](../DESIGN.md) before making architectural changes — the 4-class structure is intentional

**Before opening a PR, run:**
```bash
pytest                              # all local tests green
pytest --cov=src                    # check coverage on changed files
cd huggingface && pytest            # HF tests green if you changed shared logic
```
