# CLAUDE.md — RAG Agent

This file tells Claude Code how to work with this codebase.
Read it fully before making any changes.

---

## Core Principle — Always Apply

> **Write structured code that is easy to understand, change, and read.**
> This applies to every file, every function, every comment — always.
> Code must be readable by someone who is not a software engineer.

- Every file has **one clear job** — if it does two things, split it
- Every function does **one thing** and fits in ~30 lines
- Entry points (`app.py`, `main.py`) stay **under 50 lines** — they only wire things together; split immediately if exceeded
- Handler and logic code goes in **dedicated modules** (`src/ui/handlers.py`, `src/cli/runner.py`, `src/handlers.py`)
- **Plain English names** everywhere: `document_type` not `dtype`, `chunk_total` not `n`
- **Type hints** on every function so readers know what goes in and comes out
- **Docstrings** explain what and why in plain language
- **Structured logging** (`logging` module) — not bare `print()` statements
- **Environment variables** for all config that may differ between environments
- **Pinned dependency versions** so the app builds the same way every time
- **CI/CD** so every push is automatically tested

---

## Project Overview

RAG Agent is a fully local, production-grade Retrieval-Augmented Generation system.
It ingests documents in 8 formats, retrieves using hybrid BM25 + dense vector search,
reranks with a type-aware LLM reranker, and answers via LLaMA 3.2 — all on-device via Ollama.

**Live demo:** https://huggingface.co/spaces/anjanatiha2024/Rag-Agent
**GitHub:** https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent

**Entry points:**
- `python main.py` — terminal chatbot
- `python main.py --agent` — agent mode
- `python main.py --benchmark` — benchmark evaluation
- `streamlit run app.py` — Streamlit web UI

---

## Models

| Role | Model |
|------|-------|
| Embeddings | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` |
| Language / Reranker | `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF` |

---

## Design Decision — 4 Classes

This codebase uses **4 classes and 4 modules**. This is a deliberate choice.

Components that have no meaningful state of their own — query classification,
response generation, conversation memory, individual tools — are implemented
as private methods (`_`) on the class that owns the state they operate on.
This reduces indirection, keeps related logic together, and makes the codebase
navigable without sacrificing clarity.

**Classes own state. Modules own constants and stateless functions.**
This distinction is strict and must always be preserved.

---

## Dependencies

### Runtime — `requirements.txt`
```
ollama
rank_bm25
chromadb
streamlit
requests
pymupdf           # PDF
python-docx       # DOCX
openpyxl          # XLSX
xlrd              # XLS legacy only — NOT for .xlsx
python-pptx       # PPTX
beautifulsoup4    # HTML + URL parsing
lxml              # BeautifulSoup parser — add this, it is missing from original
```

### Dev only — `pyproject.toml`
```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "pytest-mock"]
```

**Rules:**
- Do NOT add any package to `requirements.txt` during the refactor
- `lxml` is the only addition — it is a missing but needed dependency
- Dev dependencies go in `pyproject.toml` only, never `requirements.txt`
- `xlrd` is for `.xls` legacy files only — `openpyxl` handles `.xlsx`

---

## Target Project Structure

```
rag/
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── config.py              ← MODULE: all constants
│   │   ├── logger.py              ← MODULE: stateless log functions
│   │   ├── document_loader.py     ← CLASS: DocumentLoader
│   │   ├── vector_store.py        ← CLASS: VectorStore
│   │   ├── agent.py               ← CLASS: Agent
│   │   └── benchmarker.py         ← CLASS: Benchmarker
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── handlers.py            ← MODULE: Streamlit event handlers
│   │   ├── theme.py               ← MODULE: CSS + style constants
│   │   └── session.py             ← MODULE: session state helpers
│   └── cli/
│       ├── __init__.py
│       └── runner.py              ← MODULE: CLI entry functions
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   ├── test_agent.py
│   ├── test_benchmarker.py
│   └── test_integration.py
├── app.py                         ← Streamlit UI thin wrapper (<50 lines)
├── main.py                        ← CLI thin wrapper (<50 lines)
├── requirements.txt               ← add lxml only
├── pyproject.toml                 ← new file
├── DESIGN.md                      ← new file: architectural decisions
├── .gitignore                     ← preserve existing
├── .streamlit/config.toml         ← preserve existing Ocean Blue theme
├── CLAUDE.md
└── README.md
```

---

## Class Responsibilities

### `DocumentLoader` — owns all ingestion

**State:**
```python
self.doc_folders  # from config
self.ext_to_type  # from config
self.chunk_sizes  # all chunk size constants from config
```

**Public methods:**
```python
def ensure_folders()
def scan_all_files()           # misplaced file detection
def chunk_all_documents()
def chunk_url(url)             # 4-priority type detection
```

**Private methods:**
```python
def _dispatch_chunker(file_info)
def _chunk_txt(filepath, filename)
def _chunk_md(filepath, filename)    # strip headings/bold/italic/code/images/links
def _chunk_pdf(filepath, filename)   # page-level isolation, sentence windows
def _chunk_docx(filepath, filename)  # paragraphs + table rows, dedup merged cells
def _chunk_xlsx(filepath, filename)  # key=value pairs per row, per sheet
def _chunk_xls(filepath, filename)   # xlrd fallback for legacy .xls
def _chunk_csv(filepath, filename)   # DictReader key=value pairs
def _chunk_pptx(filepath, filename)  # text shapes per slide
def _chunk_html(filepath, filename)  # BeautifulSoup tag strip + sentence windows
def _truncate_chunk(text)            # 300 words OR 1200 chars, whichever shorter
```

**URL type detection — 4 priorities in exact order:**
1. Content-Type header + fuzzy fallback
2. File extension in URL path (strip query strings first)
3. PDF magic bytes sniff (`content[:4] == b'%PDF'`)
4. Default to 'html'

**chunk_url dispatch:**
- Binary formats (pdf, docx, xlsx, pptx, xls) → write to tempfile → chunker → delete tempfile
- Text formats (csv, md) → write to tempfile → chunker → delete tempfile
- txt → line-based directly from decoded content
- html/webpage → BeautifulSoup directly from decoded content

---

### `VectorStore` — owns all retrieval, search, and response generation

This is the largest class. It owns ChromaDB, BM25, the full query pipeline,
response generation, and conversation history. These are all grouped here
because they all operate on the same core state: the vector index and chunk list.

**State:**
```python
self.client          # chromadb.PersistentClient
self.collection      # ChromaDB collection
self.chunks          # list of all chunks (local + URL/file uploaded)
self.bm25_index      # BM25Okapi index
self.conversation    # list of {'role': ..., 'content': ...} dicts
```

**Public methods:**
```python
def build_or_load(chunks)          # persistent ChromaDB, batch embed (size 50), rebuild logic
def add_chunks(chunks, id_prefix)  # runtime URL/file upload
def rebuild_bm25(all_chunks)       # after URL/file upload adds new chunks
def run_pipeline(query, streamlit_mode=False)   # full chat pipeline
def stream_response(stream)        # terminal: typing dots then token stream
def clear_conversation()           # wipe conversation history
```

**Private — vector and search:**
```python
def _embed(text)                   # ollama.embed(model=..., input=...)['embeddings'][0]
def _truncate_for_embedding(text)  # 200 words AND 1200 chars
def _cosine_similarity(a, b)       # manual dot product
def _hybrid_retrieve(queries, top_n, alpha=0.5)   # BM25 + dense fusion
def _rerank(query, candidates, top_n)              # type-aware LLM reranker
def _rerank_prompt(query, entry)   # 7 prompt variants — preserve exactly
```

**Private — query processing:**
```python
def _classify_query(query)         # summarise → comparison → factual → general
def _expand_query(query)           # LLM 2 rewrites + original = 3 queries
def _check_confidence(results)     # results[0][1] >= SIMILARITY_THRESHOLD
def _smart_top_n(query_type)       # factual:5, comparison:15, general:10, summarise:TOP_RETRIEVE
```

**Private — response generation:**
```python
def _build_instruction_prompt(context)
def _source_label(entry)           # pdf→p{n}, xlsx/csv→row{n}, pptx→slide{n}, html→s{n}, else→L{s}-{e}
def _synthesize(question, context) # anti-hallucination LLM synthesis
def _filter_hallucination(response)  # truncate on pivot phrases after no-info phrases
```

**Hybrid fusion logic:**
```python
# For each query:
# 1. Dense: query ChromaDB, get distances, convert to similarity (1 - dist)
# 2. BM25: get scores, normalize by max score
# 3. Fuse: score = alpha * dense + (1 - alpha) * bm25
# 4. Keep best score per chunk across all expanded queries
# 5. Sort descending, return top_n
```

**VectorStore rebuild logic:**
```python
if existing >= len(chunks):    # DB up to date or has extra URL chunks → skip
    return collection
if existing > 0:               # local files grew → delete all → rebuild
    collection.delete(ids=collection.get()['ids'])
# embed all chunks in batches of 50
```

**Hallucination filter — preserve both lists exactly:**
```python
_no_info_phrases = [
    "there is no information", "i couldn't find", "i could not find",
    "the provided context does not", "the provided documents do not",
    "no information in the provided", "not mentioned in the", "not found in the",
]
_hallucination_pivots = [
    "however,", "but i can", "but,", "that said,",
    "nevertheless,", "i can tell you", "i can provide"
]
```

**Low confidence path:** if `not is_confident` → return fixed message, do NOT call LLM.
**run_pipeline rerank top_n:** `10` if query_type == 'summarise', else `TOP_RERANK`.

---

### `Agent` — owns the ReAct loop and all 6 tools

Tools (calculator, summarise, sentiment, translate) are private methods — not separate classes.
They have no state of their own and are implementation details of the agent.

**State:**
```python
self.store           # VectorStore reference
self.messages        # ReAct message history
self.collected_context  # accumulated search results for final synthesis
self.max_steps       # default 8
```

**Class constant:**
```python
AGENT_SYSTEM_PROMPT: str   # preserve every word exactly — do not change
```

**Public methods:**
```python
def run(user_query, streamlit_mode=False)
```

**Private — ReAct loop:**
```python
def _parse_tool_call(response_text)    # two regex patterns — preserve both
def _dispatch_tool(tool_name, tool_arg)
def _synthesize_final_answer(query, collected_context)
def _fast_path_summarise(query, streamlit_mode)   # 4-term multi-search → synthesize
def _fast_path_sentiment(query, streamlit_mode)   # search → strip labels → sentiment
```

**Private — tools (were separate classes, now private methods):**
```python
def _tool_rag_search(query)            # expand → hybrid retrieve → rerank → format
def _tool_calculator(expression)       # safe eval, ALLOWED_CHARS whitelist
def _tool_summarise(text)              # adaptive length hint + LLM
def _tool_sentiment(text_or_query)     # < 10 words → search first; else analyse directly
def _tool_translate(language_and_text) # "Language: text" → translate; short → search first
```

**6 tools:** `rag_search`, `calculator`, `summarise`, `sentiment`, `translate`, `finish`

**Translate tool routing:**
```python
# Input format: "TargetLanguage: text to translate"
# No colon → defaults to English
# < 15 words after colon → _tool_rag_search first, then translate retrieved content
# ≥ 15 words → translate directly (content is a full passage, not a query)
```

**parse_tool_call — two patterns (preserve both):**
```python
# Pattern 1: with parentheses
re.search(r'(?i)TOOL:\s*(\w+)\s*\(\s*(.+?)\s*\)', text, re.DOTALL)
# Pattern 2: without parentheses (fallback)
re.search(r'(?i)TOOL:\s*(\w+)\s+(.+)', text)
```

**Calculator allowed chars:** `set('0123456789+-*/(). ')`

**Summariser length hints:**
```python
< 100 words  → "2-3 sentences"
< 300 words  → "4-5 sentences"
else         → "6-8 sentences covering all key points"
```

**Sentiment prompt output format (preserve exactly):**
```
Sentiment: <Positive / Negative / Neutral / Mixed>
Tone: <one short phrase>
Key phrases: <2-4 phrases>
Explanation: <1-2 sentences>
```

**Agent fast path — summarise:**
- Detected by keyword list (preserve exactly from original)
- 4 search terms: `['work experience', 'education', 'skills projects', 'summary contact']`
- Multi-search → collect context → synthesize → return

**Agent fast path — sentiment:**
- Detected by keyword list (preserve exactly)
- Strip sentiment keywords from query to get search subject
- Search → strip chunk metadata labels → sentiment analysis → return

**Bad format recovery:** up to 2 retries with correction prompt, then use raw text.
**Calculator auto-finish:** after successful calculator call → auto-finish with `{expr} = {result}`
**RAG search auto-finish:** for non-summarise queries → synthesize after first rag_search

---

### `Benchmarker` — owns all evaluation

**State:**
```python
self.store           # VectorStore reference
self.results_file    # BENCHMARK_FILE from config
```

**Class constant:**
```python
DEFAULT_TEST_CASES = [   # 5 cat facts — preserve exactly for backward compat
    {'question': 'How many hours do cats sleep per day?',     'expected_keywords': ['sleep', '16']},
    {'question': 'Can cats see in dim light?',                'expected_keywords': ['dim', 'light', 'see']},
    {'question': 'How many toes do cats have on front paws?', 'expected_keywords': ['five', 'toes', 'front']},
    {'question': 'How many whiskers does a cat have?',        'expected_keywords': ['whiskers', '12']},
    {'question': 'Can cats taste sweet food?',                'expected_keywords': ['sweet', 'taste']},
]
```

**Public methods:**
```python
def run(test_cases=None)
```

**Private methods:**
```python
def _score_faithfulness(response, reranked)      # word overlap, faithfulness stopwords
def _score_relevancy(question, response)          # F1 word overlap, relevancy stopwords
def _score_keyword_recall(response, keywords)     # fraction of keywords found
def _score_context_relevance(reranked)            # mean sim of top TOP_RERANK chunks
def _save_results(results)
def _compare_runs(current, previous)              # delta with ▲/▼/─ indicators
def _read_results()
```

**Two separate stopword sets (preserve exactly):**
```python
# Faithfulness
{'a','an','the','is','are','was','were','do','does','it','its',
 'to','of','in','for','and','or','not','with','on','at','by',
 'this','that','be','as','i','you','we','they','but','so','if'}

# Relevancy
{'a','an','the','is','are','was','were','do','does','did','have',
 'has','can','what','how','why','when','where','who','to','of','in',
 'it','its','for','and','or','not','with','on','at','by','from'}
```

**Bar chart format (preserve):** `'[' + '█'*int(s*20) + '░'*(20-int(s*20)) + ']'`

---

## Module Responsibilities

### `config.py` — constants only, no functions, no classes
```python
EMBEDDING_MODEL      = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL       = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
DOCS_ROOT            = './docs'
DOC_FOLDERS          = { 'pdf': ..., 'txt': ..., ... }
EXT_TO_TYPE          = { '.pdf': 'pdf', '.txt': 'txt', ... }
CHROMA_DIR           = './chroma_db'
CHROMA_COLLECTION    = 'rag_docs'
LOG_FILE             = 'rag_logs.json'
BENCHMARK_FILE       = 'benchmark_results.json'
SIMILARITY_THRESHOLD = 0.55
TOP_RETRIEVE         = 20
TOP_RERANK           = 5
TXT_CHUNK_SIZE       = 1
TXT_CHUNK_OVERLAP    = 0
PDF_CHUNK_SENTENCES  = 5
DOCX_CHUNK_PARAS     = 3
PPTX_CHUNK_SLIDES    = 1
HTML_CHUNK_SENTENCES = 5
```

### `logger.py` — stateless file I/O
```python
def log_interaction(query, qtype, chunks_used, sim_scores, response)
def _read_log() -> list      # safe JSON read, return [] on any error
def _write_log(entries)      # JSON dump with indent=2
# Log entry format:
# { timestamp, query, query_type, chunks_used,
#   top_similarity, avg_similarity, response_length }
```

### `src/ui/theme.py` — style constants
```python
CSS: str              # full IBM Plex Mono stylesheet — preserve every rule exactly
BADGE_CLASSES = {
    'factual':    'b-fact',
    'comparison': 'b-comp',
    'general':    'b-gen',
    'summarise':  'b-gen',
}
CONFIDENCE_BADGE = { True: 'b-ok', False: 'b-warn' }
AVATAR = { 'user': '🧑', 'assistant': '💬', 'agent': '🤖' }
```

### `src/ui/session.py` — Streamlit session helpers
```python
SESSION_DEFAULTS = {
    'conv': [], 'display': [], 'total': 0, 'last': None,
    'mode': 'chat', 'url_chunks': [], 'bm25_index': None,
    'url_msg': None, 'file_msg': None,
}
def init_session_state()           # set defaults for missing keys only
def get_active_bm25(base_bm25)     # return session bm25 if updated, else base
```

---

## Workflow Rules

- Work **ONE step at a time**
- After each step, **STOP and wait for my approval**
- Show exactly what files were created or changed
- Do **NOT** move to the next step until I say "continue" or "next"
- If you encounter a decision point, **ask** — do not assume
- Always run tests and the app before asking me to proceed
- **When asked to add comments, docstrings, or any improvements — do it completely across ALL relevant files without asking. Do not ask for permission or confirmation mid-task.**
- **After every feature or fix, keep all relevant MD files and requirements files up to date** — do not leave them stale. Update them as part of the same task, not as a separate step.
- **Every new feature or fix must have tests** — written in the same pass, not after. Use the same test types already present in the suite: unit, functional, integration, regression, boundary, negative, parametrized combination. Sync tests to both `tests/` and `huggingface/tests/`.

---

## Refactoring Rules

### Non-negotiable
- **`rag_app.py` is read-only** — never modify it, only extract from it
- **Copy, do not rewrite** — copy logic from `rag_app.py` verbatim.
  Only change what is strictly necessary to make it work in its new location
  (e.g. adding `self.`, updating import paths, passing dependencies via
  constructor). No simplifying, no improving, no renaming, no reformatting.
- **Change structure, not behavior** — app must work identically after every step
- **One file at a time** — extract, verify, commit, then move to next
- **Run both entry points after every extraction:**
  `python main.py` AND `streamlit run app.py`
- **No circular imports** — one direction only:
  ```
  config → logger → document_loader → vector_store → agent → benchmarker
  ```

### Code style — match original exactly
- Same variable names as original `rag_app.py`
- Same comment style
- Same string formatting (f-strings)
- Same error handling patterns (try/except with print warnings)
- Google-style docstrings on every class and public method
- Type hints on all method signatures
- Private methods prefixed with `_`
- `__all__` defined in each module

---

## TDD Approach — Every File

```
1. Create file with stubs only
2. Write unit tests → pytest → RED (expected)
3. Implement logic from rag_app.py → pytest → GREEN
4. Run integration tests → all green
5. Run streamlit automated tests → all green
6. STOP: "All tests passing. Please test the UI now."
7. Wait for "UI looks good, commit"
8. Commit — then move to next step
```

---

## Testing Responsibilities

### Claude Code — automated:
- Unit tests per file
- Integration tests — all 8 file types, xls, all URL types, all 5 agent tools
- Streamlit automated tests (`streamlit.testing.v1`)
- All GREEN before asking me

### Me — manual browser only:
- Upload each of the 8 file types
- Paste URLs (webpage + remote file types)
- Chat mode — factual, comparison, summarise queries
- Agent mode — all 6 tools: rag_search, calculator, summarise, sentiment, translate, finish
- Check sidebar: pre/post rerank, confidence badge, query type badge, session stats
- Say **"UI looks good, commit"** or report what is broken

### Gate before every commit:
```
1. pytest → all green ✅
2. streamlit tests → all green ✅
3. Claude: "Please test the UI now"
4. Me: test in browser
5. Me: "looks good, commit" OR "fix X"
6. Claude: commit ONLY after my approval
```

---

## Integration Tests — `tests/test_integration.py`

### 1. All 8 file types + xls (real libraries, tmp files)
- `test_load_pdf/txt/docx/xlsx/xls/pptx/csv/md/html()`
- `test_docx_table_rows_extracted()`
- `test_docx_merged_cells_deduplicated()`
- `test_misplaced_file_detected_and_chunked()`
- `test_truncate_chunk_300_words()`
- `test_truncate_chunk_1200_chars()`

### 2. URL ingestion (mock requests.get only)
- `test_url_html_webpage()`
- `test_url_remote_pdf/docx/xlsx/csv/pptx()`
- `test_url_type_by_content_type()`
- `test_url_type_by_extension()`
- `test_url_type_by_pdf_magic_bytes()`
- `test_url_defaults_to_html()`
- `test_url_connection_error_returns_empty()`
- `test_url_source_label_truncated_60_chars()`

### 3. VectorStore pipeline (mock ollama + EphemeralClient)
- `test_hybrid_retrieve_fuses_bm25_and_dense()`
- `test_hybrid_retrieve_alpha_weighting()`
- `test_expand_query_returns_3()`
- `test_classify_summarise_checked_first()`
- `test_classify_factual/comparison/general()`
- `test_rerank_orders_by_llm_score()`
- `test_confidence_below_threshold()`
- `test_confidence_above_threshold()`
- `test_smart_top_n_all_4_types()`
- `test_source_label_all_types()`
- `test_hallucination_filter_truncates()`
- `test_hallucination_filter_clean_response_unchanged()`
- `test_low_confidence_skips_llm()`
- `test_rebuild_logic_skips_if_existing_gte_chunks()`
- `test_rebuild_logic_deletes_and_rebuilds_if_local_grew()`

### 4. All 5 agent tools
- `test_tool_calculator_basic()`
- `test_tool_calculator_complex()`
- `test_tool_calculator_unsafe_chars_rejected()`
- `test_tool_summarise_short_2_3_sentences()`
- `test_tool_summarise_medium_4_5_sentences()`
- `test_tool_summarise_long_6_8_sentences()`
- `test_tool_sentiment_short_query_searches_first()`
- `test_tool_sentiment_long_text_direct()`
- `test_tool_sentiment_output_4_fields()`
- `test_parse_tool_call_with_parens()`
- `test_parse_tool_call_without_parens()`
- `test_parse_tool_call_malformed_returns_none()`
- `test_fast_path_summarise_4_searches()`
- `test_fast_path_sentiment_strips_labels()`
- `test_calculator_auto_finish()`
- `test_rag_search_auto_finish()`
- `test_bad_format_retry_max_2()`
- `test_step_limit_reached()`
- `test_collected_context_used_for_final_answer()`

### 5. Streamlit UI automated
- `test_ui_loads_without_error()`
- `test_chat_mode_is_default()`
- `test_agent_mode_toggle()`
- `test_chat_query_returns_response()`
- `test_agent_query_shows_steps_and_answer()`
- `test_url_ingestion_updates_chunk_count()`
- `test_file_upload_updates_chunk_count()`
- `test_all_8_file_types_upload()`
- `test_clear_button_resets_state()`
- `test_sidebar_pre_rerank_chunks()`
- `test_sidebar_post_rerank_chunks()`
- `test_confidence_badge_low/high()`
- `test_query_type_badge()`
- `test_agent_tools_panel_in_agent_mode()`
- `test_session_stats_increment()`

### Mock strategy
```python
# Always mock:
ollama.embed  → {'embeddings': [[0.1, 0.2, ...]]}  # NOT ollama.embeddings
ollama.chat   → {'message': {'content': 'mock'}}
chromadb      → chromadb.EphemeralClient()
requests.get  → Mock with .content, .headers, .encoding, .raise_for_status()

# Never mock:
fitz, python-docx, openpyxl, xlrd, python-pptx, beautifulsoup4
BM25Okapi, chunk truncation, misplaced detection, calculator eval
```

---

## Step-by-Step Refactor Order

```
Step 0:  Create structure + pyproject.toml + __init__.py + add lxml to requirements.txt
         → show every file created
         STOP: wait for "continue"

Step 1:  Extract config.py (module)
         → copy all constants exactly from rag_app.py
         → run python rag_app.py → confirm still works
         STOP: wait for "continue"
         commit: "refactor: extract config module"

Step 2:  Extract logger.py (module)
         → stubs → tests (red) → implement → green
         STOP: wait for "continue"
         commit: "refactor: extract logger module"

Step 3:  Extract src/ui/theme.py + src/ui/session.py + src/ui/handlers.py (modules)
         → CSS string, badge/avatar dicts, init_session_state, get_active_bm25, all Streamlit handlers
         → run streamlit → confirm UI identical
         STOP: "please check UI looks the same in browser"
         → wait for "UI looks good, commit"
         commit: "refactor: extract UI modules"

Step 4:  Extract DocumentLoader (class)
         → stubs → tests (red) → implement → green
         → integration tests: all 8 types + xls + URL + misplaced + truncation
         STOP: "please test UI — upload each file type and paste a URL"
         → wait for "UI looks good, commit"
         commit: "refactor: extract DocumentLoader class"

Step 5:  Extract VectorStore (class)
         → stubs → tests (red) → implement → green
         → integration tests: hybrid retrieve, rerank, pipeline, hallucination filter
         → confirm ChromaDB loads, queries work, pipeline produces answers
         STOP: "please test UI — ask a factual, comparison, and summarise question"
         → wait for "UI looks good, commit"
         commit: "refactor: extract VectorStore class"

Step 6:  Extract Agent (class)
         → stubs → tests (red) → implement → green
         → integration tests: all 6 tools, fast paths, bad format recovery
         STOP: "please test UI — agent mode: calculator, summarise, sentiment, translate, rag_search"
         → wait for "UI looks good, commit"
         commit: "refactor: extract Agent class"

Step 7:  Extract Benchmarker (class)
         → stubs → tests (red) → implement → green
         → run python main.py --benchmark → confirm scores + bar chart output
         STOP: wait for "continue"
         commit: "refactor: extract Benchmarker class"

Step 8:  Slim app.py and main.py to <50 lines each
         → wire all classes, @st.cache_resource, argparse
         → run all three entry points
         → run full pytest suite → all green
         STOP: "please do a full UI test — every feature"
         → wait for "UI looks good, commit"
         commit: "refactor: slim entry points"

Step 9:  Write DESIGN.md
         → architectural decisions, tradeoffs, how to scale to production
         → what each class owns and why
         STOP: wait for "continue"
         commit: "docs: add DESIGN.md"

Step 10: Full integration tests — tests/test_integration.py
         → all tests in list above → all green
         STOP: show full test results summary
         → wait for "continue"
         commit: "test: add full integration test suite"

Step 11: Final verification
         → full pytest suite → all green
         → python main.py, --agent, --benchmark
         → streamlit run app.py
         STOP: "please do a final complete UI test — every feature"
         → wait for "all good, final commit"
         commit: "refactor: complete — 4 classes, 4 modules, full test suite"
```

---

## Key Things to Preserve Exactly

1. `AGENT_SYSTEM_PROMPT` — every word, every rule, every example
2. All 7 `_rerank_prompt()` variants — one per doc type
3. Hallucination filter — both phrase lists (no-info and pivot)
4. Sentiment prompt — 4-field structured output format
5. URL type detection — 4-priority order including PDF magic bytes
6. Agent fast paths — summarise (4-term) and sentiment (strip labels)
7. `@st.cache_resource` — never cause re-initialization on Streamlit rerun
8. Deferred `_needs_rerun` flag — prevents column rendering cutoff
9. Chunk truncation: 300 words OR 1200 chars (whichever shorter)
10. Embed truncation: 200 words AND 1200 chars (both enforced)
11. BM25 rebuild after every URL/file upload
12. DOCX table extraction with merged cell deduplication
13. Benchmark stopword sets — two separate sets
14. IBM Plex Mono CSS — every rule, every class name
15. `ollama.embed(...)['embeddings'][0]` — not `ollama.embeddings`

---

## Coding Standards

> **All code must be thoroughly commented at all times.**
> This applies to source files AND test files, in this repo AND in the huggingface/ folder.
> Every class, every public method, every private method with non-obvious logic, every fixture, every test method must have a docstring or comment. No exceptions.
> When writing new code or editing existing code, always add comments as part of the same task — never leave uncommented code behind.

### File Size — Hard Limits (enforced automatically, no exceptions)
- **Source files: 500 lines maximum.** The moment any source file exceeds 500 lines, stop and split it before continuing. Do not finish the current task first. Do not wait to be asked.
- **Entry points (`main.py`, `app.py`): 50 lines maximum.** If either exceeds 50 lines, move the excess into a handler or runner module immediately.
- **Test files: 500 lines soft limit.** Split by concern (one class or feature per file) when exceeded — do not split arbitrarily by line count alone.
- After every code change, mentally check: "Did any file cross its limit?" If yes, split before committing.

### Modularity
- One responsibility per file. A file that does two things should be two files.
- Classes own state and the operations that depend on that state. Stateless operations belong in modules.
- Private methods (`_`) are implementation details — do not call them from outside the class.
- Do not create helper classes or utility classes for one-time use. A plain function in a module is clearer.

### Commenting Convention (PEP 257 / Google style)
Every public class and every public method must have a docstring. Private methods get a one-line comment only when the logic is non-obvious.

**Class docstring** — describes what state the class owns and its public API:
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
```

**Public method docstring** — one-line summary, Args, Returns:
```python
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

**Inline comment** — explains WHY, not what:
```python
i += 1  # skip the header row          ← GOOD
i += 1  # increment i by 1             ← BAD (restates the code)
```

### Naming

Follow **PEP 8** and the **Google Python Style Guide** for all naming. Quick reference:

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | `UpperCamelCase` | `DocumentLoader`, `VectorStore` |
| Public methods / functions | `lower_case_with_underscores` | `run_pipeline`, `chunk_url` |
| Private methods | `_lower_case_with_underscores` | `_embed`, `_hybrid_retrieve` |
| Module-level constants | `ALL_CAPS_WITH_UNDERSCORES` | `EMBEDDING_MODEL`, `TOP_RETRIEVE` |
| Local variables / parameters | `lower_case_with_underscores`, full plain English names | `chunk_size` not `cs`, `query_type` not `qt` |
| Modules / files | `lower_case_with_underscores` | `document_loader.py`, `vector_store.py` |

### Readability — write for non-technical readers
All code must be readable by someone who is not a software engineer.

- **Plain English names** — `chunk_size` not `cs`, `is_confident` not `conf_flag`, `number_of_chunks` not `n`
- **Plain English docstrings** — the first line of every docstring must be understandable without technical background
- **Plain English comments** — explain WHY and WHAT in simple language; avoid jargon
- **No clever one-liners** — prefer clear multi-line code over compact expressions that need decoding
- **Every non-obvious step gets a comment** a non-programmer could understand

**Example — BAD (technical, hard to follow):**
```python
def _trunc(t, mw=300, mc=1200):
    w = t.split()
    r = ' '.join(w[:mw]) if len(w) > mw else t
    return r[:mc] if len(r) > mc else r
```

**Example — GOOD (readable by anyone):**
```python
def _truncate_chunk(text, max_words=300, max_chars=1200):
    """Cut text down to a safe size before storing it.

    We limit both word count AND character count because some words
    are very long. Whichever limit is hit first wins.
    """
    # Split the text into individual words
    words = text.split()

    # If there are too many words, keep only the first 300
    if len(words) > max_words:
        text = ' '.join(words[:max_words])

    # If the result is still too long in characters, cut it there too
    return text[:max_chars]
```

### What to Avoid
- Do not write more than ~30 lines in a single method. Extract a private helper if longer.
- Do not nest more than 3 levels of indentation. Flatten with early returns.
- Do not add error handling for things that cannot fail in normal operation. Only validate at system boundaries.

---

## UI Standards — Always Apply

Every UI screen, panel, or component must meet **the highest commercial product standard**.

- **Pleasant color palette** — sea blue + teal green gradient (`--blue-600: #0891b2`, `--teal-600: #0d9488`). No raw Streamlit or Gradio defaults.
- **Clean visual hierarchy** — clear headings, labels, consistent spacing, intentional whitespace
- **Helpful notes on every input** — placeholder text with real examples, captions, and `help=` tooltips so the user never has to guess what a control does
- **Convenient UX** — smart defaults, inline guidance, success/error feedback after every action
- **Elegant proportions** — group related things, separate unrelated things; never cram controls together
- **SaaS-quality standard** — aim for the level of polish seen in Hugging Face Spaces showcases and commercial tools
- **Concise UI text** — all button labels, sidebar headings, note boxes, captions, and help text must be short and informative. Say exactly what the user needs to know — no padding, no repetition. If a label takes more than one line to read, it is too long.

Any feature that affects the user's workflow (e.g. topic filter, depth slider, document type checkboxes) **must** have a short note or caption explaining its purpose and how to use it effectively.

---

## What NOT to Do

- Do not modify `rag_app.py` — it is the source of truth, read-only
- Do not change any logic, prompt text, formula, or constant value
- Do not add packages to `requirements.txt` (lxml only exception)
- Do not make logger, theme, or session into classes
- Do not create extra classes beyond the 4
- Do not put business logic in `app.py` or `main.py`
- Do not merge multiple extractions into one commit
- Do not commit without my "UI looks good, commit"
- Do not skip TDD cycle for any file
- Do not proceed without my "continue"
- Do not mock `ollama.embeddings` — mock `ollama.embed`
- Do not remove the deferred `_needs_rerun` pattern
- Do not hardcode constants — always import from `config.py`

---

## Virtual Environment

All Python commands — `pytest`, `python main.py`, `streamlit run app.py`, `pip install` — must be run inside the project virtual environment:

```bash
source rag_env_311/bin/activate        # Mac/Linux
rag_env_311\Scripts\activate           # Windows
```

The venv is located at `./rag_env_311/` in the repo root. Never run commands against the system Python or any other interpreter — the system Python will not have the project's dependencies installed.
