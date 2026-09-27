# DESIGN.md — Architectural Decisions

## Why 4 Classes Instead of More Granular Decomposition

The system uses exactly **4 classes** and **4 modules**. This is a deliberate choice, not a limitation.

**Classes own state. Modules own constants and stateless functions.**

The 4 classes map directly to the 4 units of meaningful state in the system:

| Class | State it owns |
|-------|--------------|
| `DocumentLoader` | File system paths, extension map, chunk size config |
| `VectorStore` | ChromaDB collection, BM25 index, chunk list, conversation history |
| `Agent` | ReAct message history, collected context, tool dispatch |
| `Benchmarker` | VectorStore reference, results file path |

A finer decomposition (e.g. a `QueryExpander` class, a `Reranker` class, a `HybridSearcher` class) would split logic that shares the same state — the vector collection and BM25 index — across multiple objects that would all need to reference each other. This creates indirection and circular dependency risk without adding clarity.

---

## What Each Class Owns and Why

### `DocumentLoader`
Owns all ingestion logic: reading files from disk, misplaced file detection, URL fetching, and dispatching to the correct format chunker. The 9 format-specific chunker functions are split across two modules: text-based formats (txt, md, csv, html) live in `chunkers.py`, and binary formats (pdf, docx, xlsx, xls, pptx) live in `binary_chunkers.py`. Both modules contain stateless module-level functions — they take a file path and return a list of chunk dicts with no class state of their own. `DocumentLoader._dispatch_chunker` routes each file to the right function. URL crawling and DuckDuckGo topic search functions live in `url_crawl.py`; URL type detection, link extraction, and topic filtering helpers live in `url_utils.py`. This split keeps `DocumentLoader` focused on ingestion orchestration while each module handles one clear concern.

`scan_all_files()` uses `os.walk` for unlimited-depth recursive scanning. A folder dropped anywhere under `./docs/` at any nesting level — containing any mix of file types — will have all its files detected and processed. Type is determined by file extension, not folder name. The relative path from `docs_root` is stored as the chunk source (e.g. `project/data/q1.xlsx`) so files with the same name in different subfolders remain distinguishable in citations.

### `VectorStore`
Owns ChromaDB, BM25, hybrid retrieval, reranking, query expansion, query classification, response generation, hallucination filtering, and conversation history. These all operate on the same two data structures — the vector collection and the chunk list. Splitting them would require passing the collection and chunks to every caller, which is worse than grouping them under one owner.

### `Agent`
Owns the ReAct loop and all 6 tools as private methods. The tools (calculator, summarise, sentiment, translate, rag_search) have no independent state — they are pure operations that happen to live inside the agent's context. Making each tool a class would add 6 empty `__init__` methods and 6 objects that exist only to call one function.

### `Benchmarker`
Owns all RAG pipeline evaluation orchestration: running the pipeline per test case, printing the report, persisting JSON history, and exporting to CSV. All 7 scoring functions live as stateless module-level functions in `metrics.py` — they have no state of their own and are imported by `Benchmarker`. Takes a `VectorStore` as a dependency so it runs the exact same pipeline the user runs in production.

Agent tool benchmarks (calculator, sentiment, summarise, translate, topic_search) live in `tool_benchmarks.py` as a module — not a class — because they have no shared state. `run_tool_benchmarks(store)` is a single public function that runs 18 tests across 5 tools and saves results to `tool_benchmark_results.json`.

The 4 terminal print functions (`print_per_query_table`, `print_summary_table`, `print_by_query_type`, `format_run_comparison`) were extracted to `benchmark_report.py` when `benchmarker.py` exceeded the 500-line limit. All 4 functions are stateless — they only use the dicts passed to them — so they belong in a module, not a class.

The single `rerank_prompt(query, entry)` function was extracted to `reranker.py` when `vector_store.py` exceeded the 500-line limit. It is a pure function — takes a query and chunk entry, returns a prompt string — with no dependency on `VectorStore` state. Extracting it reduced `vector_store.py` from 527 to 473 lines while keeping the 7 prompt variants in one focused module.

---

## Why Tools Are Private Methods on Agent, Not Separate Classes

The 6 agent tools have no state of their own. `_tool_calculator` is `eval()` with a safety check. `_tool_summarise` is one `ollama.chat` call. `_tool_sentiment` is one `ollama.chat` call with an optional prior RAG search. `_tool_translate` routes through RAG search for short queries, then translates. None of them maintain anything between calls.

Making them classes would require:
- An empty `__init__` or a store reference injected into each
- An `execute()` or `run()` method containing one function call
- An import chain that risks circular dependencies

A private method is better than a premature abstraction. The private method prefix (`_`) makes the boundary clear: these are implementation details of the agent, not public API.

---

## URL Crawling Architecture

### Same-Domain Constraint

Recursive crawling from a seed URL is constrained to stay within the **same hostname** as the seed. Cross-domain links are silently skipped.

**Why:** Without this constraint, following links from a single Wikipedia article at depth 2 results in crawling external news sites, government pages, and unrelated domains — none of which the user intended to index. The constraint was added after observing real crawl logs that included paywall pages (`premium.britannica.com`), subscription login pages (`w1.buysub.com`), and magazine sales pages — all followed from a single legitimate seed URL.

**Implementation:** `is_same_domain(url, seed_domain)` in `url_utils.py` strips `www.` from both sides before comparing hostnames. The seed domain is computed once from the final (post-redirect) URL of the first page and passed through all recursive `_crawl_url` calls as `seed_domain`.

### Utility URL Filtering

Before crawling a page, the URL is checked against `_UTILITY_URL_KEYWORDS` in `url_utils.py`. If any keyword appears in the URL path, the page is skipped without fetching.

Blocked categories: login/auth pages, shopping carts, media download redirects, subscription flows, search results pages, and category index pages. These pages contain no document content and waste the page budget.

### Topic Search — Why HTML Form Endpoint Instead of Library

`chunk_topic_search` uses a direct `requests.POST` to `html.duckduckgo.com/html/` rather than the `duckduckgo-search` Python library.

**Why:** `duckduckgo-search` v6.4.2 uses `primp` internally for browser impersonation. The `safari_17.4.1` impersonation profile that the library targets does not exist in the installed `primp` version, causing a persistent `RatelimitException` (HTTP 202) on every query — even with `backend='lite'`. The HTML form endpoint at `html.duckduckgo.com/html/` responds to a standard `POST` with a User-Agent header and never rate-limits. Results are parsed from `<a class="result__a">` anchor tags using BeautifulSoup. Ad links (containing `/y.js?`) are filtered out.

### Live Crawl Progress (st.status)

Both the recursive URL crawl and the topic search show a live `st.status()` log during crawling. Each page fetched calls `progress_callback(url, dtype, chunk_count)` which appends a line to the log and shows the last 8 lines in real time. This gives the user continuous feedback on a potentially long-running operation without blocking the Streamlit event loop.

---

## Tradeoffs

### ChromaDB vs Pinecone / Weaviate
ChromaDB runs entirely locally with no API key, no network call, and no cost. For a fully local system this is the right choice. The tradeoff is that ChromaDB does not scale horizontally — at production scale with millions of chunks, a managed vector store (Pinecone, Weaviate, Qdrant) would be preferable.

### Local LLM (Ollama) vs API (OpenAI / Anthropic)
Local inference via Ollama means zero data leaves the machine. The tradeoff is speed and quality — `Llama-3.2-3B-Instruct` is much smaller than GPT-4 or Claude. For a demo or private document system this is acceptable. For production with latency SLAs, an API call to a larger model would produce better answers faster.

### BM25 + Dense Hybrid vs Pure Dense
Pure dense retrieval misses exact keyword matches — searching for "NLP" may not retrieve a chunk that says "natural language processing" at the same score as a semantically similar but lexically different chunk. BM25 is excellent at exact matches. The hybrid (alpha=0.5) gets the best of both. The tradeoff is that BM25 must be rebuilt in-memory when new chunks are added, which is O(n) in the number of chunks.

### Sentence-level Chunking vs Fixed-size Chunking
Sentence windows (PDF: 5 sentences, HTML: 5 sentences) produce chunks with complete thoughts rather than arbitrary byte cuts. This improves embedding quality because the model sees coherent context. The tradeoff is variable chunk size, which makes batch embedding slightly less predictable.

---

## What the Type-Aware Reranker Solves

A generic reranker prompt scores all chunks the same way. This works poorly for structured data:

- A spreadsheet row like `Name=Alice | Role=Engineer | Salary=90000` looks like a key-value dump to a generic prompt, which scores it low because it is not fluent prose.
- A slide title and three bullet points looks incomplete without its presentation context.
- A CSV row without column headers is meaningless unless framed correctly.

The type-aware reranker uses 7 different prompts — one per document type — that frame the content appropriately. The xlsx/csv prompt says "does this **spreadsheet row** contain relevant information?" The pptx prompt says "does this **presentation slide** contain relevant information?" This framing dramatically improves reranking accuracy on structured data.

The 7 prompt variants live in `src/rag/reranker.py` as a single public function `rerank_prompt(query, entry)`. `VectorStore._rerank()` calls it once per candidate chunk. The HF deployment uses a cross-encoder instead of LLM prompts for reranking (faster on serverless), so HF's `vector_store.py` does not import `reranker.py` — but the module exists in the HF codebase for test parity.

---

## Benchmark Metrics

The benchmark suite has two parts:

**Part 1 — RAG pipeline (15 questions, 4 domains):** 7 metrics per question. Two use the language model as a judge (more reliable but slower); five are computed directly from text and vector scores (fast, no extra LLM call).

**Part 2 — Agent tools (18 tests):** Deterministic correctness for calculator, format compliance for sentiment, keyword coverage for summarise, non-empty output for translate, mocked-network pipeline for topic_search. Results saved to `tool_benchmark_results.json` separately.

| Metric | Kind | What it measures |
|--------|------|-----------------|
| `faithfulness_llm` | LLM-as-judge (1–5 → 0–1) | Whether every claim in the answer is grounded in the retrieved context |
| `answer_relevancy_llm` | LLM-as-judge (1–5 → 0–1) | Whether the answer directly addresses the question |
| `ground_truth_match` | F1 word overlap | How closely the response matches the known correct answer |
| `keyword_recall` | Fraction found | Whether expected factual keywords appear in the response |
| `context_relevance` | Mean cosine similarity | Whether the retrieval pipeline fetched relevant chunks |
| `precision_at_5` | Fraction of top-5 | Whether the top-5 retrieved chunks actually contain relevant information |
| `mrr` | 1 / rank | How high up the list the first relevant chunk appeared |
| `overall` | Mean of all 7 | Single headline metric for pipeline health |

**Diagnostic patterns:**
- Low `faithfulness_llm` + high `context_relevance` → model is hallucinating despite receiving good chunks. Check the system prompt and `SIMILARITY_THRESHOLD`.
- Low `context_relevance` + low `precision_at_5` → retrieval is broken upstream. Check document indexing and embedding quality.
- Low `mrr` + adequate `context_relevance` → relevant chunks exist but are ranking low. Reranking quality may have degraded.
- Low `ground_truth_match` + high `faithfulness_llm` → the answer is correct but phrased differently from the expected answer. May not indicate a real problem.

All scoring functions are stateless module-level functions in `metrics.py`. They take plain Python values and return a float in [0.0, 1.0]. This makes them independently testable and reusable outside the benchmark pipeline.

---

## How to Scale to Production

1. **Async embedding**: Replace sequential `ollama.embed` calls with a batched async queue. At 50 chunks/batch this is the main bottleneck at index time.

2. **Managed vector store**: Replace `chromadb.PersistentClient` with Pinecone or Weaviate for horizontal scaling and filtering. The `VectorStore` interface isolates this behind `build_or_load` and `add_chunks` — swapping the backend requires changing two methods.

3. **Streaming responses**: The pipeline already supports `stream=True`. In the Streamlit UI, pipe the stream token-by-token into `st.write_stream` for live token output.

4. **Caching embeddings**: Cache `_embed` results by text hash to avoid re-embedding duplicate content across sessions.

5. **Larger language model**: Swap `LANGUAGE_MODEL` in `config.py` to a larger Ollama model or an API model. The model name is the only change required.

6. **Distributed BM25**: At millions of documents, replace `BM25Okapi` with Elasticsearch or OpenSearch for distributed keyword search. The `_hybrid_retrieve` method isolates the BM25 call — swap the implementation behind the same interface.

7. **Authentication and multi-tenancy**: Add a user ID to ChromaDB metadata filters so each user only retrieves from their own documents.

---

## What Would Be Done Differently with More Time

- **Streaming in the Streamlit chat**: Currently the full response is collected before display. Token-level streaming would make the UI feel much faster.
- **Persistent conversation history**: Conversation history lives in memory and resets on restart. A database-backed store (SQLite or Redis) would enable cross-session memory.
- **Evaluation dataset**: The current 15-question benchmark covers 4 domains (cat facts, Python language, team members CSV, machine learning) using committed sample files. A real production evaluation dataset with 50-100 domain-specific questions, human-verified ground truth answers, and a mix of factual/comparison/summarise query types would give stronger signal. The infrastructure is already in place — `ground_truth`, `query_type`, and `chunk_directory()` — only the domain-specific questions and documents need to be added.
- **Document update detection**: Currently, if a document changes on disk, the system does not detect it. A file hash comparison at startup would trigger targeted re-embedding of changed files without a full rebuild.
- **Reranker fine-tuning**: The LLM reranker uses a zero-shot prompt. Fine-tuning a small cross-encoder (e.g. `ms-marco-MiniLM`) on domain-specific relevance pairs would give faster and more accurate reranking than a generative model.

---

## Coding Standards

### File Size
- **Maximum 500 lines per file.** If a file exceeds 500 lines, it must be split along a clear conceptual boundary.
- Entry points (`main.py`, `app.py`) must stay under 50 lines — they wire classes together and nothing else.
- Test files must be split by test category, not by line count alone. Each test file covers one concern: mock-chunk tests, file upload tests, or URL ingestion tests.

### Modularity
- **One responsibility per file.** A file that does two things should be two files.
- Classes own state and the operations that depend on that state. Stateless operations belong in modules (functions, not classes).
- Private methods (prefix `_`) are implementation details. They should not be called from outside the class. If you find yourself needing to call a private method from another class, it is a sign the method belongs in a shared module.
- Do not create helper classes or utility classes for one-time use. A plain function in a module is clearer and cheaper.

### Commenting Convention (PEP 257 / Google style)
Every public class and every public method must have a docstring. Private methods get a one-line comment only when the logic is non-obvious.

**Class docstring — describes what state the class owns and its public API:**
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

**Public method docstring — one-line summary, Args, Returns:**
```python
def run_pipeline(self, query: str, streamlit_mode: bool = False) -> dict:
    """Run the full RAG pipeline for a user query.

    Args:
        query: The user's question.
        streamlit_mode: If True, return a dict with pipeline metadata
            (retrieved chunks, reranked chunks, confidence, query type).
            If False, stream the response to the terminal.

    Returns:
        dict with keys: response, retrieved, reranked, is_confident,
        best_score, query_type.
    """
```

**Private method — one-line comment only when non-obvious:**
```python
def _cosine_similarity(self, a: list, b: list) -> float:
    # Manual dot product — avoids numpy import for a single operation.
    dot = sum(x * y for x, y in zip(a, b))
    ...
```

**Inline comment — explains WHY, not what:**
```python
# Greedy (.+) without DOTALL: captures up to the last ')' on the same line.
# Non-greedy (.+?) would stop at the first ')' and truncate nested parens.
match = re.search(r'(?i)TOOL:\s*(\w+)\s*\(\s*(.+)\s*\)', response_text)
```

Do not add comments that restate what the code does:
```python
# BAD — restates the code
i += 1  # increment i by 1

# GOOD — explains the intent
i += 1  # skip the header row
```

### Naming

Follow **[PEP 8](https://peps.python.org/pep-0008/)** (Python's official style guide) and the **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)** for all naming. The table below is a quick reference — PEP 8 is authoritative when in doubt.

| Kind | Convention | PEP 8 reference | Example |
|------|-----------|-----------------|---------|
| Classes | `UpperCamelCase` (a.k.a. `PascalCase`) | [Class Names](https://peps.python.org/pep-0008/#class-names) | `DocumentLoader`, `VectorStore` |
| Public functions / methods | `lower_case_with_underscores` | [Function and Variable Names](https://peps.python.org/pep-0008/#function-and-variable-names) | `run_pipeline`, `chunk_url` |
| Private methods | `_lower_case_with_underscores` | [Naming Conventions — single leading underscore](https://peps.python.org/pep-0008/#naming-conventions) | `_embed`, `_hybrid_retrieve` |
| Module-level constants | `ALL_CAPS_WITH_UNDERSCORES` | [Constants](https://peps.python.org/pep-0008/#constants) | `EMBEDDING_MODEL`, `TOP_RETRIEVE` |
| Local variables / parameters | `lower_case_with_underscores`, full plain English names | [Function and Variable Names](https://peps.python.org/pep-0008/#function-and-variable-names) | `chunk_size` not `cs`, `query_type` not `qt` |
| Modules / files | `lower_case_with_underscores` | [Package and Module Names](https://peps.python.org/pep-0008/#package-and-module-names) | `document_loader.py`, `vector_store.py` |

### Readability — write for non-technical readers

All code must be readable by someone without a software engineering background.

- **Plain English names** — `chunk_size` not `cs`, `is_confident` not `conf_flag`. Never sacrifice clarity for brevity.
- **Plain English docstrings** — the first line of every docstring must be understandable without technical background. Avoid jargon.
- **Plain English inline comments** — explain WHY and WHAT in simple language. Every non-obvious step must have a comment a non-programmer could understand.
- **No clever one-liners** — prefer clear multi-line code over compact expressions. Code is read far more often than it is written.

```python
# BAD — saves two lines, hard to follow
result = [c for c in chunks if c.get('score', 0) >= t][:n]

# GOOD — takes three lines, anyone can follow it
passing_chunks = [chunk for chunk in chunks if chunk.get('score', 0) >= threshold]
top_chunks = passing_chunks[:max_results]
```

### What to Avoid
- Do not write more than ~30 lines in a single method. If a method is longer, extract a private helper.
- Do not nest more than 3 levels of indentation. Flatten with early returns.
- Do not repeat the same import inside every function. Import once at the top of the file (or once at the top of a method that is conditionally executed to avoid circular imports).
- Do not add error handling for things that cannot fail in normal operation. Only validate at system boundaries (user input, external APIs, file I/O).
