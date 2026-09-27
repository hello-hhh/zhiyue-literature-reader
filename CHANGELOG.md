# Changelog

All notable changes to this project are documented here.
Versions follow [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH.

---

## [v1.4.3] — 2026-03-31

### Documentation
- Added `SECURITY.md` — vulnerability reporting policy and scope
- Added GitHub issue templates (Bug Report, Feature Request)
- Added pull request template with contribution checklist

No functional changes.

---

## [v1.4.2] — 2026-03-28

### Features
- **Multi-turn conversation** — chatbot maintains context across turns within a session
- Follow-up questions resolve pronouns, references, and continuation phrases from prior turns
- Anti-hallucination constraint preserved — all claims remain grounded in retrieved context

### Infrastructure
- **Docker support** — full stack runs with `docker compose up`, no Python setup required
- ChromaDB and Ollama models persisted in named Docker volumes
- NVIDIA GPU support via `docker-compose.gpu.yml`

### Tests
859 local · 385 HF Space · **1,244 total** — all passing

---

## [v1.4.1] — 2026-03-28

### Infrastructure
- Docker support added — app and Ollama run as separate containers
- `./docs/` mounted as a bind mount — add documents without rebuilding the image

### Tests
859 local · 385 HF Space · **1,244 total** — all passing

---

## [v1.4.0] — 2026-03-28

### Features
- **RAGAS evaluation** — `python main.py --ragas` runs four LLM-as-judge metrics: Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall
- **Pre-commit hooks** — automated code quality checks before every commit (whitespace, isort, flake8, YAML/JSON/TOML validation)

### Tests
859 local · 385 HF Space · **1,244 total** (+16 RAGAS evaluation tests)

---

## [v1.3.2] — 2026-03-28

### Features
- **Streaming responses** — answers appear word by word as generated; no waiting for full response

### Engineering
- Pipeline split into prepare and finalize phases to support streaming
- All source files within 500-line limit

### Tests
843 local · 385 HF · **1,228 total**

---

## [v1.3.1] — 2026-03-28

### Quality
- Type hints added to all method signatures across `src/rag/`
- Docstrings expanded to Google style on all public and non-obvious private methods

No logic changes.

### Tests
843 local · 385 HF · **1,228 total**

---

## [v1.3.0] — 2026-03-28

### Features
- **Translate tool** — agent tool #6, any-to-any language translation
- **Topic search** — search DuckDuckGo and auto-index top results, no API key required
- **Recursive URL crawl** — configurable depth, page budget, and keyword filter
- **Clear Added Content** — one-click reset for runtime-ingested documents
- **URL scheme auto-correction** — bare domains auto-prefixed with `https://`

### Benchmark Improvements
| Metric | Before | After |
|--------|--------|-------|
| Overall | 0.789 | **0.808** ▲ |
| Faithfulness | 0.802 | **0.967** ▲ |
| MRR | 0.900 | **1.000** ▲ |
| Tool benchmark | 12/12 | **18/18** ▲ |

### Engineering
- 500-line file size limit enforced — 6 new modules extracted

### Tests
848 local · 385 HF · **1,233 total**

---

## [v1.2.2] — 2026-03-28

### Features
- **Translate tool** — translate to any language; short queries search knowledge base first
- **Modular reranker** — type-aware reranking extracted into its own module

### Fixes
- 19 CI test failures resolved
- Entry point slimmed to under 50 lines

### Tests
843 local · 387 HF · **1,230 total**

---

## [v1.1.1] — 2026-03-28

### Documentation
- README rewritten from 1,859 → 341 lines
- `ARCHITECTURE.md` added — full pipeline and algorithm reference
- `BENCHMARK.md` added — methodology, metric formulas, and diagnostic guide
- `CONTRIBUTING.md` moved to repo root

---

## [v1.1.0] — 2026-03-28

### Features
- Benchmark expanded from 5 to 15 questions across 4 domains
- Agent tool benchmark added (calculator, sentiment, summarise)
- New `chunk_directory()` method for chunking arbitrary folders

### Tests
**675 tests** across 27 files

---

## [v1.0.0] — 2026-03-28

Initial production release.

### Features
- Hybrid retrieval — BM25 + dense vector search with query expansion and type-aware reranking
- ReAct agent — 5 tools: `rag_search`, `calculator`, `summarise`, `sentiment`, `finish`
- 9 document formats — PDF, DOCX, XLSX, XLS, PPTX, CSV, TXT, Markdown, HTML
- URL ingestion, multi-file upload, recursive folder scan
- Streamlit web UI and CLI

### Tests
**828 tests** across 36 files

[Live Demo](https://huggingface.co/spaces/anjanatiha2024/Rag-Agent)
