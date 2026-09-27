---
title: RAG Agent
emoji: 🐱
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
python_version: 3.11
license: mit
tags:
  - RAG
  - retrieval-augmented-generation
  - document-qa
  - NLP
  - LLM
  - hybrid-search
  - chromadb
  - sentence-transformers
  - gradio
---

# RAG Agent — Ask Your Documents

A production-grade, fully cloud-hosted RAG (Retrieval-Augmented Generation) chatbot. Upload any document and ask questions in natural language. Supports **PDF, TXT, DOCX, XLSX, PPTX, CSV, Markdown, and HTML**, including structured/tabular formats like resumes and spreadsheets.

**Built by [Anjana Tiha](https://github.com/anjanatiha)** — [LinkedIn](https://www.linkedin.com/in/anjanatiha/) · [Blog Post](https://medium.com/@anjanatiha/rag-58d2497529d0)

**GitHub:** [Retrieval-Augmented-Generation-RAG-Agent](https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent)

---

## Features

| Feature | Details |
|---------|---------|
| **Multi-format support** | PDF, TXT, DOCX, XLSX, XLS, PPTX, CSV, Markdown, HTML |
| **Hybrid search** | BM25 (lexical) + dense vector (semantic) fusion |
| **Query expansion** | LLM-generated query rewrites for higher recall |
| **Type-aware reranker** | 7 different reranking prompts — one per document type |
| **Query classification** | Auto-classifies as factual / comparison / summarise / general |
| **Confidence filter** | Similarity threshold guard — no hallucination on missing info |
| **Agent mode** | ReAct-style loop with 6 tools: rag_search, calculator, summarise, sentiment, translate, finish |
| **URL ingestion** | Fetch any public URL — webpage, PDF, DOCX, XLSX, CSV, PPTX |
| **Recursive crawl** | Follow links from a seed URL (depth 1–3, same domain only) |
| **Topic search** | Enter a query → DuckDuckGo → crawl top results → index automatically |
| **Structured retrieval** | Accurate answers from resumes, spreadsheets, and tables |
| **Conversation memory** | Multi-turn memory across the session |
| **Source citations** | Type-aware location labels (page, row, slide, line) |

---

## How to Use

1. **Upload a document** using the file upload panel (PDF, TXT, DOCX, XLSX, PPTX, CSV, MD, HTML)
2. **Or paste a URL** to fetch a webpage or remote document (with optional recursive crawl)
3. **Or search a topic** — enter any query to search DuckDuckGo and index the top results automatically
4. **Ask a question** in the chat input
5. Switch to **Agent mode** for multi-step reasoning with tool calling

**Set your HF_TOKEN** in Space Secrets (Settings → Repository secrets → `HF_TOKEN`) to enable the LLM. Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Architecture

This Space uses the same 4-class architecture as the local version:

| Class | Owns |
|-------|------|
| `DocumentLoader` | All ingestion — 9 format chunkers across two modules, misplaced file detection, URL fetching |
| `VectorStore` | ChromaDB (ephemeral), BM25, hybrid retrieval, reranking, query pipeline |
| `Agent` | ReAct loop, all 6 tools as private methods |

**Models used:**
- **Embeddings**: `BAAI/bge-base-en-v1.5` via `sentence-transformers` (runs locally in the Space)
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.3` via HF Inference Providers router (with automatic fallback to `zephyr-7b-beta`, `Llama-3.2-3B-Instruct`, `gemma-2-2b-it` if unavailable)

**Differences from local version:**
- No Ollama — uses `sentence-transformers` + HF Inference API instead
- ChromaDB uses `EphemeralClient` (in-memory, resets on restart)
- Gradio UI instead of Streamlit
- Starts with empty knowledge base — upload documents to begin

---

## Local Setup

To run the full local version with persistent storage and on-device LLM inference, see the [GitHub repo](https://github.com/anjanatiha/Retrieval-Augmented-Generation-RAG-Agent).
