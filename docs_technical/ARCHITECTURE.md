# ARCHITECTURE.md — Pipeline and Algorithm Reference

This document explains every algorithm in the RAG Agent pipeline — what each one does, why it exists, and how it solves a specific problem.

**Quick navigation:**
1. [Full Pipeline Diagram](#full-pipeline-diagram)
2. [Document Chunking](#1-document-chunking)
3. [Embedding](#2-embedding)
4. [Query Expansion](#3-query-expansion)
5. [Query Classification](#4-query-classification)
6. [Hybrid Search](#5-hybrid-search)
7. [Type-Aware LLM Reranking](#6-type-aware-llm-reranking)
8. [Confidence Check](#7-confidence-check)
9. [Hallucination Filter](#8-hallucination-filter)
10. [ReAct Agent Loop](#9-react-agent-loop)
11. [URL Type Detection](#10-url-type-detection)
12. [Source Citation Labels](#11-source-citation-labels)
13. [Embedding Rebuild Decision](#12-embedding-rebuild-decision)
14. [BM25 Index Rebuild](#13-bm25-index-rebuild)
15. [Conversation Memory](#14-conversation-memory)

---

## How RAG Works — Plain English

RAG stands for **Retrieval-Augmented Generation**. It solves a fundamental problem with large language models: they hallucinate answers when they don't know something, because they can only draw on what they learned during training.

RAG fixes this by giving the model a **reference library** — your documents — and forcing it to look things up before answering:

```
Without RAG:  Question → LLM → Answer (may be hallucinated)
With RAG:     Question → Search documents → Find relevant passages
                      → Feed passages to LLM → Grounded answer with citations
```

**A note on hallucination:** RAG reduces hallucination by grounding answers in retrieved context, but does not eliminate it entirely. The confidence gate (skipping the LLM when no relevant chunk is found) and the hallucination filter (truncating pivot phrases) further reduce but do not guarantee elimination of hallucinated content.

**The three steps:**

1. **Index** — Your documents are split into chunks, converted to numbers (embeddings), and stored in a database. This happens once at startup.
2. **Retrieve** — When you ask a question, the system finds the most relevant chunks using both keyword search (BM25) and semantic search (vector similarity).
3. **Generate** — The retrieved chunks are given to the LLM as context. The LLM reads them and writes an answer grounded in your documents.

**Why this system goes further than basic RAG:**

Most RAG systems stop at retrieve-and-generate. This system adds:
- Query expansion — searches with 3 versions of your question for better recall
- Query classification — adjusts how many chunks to retrieve based on query type
- Type-aware reranking — a second LLM pass re-scores chunks with prompts tailored per document type
- Confidence check — skips the LLM entirely if no relevant chunks are found
- Hallucination filter — catches and truncates responses where the model starts fabricating

---

## Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEX TIME (once)                           │
│                                                                     │
│  Documents / URLs                                                   │
│       │                                                             │
│       ▼                                                             │
│  ① CHUNKING ──────── PDF: 5-sentence windows per page              │
│       │               DOCX: paragraph groups + table rows          │
│       │               XLSX/CSV: one row → col=value pairs          │
│       │               PPTX: one slide per chunk                    │
│       │               HTML: tag-stripped sentence windows          │
│       │               TXT/MD: one line per chunk                   │
│       │               → All chunks truncated: 300 words / 1200 chars│
│       ▼                                                             │
│  ② EMBEDDING ─────── bge-base-en-v1.5 → 768-dim vector            │
│       │               Batches of 50 → stored in ChromaDB on disk   │
│       ▼                                                             │
│  ChromaDB vector index  +  BM25 keyword index (in-memory)          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       QUERY TIME (every question)                   │
│                                                                     │
│  User question: "How many hours do cats sleep?"                     │
│       │                                                             │
│       ▼                                                             │
│  ④ CLASSIFY ───────── summarise / comparison / factual / general   │
│       │               → sets retrieval depth (5 / 10 / 15 / 20)   │
│       ▼                                                             │
│  ③ EXPAND ─────────── LLM generates 2 rewrites → 3 total queries  │
│       │               "sleep duration cats", "feline rest habits"  │
│       ▼                                                             │
│  ⑤ HYBRID SEARCH ──── (run 3× for each expanded query)            │
│       ├── Dense:  query embedding → ChromaDB cosine similarity      │
│       ├── BM25:   term frequency × IDF → keyword score             │
│       └── Fuse:   0.5 × dense + 0.5 × BM25                        │
│                   best score per chunk across all 3 queries         │
│                   → top 20 chunks                                   │
│       ▼                                                             │
│  ⑦ CONFIDENCE ──────── best_score >= 0.40?                        │
│       │               NO  → "I don't have enough information"       │
│       │               YES → continue                                │
│       ▼                                                             │
│  ⑥ RERANK ─────────── LLM scores each of top 20 chunks (1–10)     │
│       │               7 different prompts — one per document type   │
│       │               → top 5 most relevant chunks                  │
│       ▼                                                             │
│  SYNTHESIZE ────────── LLM reads top 5 chunks + conversation memory│
│       │               writes grounded answer with source citations  │
│       ▼                                                             │
│  ⑧ HALLUCINATION FILTER ── scan for no-info + pivot phrases       │
│       │                     truncate at pivot if both found         │
│       ▼                                                             │
│  ⑪ SOURCE CITATIONS ─── type-aware label per chunk                 │
│       │               pdf→p3, xlsx→row12, pptx→slide4, html→s2     │
│       ▼                                                             │
│  Answer: "Cats sleep 12–16 hours [cat_facts.pdf p1]"               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    STARTUP / INDEX MANAGEMENT                       │
│                                                                     │
│  ⑫ REBUILD DECISION                                                │
│       ├── existing >= current → SKIP (load from disk, ~1 sec)      │
│       ├── existing > 0, stale → DELETE all → RE-EMBED all          │
│       └── existing == 0      → EMBED all (batches of 50)           │
│                                                                     │
│  ⑬ BM25 REBUILD                                                    │
│       ├── Built at startup from all local chunks                    │
│       └── Rebuilt after every URL/file upload (once per batch)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      AGENT MODE (multi-step)                        │
│                                                                     │
│  Complex question: "What is 20% of the salary in the resume?"       │
│       │                                                             │
│  ⑨ REACT LOOP                                                      │
│       ├── THINK: "I need the salary → use rag_search"              │
│       ├── ACT:   TOOL: rag_search(candidate salary)                │
│       ├── OBSERVE: "Annual salary: $95,000 [resume.pdf p2]"        │
│       ├── THINK: "Now calculate 20% of 95000"                      │
│       ├── ACT:   TOOL: calculator(95000 * 0.20)                    │
│       ├── OBSERVE: 19000.0                                          │
│       └── ACT:   TOOL: finish(20% of $95,000 = $19,000)           │
│                                                                     │
│  Tools: rag_search │ calculator │ summarise │ sentiment │ translate │ finish │
│                                                                     │
│  ⑭ CONVERSATION MEMORY                                             │
│       ├── Every Q&A turn stored as {role, content}                 │
│       ├── Full history prepended to every LLM synthesis call       │
│       └── Cleared by user (Clear button) or on restart             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      URL INGESTION                                   │
│                                                                     │
│  Paste URL → ⑩ TYPE DETECTION                                      │
│       ├── 1. Content-Type header   (application/pdf → pdf)         │
│       ├── 2. File extension in URL (/report.pdf → pdf)             │
│       ├── 3. PDF magic bytes       (content[:4] == b'%PDF' → pdf)  │
│       └── 4. Default → html                                         │
│       → fetch → write tempfile → chunker → index → BM25 rebuilt    │
│                                                                     │
│  Recursive crawl (optional):                                         │
│       seed URL → depth 1–3 → same-domain links only                │
│       → utility URL filter (login, cart, search, premium …)        │
│       → optional topic_filter (keyword in URL path)                │
│       → progress_callback per page → live st.status() log          │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                      TOPIC SEARCH                                    │
│                                                                     │
│  User query → DuckDuckGo HTML POST endpoint (no API key)           │
│       → parse <a class="result__a"> anchors                         │
│       → filter ad links (/y.js?)                                    │
│       → top N result URLs                                           │
│       → each URL crawled via chunk_url_recursive                    │
│       → same-domain constraint + utility filter apply               │
│       → all chunks merged → index → BM25 rebuilt                   │
└─────────────────────────────────────────────────────────────────────┘
```

The circled numbers ① – ⑭ match the sections below.

---

## 1. Document Chunking

**What it is:** Breaking a document into small, individually searchable pieces called chunks.

**Why it is needed:** A language model and a vector database can only process a limited amount of text at once. A 100-page PDF cannot be fed to the LLM as a single blob — it must be broken into pieces so that the most relevant pieces can be retrieved for any given question. The quality of chunking directly determines the quality of retrieval.

**The core challenge — chunk size matters:**
- Too small (1–2 sentences): each chunk lacks enough context to carry meaning
- Too large (whole pages): too much irrelevant text gets included, embedding limit exceeded
- The right size depends on the document format — prose, structured data, and slides all need different strategies

**How each format is handled:**

| Format | Strategy | Reasoning |
|--------|----------|-----------|
| **TXT** | One line per chunk | Plain text files typically store one fact or note per line. Line boundaries are the natural semantic unit. |
| **Markdown** | One line per chunk, syntax stripped | Same as TXT, but heading markers (`#`), bold (`**`), italic (`_`), code fences, image tags, and link syntax are removed so the model sees clean prose. |
| **PDF** | 5-sentence sliding window per page | PDFs contain continuous prose. A 5-sentence window keeps a complete thought together. Page boundaries are respected — no window spans two pages, preventing cross-page contamination. |
| **DOCX** | Groups of 3 paragraphs + table rows individually | Word documents mix narrative paragraphs with tables. Paragraphs are grouped in threes. Each table row is extracted as `column_name=value` pairs. Merged cells are deduplicated. |
| **XLSX / XLS** | One row per chunk as `col=value \| col=value` | Spreadsheets are structured records, not prose. Converting to `Name=Alice \| Role=Engineer \| Salary=90000` preserves column context so a query like "Alice's salary" can match the row. |
| **CSV** | One row per chunk as `col=value \| col=value` | Identical to XLSX. DictReader uses the header row for column names automatically. |
| **PPTX** | All text shapes on one slide per chunk | A slide is a self-contained idea. Merging all text shapes (titles, bullets, text boxes) from one slide keeps related content together. |
| **HTML** | 5-sentence sliding window, tags stripped | BeautifulSoup strips all tags and boilerplate first. The clean text is chunked with the same 5-sentence window used for PDFs. |

**Chunk truncation — the safety limit:**

Every chunk is clipped before storage:
```
Maximum 300 words  AND  Maximum 1200 characters
```
Both limits are enforced — whichever is hit first cuts the chunk. This keeps chunks within the BGE model's effective range (512 tokens ≈ 300–350 words of typical English).

**What a finished chunk looks like:**
```python
{
  'text':   "Cats typically sleep 12 to 16 hours per day...",
  'source': "pdfs/cat_facts.pdf",
  'type':   "pdf",
  'page':   3,
  'start':  0,   # sentence index within the page
  'end':    5,
}
```

---

## 2. Embedding

**What it is:** Converting a chunk of text into a list of 768 numbers (a vector) that captures its meaning.

**Why numbers instead of text:** A database can compare numbers mathematically. By converting text to vectors, the system can measure how close any two pieces of text are in meaning — using cosine similarity.

**How it works:**

`bge-base-en-v1.5` is a BERT-based bi-encoder model trained specifically for retrieval. It reads a sentence and produces 768 numbers encoding its meaning. Each dimension captures a different abstract property of the text.

```
"Cats sleep 16 hours a day"           → [0.12, -0.34,  0.89, ...,  0.05]
"Felines rest for most of the day"    → [0.11, -0.31,  0.87, ...,  0.06]
"The stock market crashed in October" → [0.87,  0.42, -0.21, ..., -0.73]
```

The first two are about the same thing — their vectors are close together. The third is unrelated — its vector is far away.

**Cosine similarity — measuring closeness:**

```
similarity = cos(θ) = (A · B) / (|A| × |B|)

Result: 1.0 = identical meaning
        0.9 = very similar
        0.5 = loosely related
        0.0 = unrelated
```

**Storage and batching:**

Embeddings are stored in ChromaDB (persistent vector database). On restart, existing embeddings are loaded — documents are only re-embedded if new files were added. Embedding is done in batches of 50 chunks.

**Embedding truncation (separate from chunk truncation):**

Before embedding, text is truncated to **200 words AND 1200 chars** — both enforced simultaneously. This is stricter than chunk storage truncation (300 words OR 1200 chars) because the embedding model's effective context window is smaller than the storage limit.

---

## 3. Query Expansion

**What it is:** Generating 2 alternative phrasings of the user's question and running all 3 versions through retrieval.

**The problem it solves — vocabulary mismatch:**

A document might say: *"Domestic cats typically rest for 12 to 16 hours daily."*

But the user asks: *"How many hours do cats sleep?"*

"Rest" and "sleep" mean the same thing but are different tokens. BM25 will miss the chunk. Dense search might find it — but is not guaranteed for domain-specific terminology or acronyms.

**How the LLM generates rewrites:**

```
Original:  "How many hours do cats sleep?"
Rewrite 1: "What is the daily sleep duration of domestic cats?"
Rewrite 2: "Cat resting habits — how long do they sleep each day?"
```

All 3 queries are run through the full hybrid retrieval pipeline independently. For each chunk, only the **highest score across all 3 queries** is kept — preventing the same chunk from being counted three times.

**Result:** Higher recall with no reduction in precision (low-scoring chunks still get filtered by the reranker).

---

## 4. Query Classification

**What it is:** Automatically detecting what kind of question is being asked and adjusting how many chunks to retrieve.

**Why retrieval depth should vary:**

- *"What is the candidate's GPA?"* — factual, needs exactly one precise answer. Retrieving 20 chunks adds noise.
- *"Summarise this resume"* — needs as much content as possible. Retrieving 5 chunks gives an incomplete summary.

**The 4 types and their retrieval depths:**

| Type | Detection keywords (examples) | Chunks retrieved | Why |
|------|-------------------------------|-------------------|-----|
| `summarise` | "summarise", "summary", "overview", "describe" | Top 20 | Wide coverage needed — must draw from many document sections |
| `comparison` | "compare", "difference", "versus", "vs", "better" | Top 15 | Two subjects need separate supporting evidence |
| `factual` | "who", "when", "how many", "what is", "which" | Top 5 | Precise answer needed — fewer chunks means less noise |
| `general` | (everything else) | Top 10 | Balanced default |

**Why the priority order matters:**

Classification is checked in this exact order: **summarise → comparison → factual → general**.

*"Summarise the differences between the two candidates"* matches both "summarise" and "difference" (comparison). Because summarise is checked first, it wins — and 20 chunks are retrieved. Reversing the order would retrieve only 15, missing some content.

---

## 5. Hybrid Search

**What it is:** Running two completely different search algorithms simultaneously and combining their scores.

**Why two algorithms:**

- **Dense (semantic) search** understands meaning but can miss exact keywords.
- **BM25 (lexical) search** is excellent at exact keyword matches but has no concept of meaning.

Running both gives the system the strengths of each.

**How Dense Retrieval works:**

The query is embedded into a 768-dim vector using the same BGE model. ChromaDB searches for the most similar chunk vectors and returns cosine distance, converted to similarity:
```
dense_score = 1 - cosine_distance
```

**How BM25 (Best Match 25) works:**

BM25 scores each chunk by asking: *"Does this chunk contain the query words, and are those words rare enough to be meaningful?"*

```
BM25 score = Σ (for each query word):
    IDF(word) × TF(word, chunk) × (k1 + 1)
               ─────────────────────────────────────────
               TF(word, chunk) + k1 × (1 - b + b × chunk_length / avg_length)
```

In plain terms:
- **IDF** (Inverse Document Frequency): rare words score higher. "Salary" in 3 of 1000 chunks is meaningful. "The" in every chunk is not.
- **TF** (Term Frequency): a word appearing 3 times scores higher than once, but with diminishing returns.
- **Length normalisation**: a short chunk that mentions a keyword once scores comparably to a long chunk that mentions it once.

BM25 scores are normalised to [0, 1] by dividing by the highest score in the result set.

**Score fusion:**

```
final_score = 0.5 × dense_score + 0.5 × bm25_score
```

Equal weighting (alpha = 0.5) gives both signals equal influence. The system takes the **best score per chunk across all 3 expanded queries** and returns the top 20.

**Concrete example — why hybrid wins:**

Spreadsheet row: `Department=Engineering | Headcount=42`
Query: *"How many engineers does the company have?"*

| Method | Score | Why |
|--------|-------|-----|
| Dense only | 0.61 | Captures "engineers ≈ Engineering" but "how many ≈ Headcount" is weak |
| BM25 only | 0.38 | Partial match on "Engineering" but "how many" matches nothing |
| **Hybrid** | **0.50** | Combines both signals — each lifts the weakness of the other |

---

## 6. Type-Aware LLM Reranking

**What it is:** A second-pass relevance scoring step where the LLM reads each retrieved chunk and scores it 1–10 based on how relevant it is to the query.

**Why a second pass is needed:**

Hybrid search retrieves the top 20 chunks efficiently but imprecisely. Embedding similarity and BM25 are approximate signals. The reranker uses the full LLM for a deeper read of each chunk.

**The fundamental problem with generic reranking:**

A standard reranker prompt asks: *"Is this passage relevant?"* This works for prose but breaks for structured data.

Consider a spreadsheet row:
```
Name=Alice Chen | Title=Senior ML Engineer | Salary=115000 | YearsExp=7
```

To a reranker expecting prose, this looks like meaningless key-value noise. The actual answer (115000) is right there — but a generic prompt cannot see it clearly.

**The 7-prompt solution:**

| Document type | How the prompt frames the content |
|--------------|----------------------------------|
| **PDF** | *"This is a passage from a PDF document."* |
| **DOCX** | *"This is a paragraph from a Word document."* |
| **XLSX / CSV** | *"This is a row of structured data from a spreadsheet. Each field is shown as column_name=value."* |
| **PPTX** | *"This is the text content of a presentation slide."* |
| **HTML** | *"This is a section from a webpage."* |
| **TXT** | *"This is a passage of plain text."* |
| **MD** | *"This is a section from a Markdown document."* |

By telling the LLM what kind of content it is looking at, the model understands the format and scores structured data correctly.

**The output:**

The LLM returns a score from 1–10 for each chunk. The 20 chunks are sorted by this score. The top 5 go to answer synthesis. The UI sidebar shows both the pre-rerank list (top 20) and post-rerank list (top 5) so you can see the reranker's impact.

---

## 7. Confidence Check

**What it is:** A threshold gate that decides whether retrieval found anything good enough to use.

**The problem it prevents:**

Without a confidence gate, the system would always call the LLM — even when no relevant documents exist. The retrieval pipeline still returns the top chunks (they are just the least-bad matches from unrelated documents). The LLM receives irrelevant context and, rather than saying "I don't know," makes up a plausible-sounding answer. This is hallucination.

**How it works:**

After reranking, the system checks the best-scoring chunk's similarity:
```
if best_reranked_similarity >= 0.40:
    is_confident = True   → proceed to LLM synthesis
else:
    is_confident = False  → return fixed "I don't have enough information" message
```

The threshold of 0.40 (on a 0–1 cosine similarity scale) is permissive enough to handle imperfect vocabulary matches while still blocking clearly irrelevant content.

**What the UI shows:** A confidence badge — green (✓ Confident) or amber (⚠ Low confidence).

---

## 8. Hallucination Filter

**What it is:** A post-generation text scan that detects and truncates responses where the LLM started to hallucinate after acknowledging it had no information.

**The specific failure mode it catches:**

Even when the confidence check passes, language models have a known failure pattern — they start a response honestly but then "helpfully" continue beyond what the context supports:

```
"There is no information in the provided documents about the Q3 revenue figures.
However, I can tell you that based on typical industry trends, revenue likely..."
```

The first sentence is correct. The second sentence is pure fabrication. *"However, I can tell you"* is the pivot point.

**How the two-list filter works:**

**No-info phrases** (the model is signalling it has no context):
```
"there is no information"
"i couldn't find"
"i could not find"
"the provided context does not"
"the provided documents do not"
"no information in the provided"
"not mentioned in the"
"not found in the"
```

**Pivot phrases** (the model is about to hallucinate):
```
"however,"
"but i can"
"but,"
"that said,"
"nevertheless,"
"i can tell you"
"i can provide"
```

If a no-info phrase is found **followed by** a pivot phrase, the response is truncated at the pivot:

```
Before: "There is no information about revenue figures. However, I can tell you..."
                                                          ↑ truncate here
After:  "There is no information about revenue figures."
```

**What it does not do:** If there is no no-info phrase (the model answered confidently from context), pivot phrases are ignored. The filter only activates when the model has already signalled uncertainty.

---

## 9. ReAct Agent Loop

**What it is:** An autonomous loop where the LLM decides what to do next, executes a tool, observes the result, and repeats until it has a complete answer.

**What ReAct stands for:** Reasoning + Acting. The model alternates between thinking (internal reasoning) and acting (calling a tool).

**Why an agent loop is needed:**

Some questions cannot be answered in one retrieval step:
- *"What is 20% of the salary in the resume?"* — requires RAG search AND calculator
- *"Summarise the entire resume"* — requires searching multiple sections AND combining
- *"What is the sentiment of the job description?"* — requires RAG search AND sentiment analysis

**The loop in detail:**

```
Step 1 — THINK
  LLM receives the question and system prompt.
  Output: "TOOL: rag_search(candidate salary)"

Step 2 — ACT
  System parses "rag_search" and "candidate salary".
  Runs full RAG pipeline for "candidate salary".
  Result: "Source: resume.pdf p2 — Annual salary: $95,000"

Step 3 — OBSERVE
  Tool result is added to the conversation context.

Step 4 — THINK again
  LLM now has the salary.
  Output: "TOOL: calculator(95000 * 0.20)"

Step 5 — ACT
  Calculator evaluates: 95000 * 0.20 = 19000.0

Step 6 — FINISH
  Output: "TOOL: finish(20% of the salary is $19,000)"
```

**The 6 tools:**

| Tool | Input | What it does |
|------|-------|-------------|
| `rag_search` | A search query | Runs expand → classify → hybrid retrieve → rerank → return formatted chunks |
| `calculator` | An arithmetic expression | Evaluates safely — only `0123456789+-*/(). ` allowed |
| `summarise` | A passage of text | Adaptive length: 2–3 sentences for short, 6–8 for long text |
| `sentiment` | Text or a query | Returns Sentiment, Tone, Key phrases, Explanation |
| `translate` | `"Language: text"` or just text | Translates to any target language; queries under 15 words search the knowledge base first |
| `finish` | The final answer | Ends the loop and returns to the user |

**Translate tool routing logic:**

The translate tool uses a two-path strategy depending on whether the input looks like a search query or a full passage of text:

```
Input < 15 words (query) → _tool_rag_search first → translate retrieved content
Input ≥ 15 words (text)  → translate directly, no search needed

If no language prefix:   default to English
If search returns empty: fall back to translating the original input
```

This avoids translating a bare keyword like "machine learning" when the user actually wants a translated explanation of the topic from their documents.

**Tool call parsing — two regex patterns:**

The LLM does not always produce perfectly formatted tool calls. Two fallback patterns are used:
```
Pattern 1 — with parentheses (preferred):
  TOOL: rag_search(what is the GPA)
  regex: TOOL:\s*(\w+)\s*\(\s*(.+?)\s*\)

Pattern 2 — without parentheses (fallback):
  TOOL: rag_search what is the GPA
  regex: TOOL:\s*(\w+)\s+(.+)
```

If neither matches (malformed output), the system injects a correction prompt and retries up to 2 times. After 2 failed retries, it uses the raw LLM text as the answer.

**Fast paths — bypassing the loop for known patterns:**

- **Summarise fast path:** Runs 4 targeted searches simultaneously: `work experience`, `education`, `skills projects`, `summary contact`. Combines all results and feeds to LLM for a single synthesis. Faster and more complete than the loop.
- **Sentiment fast path:** If the query is under 10 words (a question rather than a passage), the system searches for the subject first and strips chunk metadata labels before analysis.

---

## 10. URL Type Detection

**What it is:** A fallback chain that determines what kind of document a URL points to, even when the server does not tell you clearly.

**Why this is harder than it sounds:**

- Many servers return `application/octet-stream` for all binary files
- Some servers return `text/html` for everything including PDFs
- Some URLs have no file extension (`/download?id=123`)
- Some URLs have misleading extensions due to URL rewriting

**The 4 checks in order:**

```
1. Content-Type header
   → "application/pdf"        → pdf
   → "text/html"              → html
   → "application/vnd...."   → xlsx / docx / etc.
   → If ambiguous → fall through

2. File extension in the URL path
   → Strip query params: "/files/report.pdf?v=2" → "/files/report.pdf"
   → Extract extension: ".pdf" → pdf
   → If no recognisable extension → fall through

3. PDF magic bytes
   → Read the first 4 bytes of the downloaded content
   → If content[:4] == b'%PDF' → pdf
   → This catches PDFs served with wrong Content-Type headers
   → If not PDF magic bytes → fall through

4. Default → html
   → Correct for the vast majority of bare URLs with no extension
```

**Why magic bytes specifically for PDF:**

The PDF specification mandates that every valid PDF starts with `%PDF`. This signature is present in 100% of valid PDFs regardless of server headers. No other common format has an equivalently reliable signature at byte 0, so magic byte detection is only implemented for PDF.

**After type detection:**

Binary formats (PDF, DOCX, XLSX, PPTX, XLS) are written to a temporary file, processed, then the temp file is deleted. Text formats (HTML, MD, CSV, TXT) are decoded from response bytes and processed in memory.

---

## 11. Source Citation Labels

**What it is:** A type-aware label generator that produces a human-readable location reference for every chunk used in an answer.

**Why citations need to be type-aware:**

"Page 3" is meaningful in a PDF. It is meaningless in a spreadsheet. "Row 47" is meaningful in a spreadsheet. It is meaningless in a PDF.

**How the label is constructed — one rule per document type:**

```
PDF   → "report.pdf p3"          ← page number
XLSX  → "data.xlsx row12"         ← row index within the sheet
CSV   → "contacts.csv row8"       ← row index
PPTX  → "deck.pptx slide4"        ← slide number
HTML  → "article.html s2"         ← sentence-window index
DOCX  → "resume.docx L4-6"        ← paragraph line range
TXT   → "notes.txt L10-14"        ← line range
MD    → "readme.md L22-25"        ← line range
URL   → "https://example.com/..."  ← URL truncated to 60 chars
```

**Example citations in answers:**
```
"Cats typically sleep 12 to 16 hours per day. [cat_facts.pdf p1]"
"Alice Chen holds the title of Senior ML Engineer. [employees.xlsx row5]"
"The company was founded in 2018. [about.html s3]"
```

---

## 12. Embedding Rebuild Decision

**What it is:** A startup check that decides whether to reuse existing embeddings from disk or rebuild the vector index from scratch.

**Why this matters:**

Embedding is the most expensive operation. For 500 chunks, embedding takes 30–60 seconds. The system avoids unnecessary re-embedding by comparing the number of chunks in ChromaDB against the number found on disk.

**The three cases:**

```
Case 1 — No changes since last run
  existing_count >= current_chunk_count
  → SKIP embedding. Load existing collection. Starts in ~1 second.

Case 2 — New documents added to ./docs/
  existing_count > 0  AND  existing_count < current_chunk_count
  → DELETE all existing embeddings.
  → RE-EMBED everything from scratch.
  → Why delete all? BM25 must rebuild over the complete set anyway.
    A delta update for ChromaDB would leave the two indexes out of sync.

Case 3 — First run / database deleted
  existing_count == 0
  → EMBED all chunks in batches of 50.
  → ChromaDB persists to disk automatically.
```

**The >= comparison (not ==) in Case 1:**

URL chunks are added to ChromaDB live during a session. On the next startup, `existing` is higher than `current` (local file count), so the system correctly skips re-embedding and keeps the previously indexed URL chunks.

---

## 13. BM25 Index Rebuild

**What it is:** An in-memory keyword search index that is rebuilt from scratch every time new documents are added.

**Why BM25 is rebuilt instead of updated incrementally:**

BM25 is a statistical model — its scores for every existing chunk change whenever the collection changes. This is because BM25 uses IDF (Inverse Document Frequency), which depends on how rare a word is **across the entire collection**:

```
IDF(word) = log( total_chunks / chunks_containing_word )
```

If "salary" appears in 5 of 100 chunks, IDF = log(100/5) ≈ 3.0.
After adding 50 chunks that all mention "salary": IDF = log(150/55) ≈ 1.0.

Every existing chunk that contains "salary" now has a lower BM25 score — even though the chunk itself has not changed. This global dependency means there is no valid incremental update — the entire index must be rebuilt.

**When BM25 is rebuilt:**
```
1. Application startup  → built once from all local document chunks
2. URL ingestion        → rebuilt after the URL chunks are added
3. File upload via UI   → rebuilt once for the whole batch (not once per file)
```

Rebuilding once per batch (not per file) avoids doing O(n) work multiple times when uploading 10 files at once.

---

## 14. Conversation Memory

**What it is:** A rolling list of previous messages included in every LLM call so the model can answer follow-up questions.

**Why it is needed:**

Without memory, every question is answered in isolation. The user cannot ask:
- *"What is her salary?"* (referring to someone mentioned in the previous answer)
- *"Summarise that in 2 sentences"* (referring to the previous response)
- *"What about the other candidate?"* (continuing a comparison thread)

**How history is structured:**

```python
conversation = [
    {'role': 'user',      'content': "What is Alice's job title?"},
    {'role': 'assistant', 'content': "Alice holds the title of Senior ML Engineer. [resume.pdf p2]"},
    {'role': 'user',      'content': "What is her salary?"},
    # "her" refers to Alice — the model knows because the previous turn is included
]
```

**What is stored and what is not:**
```
Stored:     every user question and every assistant response (final answer text only)
Not stored: retrieved chunks, reranker scores, intermediate pipeline steps
```

Retrieved chunks are included in the synthesis prompt for the current turn only. They are not carried into history. This keeps the context window manageable as the conversation grows.

**Session scope:** History lives in memory on the `VectorStore` instance. It resets on restart or when the user clicks Clear. Clearing only wipes the history — ChromaDB, BM25, and indexed documents are unaffected.
