"""config.py — All configuration constants for the local version.

WHY THIS FILE EXISTS:
    Every value that might change between environments (local dev, CI, Docker)
    lives here in one place. Nothing else should hardcode paths or thresholds.

HOW ENVIRONMENT VARIABLES WORK:
    Each constant checks for an environment variable first, then falls back to
    a sensible default. This means you can override any value without changing code:

        RAG_DOCS_ROOT=./my_docs python main.py
        RAG_CHROMA_DIR=/tmp/chroma streamlit run app.py

CONSTANTS (never changed by environment variables):
    Model names, chunk sizes, retrieval thresholds — these are tuned values
    and should only change through a deliberate code edit and benchmark check.
"""

import os

__all__ = [
    'EMBEDDING_MODEL', 'LANGUAGE_MODEL',
    'DOCS_ROOT', 'DOC_FOLDERS', 'EXT_TO_TYPE',
    'CHROMA_DIR', 'CHROMA_COLLECTION', 'LOG_FILE', 'BENCHMARK_FILE', 'BENCHMARK_CSV',
    'BENCHMARK_DOCS_DIR', 'TOOL_BENCHMARK_FILE',
    'SIMILARITY_THRESHOLD', 'TOP_RETRIEVE', 'TOP_RERANK',
    'TXT_CHUNK_SIZE', 'TXT_CHUNK_OVERLAP', 'PDF_CHUNK_SENTENCES',
    'DOCX_CHUNK_PARAS', 'PPTX_CHUNK_SLIDES', 'HTML_CHUNK_SENTENCES',
    'URL_CRAWL_MAX_DEPTH', 'URL_CRAWL_MAX_PAGES',
]

# ── Models ─────────────────────────────────────────────────────────────────────

# The embedding model turns text into numbers for similarity search
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

# The language model generates answers and performs reranking
LANGUAGE_MODEL  = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'

# ── File system paths ──────────────────────────────────────────────────────────
# Override with environment variables to run from a different location
# e.g.  RAG_DOCS_ROOT=/data/documents python main.py

# Root folder where all document subfolders live
DOCS_ROOT = os.environ.get('RAG_DOCS_ROOT', './docs')

# Each document type has its own subfolder under DOCS_ROOT
DOC_FOLDERS = {
    'pdf':  os.path.join(DOCS_ROOT, 'pdfs'),
    'txt':  os.path.join(DOCS_ROOT, 'txts'),
    'docx': os.path.join(DOCS_ROOT, 'docx'),
    'xlsx': os.path.join(DOCS_ROOT, 'xlsx'),
    'pptx': os.path.join(DOCS_ROOT, 'pptx'),
    'csv':  os.path.join(DOCS_ROOT, 'csv'),
    'md':   os.path.join(DOCS_ROOT, 'md'),
    'html': os.path.join(DOCS_ROOT, 'html'),
}

# Maps every supported file extension to its canonical type key
EXT_TO_TYPE = {
    '.pdf':      'pdf',
    '.txt':      'txt',
    '.docx':     'docx',
    '.doc':      'docx',      # legacy Word — treated the same as .docx
    '.xlsx':     'xlsx',
    '.xls':      'xlsx',      # routed to _chunk_xls (xlrd) inside the dispatcher
    '.pptx':     'pptx',
    '.ppt':      'pptx',
    '.csv':      'csv',
    '.md':       'md',
    '.markdown': 'md',
    '.html':     'html',
    '.htm':      'html',
}

# Where ChromaDB stores its persistent vector index on disk
CHROMA_DIR        = os.environ.get('RAG_CHROMA_DIR', './chroma_db')

# Name of the ChromaDB collection that holds all document chunks
CHROMA_COLLECTION = 'rag_docs'

# ── Log and benchmark files ────────────────────────────────────────────────────

# Every query is appended here as a JSON entry for analytics
LOG_FILE       = os.environ.get('RAG_LOG_FILE', 'rag_logs.json')

# Benchmark scores from each run are saved here for before/after comparison
BENCHMARK_FILE = os.environ.get('RAG_BENCHMARK_FILE', 'benchmark_results.json')

# Full per-question results exported as a spreadsheet-friendly CSV after each run
BENCHMARK_CSV  = os.environ.get('RAG_BENCHMARK_CSV',  'benchmark_results.csv')

# Folder containing sample documents committed to the repo for benchmarking.
# These are loaded automatically when running python main.py --benchmark.
BENCHMARK_DOCS_DIR = os.environ.get('RAG_BENCHMARK_DOCS_DIR', './benchmark_docs')

# Tool benchmark results (calculator, sentiment, summarise) saved here as JSON
TOOL_BENCHMARK_FILE = os.environ.get('RAG_TOOL_BENCHMARK_FILE', 'tool_benchmark_results.json')

# ── Retrieval thresholds ───────────────────────────────────────────────────────
# Do NOT change these without running a benchmark first — small changes
# can significantly affect faithfulness and recall scores.

# Minimum cosine similarity for a retrieval result to be considered "confident"
SIMILARITY_THRESHOLD = 0.40

# How many chunks to pull from hybrid search before reranking
TOP_RETRIEVE         = 20

# How many chunks to keep after reranking (these go to the LLM)
TOP_RERANK           = 5

# ── Chunk size constants ───────────────────────────────────────────────────────

# TXT / MD: 1 line per chunk — preserves individual data points
TXT_CHUNK_SIZE       = 1
TXT_CHUNK_OVERLAP    = 0

# Number of sentences per sliding window for PDF pages
PDF_CHUNK_SENTENCES  = 5

# Number of paragraphs (or table rows) per DOCX chunk
DOCX_CHUNK_PARAS     = 3

# One slide per chunk for maximum granularity in presentations
PPTX_CHUNK_SLIDES    = 1

# Number of sentences per sliding window for HTML pages
HTML_CHUNK_SENTENCES = 5

# ── Recursive URL crawl defaults ──────────────────────────────────────────────
# These are default values only — the user can override both from the UI
# when recursive crawl is enabled. Changing these does not affect single-page
# URL fetching (chunk_url), only recursive crawl (chunk_url_recursive).

# How many link-levels deep to follow from the seed URL (1 = direct links only)
URL_CRAWL_MAX_DEPTH = 2

# Maximum total pages to fetch and index in one recursive crawl session
URL_CRAWL_MAX_PAGES = 25
