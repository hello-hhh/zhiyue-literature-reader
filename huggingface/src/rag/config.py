"""config.py — constants only, no functions, no classes. HF Space version.

All tunable values for the HF Space deployment live here.
Values are intentionally smaller than the local version to avoid CPU timeouts
on HuggingFace free-tier hardware.
"""

__all__ = [
    'EMBEDDING_MODEL', 'LANGUAGE_MODEL', 'LANGUAGE_MODEL_FALLBACKS',
    'CHROMA_COLLECTION',
    'EXT_TO_TYPE',
    'SIMILARITY_THRESHOLD', 'TOP_RETRIEVE', 'TOP_RERANK',
    'TXT_CHUNK_SIZE', 'TXT_CHUNK_OVERLAP', 'PDF_CHUNK_SENTENCES',
    'DOCX_CHUNK_PARAS', 'PPTX_CHUNK_SLIDES', 'HTML_CHUNK_SENTENCES',
    'URL_CRAWL_MAX_DEPTH', 'URL_CRAWL_MAX_PAGES',
]

# ── Models ────────────────────────────────────────────────────────────────────

# sentence-transformers model — runs locally on CPU inside the Space
EMBEDDING_MODEL      = 'BAAI/bge-base-en-v1.5'

# Primary LLM — confirmed working on featherless-ai provider
LANGUAGE_MODEL       = 'HuggingFaceH4/zephyr-7b-beta'

# Tried in order until one responds; featherless-ai is the most reliable provider
# for zephyr on the free tier. Only non-gated models are listed here.
LANGUAGE_MODEL_FALLBACKS = [
    'HuggingFaceH4/zephyr-7b-beta',          # confirmed working on featherless-ai
    'mistralai/Mistral-7B-Instruct-v0.1',    # older fallback — not gated
]

# ── ChromaDB ──────────────────────────────────────────────────────────────────

CHROMA_COLLECTION    = 'rag_docs'

# ── Retrieval thresholds ──────────────────────────────────────────────────────

# Lower than the local 0.55 because sentence-transformers similarity distributions
# differ from Ollama's embedding model distributions.
SIMILARITY_THRESHOLD = 0.40

# Reduced from 20/5 in the local version to avoid timeout on HF free CPU
TOP_RETRIEVE         = 5   # max chunks returned from hybrid search
TOP_RERANK           = 3   # max chunks passed to the LLM after reranking

# ── File extension → type mapping ─────────────────────────────────────────────

# Maps every supported file extension → canonical type key used by DocumentLoader
EXT_TO_TYPE = {
    '.pdf':      'pdf',
    '.txt':      'txt',
    '.docx':     'docx',
    '.doc':      'docx',    # legacy Word — treated identically to .docx
    '.xlsx':     'xlsx',
    '.xls':      'xlsx',    # routed to _chunk_xls (xlrd) inside dispatcher
    '.pptx':     'pptx',
    '.ppt':      'pptx',
    '.csv':      'csv',
    '.md':       'md',
    '.markdown': 'md',
    '.html':     'html',
    '.htm':      'html',
}

# ── Chunk size constants ──────────────────────────────────────────────────────

# TXT / MD: 1 line per chunk — preserves individual data points
TXT_CHUNK_SIZE       = 1
TXT_CHUNK_OVERLAP    = 0

# Format chunk sizes — tuned per type for semantic coherence
PDF_CHUNK_SENTENCES  = 5   # sentences per sliding window on a PDF page
DOCX_CHUNK_PARAS     = 3   # paragraphs (and table rows) per chunk
PPTX_CHUNK_SLIDES    = 1   # one chunk per slide for maximum granularity
HTML_CHUNK_SENTENCES = 5   # sentences per window after tag stripping

# ── Recursive URL crawl defaults ──────────────────────────────────────────────
# Smaller than the local defaults — HF free CPU is slower, so fewer pages
# to avoid timeout. Users can override both from the UI.

# How many link-levels deep to follow from the seed URL (1 = direct links only)
URL_CRAWL_MAX_DEPTH = 1

# Maximum total pages to fetch and index in one recursive crawl session
URL_CRAWL_MAX_PAGES = 10
