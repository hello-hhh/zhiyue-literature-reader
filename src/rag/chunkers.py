"""chunkers.py — Stateless chunkers for text-based document formats.

Handles plain text (TXT), Markdown (MD), CSV, and HTML — formats that
can be parsed without binary libraries. Binary formats (PDF, DOCX, XLSX,
XLS, PPTX) live in binary_chunkers.py.

Each function takes a file path and filename, reads the file, and returns
a list of chunk dicts. Chunk size parameters default to the values in
config.py but can be overridden in tests or special cases.

Third-party imports (beautifulsoup4) are done inside each function so a
missing library skips only that format instead of crashing on import.
"""

import csv as _csv
import os
import re
from typing import List

from src.rag.config import HTML_CHUNK_SENTENCES, TXT_CHUNK_OVERLAP, TXT_CHUNK_SIZE

__all__ = [
    'chunk_txt', 'chunk_md', 'chunk_csv', 'chunk_html',
    'chunk_txt_from_string', 'chunk_html_from_string',
    'truncate_chunk',
]


def chunk_txt(
    filepath: str,
    filename: str,
    chunk_size: int = TXT_CHUNK_SIZE,
    overlap: int = TXT_CHUNK_OVERLAP,
) -> List[dict]:
    """Chunk a plain text file using a sliding line window.

    Lines are grouped into windows of size chunk_size with a step of
    chunk_size - overlap so consecutive windows share a few lines.

    Args:
        filepath:   Absolute path to the .txt file.
        filename:   Display name stored in each chunk's 'source' field.
        chunk_size: Number of lines per chunk window.
        overlap:    Number of lines shared between consecutive windows.

    Returns:
        List of chunk dicts with keys: text, source, start_line, end_line, type.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"  [ERROR] Could not read '{filename}': {e}")
        return []

    step   = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(lines), step):
        window = lines[i: i + chunk_size]
        if not window:
            continue
        chunks.append({
            'text':       ' '.join(window),
            'source':     filename,
            'start_line': i + 1,
            'end_line':   i + len(window),
            'type':       'txt',
        })
    return chunks


def chunk_md(
    filepath: str,
    filename: str,
    chunk_size: int = TXT_CHUNK_SIZE,
    overlap: int = TXT_CHUNK_OVERLAP,
) -> List[dict]:
    """Chunk a Markdown file by stripping all Markdown syntax first.

    Heading markers, bold/italic stars, backticks, image tags, and link
    syntax are all removed so the embedding model sees clean prose.
    Then the clean text is chunked exactly like a plain text file.

    Args:
        filepath: Absolute path to the .md file.
        filename: Display name stored in each chunk's 'source' field.

    Returns:
        List of chunk dicts with type='txt'.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except Exception as e:
        print(f"  [ERROR] Could not read '{filename}': {e}")
        return []

    # Strip heading markers (# ## ###), bold/italic markers (* _),
    # inline code backticks, image syntax, and link syntax
    clean = re.sub(r'^#{1,6}\s*', '', raw, flags=re.MULTILINE)
    clean = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', clean)
    clean = re.sub(r'_{1,3}(.*?)_{1,3}',   r'\1', clean)
    clean = re.sub(r'`{1,3}[^`]*`{1,3}',   '',    clean)
    clean = re.sub(r'!\[.*?\]\(.*?\)',      '',    clean)
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)

    lines  = [line.strip() for line in clean.splitlines() if line.strip()]
    step   = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(lines), step):
        window = lines[i: i + chunk_size]
        if not window:
            continue
        chunks.append({
            'text':       ' '.join(window),
            'source':     filename,
            'start_line': i + 1,
            'end_line':   i + len(window),
            'type':       'md',
        })
    return chunks


def chunk_csv(filepath: str, filename: str) -> List[dict]:
    """Chunk a CSV file — each row becomes one key=value pair chunk.

    Uses csv.DictReader so column names are preserved in each chunk,
    giving the embedding model full context for each value.

    Args:
        filepath: Absolute path to the .csv file.
        filename: Display name stored in each chunk's 'source' field.

    Returns:
        List of chunk dicts with type='csv'.
    """
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = _csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                pairs = '; '.join(
                    f"{k}={v}" for k, v in row.items() if v and str(v).strip()
                )
                if not pairs:
                    continue
                chunks.append({
                    'text':       pairs,
                    'source':     filename,
                    'start_line': row_idx,
                    'end_line':   row_idx,
                    'type':       'csv',
                })
    except Exception as e:
        print(f"  [ERROR] Could not read '{filename}': {e}")
    return chunks


def chunk_html(
    filepath: str,
    filename: str,
    sentences_per_chunk: int = HTML_CHUNK_SENTENCES,
) -> List[dict]:
    """Chunk an HTML file using BeautifulSoup tag stripping and sentence windows.

    All HTML tags are stripped first; then the plain text is split into
    sentences and grouped into sliding windows.

    Args:
        filepath:            Absolute path to the .html file.
        filename:            Display name stored in each chunk's 'source' field.
        sentences_per_chunk: Number of sentences per chunk window.

    Returns:
        List of chunk dicts with type='html'. Empty list on parse error.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [WARNING] beautifulsoup4 not installed — skipping HTML. Install: pip install beautifulsoup4")
        return []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except Exception as e:
        print(f"  [ERROR] Could not open '{filename}': {e}")
        return []

    text      = soup.get_text(separator=' ', strip=True)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks    = []
    for i in range(0, len(sentences), sentences_per_chunk):
        window = sentences[i: i + sentences_per_chunk]
        if not window:
            continue
        chunks.append({
            'text':       ' '.join(window),
            'source':     filename,
            'start_line': i + 1,
            'end_line':   i + len(window),
            'type':       'html',
        })
    return chunks


def chunk_txt_from_string(
    text: str,
    source_name: str,
    chunk_size: int = TXT_CHUNK_SIZE,
    overlap: int = TXT_CHUNK_OVERLAP,
) -> List[dict]:
    """Chunk a plain-text string using a sliding line window.

    This is the string-based equivalent of chunk_txt() for use when the
    content has already been decoded from a URL response — no file on disk.

    Args:
        text:        Decoded text string to chunk.
        source_name: Short citation label (e.g. 'example.com/page').
        chunk_size:  Number of lines per chunk window.
        overlap:     Number of lines shared between consecutive windows.

    Returns:
        List of chunk dicts with keys: text, source, start_line, end_line, type.
    """
    lines  = [line.strip() for line in text.splitlines() if line.strip()]
    step   = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(lines), step):
        window = lines[i: i + chunk_size]
        if not window:
            continue
        chunks.append({
            'text':       ' '.join(window),
            'source':     source_name,
            'start_line': i + 1,
            'end_line':   i + len(window),
            'type':       'txt',
        })
    return chunks


def chunk_html_from_string(
    html_string: str,
    source_name: str,
    sentences_per_chunk: int = HTML_CHUNK_SENTENCES,
) -> List[dict]:
    """Chunk an HTML string using BeautifulSoup tag stripping and sentence windows.

    This is the string-based equivalent of chunk_html() for use when the
    content has already been decoded from a URL response — no file on disk.

    Args:
        html_string:         Decoded HTML string.
        source_name:         Short citation label (e.g. 'example.com/page').
        sentences_per_chunk: Number of sentences per chunk window.

    Returns:
        List of chunk dicts with type='html'. Empty list on parse error.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [WARNING] beautifulsoup4 not installed. pip install beautifulsoup4")
        return []

    try:
        soup = BeautifulSoup(html_string, 'html.parser')
    except Exception as error:
        print(f"  [ERROR] Could not parse HTML from '{source_name}': {error}")
        return []

    text      = soup.get_text(separator=' ', strip=True)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks    = []
    for i in range(0, len(sentences), sentences_per_chunk):
        window = sentences[i: i + sentences_per_chunk]
        if not window:
            continue
        chunks.append({
            'text':       ' '.join(window),
            'source':     source_name,
            'start_line': i + 1,
            'end_line':   i + len(window),
            'type':       'html',
        })
    return chunks


def truncate_chunk(text: str, max_words: int = 300, max_chars: int = 1200) -> str:
    """Truncate text to 300 words OR 1200 characters, whichever limit is hit first.

    This keeps chunk sizes within the embedding model's context window.
    Both limits are checked because some words are very long.

    Args:
        text:      The text to truncate.
        max_words: Maximum number of words allowed.
        max_chars: Maximum number of characters allowed.

    Returns:
        Truncated text string.
    """
    words     = text.split()
    truncated = ' '.join(words[:max_words]) if len(words) > max_words else text
    return truncated[:max_chars]
