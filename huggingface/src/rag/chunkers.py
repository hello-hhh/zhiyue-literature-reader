"""chunkers.py — Stateless chunkers for text-based document formats. HF Space version.

Handles plain text (TXT), Markdown (MD), CSV, and HTML — formats that
can be parsed without binary libraries. Binary formats (PDF, DOCX, XLSX,
XLS, PPTX) live in binary_chunkers.py.

HF difference from local:
  chunk_html strips navigation boilerplate (nav, header, footer, sidebar tags)
  and filters out short lines (<40 chars) to remove menu items.
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
    """Chunk a plain-text file using a sliding line window.

    Args:
        filepath:   Absolute path to the .txt file.
        filename:   Display name stored in each chunk's 'source' field.
        chunk_size: Number of lines per chunk window.
        overlap:    Number of lines shared between consecutive windows.

    Returns:
        List of chunk dicts with keys: text, source, start_line, end_line, type.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

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
    """Strip Markdown syntax and chunk like a plain-text file.

    Args:
        filepath:   Absolute path to the .md file.
        filename:   Display name stored in each chunk's 'source' field.
        chunk_size: Number of lines per chunk window.
        overlap:    Number of lines shared between consecutive windows.

    Returns:
        List of chunk dicts with type='md'.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    clean = re.sub(r'#{1,6}\s*', '', raw)
    clean = re.sub(r'[*_`~]{1,3}', '', clean)
    clean = re.sub(r'!\[.*?\]\(.*?\)', '', clean)
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    step  = max(1, chunk_size - overlap)
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
    """Chunk an HTML file — strip navigation boilerplate, then sentence windows.

    HF-specific: removes nav/header/footer/sidebar tags and filters lines
    shorter than 40 characters to remove menu items and short labels.
    This reduces noise from Wikipedia-style and documentation pages.

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
        print("  [WARNING] beautifulsoup4 not installed — skipping HTML.")
        return []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
    except Exception as e:
        print(f"  [ERROR] Could not open '{filename}': {e}")
        return []

    # Remove navigation boilerplate before extracting text
    for tag in soup.find_all(['nav', 'header', 'footer', 'aside',
                               'script', 'style', 'noscript']):
        tag.decompose()
    for tag in soup.find_all(attrs={'role': ['navigation', 'banner', 'complementary']}):
        tag.decompose()
    for tag in soup.find_all(attrs={'id': ['mw-navigation', 'mw-head', 'mw-panel',
                                           'p-navigation', 'p-tb', 'footer',
                                           'toc', 'catlinks', 'mw-page-base']}):
        tag.decompose()
    for tag in soup.find_all(attrs={'class': ['navbox', 'sidebar', 'mw-editsection',
                                               'reflist', 'reference', 'noprint']}):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    # 40-char threshold filters single-word nav items and short menu labels
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 40]
    text  = ' '.join(lines)

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
    """Chunk a decoded plain-text string using a sliding line window.

    Used when content has already been decoded from a URL response — no file on disk.

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
    """Chunk a decoded HTML string using BeautifulSoup tag stripping and sentence windows.

    Used when content has already been decoded from a URL response — no file on disk.

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

    Args:
        text:      The text to truncate.
        max_words: Maximum number of words allowed.
        max_chars: Maximum number of characters allowed.

    Returns:
        Truncated text string.
    """
    words     = text.split()
    truncated = ' '.join(words[:max_words]) if len(words) > max_words else text
    return truncated[:max_chars] if len(truncated) > max_chars else truncated
