"""reranker.py — Stateless rerank prompt builder.

Contains one public function: rerank_prompt(query, entry).
Returns a type-specific LLM prompt that asks the model to score a
chunk's relevance to a user query on a 1-to-10 scale.

Seven document types are handled: xlsx/csv, pptx, pdf, docx, html, md,
and a generic fallback for txt and any other type.

Extracted from VectorStore to keep vector_store.py under the 500-line limit.
The prompt text is preserved verbatim — no wording changes.
"""

__all__ = ['rerank_prompt']


def rerank_prompt(query: str, entry: dict) -> str:
    """Return a reranking prompt tailored to the document type of the chunk.

    The prompt instructs the LLM to output a single integer from 1 to 10
    indicating how relevant the chunk is to the query. This score is used
    by VectorStore._rerank() to sort candidates before the final answer.

    Args:
        query: The user's original search query.
        entry: A chunk dict containing at minimum 'text' and 'type' keys.

    Returns:
        A prompt string ready to be sent to the language model.
    """
    text     = entry['text']
    doc_type = entry.get('type', 'txt')

    if doc_type in ('xlsx', 'csv'):
        return (
            f"A user is searching for: {query}\n"
            f"Does this spreadsheet row contain relevant information to answer the query?\n"
            f"Row data: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
    elif doc_type == 'pptx':
        return (
            f"A user is searching for: {query}\n"
            f"Does this presentation slide contain relevant information to answer the query?\n"
            f"Slide text: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
    elif doc_type == 'pdf':
        return (
            f"A user is searching for: {query}\n"
            f"Does this PDF page extract contain relevant information to answer the query?\n"
            f"Page text: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
    elif doc_type == 'docx':
        return (
            f"A user is searching for: {query}\n"
            f"Does this document paragraph contain relevant information to answer the query?\n"
            f"Paragraph: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
    elif doc_type == 'html':
        return (
            f"A user is searching for: {query}\n"
            f"Does this webpage content contain relevant information to answer the query?\n"
            f"Content: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
    elif doc_type == 'md':
        return (
            f"A user is searching for: {query}\n"
            f"Does this markdown document section contain relevant information to answer the query?\n"
            f"Section: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
    else:
        # txt and any other type — generic prompt
        return (
            f"On a scale of 1-10, how relevant is the following text to the query?\n"
            f"Query: {query}\nText: {text}\n"
            f"Reply with a single integer from 1 to 10 and nothing else."
        )
