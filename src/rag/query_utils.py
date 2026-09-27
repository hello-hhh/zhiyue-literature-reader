"""query_utils.py — Stateless query classification and prompt-building functions.

WHY THIS FILE EXISTS:
    These functions have no dependency on VectorStore's internal state — they are
    pure transformations of their input parameters. Extracting them here keeps
    vector_store.py within the 500-line file size limit and makes these functions
    independently testable without a full VectorStore instance.
"""

from src.rag.config import TOP_RETRIEVE

__all__ = [
    'classify_query',
    'smart_top_n',
    'build_instruction_prompt',
]


def classify_query(query: str) -> str:
    """Classify a query as summarise, comparison, factual, or general.

    Checks for keyword signals in order of specificity — summarise first
    (most specific intent), then comparison, factual, and finally general
    as the default fallback.

    Args:
        query: The user's question in plain text.

    Returns:
        One of: 'summarise', 'comparison', 'factual', 'general'.
    """
    q = query.lower()
    summarise_signals  = ['summarise', 'summarize', 'summary', 'overview',
                          'tell me about', 'what is in', 'describe', 'explain',
                          'give me a summary', 'resume']
    comparison_signals = ['compare', 'difference', 'vs', 'versus', 'better', 'worse',
                          'pros and cons', 'which is', 'how does', 'contrast']
    factual_signals    = ['what is', 'what are', 'who is', 'who are', 'when did',
                          'where is', 'how many', 'how much', 'does', 'did', 'has',
                          'have', 'list', 'name', 'define', 'tell me']
    if any(s in q for s in summarise_signals):
        return 'summarise'
    if any(s in q for s in comparison_signals):
        return 'comparison'
    if any(s in q for s in factual_signals):
        return 'factual'
    return 'general'


def smart_top_n(query_type: str) -> int:
    """Return the retrieval budget that fits the query's complexity.

    Factual queries have a clear correct answer so 5 chunks is enough.
    Comparison queries need both sides of the argument (15).
    Summarise queries need the full document scope (TOP_RETRIEVE).

    Args:
        query_type: One of 'factual', 'comparison', 'general', 'summarise'.

    Returns:
        Number of chunks to retrieve.
    """
    return {'factual': 5, 'comparison': 15, 'general': 10,
            'summarise': TOP_RETRIEVE}.get(query_type, TOP_RETRIEVE)


def build_instruction_prompt(context: str, query_type: str = 'factual') -> str:
    """Build the system prompt that instructs the LLM to answer from context only.

    Includes strict anti-hallucination rules and a length hint scaled to the
    query type — factual stays short, summarise gets room to cover all key points.

    Args:
        context:    The retrieved and reranked document passages.
        query_type: One of 'factual', 'comparison', 'general', 'summarise'.

    Returns:
        The full system instruction prompt string.
    """
    # Length hints keep answers proportional to query complexity
    _length_hints = {
        'factual':    '2-3 sentences',
        'general':    '3-4 sentences',
        'comparison': '4-6 sentences covering each item being compared',
        'summarise':  '6-8 sentences covering all key points',
    }
    length_hint = _length_hints.get(query_type, '2-3 sentences')
    return (
        "You are a document question-answering assistant in a multi-turn conversation.\n"
        "Answer the question using ONLY the context passages provided below.\n"
        "STRICT RULES:\n"
        "- Do NOT use your training data or general knowledge under any circumstances.\n"
        "- If the context does not contain the answer, say exactly: "
        "'The provided documents do not contain information about this topic.'\n"
        "- Do NOT speculate, infer, or elaborate beyond what the context states.\n"
        "- You MAY use prior conversation turns to resolve references such as pronouns "
        "('she', 'it', 'that document') or follow-up phrases ('what about her role?'), "
        "but all factual claims must come from the CONTEXT passages below.\n"
        f"- Answer in {length_hint}.\n"
        "- At the end of your answer, cite ONLY the bracketed source labels from the context "
        "(e.g. [filename.pdf p3] or [example.com/page s12]). "
        "Do NOT copy any bibliographic references, footnotes, or citations that appear "
        "inside the text.\n\n"
        f"CONTEXT:\n{context}"
    )
