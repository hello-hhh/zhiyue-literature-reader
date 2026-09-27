"""vector_store.py — VectorStore class. HF Space version.

Replaces Ollama with:
  - sentence-transformers (BAAI/bge-base-en-v1.5) for embeddings
  - HF Inference Providers router (OpenAI-compatible) for LLM calls
  - chromadb.EphemeralClient for in-memory vector store
"""

import os
import re

import chromadb
from rank_bm25 import BM25Okapi

from src.rag.config import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    LANGUAGE_MODEL,
    LANGUAGE_MODEL_FALLBACKS,
    SIMILARITY_THRESHOLD,
    TOP_RERANK,
    TOP_RETRIEVE,
)

__all__ = ['VectorStore']


def _load_st_model():
    """Load sentence-transformers model once at module level."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device='cpu')


def _load_cross_encoder():
    """Load cross-encoder reranker once at module level."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)


# Module-level singletons — loaded once when the module is first imported
_ST_MODEL = None
_CROSS_ENCODER = None


def _get_st_model():
    """Return the module-level SentenceTransformer singleton, loading it on first call."""
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = _load_st_model()
    return _ST_MODEL


def _get_cross_encoder():
    """Return the module-level CrossEncoder singleton, loading it on first call."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = _load_cross_encoder()
    return _CROSS_ENCODER


def _llm_call(messages, max_tokens=512, temperature=0.1):
    """Call HF Inference via InferenceClient. Tries providers then models in order.
    messages: list of {'role': ..., 'content': ...} dicts (OpenAI format).
    """
    from huggingface_hub import InferenceClient
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("[WARNING] HF_TOKEN not set.")
        return "[LLM error: HF_TOKEN not set]"

    last_error = ""

    # Try each provider × model combination until one works
    providers = ["featherless-ai", "novita", "hf-inference"]
    for provider in providers:
        for model in LANGUAGE_MODEL_FALLBACKS:
            try:
                client = InferenceClient(provider=provider, api_key=token)
                result = client.chat_completion(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.1),
                    stream=False,
                )
                content = result.choices[0].message.content
                if content and content.strip():
                    print(f"[LLM] Used {provider}/{model}")
                    return content.strip()
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"[LLM] {provider}/{model} failed: {last_error}")
                continue

    return f"[LLM error: all providers/models failed. Last: {last_error}]"


class VectorStore:
    """Owns ChromaDB, BM25, hybrid search, reranking, query pipeline,
    response generation, and conversation history.

    State:
        collection (chromadb.Collection): In-memory ChromaDB collection.
        chunks (list): All chunk dicts currently indexed (local + URL/upload).
        bm25_index (BM25Okapi | None): BM25 index over all chunks; None when empty.
        conversation_history (list): Multi-turn chat history as role/content dicts.

    Public API:
        build_or_load(chunks)          -- Initialise collection and embed chunks.
        add_chunks(chunks, id_prefix)  -- Add new chunks at runtime.
        rebuild_bm25(all_chunks)       -- Rebuild BM25 after any chunk additions.
        run_pipeline(query, ...)       -- Full classify→retrieve→rerank→generate pipeline.
        clear_conversation()           -- Wipe conversation history.
    """

    def __init__(self):
        """Initialise all state to empty — call build_or_load() before querying."""
        self.collection          = None
        self.chunks              = []
        self._local_chunks       = []   # snapshot of local-doc chunks set at build time
        self.bm25_index          = None
        self.conversation_history = []

    # ── Public ──────────────────────────────────────────────────────────────

    def build_or_load(self, chunks: list) -> None:
        """Initialize an in-memory ChromaDB collection and embed any provided chunks.

        Args:
            chunks: List of chunk dicts to embed and store.  Pass an empty list to
                    create a collection without any documents (e.g. at startup).
        """
        client     = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        if chunks:
            print(f"Embedding {len(chunks)} chunks...")
            # Batch size of 50 keeps memory usage bounded on CPU
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch  = chunks[i: i + batch_size]
                ids    = [f"chunk_{i+j}" for j in range(len(batch))]
                texts  = [c['text'] for c in batch]
                metas  = [{'source': c['source'], 'start_line': c['start_line'],
                           'end_line': c['end_line'], 'type': c.get('type', 'txt')}
                          for c in batch]
                embeds = [self._embed(self._truncate_for_embedding(t)) for t in texts]
                collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
            print(f"Ready — {collection.count()} chunks stored.\n")

        self.collection    = collection
        self.chunks        = list(chunks)
        self._local_chunks = list(chunks)   # frozen snapshot — used by clear_added_chunks()
        self.bm25_index    = BM25Okapi([c['text'].lower().split() for c in chunks]) if chunks else None

    def add_chunks(self, chunks: list, id_prefix: str) -> None:
        """Add new chunks from a URL or file upload to the live collection.

        Args:
            chunks:    List of chunk dicts to embed and append.
            id_prefix: String prefix for generated ChromaDB IDs (e.g. 'url', 'file').
        """
        if not chunks:
            return
        # Use current count as offset so new IDs never collide with existing ones
        offset = self.collection.count()
        ids    = [f"{id_prefix}_{offset+i}" for i in range(len(chunks))]
        texts  = [c['text'] for c in chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c.get('type', 'txt')}
                  for c in chunks]
        embeds = [self._embed(self._truncate_for_embedding(t)) for t in texts]
        self.collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
        self.chunks = self.chunks + list(chunks)

    def rebuild_bm25(self, all_chunks: list) -> None:
        """Rebuild the BM25 index over all current chunks after any addition.

        Args:
            all_chunks: Complete list of chunks (base + newly added).
        """
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in all_chunks]) if all_chunks else None

    def run_pipeline(self, query: str, streamlit_mode: bool = False) -> dict:
        """Execute the full RAG pipeline for a single user query.

        Steps: classify → expand → hybrid retrieve → confidence check →
               cross-encoder rerank → build context → LLM synthesize →
               hallucination filter → return.

        Args:
            query:          User's question string.
            streamlit_mode: Unused in the HF version (kept for API compatibility).

        Returns:
            Dict with keys: response, query_type, queries, is_confident,
            best_score, retrieved, reranked.
        """
        qtype      = self._classify_query(query)
        top_n      = self._smart_top_n(qtype)
        # Use more chunks for summarise to cover the whole document
        top_rerank = 10 if qtype == 'summarise' else TOP_RERANK
        queries    = self._expand_query(query)
        retrieved  = self._hybrid_retrieve(queries, top_n=top_n)
        is_confident, best_score = self._check_confidence(retrieved)
        reranked   = self._rerank(query, retrieved, top_n=top_rerank)

        context_lines = []
        for e, _, _ in reranked:
            label = self._source_label(e)
            context_lines.append(f" - [{e['source']} {label}] {e['text']}")
        context = '\n'.join(context_lines)

        if not is_confident:
            full_response = (
                "I could not find relevant information in the provided documents to answer this question. "
                "Please upload a document or add a URL that contains the relevant information."
            )
            self.conversation_history.append({'role': 'user', 'content': query})
            self.conversation_history.append({'role': 'assistant', 'content': full_response})
            return {
                'response':     full_response,
                'query_type':   qtype,
                'queries':      queries,
                'is_confident': False,
                'best_score':   best_score,
                'retrieved':    retrieved,
                'reranked':     reranked,
            }

        instruction_prompt = self._build_instruction_prompt(context)
        self.conversation_history.append({'role': 'user', 'content': query})

        messages = [{'role': 'system', 'content': instruction_prompt},
                    *self.conversation_history]
        full_response = self._llm_chat(messages, temperature=0.1)
        full_response = self._filter_hallucination(full_response)

        self.conversation_history.append({'role': 'assistant', 'content': full_response})

        return {
            'response':     full_response,
            'query_type':   qtype,
            'queries':      queries,
            'is_confident': is_confident,
            'best_score':   best_score,
            'retrieved':    retrieved,
            'reranked':     reranked,
        }

    def clear_added_chunks(self) -> int:
        """Remove all URL and file-upload chunks added at runtime.

        Deletes every chunk whose ChromaDB ID starts with 'url_' or 'file_'
        (the prefixes used by add_chunks()). Local document chunks — loaded
        at startup — are kept. BM25 is rebuilt from the remaining local chunks.

        Returns:
            Number of chunks removed.
        """
        all_ids     = self.collection.get()['ids']
        runtime_ids = [id_ for id_ in all_ids
                       if id_.startswith('url_') or id_.startswith('file_')]

        if runtime_ids:
            self.collection.delete(ids=runtime_ids)

        self.chunks     = list(self._local_chunks)
        self.bm25_index = BM25Okapi([c['text'].lower().split() for c in self.chunks]) if self.chunks else None
        return len(runtime_ids)

    def clear_conversation(self) -> None:
        """Wipe the multi-turn conversation history so the next query starts fresh."""
        self.conversation_history = []

    # ── Private — LLM ────────────────────────────────────────────────────────

    def _llm_chat(self, messages, temperature=0.0, max_tokens=512):
        """Single point of contact for all LLM calls via HF Inference API."""
        try:
            content = _llm_call(messages, max_tokens=max_tokens, temperature=temperature)
            if not content:
                print("[WARNING] LLM returned empty response.")
                return "[LLM error: empty response from model]"
            return content.strip()
        except Exception as e:
            print(f"[ERROR] LLM call failed: {type(e).__name__}: {e}")
            return f"[LLM error: {type(e).__name__}: {e}]"

    # ── Private — vector/search ──────────────────────────────────────────────

    def _embed(self, text: str) -> list:
        """Embed text using sentence-transformers (runs locally)."""
        model = _get_st_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def _truncate_for_embedding(self, text: str, max_words: int = 200, max_chars: int = 1200) -> str:
        """Truncate to stay within bge-base-en 512 token limit."""
        words     = text.split()
        truncated = ' '.join(words[:max_words]) if len(words) > max_words else text
        return truncated[:max_chars] if len(truncated) > max_chars else truncated

    def _cosine_similarity(self, a: list, b: list) -> float:
        """Compute cosine similarity between two plain Python float lists."""
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x**2 for x in a)**0.5
        nb  = sum(x**2 for x in b)**0.5
        return dot / (na * nb) if na and nb else 0.0

    def _hybrid_retrieve(self, queries: list, top_n: int, alpha: float = 0.5) -> list:
        """True hybrid search: fuses BM25 (lexical) + ChromaDB (dense) scores."""
        if self.collection is None or self.collection.count() == 0:
            return []

        # fused maps doc_text → (entry, best_score_across_all_queries)
        fused = {}

        for query in queries:
            q_emb   = self._embed(query)
            # Fetch 2× top_n candidates so fusion has room to reshuffle
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_n * 2, self.collection.count())
            )

            # ChromaDB returns cosine distance [0,2]; convert to similarity [0,1]
            dense_map = {}
            for doc, meta, dist in zip(results['documents'][0],
                                        results['metadatas'][0],
                                        results['distances'][0]):
                entry = {
                    'text':       doc,
                    'source':     meta.get('source', '?'),
                    'start_line': meta.get('start_line', 0),
                    'end_line':   meta.get('end_line', 0),
                    'type':       meta.get('type', 'txt'),
                }
                dense_map[doc] = (entry, 1 - dist)

            if self.bm25_index is not None:
                tokenized       = query.lower().split()
                bm25_scores_raw = self.bm25_index.get_scores(tokenized)
                # Normalise BM25 to [0,1] so it's on the same scale as dense scores
                bm25_max        = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
                bm25_norm       = [s / bm25_max for s in bm25_scores_raw]
            else:
                bm25_norm = [0.0] * len(self.chunks)

            for doc, (entry, dense_score) in dense_map.items():
                # Linear search to map chunk text → BM25 index position
                bm25_score = 0.0
                for idx, c in enumerate(self.chunks):
                    if c['text'] == doc:
                        bm25_score = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
                        break
                # Alpha-weighted fusion: keep the best score this doc ever receives
                score = alpha * dense_score + (1 - alpha) * bm25_score
                if doc not in fused or score > fused[doc][1]:
                    fused[doc] = (entry, score)

        return sorted(fused.values(), key=lambda x: x[1], reverse=True)[:top_n]

    def _rerank(self, query, candidates, top_n):
        """Rerank candidates using a local cross-encoder (no LLM API call needed).

        Falls back to the original similarity score if the cross-encoder fails.

        Args:
            query:      The user's question string.
            candidates: List of (entry, similarity) tuples from _hybrid_retrieve.
            top_n:      Number of results to return after reranking.

        Returns:
            List of (entry, similarity, rerank_score) tuples, sorted descending.
        """
        try:
            ce     = _get_cross_encoder()
            pairs  = [(query, entry['text']) for entry, sim in candidates]
            scores = ce.predict(pairs)
            scored = [(entry, sim, float(score))
                      for (entry, sim), score in zip(candidates, scores)]
        except Exception as e:
            # Degrade gracefully — cross-encoder is a latency optimisation, not critical
            print(f"[Rerank] Cross-encoder failed: {e} — using similarity score")
            scored = [(entry, sim, sim) for entry, sim in candidates]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    # ── Private — query ──────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """Classifies query as summarise / comparison / factual / general."""
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

    def _expand_query(self, query):
        """Query expansion disabled on HF free CPU — saves 1 LLM call per query."""
        return [query]

    def _check_confidence(self, results):
        """Return (is_confident, best_score) based on the top-ranked chunk's similarity.

        Args:
            results: List of (entry, score) tuples from _hybrid_retrieve.

        Returns:
            Tuple of (bool, float).  False/0.0 when results is empty.
        """
        if not results:
            return False, 0.0
        best = results[0][1]
        return best >= SIMILARITY_THRESHOLD, best

    def _smart_top_n(self, query_type: str) -> int:
        """Map query type to the number of chunks to retrieve before reranking."""
        return {'factual': 5, 'comparison': 15, 'general': 10,
                'summarise': TOP_RETRIEVE}.get(query_type, TOP_RETRIEVE)

    # ── Private — response ───────────────────────────────────────────────────

    def _build_instruction_prompt(self, context):
        """Build the system-level anti-hallucination instruction that wraps each LLM call."""
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
            "- Do NOT generate examples, hypothetical scenarios, or additional text.\n"
            "- Stop writing immediately after your answer. Do not add anything after.\n"
            "- At the end of your answer, cite ONLY the bracketed source labels from the context "
            "(e.g. [filename.pdf p3] or [example.com/page s12]). "
            "Do NOT copy any bibliographic references, footnotes, or citations that appear "
            "inside the text.\n\n"
            f"CONTEXT:\n{context}"
        )

    def _source_label(self, entry: dict) -> str:
        """Returns a consistent location label for any doc type."""
        t = entry.get('type', 'txt')
        if t == 'pdf':
            return f"p{entry['start_line']}"
        elif t in ('xlsx', 'csv'):
            return f"row{entry['start_line']}"
        elif t == 'pptx':
            return f"slide{entry['start_line']}"
        elif t == 'html':
            return f"s{entry['start_line']}"
        else:
            return f"L{entry['start_line']}-{entry['end_line']}"

    def _synthesize(self, question, context):
        """Takes raw retrieved context and asks LLM to produce a clean direct answer."""
        prompt = (
            "You are a helpful assistant. Answer the question below using ONLY the "
            "provided context. Be concise and direct. Do not repeat the context — "
            "just answer the question. Cite the source filename at the end.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        try:
            return self._llm_chat([{'role': 'user', 'content': prompt}], temperature=0)
        except Exception:
            return context

    def _filter_hallucination(self, response: str) -> str:
        """Truncate at hallucination pivot if model admitted no-info then hallucinated.

        Also strips instruction-template tokens ([/USER], [/INST], [/ASSIST]) that
        some instruction-tuned models (e.g. Zephyr, Mistral) leak into their output,
        causing fake Q&A continuations to appear after the real answer.
        """
        _no_info_phrases = [
            "there is no information",
            "i couldn't find",
            "i could not find",
            "the provided context does not",
            "the provided documents do not",
            "no information in the provided",
            "not mentioned in the",
            "not found in the",
        ]
        _hallucination_pivots = [
            "however,", "but i can", "but,", "that said,",
            "nevertheless,", "i can tell you", "i can provide",
        ]
        # Template bleed tokens — instruction-tuned models sometimes continue
        # generating fake conversation turns using their training format tokens
        _template_tokens = [
            "[/USER]", "[/INST]", "[/ASSIST]", "[INST]", "[USER]",
            "</s>", "<|user|>", "<|assistant|>", "<|im_end|>",
        ]

        # Truncate at the first template token — everything after is fabricated
        for token in _template_tokens:
            idx = response.find(token)
            if idx != -1:
                response = response[:idx].strip()

        lower_resp = response.lower()
        # Only search for pivots when the model has already admitted it couldn't find info;
        # this avoids false-positive truncation on responses that happen to use pivot words
        if any(p in lower_resp for p in _no_info_phrases):
            for pivot in _hallucination_pivots:
                idx = lower_resp.find(pivot)
                if idx != -1:
                    # Keep the honest "no info" sentence, drop the hallucinated continuation
                    return (
                        response[:idx].strip() + "\n\n"
                        "I can only answer based on the uploaded documents. "
                        "Please add a relevant document or URL to get an answer."
                    )
        return response
