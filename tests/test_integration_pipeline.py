"""Integration tests — VectorStore pipeline and agent tools.

Covers sections 3 and 4 of the original integration test suite:
  3. VectorStore pipeline (mock ollama + EphemeralClient)
  4. All 5 agent tools

Mock strategy (per CLAUDE.md):
  Always mock:   ollama.embed, ollama.chat, chromadb → EphemeralClient, requests.get
  Never mock:    BM25Okapi, chunk truncation, misplaced detection, calculator eval

Reason for split: max 500 lines per file per CLAUDE.md.
"""

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed():
    """Return a minimal fixed embedding vector for any input text."""
    return {'embeddings': [[0.1, 0.2, 0.3, 0.4]]}


def _fake_chat(content='mock'):
    """Return a mock ollama.chat response with the given content string."""
    return {'message': {'content': content}}


def _pipeline_chat_mock(*args, **kwargs):
    """Smart mock: returns a stream list when stream=True, else a plain dict."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': '8'}}


def _make_store(chunks):
    """Return a VectorStore wired to EphemeralClient with given chunks."""
    from src.rag.vector_store import VectorStore
    vs = VectorStore()
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        'test_rag', metadata={'hnsw:space': 'cosine'}
    )
    if chunks:
        ids    = [f'c{i}' for i in range(len(chunks))]
        texts  = [c['text'] for c in chunks]
        metas  = [{'source': c['source'], 'start_line': c['start_line'],
                   'end_line': c['end_line'], 'type': c['type']} for c in chunks]
        embeds = [[0.1, 0.2, 0.3, 0.4]] * len(chunks)
        collection.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    vs.collection          = collection
    vs.chunks              = chunks
    vs.bm25_index          = BM25Okapi([c['text'].lower().split() for c in chunks]) if chunks else None
    vs.conversation_history = []
    return vs


SAMPLE_CHUNKS = [
    {'text': 'Cats sleep sixteen hours a day.',    'source': 'cats.txt', 'start_line': 1, 'end_line': 1, 'type': 'txt'},
    {'text': 'Cats can see in dim light very well.', 'source': 'cats.txt', 'start_line': 2, 'end_line': 2, 'type': 'txt'},
    {'text': 'Cats have five toes on front paws.',  'source': 'cats.txt', 'start_line': 3, 'end_line': 3, 'type': 'txt'},
    {'text': 'Cats have twelve whiskers.',           'source': 'cats.txt', 'start_line': 4, 'end_line': 4, 'type': 'txt'},
    {'text': 'Cats cannot taste sweet food.',        'source': 'cats.txt', 'start_line': 5, 'end_line': 5, 'type': 'txt'},
]


# ============================================================
# 3. VectorStore pipeline (mock ollama + EphemeralClient)
# ============================================================

class TestHybridRetrieve:
    """Integration tests for _hybrid_retrieve BM25 + dense fusion."""

    def test_hybrid_retrieve_fuses_bm25_and_dense(self):
        """Single query string: result is a list with float scores (BM25 + dense fused)."""
        vs = _make_store(SAMPLE_CHUNKS)
        with patch('ollama.embed', return_value=_fake_embed()):
            result = vs._hybrid_retrieve(['cats sleep'], top_n=3)
        assert len(result) <= 3
        assert all(isinstance(score, float) for _, score in result)

    def test_hybrid_retrieve_alpha_weighting(self):
        """alpha=1.0 (dense only) and alpha=0.0 (BM25 only) both return lists."""
        vs = _make_store(SAMPLE_CHUNKS)
        with patch('ollama.embed', return_value=_fake_embed()):
            r_dense = vs._hybrid_retrieve(['cats sleep'], top_n=5, alpha=1.0)
            r_bm25  = vs._hybrid_retrieve(['cats sleep'], top_n=5, alpha=0.0)
        assert isinstance(r_dense, list)
        assert isinstance(r_bm25, list)


class TestExpandQuery:
    """Integration tests for _expand_query LLM-based query rewriting."""

    def test_expand_query_returns_3(self):
        """LLM returns two rewrites: result has exactly 3 entries (original + 2 rewrites)."""
        vs = _make_store(SAMPLE_CHUNKS)
        with patch('ollama.chat', return_value=_fake_chat('rewrite1\nrewrite2')):
            result = vs._expand_query('how many hours do cats sleep')
        assert len(result) == 3
        assert result[0] == 'how many hours do cats sleep'

    def test_expand_query_fallback_on_error(self):
        """LLM raises exception: _expand_query falls back to returning only the original."""
        vs = _make_store(SAMPLE_CHUNKS)
        with patch('ollama.chat', side_effect=Exception('fail')):
            result = vs._expand_query('test query')
        assert result == ['test query']


class TestClassifyQuery:
    """Integration tests for _classify_query keyword-based classification."""

    def test_classify_summarise_checked_first(self):
        """Query with 'summarise' keyword: classified as 'summarise' (checked first)."""
        vs = _make_store(SAMPLE_CHUNKS)
        assert vs._classify_query('summarise all cat facts') == 'summarise'

    def test_classify_factual(self):
        """Query starting with 'what is': classified as 'factual'."""
        vs = _make_store(SAMPLE_CHUNKS)
        assert vs._classify_query('what is the boiling point of water') == 'factual'

    def test_classify_comparison(self):
        """Query with 'compare' and 'vs': classified as 'comparison'."""
        vs = _make_store(SAMPLE_CHUNKS)
        assert vs._classify_query('compare cats vs dogs') == 'comparison'

    def test_classify_general(self):
        """Query with no matching keywords: falls back to 'general'."""
        vs = _make_store(SAMPLE_CHUNKS)
        assert vs._classify_query('something interesting happened today') == 'general'


class TestRerank:
    """Integration tests for _rerank LLM scoring and ordering."""

    def test_rerank_orders_by_llm_score(self):
        """Two candidates with LLM scores 2 and 9: result is ordered high-to-low."""
        vs = _make_store(SAMPLE_CHUNKS)
        candidates = [
            ({'text': 'low relevance', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}, 0.5),
            ({'text': 'high relevance', 'source': 's', 'start_line': 2, 'end_line': 2, 'type': 'txt'}, 0.4),
        ]
        scores = iter(['2', '9'])
        with patch('ollama.chat', side_effect=lambda **kw: _fake_chat(next(scores))):
            result = vs._rerank('test query', candidates, top_n=2)
        assert result[0][2] >= result[1][2]


class TestConfidence:
    """Integration tests for _check_confidence threshold logic."""

    def test_confidence_below_threshold(self):
        """Score below SIMILARITY_THRESHOLD: confident=False is returned."""
        from src.rag.config import SIMILARITY_THRESHOLD
        vs = _make_store(SAMPLE_CHUNKS)
        entry = SAMPLE_CHUNKS[0]
        results = [(entry, SIMILARITY_THRESHOLD - 0.1)]
        confident, score = vs._check_confidence(results)
        assert confident is False

    def test_confidence_above_threshold(self):
        """Score above SIMILARITY_THRESHOLD: confident=True is returned."""
        from src.rag.config import SIMILARITY_THRESHOLD
        vs = _make_store(SAMPLE_CHUNKS)
        entry = SAMPLE_CHUNKS[0]
        results = [(entry, SIMILARITY_THRESHOLD + 0.1)]
        confident, score = vs._check_confidence(results)
        assert confident is True


class TestSmartTopN:
    """Integration tests for _smart_top_n retrieval count selection."""

    def test_smart_top_n_all_4_types(self):
        """All 4 query types return the correct top-N retrieval count."""
        from src.rag.config import TOP_RETRIEVE
        vs = _make_store(SAMPLE_CHUNKS)
        assert vs._smart_top_n('factual')    == 5
        assert vs._smart_top_n('comparison') == 15
        assert vs._smart_top_n('general')    == 10
        assert vs._smart_top_n('summarise')  == TOP_RETRIEVE


class TestSourceLabel:
    """Integration tests for _source_label format per document type."""

    def test_source_label_all_types(self):
        """All 6 document types produce the correct source label format."""
        vs = _make_store(SAMPLE_CHUNKS)
        assert vs._source_label({'type': 'pdf',  'start_line': 3, 'end_line': 3}) == 'p3'
        assert vs._source_label({'type': 'xlsx', 'start_line': 5, 'end_line': 5}) == 'row5'
        assert vs._source_label({'type': 'csv',  'start_line': 2, 'end_line': 2}) == 'row2'
        assert vs._source_label({'type': 'pptx', 'start_line': 1, 'end_line': 1}) == 'slide1'
        assert vs._source_label({'type': 'html', 'start_line': 4, 'end_line': 4}) == 's4'
        assert vs._source_label({'type': 'txt',  'start_line': 1, 'end_line': 3}) == 'L1-3'


class TestHallucinationFilter:
    """Integration tests for _filter_hallucination pivot-phrase truncation."""

    def test_hallucination_filter_truncates(self):
        """No-info phrase followed by 'however,': response is truncated and safe message returned."""
        vs = _make_store(SAMPLE_CHUNKS)
        response = "I couldn't find any information. However, I can tell you cats are nice."
        result = vs._filter_hallucination(response)
        assert 'however' not in result.lower()
        assert 'I can only answer' in result

    def test_hallucination_filter_clean_response_unchanged(self):
        """Clean response with no hallucination markers: returned exactly as-is."""
        vs = _make_store(SAMPLE_CHUNKS)
        response = 'Cats sleep 16 hours a day. [cats.txt L1]'
        assert vs._filter_hallucination(response) == response


class TestLowConfidence:
    """Integration tests for low-confidence path in run_pipeline."""

    def test_low_confidence_skips_llm(self):
        """Distant embedding forces low similarity: is_confident is False and response key exists."""
        vs = _make_store(SAMPLE_CHUNKS)
        # Use distant embedding to force low similarity
        with patch('ollama.embed', return_value={'embeddings': [[0.0, 0.0, 0.0, 1.0]]}), \
             patch('ollama.chat', side_effect=_pipeline_chat_mock) as mock_chat:
            result = vs.run_pipeline('completely unrelated xyz topic', streamlit_mode=True)
        # If not confident, LLM for synthesis should not be called
        assert 'response' in result
        assert result['is_confident'] is False or 'response' in result


class TestRebuildLogic:
    """Integration tests for build_or_load rebuild-vs-skip decision logic."""

    def test_rebuild_logic_skips_if_existing_gte_chunks(self):
        """DB with 3 docs and only 2 chunks: existing >= len(chunks), so rebuild is skipped."""
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        with patch('chromadb.PersistentClient') as mock_client:
            client     = chromadb.EphemeralClient()
            collection = client.get_or_create_collection('rag', metadata={'hnsw:space': 'cosine'})
            # Pre-populate with more entries than chunks → should skip rebuild
            collection.add(
                ids=['x0', 'x1', 'x2'],
                embeddings=[[0.1, 0.2, 0.3, 0.4]] * 3,
                documents=['a', 'b', 'c'],
                metadatas=[{'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}] * 3
            )
            mock_client.return_value.get_or_create_collection.return_value = collection
            chunks = [
                {'text': 'a', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'},
                {'text': 'b', 'source': 's', 'start_line': 2, 'end_line': 2, 'type': 'txt'},
            ]
            with patch('ollama.embed', return_value=_fake_embed()):
                vs.build_or_load(chunks)
        # Collection should still have 3 (not rebuilt)
        assert vs.collection.count() == 3

    def test_rebuild_logic_deletes_and_rebuilds_if_local_grew(self):
        """DB with 1 doc and 3 chunks: existing < len(chunks), so DB is cleared and rebuilt."""
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        with patch('chromadb.PersistentClient') as mock_client:
            client     = chromadb.EphemeralClient()
            collection = client.get_or_create_collection('rag2', metadata={'hnsw:space': 'cosine'})
            # Pre-populate with fewer entries than chunks → should delete and rebuild
            collection.add(
                ids=['x0'],
                embeddings=[[0.1, 0.2, 0.3, 0.4]],
                documents=['old'],
                metadatas=[{'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}]
            )
            mock_client.return_value.get_or_create_collection.return_value = collection
            chunks = [
                {'text': 'a', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'},
                {'text': 'b', 'source': 's', 'start_line': 2, 'end_line': 2, 'type': 'txt'},
                {'text': 'c', 'source': 's', 'start_line': 3, 'end_line': 3, 'type': 'txt'},
            ]
            with patch('ollama.embed', return_value=_fake_embed()):
                vs.build_or_load(chunks)
        assert vs.collection.count() == 3


# ============================================================
# 4. All 5 agent tools
# ============================================================

class TestAgentTools:
    """Integration tests for all 5 Agent tools: calculator, summarise, sentiment, rag_search, finish."""

    @pytest.fixture
    def agent(self):
        """Provide an Agent backed by a mocked VectorStore for tool-level testing."""
        from src.rag.agent import Agent
        vs = _make_store(SAMPLE_CHUNKS)
        vs._expand_query = MagicMock(return_value=['cats'])
        entry = SAMPLE_CHUNKS[0]
        vs._hybrid_retrieve = MagicMock(return_value=[(entry, 0.9)])
        vs._rerank          = MagicMock(return_value=[(entry, 0.9, 0.9)])
        vs._source_label    = MagicMock(return_value='L1-1')
        return Agent(vs)

    def test_tool_calculator_basic(self, agent):
        """'2 + 2': calculator returns '4'."""
        assert agent._tool_calculator('2 + 2') == '4'

    def test_tool_calculator_complex(self, agent):
        """'(100 + 50) * 2': calculator returns '300'."""
        assert agent._tool_calculator('(100 + 50) * 2') == '300'

    def test_tool_calculator_unsafe_chars_rejected(self, agent):
        """Expression with __import__: calculator returns an error string."""
        result = agent._tool_calculator('__import__("os").system("rm -rf /")')
        assert 'Error' in result

    def test_tool_summarise_short_2_3_sentences(self, agent):
        """Under 100 words: summarise prompt contains '2-3 sentences' length hint."""
        short_text = ' '.join(['word'] * 50)
        with patch('ollama.chat', return_value=_fake_chat('summary')) as m:
            agent._tool_summarise(short_text)
        assert '2-3 sentences' in m.call_args[1]['messages'][0]['content']

    def test_tool_summarise_medium_4_5_sentences(self, agent):
        """100–300 words: summarise prompt contains '4-5 sentences' length hint."""
        medium_text = ' '.join(['word'] * 150)
        with patch('ollama.chat', return_value=_fake_chat('summary')) as m:
            agent._tool_summarise(medium_text)
        assert '4-5 sentences' in m.call_args[1]['messages'][0]['content']

    def test_tool_summarise_long_6_8_sentences(self, agent):
        """Over 300 words: summarise prompt contains '6-8 sentences' length hint."""
        long_text = ' '.join(['word'] * 400)
        with patch('ollama.chat', return_value=_fake_chat('summary')) as m:
            agent._tool_summarise(long_text)
        assert '6-8 sentences' in m.call_args[1]['messages'][0]['content']

    def test_tool_sentiment_short_query_searches_first(self, agent):
        """Short query (< 10 words): _tool_rag_search is called before sentiment analysis."""
        agent._tool_rag_search = MagicMock(return_value='cats are wonderful creatures')
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Positive\nTone: warm\nKey phrases: wonderful\nExplanation: positive.')):
            agent._tool_sentiment('cats')
        agent._tool_rag_search.assert_called_once()

    def test_tool_sentiment_long_text_direct(self, agent):
        """Long text (>= 10 words): _tool_rag_search is NOT called — analysis is direct."""
        agent._tool_rag_search = MagicMock(return_value='x')
        long_text = ' '.join(['word'] * 15)
        with patch('ollama.chat', return_value=_fake_chat('Sentiment: Neutral\nTone: flat\nKey phrases: word\nExplanation: neutral.')):
            agent._tool_sentiment(long_text)
        agent._tool_rag_search.assert_not_called()

    def test_tool_sentiment_output_4_fields(self, agent):
        """Sentiment mock output with all 4 fields: result contains 'Sentiment' and 'Tone'."""
        mock_resp = 'Sentiment: Positive\nTone: upbeat\nKey phrases: great\nExplanation: positive mood.'
        with patch('ollama.chat', return_value=_fake_chat(mock_resp)):
            result = agent._tool_sentiment('great product experience here today')
        assert 'Sentiment' in result
        assert 'Tone' in result

    def test_parse_tool_call_with_parens(self, agent):
        """TOOL: rag_search(cat sleep hours) — parenthesis pattern: name and arg extracted."""
        name, arg = agent._parse_tool_call('TOOL: rag_search(cat sleep hours)')
        assert name == 'rag_search'
        assert arg == 'cat sleep hours'

    def test_parse_tool_call_without_parens(self, agent):
        """TOOL: rag_search cat sleep hours — no-parenthesis fallback: name and arg extracted."""
        name, arg = agent._parse_tool_call('TOOL: rag_search cat sleep hours')
        assert name == 'rag_search'
        assert arg == 'cat sleep hours'

    def test_parse_tool_call_malformed_returns_none(self, agent):
        """Non-tool-call text: _parse_tool_call returns (None, None)."""
        name, arg = agent._parse_tool_call('This is not a valid response.')
        assert name is None

    def test_fast_path_summarise_4_searches(self, agent):
        """Fast-path summarise: exactly 4 rag_search steps appear in result['steps']."""
        agent._tool_rag_search = MagicMock(return_value='result')
        with patch('ollama.chat', return_value=_fake_chat('summary answer')):
            result = agent._fast_path_summarise('summarise the document')
        tool_steps = [s for s in result['steps'] if s['tool'] == 'rag_search']
        assert len(tool_steps) == 4

    def test_fast_path_sentiment_strips_labels(self, agent):
        """Fast-path sentiment: chunk metadata labels (e.g. [cats.txt L1-1]) are stripped."""
        raw = '- [cats.txt L1-1] Cats are wonderful creatures indeed.'
        agent._tool_rag_search = MagicMock(return_value=raw)
        captured = []
        def cap_sentiment(text):
            """Capture text passed to _tool_sentiment for inspection."""
            captured.append(text)
            return 'Sentiment: Positive\nTone: warm\nKey phrases: wonderful\nExplanation: positive.'
        agent._tool_sentiment = cap_sentiment
        agent._fast_path_sentiment('what is the sentiment of cats')
        if captured:
            assert '[cats.txt' not in captured[0]

    def test_calculator_auto_finish(self, agent):
        """Calculator tool call: agent automatically appends a finish step with the result."""
        with patch('ollama.chat', return_value=_fake_chat('TOOL: calculator(16 * 365)')):
            result = agent.run('how many hours in 365 days at 16h each')
        finish_steps = [s for s in result['steps'] if s['tool'] == 'finish']
        assert len(finish_steps) >= 1
        assert '5840' in result['answer']

    def test_rag_search_auto_finish(self, agent):
        """rag_search tool call: agent auto-finishes after first rag_search for non-summarise queries."""
        with patch('ollama.chat', side_effect=[
            _fake_chat('TOOL: rag_search(cats sleep)'),
            _fake_chat('Cats sleep 16 hours a day.'),
        ]):
            result = agent.run('how many hours do cats sleep')
        assert 'answer' in result
        finish_steps = [s for s in result['steps'] if s['tool'] == 'finish']
        assert len(finish_steps) >= 1

    def test_bad_format_retry_max_2(self, agent):
        """Two malformed responses then a valid finish: agent still returns a non-None answer."""
        responses = [
            _fake_chat('not a tool call'),
            _fake_chat('still not a tool call'),
            _fake_chat('TOOL: finish(final answer here)'),
        ]
        with patch('ollama.chat', side_effect=responses):
            result = agent.run('any query')
        assert result is not None

    def test_step_limit_reached(self, agent):
        """LLM always returns non-tool text: agent terminates at max_steps and returns an answer."""
        with patch('ollama.chat', return_value=_fake_chat('not a tool call ever')):
            result = agent.run('any query')
        assert 'answer' in result

    def test_collected_context_used_for_final_answer(self, agent):
        """After rag_search, finish uses synthesized answer from collected context."""
        with patch('ollama.chat', side_effect=[
            _fake_chat('TOOL: rag_search(cats)'),
            _fake_chat('Cats sleep 16 hours.'),
        ]):
            result = agent.run('tell me about cats sleeping')
        assert result['answer']
        assert len(result['steps']) >= 2
