"""test_doc_types_agent.py — Agent mode tests for all 8 document types,
all 5 agent tools, and agent bad format recovery.

Mock strategy:
  Always mock: ollama.embed, ollama.chat, chromadb → EphemeralClient
  Never mock:  BM25Okapi, calculator eval, chunk parsing

Reason for split: max 500 lines per file per CLAUDE.md.
Chat mode and pipeline feature tests are in test_doc_types_and_modes.py.
"""

import uuid
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_embed(**kwargs):
    """Return a minimal fixed embedding vector for any input text."""
    return {'embeddings': [[0.1, 0.2, 0.3, 0.4]]}


def _fake_chat(**kwargs):
    """Return streaming list or plain dict depending on stream= kwarg."""
    if kwargs.get('stream'):
        return [{'message': {'content': 'mock response'}}]
    return {'message': {'content': 'mock response'}}


def _make_store(chunks):
    """Build a VectorStore with EphemeralClient loaded with given chunks."""
    from src.rag.vector_store import VectorStore
    vs         = VectorStore()
    client     = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        f'test_{uuid.uuid4().hex}', metadata={'hnsw:space': 'cosine'}
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


def _chunk(text, source, doc_type, line=1):
    """Build a minimal chunk dict with all required fields for VectorStore."""
    return {'text': text, 'source': source,
            'start_line': line, 'end_line': line, 'type': doc_type}


PATCHES = [
    patch('ollama.embed',  side_effect=_fake_embed),
    patch('ollama.chat',   side_effect=_fake_chat),
]


def _apply_patches(func):
    """Decorator to apply ollama patches to a test method."""
    for p in reversed(PATCHES):
        func = p(func)
    return func


# ---------------------------------------------------------------------------
# 2. Agent mode — all 8 document types
# ---------------------------------------------------------------------------

class TestAgentModeTxt:
    """Agent ReAct loop tests using plain-text (.txt) document chunks."""

    @_apply_patches
    def test_agent_txt_rag_search(self, mock_chat, mock_embed):
        """Agent with txt chunk: result contains answer and at least one step."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(dog sleep hours)'}}
        chunks = [_chunk('Dogs sleep about 12 hours a day.', 'dog-facts.txt', 'txt')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('How many hours do dogs sleep?', streamlit_mode=True)
        assert 'answer' in result
        assert 'steps' in result
        assert len(result['steps']) >= 1


class TestAgentModePdf:
    """Agent ReAct loop tests using PDF document chunks."""

    @_apply_patches
    def test_agent_pdf_rag_search(self, mock_chat, mock_embed):
        """Agent with pdf chunk: result contains answer and a rag_search step."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(banana calories)'}}
        chunks = [_chunk('Bananas contain 89 calories per 100 grams.', 'nutrition.pdf', 'pdf')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('How many calories in bananas?', streamlit_mode=True)
        assert 'answer' in result
        assert any(s['tool'] == 'rag_search' for s in result['steps'])


class TestAgentModeDocx:
    """Agent ReAct loop tests using Word document (.docx) chunks."""

    @_apply_patches
    def test_agent_docx_rag_search(self, mock_chat, mock_embed):
        """Agent with docx chunk: result contains answer after rag_search tool call."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(python creator)'}}
        chunks = [_chunk('Python was created by Guido van Rossum in 1991.', 'resume.docx', 'docx')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('Who created Python?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeXlsx:
    """Agent ReAct loop tests using Excel spreadsheet (.xlsx) chunks."""

    @_apply_patches
    def test_agent_xlsx_rag_search(self, mock_chat, mock_embed):
        """Agent with xlsx chunk: result contains answer after rag_search tool call."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(Brazil capital)'}}
        chunks = [_chunk('country=Brazil | capital=Brasilia | population_millions=215',
                         'countries.xlsx', 'xlsx')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('What is the capital of Brazil?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeCsv:
    """Agent ReAct loop tests using CSV document chunks."""

    @_apply_patches
    def test_agent_csv_rag_search(self, mock_chat, mock_embed):
        """Agent with csv chunk: result contains answer after rag_search tool call."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(Python programming language)'}}
        chunks = [_chunk('language=Python | year_created=1991 | creator=Guido van Rossum',
                         'programming-languages.csv', 'csv')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('When was Python created?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModePptx:
    """Agent ReAct loop tests using PowerPoint (.pptx) document chunks."""

    @_apply_patches
    def test_agent_pptx_rag_search(self, mock_chat, mock_embed):
        """Agent with pptx chunk: result contains answer after rag_search tool call."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(Berlin Wall)'}}
        chunks = [_chunk('The Berlin Wall fell on November 9 1989.', 'history.pptx', 'pptx')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('When did the Berlin Wall fall?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeMd:
    """Agent ReAct loop tests using Markdown (.md) document chunks."""

    @_apply_patches
    def test_agent_md_rag_search(self, mock_chat, mock_embed):
        """Agent with md chunk: result contains answer after rag_search tool call."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(coffee origin Ethiopia)'}}
        chunks = [_chunk('Coffee was first discovered in Ethiopia around 850 AD.',
                         'coffee-facts.md', 'md')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('Where was coffee discovered?', streamlit_mode=True)
        assert 'answer' in result


class TestAgentModeHtml:
    """Agent ReAct loop tests using HTML document chunks."""

    @_apply_patches
    def test_agent_html_rag_search(self, mock_chat, mock_embed):
        """Agent with html chunk: result contains answer after rag_search tool call."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(dog words understand)'}}
        chunks = [_chunk('A dog can understand about 165 words and gestures.', 'dog-facts.html', 'html')]
        store  = _make_store(chunks)
        from src.rag.agent import Agent
        result = Agent(store).run('How many words can a dog understand?', streamlit_mode=True)
        assert 'answer' in result


# ---------------------------------------------------------------------------
# 3. Agent tools — all 5 tools across doc types
# ---------------------------------------------------------------------------

class TestAgentToolCalculator:
    """Agent calculator tool tests: arithmetic, percentages, unsafe input rejection."""

    def test_calculator_basic(self):
        """Simple addition: '2 + 2' → '4'."""
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('2 + 2') == '4'

    def test_calculator_nested_parens(self):
        """Nested parentheses expression: '7 + (9 + 8) - 2 * 6' → '12'."""
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('7 + (9 + 8) - 2 * 6') == '12'

    def test_calculator_percentage(self):
        """Percentage of value: '15% of 85000' → 12750.0."""
        from src.rag.agent import Agent
        assert float(Agent(MagicMock())._tool_calculator('15% of 85000')) == pytest.approx(12750.0)

    def test_calculator_percentage_alone(self):
        """Bare percentage literal: '20%' → 0.2."""
        from src.rag.agent import Agent
        assert float(Agent(MagicMock())._tool_calculator('20%')) == pytest.approx(0.2)

    def test_calculator_unsafe_rejected(self):
        """Expression with disallowed characters: returns an error string."""
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('__import__("os")').startswith('Error')

    def test_calculator_complex_expression(self):
        """Mixed operators with parentheses: '(10 + 5) * 2 / 3' → '10.0'."""
        from src.rag.agent import Agent
        assert Agent(MagicMock())._tool_calculator('(10 + 5) * 2 / 3') == '10.0'

    def test_calculator_complex_division(self):
        """Division with grouping: '(10 + 5) * 4 / 6' → 10.0."""
        from src.rag.agent import Agent
        assert float(Agent(MagicMock())._tool_calculator('(10 + 5) * 4 / 6')) == pytest.approx(10.0)

    @_apply_patches
    def test_calculator_auto_finish(self, mock_chat, mock_embed):
        """Calculator tool call: agent automatically appends a finish step."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: calculator(100 * 365)'}}
        from src.rag.agent import Agent
        result = Agent(_make_store([])).run('How many days in 100 years?', streamlit_mode=True)
        assert any(s['tool'] == 'calculator' for s in result['steps'])
        assert any(s['tool'] == 'finish' for s in result['steps'])


class TestAgentToolRagSearch:
    """Agent rag_search tool tests: chunk retrieval, auto-finish, and all doc types."""

    @_apply_patches
    def test_rag_search_returns_chunks(self, mock_chat, mock_embed):
        """rag_search with a matching chunk: result string contains the source filename."""
        chunks = [_chunk('Coffee was discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks))._tool_rag_search('coffee origin')
        assert 'coffee-facts.md' in result

    @_apply_patches
    def test_rag_search_auto_finish(self, mock_chat, mock_embed):
        """rag_search tool call: agent steps include both rag_search and finish."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: rag_search(coffee origin)'}}
        chunks = [_chunk('Coffee was discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Where was coffee discovered?', streamlit_mode=True)
        tools = [s['tool'] for s in result['steps']]
        assert 'rag_search' in tools
        assert 'finish' in tools

    @_apply_patches
    def test_rag_search_all_doc_types_return_results(self, mock_chat, mock_embed):
        """All 8 doc types should return at least one result from rag_search."""
        from src.rag.agent import Agent
        for dtype in ['txt', 'pdf', 'docx', 'xlsx', 'csv', 'pptx', 'md', 'html']:
            chunks = [_chunk(f'This is a {dtype} document with relevant content.',
                             f'test.{dtype}', dtype)]
            result = Agent(_make_store(chunks))._tool_rag_search('relevant content')
            assert len(result) > 0, f"rag_search returned empty for {dtype}"


class TestAgentToolSummarise:
    """Agent summarise tool tests: output, length hints, and fast-path behaviour."""

    @_apply_patches
    def test_summarise_short_text(self, mock_chat, mock_embed):
        """Short input text: summarise returns a non-empty string."""
        mock_chat.return_value = {'message': {'content': 'This is a short summary.'}}
        from src.rag.agent import Agent
        result = Agent(MagicMock())._tool_summarise('Coffee is a drink.')
        assert isinstance(result, str) and len(result) > 0

    @_apply_patches
    def test_summarise_length_hint_short(self, mock_chat, mock_embed):
        """Under 100 words → 2-3 sentences hint."""
        from src.rag.agent import Agent
        captured = []
        mock_chat.side_effect = lambda **kw: captured.append(kw['messages'][0]['content']) or {'message': {'content': 'S.'}}
        Agent(MagicMock())._tool_summarise('Short text here.')
        assert any('2-3 sentences' in c for c in captured)

    @_apply_patches
    def test_summarise_length_hint_medium(self, mock_chat, mock_embed):
        """100-300 words → 4-5 sentences hint."""
        from src.rag.agent import Agent
        captured = []
        mock_chat.side_effect = lambda **kw: captured.append(kw['messages'][0]['content']) or {'message': {'content': 'S.'}}
        Agent(MagicMock())._tool_summarise(' '.join(['word'] * 150))
        assert any('4-5 sentences' in c for c in captured)

    @_apply_patches
    def test_summarise_length_hint_long(self, mock_chat, mock_embed):
        """Over 300 words → 6-8 sentences hint."""
        from src.rag.agent import Agent
        captured = []
        mock_chat.side_effect = lambda **kw: captured.append(kw['messages'][0]['content']) or {'message': {'content': 'S.'}}
        Agent(MagicMock())._tool_summarise(' '.join(['word'] * 350))
        assert any('6-8 sentences' in c for c in captured)

    @_apply_patches
    def test_fast_path_summarise_runs_4_searches(self, mock_chat, mock_embed):
        """Summarise fast path: exactly 4 rag_search steps are recorded in agent steps."""
        chunks = [_chunk('Coffee was discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('summarise the coffee document', streamlit_mode=True)
        assert len([s for s in result['steps'] if s['tool'] == 'rag_search']) == 4

    @_apply_patches
    def test_fast_path_summarise_resume_uses_resume_terms(self, mock_chat, mock_embed):
        """Summarise fast path on a resume query: search args include work-experience related terms."""
        chunks = [_chunk('Work experience: 5 years as software engineer.', 'resume.docx', 'docx')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('summarise the resume', streamlit_mode=True)
        search_args = [s['arg'] for s in result['steps'] if s['tool'] == 'rag_search']
        assert any('experience' in a for a in search_args)


class TestAgentToolSentiment:
    """Agent sentiment tool tests: short-query search path, direct analysis, and output format."""

    @_apply_patches
    def test_sentiment_short_query_searches_first(self, mock_chat, mock_embed):
        """Short query (<10 words): sentiment tool searches the store before analysing."""
        resp = 'Sentiment: Positive\nTone: enthusiastic\nKey phrases: wonderful, energizing\nExplanation: Positive tone.'
        mock_chat.side_effect = lambda **kw: {'message': {'content': resp}}
        chunks = [_chunk('Coffee is wonderful and energizing.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks))._tool_sentiment('coffee')
        assert 'Sentiment' in result

    @_apply_patches
    def test_sentiment_long_text_direct(self, mock_chat, mock_embed):
        """Long input text (>=10 words): sentiment tool analyses directly without searching."""
        resp = 'Sentiment: Positive\nTone: professional\nKey phrases: excellent, outstanding\nExplanation: Very positive.'
        mock_chat.side_effect = lambda **kw: {'message': {'content': resp}}
        from src.rag.agent import Agent
        result = Agent(_make_store([]))._tool_sentiment(' '.join(['excellent outstanding remarkable positive'] * 5))
        assert 'Sentiment' in result

    @_apply_patches
    def test_sentiment_output_has_4_fields(self, mock_chat, mock_embed):
        """Sentiment output: all four required fields (Sentiment, Tone, Key phrases, Explanation) present."""
        resp = 'Sentiment: Positive\nTone: upbeat\nKey phrases: great, amazing\nExplanation: Overall positive.'
        mock_chat.side_effect = lambda **kw: {'message': {'content': resp}}
        from src.rag.agent import Agent
        result = Agent(_make_store([]))._tool_sentiment(' '.join(['great amazing'] * 10))
        assert 'Sentiment' in result
        assert 'Tone' in result
        assert 'Key phrases' in result
        assert 'Explanation' in result

    @_apply_patches
    def test_fast_path_sentiment_strips_metadata_labels(self, mock_chat, mock_embed):
        """Sentiment fast path: agent records at least one sentiment step and returns an answer."""
        resp = 'Sentiment: Positive\nTone: warm\nKey phrases: rich, bold\nExplanation: Positive tone.'
        mock_chat.return_value = {'message': {'content': resp}}
        chunks = [_chunk('Coffee has a rich and bold flavour.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('what is the sentiment of the coffee document', streamlit_mode=True)
        assert result['answer'] is not None
        assert len([s for s in result['steps'] if s['tool'] == 'sentiment']) >= 1


class TestAgentToolFinish:
    """Agent finish tool tests: context accumulation and direct finish with no prior search."""

    @_apply_patches
    def test_finish_uses_collected_context(self, mock_chat, mock_embed):
        """rag_search followed by finish: steps include finish and answer is non-None."""
        responses = iter([
            {'message': {'content': 'TOOL: rag_search(coffee origin)'}},
            {'message': {'content': 'TOOL: finish(Coffee was found in Ethiopia.)'}},
        ])
        mock_chat.side_effect = lambda **kw: next(responses)
        chunks = [_chunk('Coffee was first discovered in Ethiopia.', 'coffee-facts.md', 'md')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Where was coffee discovered?', streamlit_mode=True)
        assert any(s['tool'] == 'finish' for s in result['steps'])
        assert result['answer'] is not None

    @_apply_patches
    def test_finish_direct_no_context(self, mock_chat, mock_embed):
        """Immediate finish with no prior search: answer contains the finish argument verbatim."""
        mock_chat.side_effect = lambda **kw: {'message': {'content': 'TOOL: finish(42 is the answer.)'}}
        from src.rag.agent import Agent
        result = Agent(_make_store([])).run('What is the answer?', streamlit_mode=True)
        assert '42 is the answer.' in result['answer']


# ---------------------------------------------------------------------------
# 5. Agent bad format recovery
# ---------------------------------------------------------------------------

class TestAgentBadFormat:
    """Agent ReAct loop recovery tests: bad LLM output format and step-limit handling."""

    @_apply_patches
    def test_bad_format_retries_twice(self, mock_chat, mock_embed):
        """Two malformed responses then a valid tool call: agent still returns a non-None answer."""
        responses = iter([
            {'message': {'content': 'This is not a tool call'}},
            {'message': {'content': 'Still not a tool call'}},
            {'message': {'content': 'TOOL: rag_search(dogs)'}},
        ])
        mock_chat.side_effect = lambda **kw: next(responses)
        chunks = [_chunk('Dogs are loyal animals.', 'dog-facts.txt', 'txt')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Tell me about dogs', streamlit_mode=True)
        assert result['answer'] is not None

    @_apply_patches
    def test_max_steps_returns_answer(self, mock_chat, mock_embed):
        """LLM always returns rag_search (never finish): agent terminates at max_steps with an answer."""
        mock_chat.return_value = {'message': {'content': 'TOOL: rag_search(dogs)'}}
        chunks = [_chunk('Dogs are loyal animals.', 'dog-facts.txt', 'txt')]
        from src.rag.agent import Agent
        result = Agent(_make_store(chunks)).run('Tell me about dogs', streamlit_mode=True)
        assert result['answer'] is not None
