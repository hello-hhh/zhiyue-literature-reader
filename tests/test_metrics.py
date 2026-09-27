"""Unit tests for src/rag/metrics.py — all 7 RAG scoring functions.

Each test class covers one function. LLM-as-judge functions (score_faithfulness_llm,
score_answer_relevancy_llm) mock ollama.chat so no Ollama instance is needed in CI.
All other functions are pure Python and require no mocking.
"""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_reranked(texts, sim=0.8):
    """Build a reranked list from a list of text strings.

    Each entry gets the same similarity score (sim) and a dummy rerank score.
    Used by tests for context_relevance, precision_at_k, and mrr.
    """
    return [
        ({'text': t, 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}, sim, sim)
        for t in texts
    ]


# ---------------------------------------------------------------------------
# score_faithfulness_llm
# ---------------------------------------------------------------------------

class TestScoreFaithfulnessLlm:
    """Tests for LLM-as-judge faithfulness scoring."""

    def test_rating_5_returns_1_0(self):
        """LLM returns '5': score normalises to 1.0 (perfectly grounded)."""
        from src.rag.metrics import score_faithfulness_llm
        with patch('ollama.chat', return_value={'message': {'content': '5'}}):
            score = score_faithfulness_llm('q', 'a', 'ctx', 'model')
        assert score == 1.0

    def test_rating_1_returns_0_0(self):
        """LLM returns '1': score normalises to 0.0 (complete hallucination)."""
        from src.rag.metrics import score_faithfulness_llm
        with patch('ollama.chat', return_value={'message': {'content': '1'}}):
            score = score_faithfulness_llm('q', 'a', 'ctx', 'model')
        assert score == 0.0

    def test_rating_3_returns_0_5(self):
        """LLM returns '3': score normalises to 0.5 (midpoint)."""
        from src.rag.metrics import score_faithfulness_llm
        with patch('ollama.chat', return_value={'message': {'content': '3'}}):
            score = score_faithfulness_llm('q', 'a', 'ctx', 'model')
        assert score == pytest.approx(0.5)

    def test_unparseable_response_returns_0_5(self):
        """LLM returns text with no valid digit: score falls back to 0.5 (neutral)."""
        from src.rag.metrics import score_faithfulness_llm
        with patch('ollama.chat', return_value={'message': {'content': 'I cannot judge'}}):
            score = score_faithfulness_llm('q', 'a', 'ctx', 'model')
        assert score == 0.5

    def test_ollama_exception_returns_0_5(self):
        """ollama.chat raises an exception: score falls back to 0.5."""
        from src.rag.metrics import score_faithfulness_llm
        with patch('ollama.chat', side_effect=RuntimeError('connection refused')):
            score = score_faithfulness_llm('q', 'a', 'ctx', 'model')
        assert score == 0.5

    def test_score_in_0_1_range(self):
        """Any valid LLM rating: result is always in [0.0, 1.0]."""
        from src.rag.metrics import score_faithfulness_llm
        for rating in ['1', '2', '3', '4', '5']:
            with patch('ollama.chat', return_value={'message': {'content': rating}}):
                score = score_faithfulness_llm('q', 'a', 'ctx', 'model')
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_answer_relevancy_llm
# ---------------------------------------------------------------------------

class TestScoreAnswerRelevancyLlm:
    """Tests for LLM-as-judge answer relevancy scoring."""

    def test_rating_5_returns_1_0(self):
        """LLM returns '5': score normalises to 1.0 (perfectly on-topic answer)."""
        from src.rag.metrics import score_answer_relevancy_llm
        with patch('ollama.chat', return_value={'message': {'content': '5'}}):
            score = score_answer_relevancy_llm('q', 'a', 'model')
        assert score == 1.0

    def test_rating_1_returns_0_0(self):
        """LLM returns '1': score normalises to 0.0 (completely off-topic)."""
        from src.rag.metrics import score_answer_relevancy_llm
        with patch('ollama.chat', return_value={'message': {'content': '1'}}):
            score = score_answer_relevancy_llm('q', 'a', 'model')
        assert score == 0.0

    def test_digit_embedded_in_text_is_parsed(self):
        """LLM returns 'Rating: 4 out of 5': first valid digit '4' is extracted."""
        from src.rag.metrics import score_answer_relevancy_llm
        with patch('ollama.chat', return_value={'message': {'content': 'Rating: 4 out of 5'}}):
            score = score_answer_relevancy_llm('q', 'a', 'model')
        assert score == pytest.approx(0.75)

    def test_ollama_exception_returns_0_5(self):
        """ollama.chat raises an exception: score falls back to 0.5."""
        from src.rag.metrics import score_answer_relevancy_llm
        with patch('ollama.chat', side_effect=RuntimeError('timeout')):
            score = score_answer_relevancy_llm('q', 'a', 'model')
        assert score == 0.5


# ---------------------------------------------------------------------------
# score_ground_truth_match
# ---------------------------------------------------------------------------

class TestScoreGroundTruthMatch:
    """Tests for F1 word-overlap against a known correct answer."""

    def test_identical_text_returns_high_score(self):
        """Response matches ground truth exactly: score is above 0.8."""
        from src.rag.metrics import score_ground_truth_match
        score = score_ground_truth_match(
            'cats sleep sixteen hours per day',
            'cats sleep sixteen hours per day',
        )
        assert score > 0.8

    def test_completely_different_text_returns_low_score(self):
        """Response shares no content words with ground truth: score is below 0.2."""
        from src.rag.metrics import score_ground_truth_match
        score = score_ground_truth_match('dogs bark loudly', 'cats sleep sixteen hours')
        assert score < 0.2

    def test_empty_response_returns_zero(self):
        """Empty response: score is 0.0."""
        from src.rag.metrics import score_ground_truth_match
        score = score_ground_truth_match('', 'cats sleep sixteen hours')
        assert score == 0.0

    def test_empty_ground_truth_returns_zero(self):
        """Empty ground truth: score is 0.0."""
        from src.rag.metrics import score_ground_truth_match
        score = score_ground_truth_match('cats sleep sixteen hours', '')
        assert score == 0.0

    def test_score_between_0_and_1(self):
        """Any two strings: score is always in [0.0, 1.0]."""
        from src.rag.metrics import score_ground_truth_match
        score = score_ground_truth_match('cats are wonderful', 'cats sleep a lot')
        assert 0.0 <= score <= 1.0

    def test_partial_overlap_is_between_0_and_1(self):
        """Response shares some words with ground truth: score is between 0.1 and 0.9."""
        from src.rag.metrics import score_ground_truth_match
        score = score_ground_truth_match(
            'cats sleep around fourteen hours',
            'cats sleep between twelve and sixteen hours per day',
        )
        assert 0.1 < score < 0.9


# ---------------------------------------------------------------------------
# score_keyword_recall
# ---------------------------------------------------------------------------

class TestScoreKeywordRecall:
    """Tests for expected-keyword recall scoring."""

    def test_all_keywords_found_returns_1_0(self):
        """All expected keywords present in response: score is 1.0."""
        from src.rag.metrics import score_keyword_recall
        score = score_keyword_recall('cats sleep 16 hours a day', ['sleep', '16'])
        assert score == 1.0

    def test_no_keywords_found_returns_0_0(self):
        """No expected keywords in response: score is 0.0."""
        from src.rag.metrics import score_keyword_recall
        score = score_keyword_recall('dogs bark a lot', ['sleep', '16'])
        assert score == 0.0

    def test_partial_keywords_returns_0_5(self):
        """One of two keywords found: score is 0.5."""
        from src.rag.metrics import score_keyword_recall
        score = score_keyword_recall('cats sleep a lot', ['sleep', '16'])
        assert score == 0.5

    def test_empty_keywords_returns_1_0(self):
        """No keywords expected: vacuously perfect score of 1.0."""
        from src.rag.metrics import score_keyword_recall
        score = score_keyword_recall('any response', [])
        assert score == 1.0

    def test_case_insensitive_matching(self):
        """Keywords in mixed case in response: matching is case-insensitive."""
        from src.rag.metrics import score_keyword_recall
        score = score_keyword_recall('Cats SLEEP 16 Hours', ['sleep', '16'])
        assert score == 1.0


# ---------------------------------------------------------------------------
# score_context_relevance
# ---------------------------------------------------------------------------

class TestScoreContextRelevance:
    """Tests for mean cosine similarity of retrieved chunks."""

    def test_empty_reranked_returns_zero(self):
        """Empty reranked list: context relevance is 0.0."""
        from src.rag.metrics import score_context_relevance
        assert score_context_relevance([], top_n=5) == 0.0

    def test_returns_mean_of_scores(self):
        """Two chunks with similarities 0.8 and 0.6: mean is 0.7."""
        from src.rag.metrics import score_context_relevance
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        reranked = [(entry, 0.8, 0.8), (entry, 0.6, 0.6)]
        score = score_context_relevance(reranked, top_n=5)
        assert abs(score - 0.7) < 0.01

    def test_top_n_limits_chunks_used(self):
        """top_n=1 with two chunks: only the first chunk's score is used."""
        from src.rag.metrics import score_context_relevance
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        reranked = [(entry, 0.9, 0.9), (entry, 0.1, 0.1)]
        score = score_context_relevance(reranked, top_n=1)
        assert abs(score - 0.9) < 0.01

    def test_score_between_0_and_1(self):
        """Single chunk with similarity 0.5: result is in [0.0, 1.0]."""
        from src.rag.metrics import score_context_relevance
        entry = {'text': 'x', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        score = score_context_relevance([(entry, 0.5, 0.5)], top_n=5)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_precision_at_k
# ---------------------------------------------------------------------------

class TestScorePrecisionAtK:
    """Tests for Precision@K — fraction of top-K chunks containing a keyword."""

    def test_all_chunks_relevant_returns_1_0(self):
        """All top-5 chunks contain the keyword: P@5 is 1.0."""
        from src.rag.metrics import score_precision_at_k
        reranked = _make_reranked(['cats sleep 16 hours'] * 5)
        score = score_precision_at_k(reranked, ['sleep'], k=5)
        assert score == 1.0

    def test_no_chunks_relevant_returns_0_0(self):
        """No top-5 chunks contain the keyword: P@5 is 0.0."""
        from src.rag.metrics import score_precision_at_k
        reranked = _make_reranked(['dogs bark loudly'] * 5)
        score = score_precision_at_k(reranked, ['sleep'], k=5)
        assert score == 0.0

    def test_partial_relevant_chunks(self):
        """2 of 4 chunks contain the keyword: P@4 is 0.5."""
        from src.rag.metrics import score_precision_at_k
        reranked = _make_reranked([
            'cats sleep a lot',
            'dogs run fast',
            'cats sleep deeply',
            'birds fly high',
        ])
        score = score_precision_at_k(reranked, ['sleep'], k=4)
        assert score == 0.5

    def test_empty_keywords_returns_0_0(self):
        """No keywords provided: score is 0.0 (nothing to check relevance against)."""
        from src.rag.metrics import score_precision_at_k
        reranked = _make_reranked(['cats sleep a lot'])
        score = score_precision_at_k(reranked, [], k=5)
        assert score == 0.0

    def test_k_limits_chunks_evaluated(self):
        """k=2: only the first 2 chunks are evaluated, not all 5."""
        from src.rag.metrics import score_precision_at_k

        # First 2 chunks are irrelevant, next 3 are relevant
        reranked = _make_reranked([
            'dogs bark',
            'birds fly',
            'cats sleep',
            'cats sleep',
            'cats sleep',
        ])
        score = score_precision_at_k(reranked, ['sleep'], k=2)
        # Only the first 2 (irrelevant) chunks are evaluated → P@2 = 0.0
        assert score == 0.0

    def test_case_insensitive_matching(self):
        """Keyword in uppercase in chunk text: matching is case-insensitive."""
        from src.rag.metrics import score_precision_at_k
        reranked = _make_reranked(['CATS SLEEP 16 HOURS'])
        score = score_precision_at_k(reranked, ['sleep'], k=1)
        assert score == 1.0


# ---------------------------------------------------------------------------
# score_mrr
# ---------------------------------------------------------------------------

class TestScoreMrr:
    """Tests for Mean Reciprocal Rank — rank of the first relevant chunk."""

    def test_first_chunk_relevant_returns_1_0(self):
        """The very first chunk contains the keyword: MRR = 1/1 = 1.0."""
        from src.rag.metrics import score_mrr
        reranked = _make_reranked(['cats sleep 16 hours', 'dogs bark'])
        score = score_mrr(reranked, ['sleep'])
        assert score == 1.0

    def test_second_chunk_relevant_returns_0_5(self):
        """First chunk irrelevant, second relevant: MRR = 1/2 = 0.5."""
        from src.rag.metrics import score_mrr
        reranked = _make_reranked(['dogs bark', 'cats sleep 16 hours'])
        score = score_mrr(reranked, ['sleep'])
        assert score == pytest.approx(0.5)

    def test_no_relevant_chunk_returns_0_0(self):
        """No chunk contains the keyword: MRR = 0.0."""
        from src.rag.metrics import score_mrr
        reranked = _make_reranked(['dogs bark', 'birds fly'])
        score = score_mrr(reranked, ['sleep'])
        assert score == 0.0

    def test_empty_keywords_returns_0_0(self):
        """No keywords provided: MRR is 0.0."""
        from src.rag.metrics import score_mrr
        reranked = _make_reranked(['cats sleep'])
        score = score_mrr(reranked, [])
        assert score == 0.0

    def test_empty_reranked_returns_0_0(self):
        """Empty reranked list: MRR is 0.0."""
        from src.rag.metrics import score_mrr
        assert score_mrr([], ['sleep']) == 0.0

    def test_third_chunk_relevant_returns_one_third(self):
        """First two chunks irrelevant, third relevant: MRR = 1/3 ≈ 0.333."""
        from src.rag.metrics import score_mrr
        reranked = _make_reranked(['dogs bark', 'birds fly', 'cats sleep 16 hours'])
        score = score_mrr(reranked, ['sleep'])
        assert score == pytest.approx(1.0 / 3, rel=1e-3)

    def test_case_insensitive_matching(self):
        """Keyword appears in uppercase in chunk: matching is case-insensitive."""
        from src.rag.metrics import score_mrr
        reranked = _make_reranked(['CATS SLEEP 16 HOURS'])
        score = score_mrr(reranked, ['sleep'])
        assert score == 1.0
