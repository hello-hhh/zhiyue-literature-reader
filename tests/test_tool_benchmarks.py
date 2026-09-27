"""Unit tests for src/rag/tool_benchmarks.py.

Tests cover:
    - TOOL_TEST_CASES structure: keys and check callables present for each entry
    - _check_sentiment_format: format validation helper
    - _check_valid_sentiment_label: label presence helper
    - _check_calculator_approx: numeric tolerance helper
    - _invoke_tool: dispatches to the correct Agent private method
    - run_tool_benchmarks: integration test with mocked Agent tools

All LLM calls are mocked so these tests run without Ollama running.
"""

from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_store():
    """Build a minimal mock VectorStore sufficient for Agent construction.

    The mock has the methods the Agent's tools need:
        _expand_query  → returns a single-element list (one query variant)
        _hybrid_retrieve → returns an empty list (no chunks needed for most tests)
        _rerank         → returns an empty list
        _source_label   → returns 'L1-1'
    """
    store = MagicMock()
    store._expand_query.return_value        = ['mock query']
    store._hybrid_retrieve.return_value     = []
    store._rerank.return_value              = []
    store._source_label.return_value        = 'L1-1'
    return store


def _mock_chat_response(content: str):
    """Return a dict that mimics an ollama.chat() non-streaming response."""
    return {'message': {'content': content}}


# ── TestToolTestCasesStructure ─────────────────────────────────────────────────

class TestToolTestCasesStructure:
    """Verify that every entry in TOOL_TEST_CASES has the required keys and types."""

    def test_list_is_not_empty(self):
        """TOOL_TEST_CASES must contain at least one entry."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        assert len(TOOL_TEST_CASES) > 0

    def test_every_entry_has_required_keys(self):
        """Every test case must have 'tool', 'input', 'check', and 'note' keys."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        required_keys = {'tool', 'input', 'check', 'note'}
        for index, tc in enumerate(TOOL_TEST_CASES):
            missing = required_keys - set(tc.keys())
            assert not missing, f"Test case {index} is missing keys: {missing}"

    def test_tool_names_are_valid(self):
        """Every 'tool' value must be one of the five directly callable tools."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES

        # All five tool names present in the benchmark suite
        valid_tools = {'calculator', 'sentiment', 'summarise', 'translate', 'topic_search'}
        for index, tc in enumerate(TOOL_TEST_CASES):
            assert tc['tool'] in valid_tools, (
                f"Test case {index} has unexpected tool name '{tc['tool']}'"
            )

    def test_check_is_callable(self):
        """Every 'check' value must be a callable (lambda or function)."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        for index, tc in enumerate(TOOL_TEST_CASES):
            assert callable(tc['check']), (
                f"Test case {index}: 'check' is not callable (got {type(tc['check'])})"
            )

    def test_input_is_non_empty_string(self):
        """Every 'input' value must be a non-empty string."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        for index, tc in enumerate(TOOL_TEST_CASES):
            assert isinstance(tc['input'], str) and tc['input'].strip(), (
                f"Test case {index} has an empty or non-string 'input'"
            )

    def test_at_least_one_calculator_case(self):
        """There must be at least one calculator test case."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        calc_cases = [tc for tc in TOOL_TEST_CASES if tc['tool'] == 'calculator']
        assert len(calc_cases) >= 1

    def test_at_least_one_sentiment_case(self):
        """There must be at least one sentiment test case."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        sentiment_cases = [tc for tc in TOOL_TEST_CASES if tc['tool'] == 'sentiment']
        assert len(sentiment_cases) >= 1

    def test_at_least_one_summarise_case(self):
        """There must be at least one summarise test case."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES
        summarise_cases = [tc for tc in TOOL_TEST_CASES if tc['tool'] == 'summarise']
        assert len(summarise_cases) >= 1


# ── TestCheckSentimentFormat ───────────────────────────────────────────────────

class TestCheckSentimentFormat:
    """Tests for the _check_sentiment_format helper."""

    def test_all_four_fields_present_returns_true(self):
        """Result with all 4 required fields: check returns True."""
        from src.rag.tool_benchmarks import _check_sentiment_format
        result = (
            "Sentiment: Positive\n"
            "Tone: enthusiastic\n"
            "Key phrases: great product, works well\n"
            "Explanation: The text expresses clear satisfaction."
        )
        assert _check_sentiment_format(result) is True

    def test_missing_tone_field_returns_false(self):
        """Result missing the 'Tone:' field: check returns False."""
        from src.rag.tool_benchmarks import _check_sentiment_format
        result = (
            "Sentiment: Positive\n"
            "Key phrases: great product\n"
            "Explanation: Satisfied customer."
        )
        assert _check_sentiment_format(result) is False

    def test_missing_key_phrases_returns_false(self):
        """Result missing the 'Key phrases:' field: check returns False."""
        from src.rag.tool_benchmarks import _check_sentiment_format
        result = (
            "Sentiment: Neutral\n"
            "Tone: factual\n"
            "Explanation: The text is neutral."
        )
        assert _check_sentiment_format(result) is False

    def test_empty_string_returns_false(self):
        """Empty string: check returns False (no fields present)."""
        from src.rag.tool_benchmarks import _check_sentiment_format
        assert _check_sentiment_format('') is False

    def test_all_fields_in_one_line_returns_true(self):
        """All 4 field labels on one line (unusual but valid): check returns True."""
        from src.rag.tool_benchmarks import _check_sentiment_format
        result = "Sentiment: X Tone: Y Key phrases: Z Explanation: W"
        assert _check_sentiment_format(result) is True


# ── TestCheckValidSentimentLabel ──────────────────────────────────────────────

class TestCheckValidSentimentLabel:
    """Tests for the _check_valid_sentiment_label helper."""

    def test_positive_label_returns_true(self):
        """Result containing 'Positive': check returns True."""
        from src.rag.tool_benchmarks import _check_valid_sentiment_label
        assert _check_valid_sentiment_label("Sentiment: Positive") is True

    def test_negative_label_returns_true(self):
        """Result containing 'Negative': check returns True."""
        from src.rag.tool_benchmarks import _check_valid_sentiment_label
        assert _check_valid_sentiment_label("Sentiment: Negative") is True

    def test_neutral_label_returns_true(self):
        """Result containing 'Neutral': check returns True."""
        from src.rag.tool_benchmarks import _check_valid_sentiment_label
        assert _check_valid_sentiment_label("Sentiment: Neutral") is True

    def test_mixed_label_returns_true(self):
        """Result containing 'Mixed': check returns True."""
        from src.rag.tool_benchmarks import _check_valid_sentiment_label
        assert _check_valid_sentiment_label("Sentiment: Mixed") is True

    def test_no_valid_label_returns_false(self):
        """Result with no recognised label: check returns False."""
        from src.rag.tool_benchmarks import _check_valid_sentiment_label
        assert _check_valid_sentiment_label("The text is somewhat unclear.") is False

    def test_empty_string_returns_false(self):
        """Empty string: check returns False."""
        from src.rag.tool_benchmarks import _check_valid_sentiment_label
        assert _check_valid_sentiment_label('') is False


# ── TestCheckCalculatorApprox ─────────────────────────────────────────────────

class TestCheckCalculatorApprox:
    """Tests for the _check_calculator_approx factory helper."""

    def test_exact_match_passes(self):
        """Result exactly equals expected: check returns True."""
        from src.rag.tool_benchmarks import _check_calculator_approx
        check = _check_calculator_approx(42.0)
        assert check('42.0') is True

    def test_within_tolerance_passes(self):
        """Result within default tolerance (0.01): check returns True."""
        from src.rag.tool_benchmarks import _check_calculator_approx
        check = _check_calculator_approx(50.0)
        assert check('50.005') is True

    def test_outside_tolerance_fails(self):
        """Result outside default tolerance: check returns False."""
        from src.rag.tool_benchmarks import _check_calculator_approx
        check = _check_calculator_approx(50.0)
        assert check('51.0') is False

    def test_non_numeric_result_fails(self):
        """Non-numeric string (e.g. error message): check returns False."""
        from src.rag.tool_benchmarks import _check_calculator_approx
        check = _check_calculator_approx(42.0)
        assert check('Error: unsafe expression') is False

    def test_custom_tolerance(self):
        """Result within a wide custom tolerance (1.0): check returns True."""
        from src.rag.tool_benchmarks import _check_calculator_approx
        check = _check_calculator_approx(100.0, tolerance=1.0)
        assert check('100.5') is True


# ── TestInvokeTool ─────────────────────────────────────────────────────────────

class TestInvokeTool:
    """Tests for _invoke_tool — verifies it calls the right Agent method."""

    def test_calculator_calls_agent_tool_calculator(self):
        """_invoke_tool with 'calculator' must call Agent._tool_calculator."""
        from src.rag.agent import Agent
        from src.rag.tool_benchmarks import _invoke_tool

        mock_agent = MagicMock(spec=Agent)
        mock_agent._tool_calculator.return_value = '42'

        result = _invoke_tool(mock_agent, 'calculator', '6 * 7')
        mock_agent._tool_calculator.assert_called_once_with('6 * 7')
        assert result == '42'

    def test_sentiment_calls_agent_tool_sentiment(self):
        """_invoke_tool with 'sentiment' must call Agent._tool_sentiment."""
        from src.rag.agent import Agent
        from src.rag.tool_benchmarks import _invoke_tool

        mock_agent = MagicMock(spec=Agent)
        mock_agent._tool_sentiment.return_value = 'Sentiment: Positive\nTone: happy\nKey phrases: great\nExplanation: Good.'

        result = _invoke_tool(mock_agent, 'sentiment', 'I love this')
        mock_agent._tool_sentiment.assert_called_once_with('I love this')
        assert 'Sentiment' in result

    def test_summarise_calls_agent_tool_summarise(self):
        """_invoke_tool with 'summarise' must call Agent._tool_summarise."""
        from src.rag.agent import Agent
        from src.rag.tool_benchmarks import _invoke_tool

        mock_agent = MagicMock(spec=Agent)
        mock_agent._tool_summarise.return_value = 'Python is a programming language.'

        result = _invoke_tool(mock_agent, 'summarise', 'Python was created by Guido.')
        mock_agent._tool_summarise.assert_called_once_with('Python was created by Guido.')
        assert result == 'Python is a programming language.'

    def test_unknown_tool_raises_value_error(self):
        """_invoke_tool with an unsupported tool name must raise ValueError."""
        from src.rag.agent import Agent
        from src.rag.tool_benchmarks import _invoke_tool

        mock_agent = MagicMock(spec=Agent)
        with pytest.raises(ValueError, match="unsupported tool"):
            _invoke_tool(mock_agent, 'finish', 'done')


# ── TestRunToolBenchmarks ──────────────────────────────────────────────────────

class TestRunToolBenchmarks:
    """Integration tests for run_tool_benchmarks with mocked LLM calls."""

    @pytest.fixture
    def mock_store(self):
        """Minimal mock VectorStore for Agent construction."""
        return _make_mock_store()

    def test_returns_dict_with_required_keys(self, mock_store):
        """run_tool_benchmarks must return a dict with total, passed, failed, pass_rate, results."""
        from src.rag.tool_benchmarks import run_tool_benchmarks

        # Mock ollama.chat to return a passing sentiment/summarise response
        mock_sentiment = (
            "Sentiment: Positive\nTone: enthusiastic\n"
            "Key phrases: great\nExplanation: Good."
        )
        with patch('ollama.chat', return_value={'message': {'content': mock_sentiment}}):
            report = run_tool_benchmarks(mock_store)

        required_keys = {'total', 'passed', 'failed', 'pass_rate', 'results'}
        assert required_keys.issubset(set(report.keys()))

    def test_total_matches_number_of_test_cases(self, mock_store):
        """The 'total' count must equal the number of entries in TOOL_TEST_CASES."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES, run_tool_benchmarks
        mock_sentiment = (
            "Sentiment: Positive\nTone: enthusiastic\n"
            "Key phrases: great\nExplanation: Good."
        )
        with patch('ollama.chat', return_value={'message': {'content': mock_sentiment}}):
            report = run_tool_benchmarks(mock_store)

        assert report['total'] == len(TOOL_TEST_CASES)

    def test_passed_plus_failed_equals_total(self, mock_store):
        """passed + failed must always equal total."""
        from src.rag.tool_benchmarks import run_tool_benchmarks
        mock_sentiment = (
            "Sentiment: Neutral\nTone: flat\n"
            "Key phrases: boiling water\nExplanation: Neutral text."
        )
        with patch('ollama.chat', return_value={'message': {'content': mock_sentiment}}):
            report = run_tool_benchmarks(mock_store)

        assert report['passed'] + report['failed'] == report['total']

    def test_pass_rate_is_float_between_0_and_1(self, mock_store):
        """pass_rate must be a float in [0.0, 1.0]."""
        from src.rag.tool_benchmarks import run_tool_benchmarks
        mock_sentiment = (
            "Sentiment: Positive\nTone: happy\n"
            "Key phrases: good\nExplanation: All good."
        )
        with patch('ollama.chat', return_value={'message': {'content': mock_sentiment}}):
            report = run_tool_benchmarks(mock_store)

        assert 0.0 <= report['pass_rate'] <= 1.0

    def test_calculator_tests_pass_without_llm(self, mock_store):
        """Calculator tests must pass using only eval() — no LLM call needed."""
        from src.rag.tool_benchmarks import TOOL_TEST_CASES, run_tool_benchmarks

        # Only run the calculator test cases to verify they pass without LLM
        calc_cases = [tc for tc in TOOL_TEST_CASES if tc['tool'] == 'calculator']

        # Patch ollama.chat to raise an error if called — it should NOT be called
        with patch('ollama.chat', side_effect=RuntimeError("LLM should not be called")):
            # Directly test the calculator tool via Agent
            from src.rag.agent import Agent
            agent = Agent(mock_store)
            for tc in calc_cases:
                result = agent._tool_calculator(tc['input'])
                # Verify the check function works on the actual calculator output
                passed = tc['check'](result)
                assert passed, (
                    f"Calculator test failed: input={tc['input']!r}, "
                    f"result={result!r}, expected: {tc['note']}"
                )

    def test_results_list_has_correct_structure(self, mock_store):
        """Each result in the results list must have tool, input, result, passed, note keys."""
        from src.rag.tool_benchmarks import run_tool_benchmarks
        mock_sentiment = (
            "Sentiment: Positive\nTone: happy\n"
            "Key phrases: great\nExplanation: Good."
        )
        with patch('ollama.chat', return_value={'message': {'content': mock_sentiment}}):
            report = run_tool_benchmarks(mock_store)

        required_result_keys = {'tool', 'input', 'result', 'passed', 'note'}
        for result in report['results']:
            missing = required_result_keys - set(result.keys())
            assert not missing, f"Result entry missing keys: {missing}"
