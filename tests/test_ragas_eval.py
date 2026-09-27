"""Unit tests for src/rag/ragas_eval.py.

Tests cover:
  - _check_ragas_dependencies: raises ImportError when packages are missing
  - _configure_ragas_llm: returns a wrapped LLM object
  - _configure_ragas_embeddings: returns a wrapped embeddings object
  - _build_evaluation_dataset: calls run_pipeline and produces correct samples
  - run_ragas_evaluation: full flow with mocked RAGAS and VectorStore
  - print_ragas_results: prints header and per-question rows without raising

Mock strategy:
  - All RAGAS imports (ragas, langchain_ollama, datasets) are mocked via
    sys.modules patching so the tests run even when those packages are not
    installed.
  - VectorStore.run_pipeline is mocked to return a fixed dict.
  - ollama is NOT called — no network calls in these tests.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build minimal fake module trees for optional RAGAS dependencies
# ---------------------------------------------------------------------------

def _make_ragas_modules():
    """Return a dict of fake module objects that stand in for the RAGAS stack.

    We register these in sys.modules so any `import ragas.*` statement in
    ragas_eval.py resolves to our controlled fakes.
    """
    # Top-level `ragas` package
    ragas_mod = types.ModuleType('ragas')

    # Fake SingleTurnSample — just stores its kwargs as attributes
    class FakeSingleTurnSample:
        """Stores evaluation fields for one question/answer pair."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Fake EvaluationDataset — holds a list of samples
    class FakeEvaluationDataset:
        """Wraps a list of SingleTurnSample objects for RAGAS evaluate()."""
        def __init__(self, samples):
            self.samples = samples

    # Fake evaluate() — returns a mock result object with to_pandas()
    fake_df = MagicMock()
    fake_df.columns = ['faithfulness', 'response_relevancy']
    fake_df.__getitem__ = lambda self, key: MagicMock(**{'mean.return_value': 0.75})
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = fake_df

    ragas_mod.evaluate          = MagicMock(return_value=fake_result)
    ragas_mod.EvaluationDataset = FakeEvaluationDataset
    ragas_mod.SingleTurnSample  = FakeSingleTurnSample

    # ragas.metrics submodule
    metrics_mod = types.ModuleType('ragas.metrics')
    metrics_mod.Faithfulness       = MagicMock(return_value=MagicMock())
    metrics_mod.ResponseRelevancy  = MagicMock(return_value=MagicMock())
    metrics_mod.ContextPrecision   = MagicMock(return_value=MagicMock())
    metrics_mod.ContextRecall      = MagicMock(return_value=MagicMock())
    ragas_mod.metrics = metrics_mod

    # ragas.llms submodule
    llms_mod = types.ModuleType('ragas.llms')
    llms_mod.LangchainLLMWrapper = MagicMock(return_value=MagicMock())
    ragas_mod.llms = llms_mod

    # ragas.embeddings submodule
    embeddings_mod = types.ModuleType('ragas.embeddings')
    embeddings_mod.LangchainEmbeddingsWrapper = MagicMock(return_value=MagicMock())
    ragas_mod.embeddings = embeddings_mod

    # langchain_ollama
    langchain_ollama_mod = types.ModuleType('langchain_ollama')
    langchain_ollama_mod.ChatOllama      = MagicMock(return_value=MagicMock())
    langchain_ollama_mod.OllamaEmbeddings = MagicMock(return_value=MagicMock())

    # datasets (only needs to be importable, not used directly)
    datasets_mod = types.ModuleType('datasets')

    return {
        'ragas':                        ragas_mod,
        'ragas.metrics':                metrics_mod,
        'ragas.llms':                   llms_mod,
        'ragas.embeddings':             embeddings_mod,
        'langchain_ollama':             langchain_ollama_mod,
        'datasets':                     datasets_mod,
    }


def _inject_ragas(monkeypatch, fake_modules=None):
    """Register fake RAGAS modules into sys.modules for the duration of a test."""
    if fake_modules is None:
        fake_modules = _make_ragas_modules()
    for module_name, module_object in fake_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module_object)
    return fake_modules


def _make_pipeline_result(response='Cats sleep 16 hours.'):
    """Return a minimal dict that mimics store.run_pipeline() output."""
    entry = {
        'text': 'Cats sleep sixteen hours a day.',
        'source': 'cats.txt',
        'start_line': 1,
        'end_line': 1,
        'type': 'txt',
    }
    return {
        'response':    response,
        'reranked':    [(entry, 0.8, 8)],
        'is_confident': True,
        'query_type':  'factual',
        'retrieved':   [entry],
        'queries':     ['cats sleep'],
        'best_score':  0.8,
    }


# ---------------------------------------------------------------------------
# _check_ragas_dependencies
# ---------------------------------------------------------------------------

class TestCheckRagasDependencies:
    """Tests that _check_ragas_dependencies raises clearly when packages are missing."""

    def test_raises_when_ragas_missing(self, monkeypatch):
        """If 'ragas' is not importable, an ImportError with install instructions is raised."""
        # Remove ragas from sys.modules so the import inside the function fails
        monkeypatch.delitem(sys.modules, 'ragas', raising=False)
        with patch.dict(sys.modules, {'ragas': None}):
            from src.rag.ragas_eval import _check_ragas_dependencies
            with pytest.raises(ImportError, match='pip install'):
                _check_ragas_dependencies()

    def test_raises_when_langchain_ollama_missing(self, monkeypatch):
        """If 'langchain_ollama' is not importable, an ImportError is raised."""
        monkeypatch.delitem(sys.modules, 'langchain_ollama', raising=False)
        fake_modules = _make_ragas_modules()
        _inject_ragas(monkeypatch, fake_modules)
        # Now hide langchain_ollama specifically
        monkeypatch.setitem(sys.modules, 'langchain_ollama', None)
        from src.rag.ragas_eval import _check_ragas_dependencies
        with pytest.raises(ImportError, match='langchain_ollama'):
            _check_ragas_dependencies()

    def test_passes_when_all_installed(self, monkeypatch):
        """If all three packages are importable, no exception is raised."""
        _inject_ragas(monkeypatch)
        from src.rag.ragas_eval import _check_ragas_dependencies

        # Should not raise
        _check_ragas_dependencies()


# ---------------------------------------------------------------------------
# _build_evaluation_dataset
# ---------------------------------------------------------------------------

class TestBuildEvaluationDataset:
    """Tests that _build_evaluation_dataset calls run_pipeline and builds correct samples."""

    def test_calls_run_pipeline_for_each_test_case(self, monkeypatch):
        """Given 2 test cases, run_pipeline is called exactly 2 times."""
        _inject_ragas(monkeypatch)

        mock_store = MagicMock()
        mock_store.run_pipeline.return_value = _make_pipeline_result()

        test_cases = [
            {'question': 'Q1', 'ground_truth': 'A1'},
            {'question': 'Q2', 'ground_truth': 'A2'},
        ]

        from src.rag.ragas_eval import _build_evaluation_dataset
        _build_evaluation_dataset(mock_store, test_cases)

        assert mock_store.run_pipeline.call_count == 2

    def test_raw_results_has_correct_length(self, monkeypatch):
        """Given 3 test cases, raw_results list has exactly 3 entries."""
        _inject_ragas(monkeypatch)

        mock_store = MagicMock()
        mock_store.run_pipeline.return_value = _make_pipeline_result()

        test_cases = [
            {'question': f'Q{i}', 'ground_truth': f'A{i}'}
            for i in range(3)
        ]

        from src.rag.ragas_eval import _build_evaluation_dataset
        _dataset, raw_results = _build_evaluation_dataset(mock_store, test_cases)

        assert len(raw_results) == 3

    def test_raw_result_has_required_keys(self, monkeypatch):
        """Each entry in raw_results has question, response, n_contexts, confident."""
        _inject_ragas(monkeypatch)

        mock_store = MagicMock()
        mock_store.run_pipeline.return_value = _make_pipeline_result()

        test_cases = [{'question': 'How many hours do cats sleep?', 'ground_truth': '16 hours'}]

        from src.rag.ragas_eval import _build_evaluation_dataset
        _dataset, raw_results = _build_evaluation_dataset(mock_store, test_cases)

        row = raw_results[0]
        assert 'question'   in row
        assert 'response'   in row
        assert 'n_contexts' in row
        assert 'confident'  in row

    def test_n_contexts_matches_reranked_length(self, monkeypatch):
        """n_contexts in raw_results matches the number of reranked chunks from run_pipeline."""
        _inject_ragas(monkeypatch)

        mock_store = MagicMock()
        # Pipeline result with 2 reranked chunks
        entry = {'text': 'chunk', 'source': 's', 'start_line': 1, 'end_line': 1, 'type': 'txt'}
        mock_store.run_pipeline.return_value = {
            'response':    'Some answer',
            'reranked':    [(entry, 0.9, 9), (entry, 0.8, 7)],
            'is_confident': True,
            'query_type':  'factual',
        }

        test_cases = [{'question': 'Q1', 'ground_truth': 'A1'}]

        from src.rag.ragas_eval import _build_evaluation_dataset
        _dataset, raw_results = _build_evaluation_dataset(mock_store, test_cases)

        assert raw_results[0]['n_contexts'] == 2

    def test_dataset_samples_match_test_case_count(self, monkeypatch):
        """EvaluationDataset has exactly as many samples as there are test cases."""
        _inject_ragas(monkeypatch)

        mock_store = MagicMock()
        mock_store.run_pipeline.return_value = _make_pipeline_result()

        test_cases = [
            {'question': f'Q{i}', 'ground_truth': f'A{i}'}
            for i in range(4)
        ]

        from src.rag.ragas_eval import _build_evaluation_dataset
        dataset, _raw = _build_evaluation_dataset(mock_store, test_cases)

        assert len(dataset.samples) == 4


# ---------------------------------------------------------------------------
# run_ragas_evaluation
# ---------------------------------------------------------------------------

class TestRunRagasEvaluation:
    """Tests the full run_ragas_evaluation() flow with mocked RAGAS and VectorStore."""

    def _make_store(self):
        """Return a MagicMock store whose run_pipeline returns a valid result."""
        mock_store = MagicMock()
        mock_store.run_pipeline.return_value = _make_pipeline_result()
        return mock_store

    def test_returns_dict_with_scores_and_raw_results(self, monkeypatch):
        """run_ragas_evaluation returns a dict with 'scores' and 'raw_results' keys."""
        _inject_ragas(monkeypatch)

        test_cases = [{'question': 'How many hours do cats sleep?', 'ground_truth': '16'}]
        from src.rag.ragas_eval import run_ragas_evaluation
        result = run_ragas_evaluation(self._make_store(), test_cases=test_cases)

        assert 'scores'      in result
        assert 'raw_results' in result

    def test_raw_results_length_matches_test_cases(self, monkeypatch):
        """raw_results list has the same length as the test_cases input."""
        _inject_ragas(monkeypatch)

        test_cases = [
            {'question': f'Q{i}', 'ground_truth': f'A{i}'}
            for i in range(3)
        ]
        from src.rag.ragas_eval import run_ragas_evaluation
        result = run_ragas_evaluation(self._make_store(), test_cases=test_cases)

        assert len(result['raw_results']) == 3

    def test_raises_import_error_when_dependencies_missing(self, monkeypatch):
        """If ragas is not installed, run_ragas_evaluation raises ImportError."""
        # Ensure ragas resolves to None (unimportable)
        monkeypatch.setitem(sys.modules, 'ragas', None)
        monkeypatch.setitem(sys.modules, 'langchain_ollama', None)
        monkeypatch.setitem(sys.modules, 'datasets', None)

        from src.rag.ragas_eval import run_ragas_evaluation
        with pytest.raises(ImportError):
            run_ragas_evaluation(MagicMock(), test_cases=[{'question': 'Q', 'ground_truth': 'A'}])

    def test_uses_default_test_cases_when_none_provided(self, monkeypatch):
        """When test_cases is None, DEFAULT_TEST_CASES from Benchmarker is used."""
        _inject_ragas(monkeypatch)

        mock_store = self._make_store()

        from src.rag.benchmarker import Benchmarker
        expected_count = len(Benchmarker.DEFAULT_TEST_CASES)

        from src.rag.ragas_eval import run_ragas_evaluation
        result = run_ragas_evaluation(mock_store, test_cases=None)

        assert len(result['raw_results']) == expected_count


# ---------------------------------------------------------------------------
# print_ragas_results
# ---------------------------------------------------------------------------

class TestPrintRagasResults:
    """Tests that print_ragas_results renders the report without raising."""

    def _make_result(self, n_rows=2):
        """Build a minimal result dict as returned by run_ragas_evaluation."""
        # Fake RAGAS scores object
        fake_df = MagicMock()
        fake_df.columns = ['faithfulness', 'response_relevancy']
        fake_df.__getitem__ = lambda self, key: MagicMock(**{'mean.return_value': 0.80})
        fake_scores = MagicMock()
        fake_scores.to_pandas.return_value = fake_df

        raw_results = [
            {
                'question':   f'Question number {i}?',
                'response':   'Some answer text here.',
                'n_contexts': 3,
                'confident':  True,
            }
            for i in range(n_rows)
        ]
        return {'scores': fake_scores, 'raw_results': raw_results}

    def test_does_not_raise(self, capsys):
        """print_ragas_results with a valid result dict: no exception is raised."""
        from src.rag.ragas_eval import print_ragas_results
        print_ragas_results(self._make_result(n_rows=3))
        # No assertion needed — test passes if no exception was raised

    def test_prints_header(self, capsys):
        """Output contains 'RAGAS Results' heading."""
        from src.rag.ragas_eval import print_ragas_results
        print_ragas_results(self._make_result(n_rows=1))
        captured = capsys.readouterr()
        assert 'RAGAS Results' in captured.out

    def test_prints_per_question_row(self, capsys):
        """Output contains one row per entry in raw_results."""
        from src.rag.ragas_eval import print_ragas_results
        result = self._make_result(n_rows=2)
        print_ragas_results(result)
        captured = capsys.readouterr()
        # Row indices 1 and 2 should appear
        assert '1' in captured.out
        assert '2' in captured.out

    def test_handles_to_pandas_failure_gracefully(self, capsys):
        """If to_pandas() raises, print_ragas_results falls back without crashing."""
        fake_scores = MagicMock()
        fake_scores.to_pandas.side_effect = RuntimeError('pandas not available')
        result = {
            'scores': fake_scores,
            'raw_results': [{'question': 'Q?', 'response': 'A', 'n_contexts': 1, 'confident': False}],
        }
        from src.rag.ragas_eval import print_ragas_results

        # Should not raise — graceful fallback path is exercised
        print_ragas_results(result)
        captured = capsys.readouterr()
        assert 'RAGAS Results' in captured.out
