"""ragas_eval.py — RAGAS evaluation for the RAG pipeline.

WHY THIS FILE EXISTS:
    The custom benchmarker (benchmarker.py) measures retrieval quality with
    hand-crafted scoring functions. RAGAS adds a second layer of evaluation
    using LLM-as-a-judge metrics that are widely recognised in the industry:
    Faithfulness, ResponseRelevancy, ContextPrecision, and ContextRecall.

    Having both gives a complete picture:
      - Custom metrics: fast, deterministic, no extra dependencies
      - RAGAS metrics:  model-backed, aligned with academic benchmarks

HOW TO USE:
    Install optional dependencies first:
        pip install "ragas>=0.2.0" langchain-ollama datasets

    Then run:
        python main.py --ragas

DEPENDENCIES (optional — not in requirements.txt):
    ragas>=0.2.0, langchain-ollama, datasets

    These live in pyproject.toml under [project.optional-dependencies] eval.
    They are NOT loaded at import time — only when run_ragas_evaluation() is called.
    This means the rest of the app still starts normally if RAGAS is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

# VectorStore is only used for type-checking — import deferred to avoid
# circular imports at module load time.
if TYPE_CHECKING:
    from src.rag.vector_store import VectorStore

from src.rag.config import EMBEDDING_MODEL, LANGUAGE_MODEL

# Width used for separator lines in terminal output
_LINE_WIDTH = 72

__all__ = ['run_ragas_evaluation', 'print_ragas_results']


# ---------------------------------------------------------------------------
# Dependency check helpers
# ---------------------------------------------------------------------------

def _check_ragas_dependencies() -> None:
    """Raise a clear ImportError if any RAGAS dependency is missing.

    Called at the start of run_ragas_evaluation() so the user gets a
    helpful message instead of a cryptic ModuleNotFoundError deep in
    the call stack.
    """
    missing = []
    for package in ('ragas', 'langchain_ollama', 'datasets'):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise ImportError(
            f"\nRAGAS dependencies not installed: {', '.join(missing)}\n"
            "Install them with:\n"
            '    pip install "ragas>=0.2.0" langchain-ollama datasets\n'
            "Or install the full eval group:\n"
            '    pip install -e ".[eval]"'
        )


# ---------------------------------------------------------------------------
# LLM + embeddings configuration
# ---------------------------------------------------------------------------

def _configure_ragas_llm():
    """Wrap the local Ollama LLM for use with RAGAS.

    RAGAS requires a LangChain-compatible LLM to run its judge prompts.
    We use the same LANGUAGE_MODEL that the pipeline uses so results are
    consistent with the rest of the system.

    Returns:
        A LangchainLLMWrapper wrapping ChatOllama.
    """
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    # ChatOllama communicates with the local Ollama daemon on port 11434.
    # temperature=0 makes the judge scores deterministic across runs.
    ollama_llm = ChatOllama(model=LANGUAGE_MODEL, temperature=0)
    return LangchainLLMWrapper(ollama_llm)


def _configure_ragas_embeddings():
    """Wrap the local Ollama embedding model for use with RAGAS.

    RAGAS uses embeddings when computing ResponseRelevancy (it embeds the
    question and the answer and computes cosine similarity).  We use the
    same EMBEDDING_MODEL as the pipeline for consistency.

    Returns:
        A LangchainEmbeddingsWrapper wrapping OllamaEmbeddings.
    """
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(ollama_embeddings)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_evaluation_dataset(
    store: 'VectorStore',
    test_cases: List[dict],
) -> tuple:
    """Run the RAG pipeline for every test case and collect inputs/outputs.

    For each question the pipeline runs in non-streaming mode to get the
    full response and the retrieved context chunks.  These are assembled
    into a RAGAS EvaluationDataset.

    Args:
        store:      An initialised VectorStore with documents already loaded.
        test_cases: List of dicts, each with 'question' and 'ground_truth'.

    Returns:
        A tuple of (EvaluationDataset, list_of_raw_results).
        raw_results contains per-question dicts for the printed table.
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    raw_results = []

    total = len(test_cases)
    for index, test_case in enumerate(test_cases, start=1):
        question     = test_case['question']
        ground_truth = test_case.get('ground_truth', '')

        print(f"  [{index}/{total}] Running pipeline for: {question[:60]}...")

        # Run the full RAG pipeline (non-streaming) to get the response
        # and the reranked context chunks that were fed to the LLM.
        pipeline_result = store.run_pipeline(question, streamlit_mode=True)

        response = pipeline_result.get('response', '')

        # Extract the plain text of each context chunk that was actually
        # retrieved and reranked — this is what the LLM was given.
        reranked_chunks = pipeline_result.get('reranked', [])
        retrieved_contexts = [
            entry['text']
            for entry, _sim_score, _llm_score in reranked_chunks
        ]

        # Build a RAGAS SingleTurnSample for this question.
        # reference_contexts is the ground-truth supporting context — we use
        # the retrieved contexts here (same as the LLM input) since we do
        # not have labelled reference passages for each question.
        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=ground_truth,
        )
        samples.append(sample)

        # Keep raw data for the printed summary table
        raw_results.append({
            'question':   question,
            'response':   response,
            'n_contexts': len(retrieved_contexts),
            'confident':  pipeline_result.get('is_confident', False),
        })

    dataset = EvaluationDataset(samples=samples)
    return dataset, raw_results


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    store: 'VectorStore',
    test_cases: Optional[List[dict]] = None,
) -> dict:
    """Run RAGAS evaluation on the RAG pipeline and return the scores.

    Runs these four LLM-as-a-judge metrics:
        Faithfulness       — does the answer stay within the retrieved context?
        ResponseRelevancy  — does the answer address the question asked?
        ContextPrecision   — are the top-ranked chunks the most relevant ones?
        ContextRecall      — does the retrieved context cover the ground truth?

    Uses the same DEFAULT_TEST_CASES as the custom benchmarker so the two
    sets of scores are directly comparable.

    Args:
        store:      An initialised VectorStore with documents loaded.
        test_cases: Optional list of test dicts to override the defaults.
                    Each dict must have 'question' and 'ground_truth'.

    Returns:
        A dict with keys 'scores' (RAGAS result object) and 'raw_results'
        (list of per-question pipeline outputs).

    Raises:
        ImportError: If ragas, langchain-ollama, or datasets are not installed.
    """
    # Fail fast with a helpful message if dependencies are missing
    _check_ragas_dependencies()

    from ragas import evaluate
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy

    # Import DEFAULT_TEST_CASES here (not at module top) to avoid a circular
    # import between ragas_eval and benchmarker at load time.
    if test_cases is None:
        from src.rag.benchmarker import Benchmarker
        test_cases = Benchmarker.DEFAULT_TEST_CASES

    print("=" * _LINE_WIDTH)
    print("  RAGAS Evaluation — LLM-as-a-Judge Metrics")
    print("=" * _LINE_WIDTH)
    print(f"  Model (judge + embeddings): {LANGUAGE_MODEL}")
    print(f"  Test cases: {len(test_cases)}")
    print()
    print("  Metrics:")
    print("    • Faithfulness       — answer grounded in retrieved context")
    print("    • ResponseRelevancy  — answer addresses the question")
    print("    • ContextPrecision   — best chunks ranked highest")
    print("    • ContextRecall      — context covers ground truth")
    print()
    print("  Running pipeline for each test case...")
    print("-" * _LINE_WIDTH)

    # Run the pipeline for every test case and collect RAGAS samples
    evaluation_dataset, raw_results = _build_evaluation_dataset(store, test_cases)

    print()
    print("  Evaluating with RAGAS metrics (this calls the LLM for each sample)...")
    print("-" * _LINE_WIDTH)

    # Set up the local Ollama judge and embeddings
    ragas_llm         = _configure_ragas_llm()
    ragas_embeddings  = _configure_ragas_embeddings()

    # Define the four metrics — each one is configured with the same local model
    metrics = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # Run RAGAS evaluation — this sends each (question, answer, context) triple
    # to the LLM as a structured judge prompt and collects numeric scores.
    scores = evaluate(dataset=evaluation_dataset, metrics=metrics)

    return {'scores': scores, 'raw_results': raw_results}


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_ragas_results(result: dict) -> None:
    """Print the RAGAS evaluation results to the terminal in a readable format.

    Prints a summary table with mean scores for all four metrics, then
    a per-question breakdown showing how many context chunks were retrieved
    and whether the pipeline was confident for each question.

    Args:
        result: The dict returned by run_ragas_evaluation().
    """
    scores      = result['scores']
    raw_results = result['raw_results']

    print()
    print("=" * _LINE_WIDTH)
    print("  RAGAS Results — Summary")
    print("=" * _LINE_WIDTH)

    # RAGAS returns a result object that can be converted to a pandas DataFrame.
    # We convert to a dict of column→list for easy iteration without requiring
    # pandas as an import in this file.
    try:
        scores_df = scores.to_pandas()

        # Compute mean for each metric across all test cases
        metric_means = {
            column: scores_df[column].mean()
            for column in scores_df.columns
            if column not in ('user_input', 'response', 'retrieved_contexts', 'reference')
        }

        for metric_name, mean_score in metric_means.items():
            # Build a simple bar chart so the score is easy to compare at a glance
            bar_filled = int(mean_score * 20)
            bar_empty  = 20 - bar_filled
            bar        = '[' + '█' * bar_filled + '░' * bar_empty + ']'
            print(f"  {metric_name:<24} {bar}  {mean_score:.3f}")

    except Exception as render_error:
        # If DataFrame conversion fails, fall back to printing the raw object
        print(f"  (Could not render table: {render_error})")
        print(f"  Raw scores: {scores}")

    print()
    print("-" * _LINE_WIDTH)
    print("  Per-question pipeline summary:")
    print(f"  {'#':<4} {'Question':<50} {'Ctx':>3} {'Conf':>5}")
    print("-" * _LINE_WIDTH)

    for index, row in enumerate(raw_results, start=1):
        question_short = row['question'][:48] + '..' if len(row['question']) > 50 else row['question']
        confident_flag = 'YES' if row['confident'] else 'NO '
        print(f"  {index:<4} {question_short:<50} {row['n_contexts']:>3} {confident_flag:>5}")

    print("=" * _LINE_WIDTH)
    print()
