"""cli/runner.py — Logic for each command-line mode.

WHY THIS FILE EXISTS:
    main.py must stay under 50 lines so it is easy to read at a glance.
    The actual work — initialising the pipeline, running the chat loop,
    running the agent loop, running benchmarks — all lives here.

ADDING A NEW MODE:
    1. Write a new run_*() function in this file.
    2. Add a new --flag to main.py's argument parser.
    3. Call your new function from main.py.
    You never need to change the existing run_* functions.
"""

import logging
from typing import Tuple

from src.rag.agent import Agent
from src.rag.benchmarker import Benchmarker
from src.rag.config import BENCHMARK_DOCS_DIR, CHROMA_DIR, DOC_FOLDERS, DOCS_ROOT, TOP_RERANK
from src.rag.document_loader import DocumentLoader
from src.rag.tool_benchmarks import run_tool_benchmarks
from src.rag.vector_store import VectorStore

# Module-level logger — replaces bare print() for non-user-facing messages
logger = logging.getLogger(__name__)

__all__ = ['initialize', 'run_benchmark', 'run_agent', 'run_chat', 'run_ragas']


def initialize() -> Tuple[DocumentLoader, VectorStore]:
    """Load all documents, build the vector index, and return the two main objects.

    Prints a startup summary to the terminal so the user knows what is loading.

    Returns:
        A tuple of (DocumentLoader, VectorStore) ready to handle queries.
    """
    print("=" * 60)
    print("  Initializing RAG Pipeline")
    print("=" * 60)
    print(f"  Docs root:  {DOCS_ROOT}/")
    for document_type, folder_path in DOC_FOLDERS.items():
        print(f"    {document_type.upper():<8} → {folder_path}/")
    print(f"  Vector DB:  ChromaDB (persistent @ {CHROMA_DIR})")
    print("  Reranker:   LLM-based (document type-aware)")
    print("  Misplaced file detection: ENABLED")
    print("=" * 60 + "\n")

    loader = DocumentLoader()
    loader.ensure_folders()
    all_chunks = loader.chunk_all_documents()

    store = VectorStore()
    store.build_or_load(all_chunks)
    return loader, store


def run_benchmark(loader: DocumentLoader, store: VectorStore) -> None:
    """Run the full automated benchmark suite and print scores to the terminal.

    The benchmark runs in two phases:

    Phase 1 — RAG pipeline benchmark (Benchmarker):
        Loads sample documents from benchmark_docs/ into the VectorStore, then
        evaluates retrieval + generation quality on 15 questions spanning 4 domains
        (cat facts, Python, team members, machine learning) across 3 file types
        (txt, csv, md). Saves results to benchmark_results.json and benchmark_results.csv.

    Phase 2 — Agent tool benchmark (tool_benchmarks):
        Tests all 3 direct-call tools in isolation:
            - calculator: 5 tests, exact numeric correctness
            - sentiment:  4 tests, 4-field format + valid label
            - summarise:  3 tests, keyword coverage

    Args:
        loader: A DocumentLoader used to chunk benchmark_docs/ sample files.
        store:  An already-initialised VectorStore with documents loaded.
    """
    # ── Phase 1 setup: load benchmark sample documents ───────────────────────
    print(f"\nLoading benchmark sample documents from '{BENCHMARK_DOCS_DIR}/'...")
    benchmark_chunks = loader.chunk_directory(BENCHMARK_DOCS_DIR)

    if benchmark_chunks:
        # Add benchmark docs to the live VectorStore so pipeline questions can
        # retrieve from them.  Use a distinct id_prefix so they can be identified.
        store.add_chunks(benchmark_chunks, id_prefix='bench')
        store.rebuild_bm25(store.chunks)
        print(f"  Added {len(benchmark_chunks)} benchmark chunks to the index.\n")
    else:
        print("  No benchmark documents found — running on existing index only.\n")

    # ── Phase 1: RAG pipeline benchmark ─────────────────────────────────────
    Benchmarker(store).run()

    # ── Phase 2: Agent tool benchmark ───────────────────────────────────────
    run_tool_benchmarks(store)


def run_agent(store: VectorStore) -> None:
    """Start an interactive agent loop in the terminal.

    The agent can use rag_search, calculator, summarise, sentiment, and finish.
    Type 'exit' or press Ctrl+C to quit.

    Args:
        store: An already-initialised VectorStore with documents loaded.
    """
    agent = Agent(store)
    print("Agent mode — type your task (or 'exit' to quit):\n")

    while True:
        try:
            user_task = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_task:
            continue  # User pressed Enter with nothing typed — ask again

        if user_task.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        result = agent.run(user_task)
        print(f"\nFinal answer: {result['answer']}\n" + "-" * 60)


def run_ragas(store: VectorStore) -> None:
    """Run RAGAS LLM-as-a-judge evaluation on the RAG pipeline.

    Runs four metrics against the same DEFAULT_TEST_CASES used by the custom
    benchmarker: Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall.

    Requires optional dependencies:
        pip install "ragas>=0.2.0" langchain-ollama datasets
    Or:
        pip install -e ".[eval]"

    Prints a scored summary table to the terminal when done.

    Args:
        store: An already-initialised VectorStore with documents loaded.
    """
    try:
        from src.rag.ragas_eval import print_ragas_results, run_ragas_evaluation
    except ImportError as error:
        # Happens if ragas / langchain-ollama / datasets are not installed.
        print(f"\nCould not import RAGAS evaluation module: {error}")
        return

    try:
        result = run_ragas_evaluation(store)
        print_ragas_results(result)
    except ImportError as error:
        # Raised by _check_ragas_dependencies() inside run_ragas_evaluation()
        # when one or more RAGAS packages are missing.
        print(str(error))
    except Exception as error:
        logger.exception("RAGAS evaluation failed: %s", error)
        print(f"\nRAGAS evaluation encountered an error: {error}")


def run_chat(store: VectorStore) -> None:
    """Start an interactive chat loop in the terminal.

    Uses the full RAG pipeline: classify → expand → retrieve → rerank → generate.
    Prefix a message with 'agent:' to run one question through the agent instead.
    Type 'exit', 'quit', or 'bye' to quit.

    Args:
        store: An already-initialised VectorStore with documents loaded.
    """
    print("=" * 60 + "\n  RAG Chatbot — Full Pipeline\n" + "=" * 60)
    print("Commands: 'exit' to quit | 'agent: <question>' for one agent turn\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_query:
            continue  # Empty line — ask again

        if user_query.lower() in ('exit', 'quit', 'bye'):
            print("Goodbye!")
            break

        # Allow a one-off agent query without switching modes permanently
        if user_query.lower().startswith('agent:'):
            agent_query = user_query[6:].strip()
            print("\n[Agent mode]")
            agent_result = Agent(store).run(agent_query)
            print(f"\nAgent answer: {agent_result['answer']}")
        else:
            pipeline_result = store.run_pipeline(user_query)

            # Warn the user if the answer might be unreliable
            if not pipeline_result['is_confident']:
                print(f"[Warning] Low confidence ({pipeline_result['best_score']:.2f})")

            print(
                f"\n[type:{pipeline_result['query_type']} | "
                f"expanded:{len(pipeline_result['queries'])} queries]"
            )
            print(
                f"Before rerank: {len(pipeline_result['retrieved'])} chunks | "
                f"After: {TOP_RERANK} chunks"
            )

        print("-" * 60)
