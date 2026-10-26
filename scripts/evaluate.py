"""Orchestrator script to evaluate retrieval, reranking, and answer generation on FiQA."""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_fiqa.config import (
    ABSTENTION_QUERIES_PATH,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    EVAL_DIR,
    EVAL_GENERATION_SAMPLE,
    EVAL_NUM_QUERIES,
    EVAL_SEED,
    EVAL_SPLITS,
    EVAL_TOP_K_DOCUMENTS,
    GENERATION_MODEL_NAME,
    INDEX_MANIFEST_PATH,
    MIN_RERANK_SCORE,
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
    QRELS_DIR,
    QUERIES_PATH,
    RERANK_TOP_K_CHUNKS,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    RETRIEVAL_TOP_K_CHUNKS,
    SUPPORTED_GENERATION_MODELS,
)
from rag_fiqa.evaluation import (
    evaluate_abstention,
    evaluate_generation,
    evaluate_retrieval,
    format_abstention_table,
    format_retrieval_table,
    get_indexed_document_ids,
    load_abstention_queries,
    load_qrels,
    load_queries,
    select_eval_queries,
)
from rag_fiqa.generation import create_ollama_client
from rag_fiqa.indexing import load_embedding_model
from rag_fiqa.reranking import load_reranker_model
from rag_fiqa.retrieval import open_vector_store
from rag_fiqa.utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=EVAL_SPLITS, default="dev", help="Qrels split to evaluate (default: dev).")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=EVAL_NUM_QUERIES,
        help=f"Number of sampled judged queries; 0 evaluates all queries in the split (default: {EVAL_NUM_QUERIES}).",
    )
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help=f"Query sampling seed (default: {EVAL_SEED}).")
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=RETRIEVAL_TOP_K_CHUNKS,
        help=f"Candidate chunks retrieved from Qdrant (default: {RETRIEVAL_TOP_K_CHUNKS}).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=RERANK_TOP_K_CHUNKS,
        help=f"Reranked chunks passed to the generator (default: {RERANK_TOP_K_CHUNKS}).",
    )
    parser.add_argument(
        "--with-generation", action="store_true", help="Also generate and check answers for a query sample."
    )
    parser.add_argument(
        "--generation-sample",
        type=int,
        default=EVAL_GENERATION_SAMPLE,
        help=f"Queries sent to the generator when --with-generation is set (default: {EVAL_GENERATION_SAMPLE}).",
    )
    parser.add_argument(
        "--generation-model",
        choices=SUPPORTED_GENERATION_MODELS,
        default=GENERATION_MODEL_NAME,
        help=f"Ollama generation model (default: {GENERATION_MODEL_NAME}).",
    )
    parser.add_argument(
        "--withholding-check",
        action="store_true",
        help="Also answer each sampled question with another query's context and record abstentions.",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=MIN_RERANK_SCORE,
        help=f"Minimum rerank score to trigger generation (default: {MIN_RERANK_SCORE}).",
    )
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--collection-name", default=QDRANT_COLLECTION_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR,
        help=f"Directory for evaluation outputs (default: {EVAL_DIR}).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional descriptive label appended to the timestamped run directory.",
    )
    parser.add_argument(
        "--abstention-only",
        "--abstention",
        dest="abstention_only",
        action="store_true",
        help="Run evaluation exclusively on unanswerable and out-of-domain queries to test abstention.",
    )
    parser.add_argument(
        "--abstention-queries",
        type=Path,
        default=ABSTENTION_QUERIES_PATH,
        help=f"Path to JSON file containing unanswerable queries (default: {ABSTENTION_QUERIES_PATH}).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (recommended for non-interactive scripts or CI).",
    )
    return parser.parse_args()


def write_json(path: Path, data: object) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_abstention(args: argparse.Namespace) -> None:
    """Run evaluation exclusively on unanswerable and out-of-domain queries."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = args.output_dir / "abstention" / run_id
    latest_dir = args.output_dir / "abstention" / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    abstention_queries = load_abstention_queries(args.abstention_queries)
    logger.info("Evaluating abstention on %d unanswerable queries", len(abstention_queries))

    client = open_vector_store(args.qdrant_path, args.collection_name)
    try:
        summary, records = evaluate_abstention(
            queries=abstention_queries,
            qdrant_client=client,
            embedding_model=load_embedding_model(),
            reranker=load_reranker_model(),
            ollama_client=create_ollama_client(),
            model_name=args.generation_model,
            retrieval_top_k=args.retrieval_top_k,
            rerank_top_k=args.rerank_top_k,
            min_rerank_score=args.min_rerank_score,
            collection_name=args.collection_name,
            show_progress=not args.no_progress,
        )
    finally:
        client.close()

    summary_payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "abstention",
        "settings": {
            "model": args.generation_model,
            "retrieval_top_k": args.retrieval_top_k,
            "rerank_top_k": args.rerank_top_k,
            "min_rerank_score": args.min_rerank_score,
        },
        "summary": summary,
    }

    for target in (run_dir, latest_dir):
        write_json(target / "eval_abstention_summary.json", summary_payload)
        write_json(target / "eval_abstention_records.json", records)

    print(f"\nAbstention Evaluation ({summary['num_queries']} out-of-domain / unanswerable queries)\n")
    print(format_abstention_table(summary))
    print("\nMetrics:")
    total_rate = summary["total_abstention_rate"]
    total_str = f"{total_rate * 100:.1f}%" if total_rate is not None else "N/A"
    print(f"- Total abstention rate: {total_str}")
    pre_rate = summary["pre_llm_abstention_rate"]
    pre_str = f"{pre_rate * 100:.1f}%" if pre_rate is not None else "N/A"
    print(f"- Pre-LLM score-gated rate: {pre_str}")
    if summary["llm_abstention_rate"] is not None:
        print(f"- LLM refusal rate (of queries reaching model): {summary['llm_abstention_rate'] * 100:.1f}%")
    halluc_rate = summary["hallucination_rate"]
    halluc_str = f"{halluc_rate * 100:.1f}%" if halluc_rate is not None else "N/A"
    print(f"- Hallucination rate: {halluc_str}")
    print(f"\nResults written to {run_dir} (and {latest_dir})")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run standard retrieval and optional generation evaluation on FiQA splits."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = args.output_dir / args.split / run_id
    latest_dir = args.output_dir / args.split / "latest"
    legacy_dir = args.output_dir / args.split
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_DIR / f"{args.split}.tsv")
    client = open_vector_store(args.qdrant_path, args.collection_name)
    try:
        indexed_doc_ids = get_indexed_document_ids(client, args.collection_name)
        eval_queries = select_eval_queries(
            queries, qrels, args.num_queries or None, args.seed, indexed_doc_ids=indexed_doc_ids
        )
        logger.info("Evaluating %d %s queries (seed=%d)", len(eval_queries), args.split, args.seed)

        retrieval_summary, per_query, contexts = evaluate_retrieval(
            eval_queries,
            client,
            load_embedding_model(),
            load_reranker_model(),
            retrieval_top_k=args.retrieval_top_k,
            rerank_top_k=args.rerank_top_k,
            collection_name=args.collection_name,
            show_progress=not args.no_progress,
        )
    finally:
        client.close()

    index_manifest = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8")) if INDEX_MANIFEST_PATH.exists() else {}
    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "seed": args.seed,
        "settings": {
            "retrieval_top_k_chunks": args.retrieval_top_k,
            "eval_top_k_documents": EVAL_TOP_K_DOCUMENTS,
            "rerank_top_k_chunks": args.rerank_top_k,
            "embedding_model": f"{EMBEDDING_MODEL_ID}@{EMBEDDING_MODEL_REVISION}",
            "reranker_model": f"{RERANKER_MODEL_ID}@{RERANKER_MODEL_REVISION}",
        },
        "index": {key: index_manifest.get(key) for key in ("input_sha256", "subset_size", "chunks_indexed")},
        "query_ids": [q["query_id"] for q in eval_queries],
        "retrieval": retrieval_summary,
    }

    generations: list[dict[str, Any]] | None = None
    if args.with_generation:
        sample_size = min(args.generation_sample, len(eval_queries))
        sample = sorted(random.Random(args.seed).sample(eval_queries, sample_size), key=lambda q: q["query_id"])
        generation_summary, generations = evaluate_generation(
            sample,
            contexts,
            ollama_client=create_ollama_client(),
            model_name=args.generation_model,
            withholding_check=args.withholding_check,
            show_progress=not args.no_progress,
        )
        summary["generation"] = generation_summary

    for target in (run_dir, latest_dir, legacy_dir):
        write_json(target / "eval_summary.json", summary)
        write_json(target / "eval_per_query.json", per_query)
        if generations is not None:
            write_json(target / "eval_generations.json", generations)

    print(f"\nRetrieval on FiQA {args.split} ({retrieval_summary['num_queries']} queries)\n")
    print(format_retrieval_table(retrieval_summary))
    if args.with_generation and "generation" in summary:
        print(f"\nGeneration checks ({summary['generation']['num_queries']} queries)\n")
        for key, value in summary["generation"].items():
            print(f"- {key}: {value}")
    print(f"\nResults written to {run_dir} (and {latest_dir})")


def run(args: argparse.Namespace) -> None:
    """Run the requested evaluation mode."""
    if args.abstention_only:
        run_abstention(args)
    else:
        run_benchmark(args)


def main() -> None:
    """Configure logging and run the evaluation."""
    args = parse_args()
    setup_logging()
    try:
        run(args)
    except Exception:
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
