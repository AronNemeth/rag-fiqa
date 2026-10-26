"""Evaluate retrieval, reranking, and generated answers against FiQA relevance judgments."""

import csv
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

from ollama import Client
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from rag_fiqa.config import (
    ABSTENTION_QUERIES_PATH,
    EVAL_TOP_K_DOCUMENTS,
    GENERATION_MODEL_NAME,
    MIN_RERANK_SCORE,
    QDRANT_COLLECTION_NAME,
    RERANK_TOP_K_CHUNKS,
    RETRIEVAL_TOP_K_CHUNKS,
)
from rag_fiqa.generation import generate_answer
from rag_fiqa.reranking import rerank_chunks
from rag_fiqa.retrieval import deduplicate_to_parent_documents, retrieve_chunks
from rag_fiqa.utils import get_logger

logger = get_logger(__name__)

METRIC_NAMES = ("hit@5", "recall@5", "recall@20", "mrr@10", "ndcg@10")
PRE_LLM_ABSTENTION_REASONS = ("no_chunks", "below_rerank_threshold")


def load_queries(path: Path) -> dict[str, str]:
    """Load query texts keyed by original query ID."""
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            queries[record["_id"]] = record["text"]
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load positive relevance judgments as `{query_id: {doc_id: score}}`."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = score
    return qrels


def load_abstention_queries(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the out-of-domain and unanswerable evaluation queries.

    Raises:
        FileNotFoundError: If the specified query file does not exist.
        ValueError: If the query file is empty or missing required keys.
    """
    target_path = path or ABSTENTION_QUERIES_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Abstention query file not found at: {target_path}")

    records = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Abstention queries file at {target_path} must be a non-empty list.")

    for item in records:
        if not isinstance(item, dict) or "query_id" not in item or "question" not in item:
            raise ValueError(f"Invalid record in abstention queries: {item}")
    return records


def get_indexed_document_ids(client: QdrantClient, collection_name: str = QDRANT_COLLECTION_NAME) -> set[str]:
    """Collect all original document IDs present in the vector store."""
    doc_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=2000,
            offset=offset,
            with_payload=["original_document_ids", "parent_document_id"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_ids.update(payload.get("original_document_ids") or [payload["parent_document_id"]])
        if offset is None:
            return doc_ids


def select_eval_queries(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    num_queries: int | None,
    seed: int,
    indexed_doc_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a reproducible sample of judged queries, independent of system results.

    When `indexed_doc_ids` is given, queries without any relevant document in the
    index are skipped, since no retriever could find their evidence.

    Raises:
        ValueError: If no eligible queries remain or num_queries is not positive.
    """
    if num_queries is not None and num_queries < 1:
        raise ValueError("num_queries must be positive when specified.")

    candidates = sorted(qid for qid in qrels if qid in queries)
    if indexed_doc_ids is not None:
        judged = len(candidates)
        candidates = [qid for qid in candidates if indexed_doc_ids.intersection(qrels[qid])]
        if len(candidates) < judged:
            logger.warning("Skipped %d queries with no relevant document in the index", judged - len(candidates))
    if not candidates:
        raise ValueError("No judged queries have relevant documents in the index.")

    if num_queries is not None and num_queries < len(candidates):
        candidates = sorted(random.Random(seed).sample(candidates, num_queries))
    return [{"query_id": qid, "question": queries[qid], "gold": qrels[qid]} for qid in candidates]


def compute_ranking_metrics(ranked_doc_ids: list[str], gold: dict[str, int]) -> dict[str, float]:
    """Compute document-level ranking metrics for one query.

    Raises:
        ValueError: If the query has no positive relevance judgments.
    """
    relevant = {doc_id for doc_id, score in gold.items() if score > 0}
    if not relevant:
        raise ValueError("Query has no positive relevance judgments.")

    reciprocal_rank = next(
        (1.0 / rank for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1) if doc_id in relevant), 0.0
    )
    dcg = sum(gold.get(doc_id, 0) / math.log2(rank + 1) for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1))
    ideal_gains = sorted((score for score in gold.values() if score > 0), reverse=True)[:10]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return {
        "hit@5": float(bool(relevant.intersection(ranked_doc_ids[:5]))),
        "recall@5": len(relevant.intersection(ranked_doc_ids[:5])) / len(relevant),
        "recall@20": len(relevant.intersection(ranked_doc_ids[:20])) / len(relevant),
        "mrr@10": reciprocal_rank,
        "ndcg@10": dcg / idcg,
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average per-query metrics, returning an empty mapping for an empty group."""
    if not rows:
        return {}
    return {name: round(mean(row[name] for row in rows), 4) for name in METRIC_NAMES}


def _chunk_doc_ids(chunk: dict[str, Any]) -> set[str]:
    """Return all original document IDs represented by a chunk."""
    return set(chunk.get("original_document_ids") or [chunk["parent_document_id"]])


def _summarize_pipeline(records: list[dict[str, Any]], metrics_key: str, latency_key: str) -> dict[str, Any]:
    """Aggregate one pipeline's metrics overall and by number of relevant documents."""
    return {
        "overall": aggregate_metrics([r[metrics_key] for r in records]),
        "single_relevant": aggregate_metrics([r[metrics_key] for r in records if r["num_relevant"] == 1]),
        "multi_relevant": aggregate_metrics([r[metrics_key] for r in records if r["num_relevant"] > 1]),
        "mean_latency_ms": round(mean(r[latency_key] for r in records), 1),
    }


def evaluate_retrieval(
    eval_queries: list[dict[str, Any]],
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    retrieval_top_k: int = RETRIEVAL_TOP_K_CHUNKS,
    rerank_top_k: int = RERANK_TOP_K_CHUNKS,
    doc_top_k: int = EVAL_TOP_K_DOCUMENTS,
    collection_name: str = QDRANT_COLLECTION_NAME,
    show_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Compare dense retrieval alone with dense retrieval plus reranking.

    Both pipelines rank the same candidate chunks and use the same parent-document
    deduplication policy, so differences come from reranking alone.

    Returns:
        A summary, per-query records, and the top reranked context chunks per query ID.
    """
    records: list[dict[str, Any]] = []
    contexts: dict[str, list[dict[str, Any]]] = {}
    iterator = tqdm(eval_queries, desc="Evaluating retrieval", unit="query", disable=not show_progress)
    for position, query in enumerate(iterator, start=1):
        start = time.perf_counter()
        candidates = retrieve_chunks(
            query["question"], qdrant_client, embedding_model, top_k=retrieval_top_k, collection_name=collection_name
        )
        dense_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        reranked = rerank_chunks(query["question"], candidates, reranker, top_k=None)
        rerank_ms = (time.perf_counter() - start) * 1000

        dense_docs = list(deduplicate_to_parent_documents(candidates, "retrieval_score", doc_top_k))
        rerank_docs = list(deduplicate_to_parent_documents(reranked, "rerank_score", doc_top_k))
        records.append(
            {
                "query_id": query["query_id"],
                "num_relevant": len(query["gold"]),
                "dense_metrics": compute_ranking_metrics(dense_docs, query["gold"]),
                "rerank_metrics": compute_ranking_metrics(rerank_docs, query["gold"]),
                "dense_latency_ms": round(dense_ms, 1),
                "rerank_pipeline_latency_ms": round(dense_ms + rerank_ms, 1),
                "dense_top5_docs": dense_docs[:5],
                "rerank_top5_docs": rerank_docs[:5],
            }
        )
        contexts[query["query_id"]] = reranked[:rerank_top_k]
        if not show_progress and position % 50 == 0:
            logger.info("Evaluated retrieval for %d/%d queries", position, len(eval_queries))

    summary = {
        "num_queries": len(records),
        "num_single_relevant": sum(r["num_relevant"] == 1 for r in records),
        "num_multi_relevant": sum(r["num_relevant"] > 1 for r in records),
        "dense_only": _summarize_pipeline(records, "dense_metrics", "dense_latency_ms"),
        "dense_plus_rerank": _summarize_pipeline(records, "rerank_metrics", "rerank_pipeline_latency_ms"),
    }
    return summary, records, contexts


def _rate(values: list[bool]) -> float | None:
    """Return the fraction of true values, or None when there is nothing to measure."""
    return round(sum(values) / len(values), 4) if values else None


def _context_records(chunks: list[dict[str, Any]], gold: set[str]) -> list[dict[str, Any]]:
    """Describe the numbered context passages for manual review."""
    return [
        {
            "citation_id": index,
            "chunk_id": chunk["chunk_id"],
            "parent_document_id": chunk["parent_document_id"],
            "is_gold": bool(_chunk_doc_ids(chunk) & gold),
            "rerank_score": chunk.get("rerank_score"),
            "text": chunk["text"],
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def evaluate_generation(
    sample_queries: list[dict[str, Any]],
    contexts: dict[str, list[dict[str, Any]]],
    ollama_client: Client | None = None,
    model_name: str = GENERATION_MODEL_NAME,
    withholding_check: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate answers for sampled queries and compute automated grounding checks.

    Automated checks measure citation validity and use of judged evidence; they do
    not establish factual correctness, which requires the manual review fields.
    The optional withholding check supplies another query's context and records
    whether the model abstains; it is not a test of corpus-level unanswerability.

    Raises:
        RuntimeError: If Ollama is unavailable.
    """
    records: list[dict[str, Any]] = []
    iterator = tqdm(sample_queries, desc=f"Generating answers ({model_name})", unit="query", disable=not show_progress)
    for position, query in enumerate(iterator):
        gold = set(query["gold"])
        chunks = contexts[query["query_id"]]
        result = generate_answer(query["question"], chunks, model_name=model_name, client=ollama_client)
        cited_doc_ids = set().union(*(set(c["original_document_ids"]) for c in result["citations"]))
        record: dict[str, Any] = {
            "query_id": query["query_id"],
            "question": query["question"],
            "answer": result["answer"],
            "abstained": result["abstained"],
            "abstention_reason": result["abstention_reason"],
            "llm_called": result["abstention_reason"] not in PRE_LLM_ABSTENTION_REASONS,
            "cited_ids": [c["citation_id"] for c in result["citations"]],
            "invalid_citations": result["invalid_citations"],
            "gold_document_ids": sorted(gold),
            "gold_in_context": any(_chunk_doc_ids(c) & gold for c in chunks),
            "cites_gold": bool(cited_doc_ids & gold),
            "generation_seconds": result["generation_seconds"],
            "context": _context_records(chunks, gold),
            "manual_review": {
                "correctness_0_2": None,
                "support_0_2": None,
                "citation_support_0_2": None,
                "clarity_0_2": None,
                "notes": "",
            },
        }

        if withholding_check and len(sample_queries) > 1:
            other_id = sample_queries[(position + 1) % len(sample_queries)]["query_id"]
            mismatched = [c for c in contexts[other_id] if not _chunk_doc_ids(c) & gold]
            # Rerank scores belong to the other question, so bypass the score gate to test the model itself.
            withheld = generate_answer(
                query["question"], mismatched, model_name=model_name, client=ollama_client, min_rerank_score=None
            )
            record["withheld_context_answer"] = withheld["answer"]
            record["withheld_context_abstained"] = withheld["abstained"]

        records.append(record)
        if show_progress:
            iterator.set_postfix(last_sec=f"{result['generation_seconds']:.1f}s")
        else:
            logger.info(
                "Generated answer %d/%d (%.1fs)", position + 1, len(sample_queries), result["generation_seconds"]
            )

    if withholding_check and len(sample_queries) < 2:
        logger.warning("Withholding check needs at least two sampled queries; skipped")

    answered = [r for r in records if not r["abstained"]]
    llm_seconds = [r["generation_seconds"] for r in records if r["llm_called"]]
    summary = {
        "num_queries": len(records),
        "model": model_name,
        "abstention_rate": _rate([r["abstained"] for r in records]),
        "citation_validity_rate": _rate([not r["invalid_citations"] for r in records]),
        "citation_presence_rate": _rate([bool(r["cited_ids"]) for r in answered]),
        "gold_in_context_rate": _rate([r["gold_in_context"] for r in records]),
        "gold_citation_rate": _rate([r["cites_gold"] for r in answered if r["gold_in_context"]]),
        "withheld_context_abstention_rate": _rate(
            [r["withheld_context_abstained"] for r in records if "withheld_context_abstained" in r]
        ),
        "mean_generation_seconds": round(mean(llm_seconds), 2) if llm_seconds else None,
    }
    return summary, records


def format_retrieval_table(summary: dict[str, Any]) -> str:
    """Render the dense vs. reranked comparison as a Markdown table."""
    lines = [
        "| Pipeline | Hit@5 | Recall@5 | Recall@20 | nDCG@10 | MRR@10 | Latency (ms/query) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Dense only", "dense_only"), ("Dense + reranker", "dense_plus_rerank")):
        pipeline = summary[key]
        m = pipeline["overall"]
        lines.append(
            f"| {label} | {m['hit@5']:.3f} | {m['recall@5']:.3f} | {m['recall@20']:.3f} | "
            f"{m['ndcg@10']:.3f} | {m['mrr@10']:.3f} | {pipeline['mean_latency_ms']:.0f} |"
        )
    return "\n".join(lines)


def evaluate_abstention(
    queries: list[dict[str, Any]],
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    ollama_client: Client | None = None,
    model_name: str = GENERATION_MODEL_NAME,
    retrieval_top_k: int = RETRIEVAL_TOP_K_CHUNKS,
    rerank_top_k: int = RERANK_TOP_K_CHUNKS,
    min_rerank_score: float = MIN_RERANK_SCORE,
    collection_name: str = QDRANT_COLLECTION_NAME,
    show_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate pipeline abstention on out-of-domain and unanswerable queries.

    Checks both the pre-LLM reranker score gate and the generator's refusal
    behavior when evidence is absent or irrelevant.

    Returns:
        A tuple of (summary_dict, per_query_records).
    """
    records: list[dict[str, Any]] = []
    iterator = tqdm(queries, desc=f"Evaluating abstention ({model_name})", unit="query", disable=not show_progress)
    for query in iterator:
        start_time = time.perf_counter()
        candidates = retrieve_chunks(
            query["question"], qdrant_client, embedding_model, top_k=retrieval_top_k, collection_name=collection_name
        )
        reranked = rerank_chunks(query["question"], candidates, reranker, top_k=None)
        top_chunks = reranked[:rerank_top_k]
        max_score = max((float(c["rerank_score"]) for c in top_chunks if "rerank_score" in c), default=None)

        result = generate_answer(
            query["question"],
            top_chunks,
            model_name=model_name,
            client=ollama_client,
            min_rerank_score=min_rerank_score,
        )
        elapsed = time.perf_counter() - start_time

        pre_llm_gated = result["abstention_reason"] in PRE_LLM_ABSTENTION_REASONS
        llm_called = result["abstention_reason"] not in PRE_LLM_ABSTENTION_REASONS
        llm_abstained = result["abstention_reason"] == "model_abstained"
        abstained = result["abstained"]
        hallucinated = not abstained

        record: dict[str, Any] = {
            "query_id": query["query_id"],
            "category": query.get("category", "unspecified"),
            "question": query["question"],
            "notes": query.get("notes", ""),
            "max_rerank_score": round(max_score, 3) if max_score is not None else None,
            "abstained": abstained,
            "abstention_reason": result["abstention_reason"],
            "pre_llm_gated": pre_llm_gated,
            "llm_called": llm_called,
            "llm_abstained": llm_abstained,
            "hallucinated": hallucinated,
            "answer": result["answer"],
            "citations": [c["citation_id"] for c in result["citations"]],
            "invalid_citations": result["invalid_citations"],
            "generation_seconds": result["generation_seconds"],
            "total_latency_seconds": round(elapsed, 3),
            "top_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "parent_document_id": c["parent_document_id"],
                    "rerank_score": round(float(c["rerank_score"]), 3) if "rerank_score" in c else None,
                    "text": c["text"][:150] + ("..." if len(c["text"]) > 150 else ""),
                }
                for c in top_chunks
            ],
        }
        records.append(record)
        if show_progress:
            iterator.set_postfix(abstained=str(abstained), pre_gated=str(pre_llm_gated))

    categories = sorted({r["category"] for r in records})
    category_breakdown: dict[str, Any] = {}
    for cat in categories:
        cat_records = [r for r in records if r["category"] == cat]
        cat_llm_called = [r for r in cat_records if r["llm_called"]]
        category_breakdown[cat] = {
            "num_queries": len(cat_records),
            "pre_llm_abstention_count": sum(r["pre_llm_gated"] for r in cat_records),
            "llm_abstention_count": sum(r["llm_abstained"] for r in cat_records),
            "total_abstention_count": sum(r["abstained"] for r in cat_records),
            "hallucination_count": sum(r["hallucinated"] for r in cat_records),
            "total_abstention_rate": _rate([r["abstained"] for r in cat_records]),
            "pre_llm_abstention_rate": _rate([r["pre_llm_gated"] for r in cat_records]),
            "llm_abstention_rate": _rate([r["llm_abstained"] for r in cat_llm_called]),
            "hallucination_rate": _rate([r["hallucinated"] for r in cat_records]),
        }

    llm_called_records = [r for r in records if r["llm_called"]]
    summary = {
        "num_queries": len(records),
        "model": model_name,
        "min_rerank_score": min_rerank_score,
        "pre_llm_abstention_count": sum(r["pre_llm_gated"] for r in records),
        "llm_abstention_count": sum(r["llm_abstained"] for r in records),
        "total_abstention_count": sum(r["abstained"] for r in records),
        "hallucination_count": sum(r["hallucinated"] for r in records),
        "total_abstention_rate": _rate([r["abstained"] for r in records]),
        "pre_llm_abstention_rate": _rate([r["pre_llm_gated"] for r in records]),
        "llm_abstention_rate": _rate([r["llm_abstained"] for r in llm_called_records]),
        "hallucination_rate": _rate([r["hallucinated"] for r in records]),
        "mean_latency_seconds": round(mean(r["total_latency_seconds"] for r in records), 3) if records else None,
        "categories": category_breakdown,
    }
    return summary, records


def format_abstention_table(summary: dict[str, Any]) -> str:
    """Render the abstention evaluation breakdown as a Markdown table."""
    lines = [
        "| Category | Queries | Pre-LLM Gated | LLM Abstained | Total Abstained | Hallucinated | Abstention Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, stats in summary.get("categories", {}).items():
        rate_str = (
            f"{stats['total_abstention_rate'] * 100:.1f}%" if stats["total_abstention_rate"] is not None else "N/A"
        )
        lines.append(
            f"| {cat} | {stats['num_queries']} | {stats['pre_llm_abstention_count']} | "
            f"{stats['llm_abstention_count']} | {stats['total_abstention_count']} | "
            f"{stats['hallucination_count']} | {rate_str} |"
        )
    total_rate_str = (
        f"{summary['total_abstention_rate'] * 100:.1f}%" if summary.get("total_abstention_rate") is not None else "N/A"
    )
    lines.append(
        f"| **Overall** | **{summary['num_queries']}** | **{summary['pre_llm_abstention_count']}** | "
        f"**{summary['llm_abstention_count']}** | **{summary['total_abstention_count']}** | "
        f"**{summary['hallucination_count']}** | **{total_rate_str}** |"
    )
    return "\n".join(lines)
