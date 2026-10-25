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

from rag_fiqa.config import (
    EVAL_TOP_K_DOCUMENTS,
    GENERATION_MODEL_NAME,
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
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Compare dense retrieval alone with dense retrieval plus reranking.

    Both pipelines rank the same candidate chunks and use the same parent-document
    deduplication policy, so differences come from reranking alone.

    Returns:
        A summary, per-query records, and the top reranked context chunks per query ID.
    """
    records: list[dict[str, Any]] = []
    contexts: dict[str, list[dict[str, Any]]] = {}
    for position, query in enumerate(eval_queries, start=1):
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
        if position % 50 == 0:
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
    for position, query in enumerate(sample_queries):
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
        logger.info("Generated answer %d/%d (%.1fs)", position + 1, len(sample_queries), result["generation_seconds"])

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
