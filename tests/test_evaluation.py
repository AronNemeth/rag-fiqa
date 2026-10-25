import math
from typing import Any

import pytest

from rag_fiqa.evaluation import (
    aggregate_metrics,
    compute_ranking_metrics,
    evaluate_generation,
    select_eval_queries,
)


def test_ranking_metrics_match_plan_example() -> None:
    # Four relevant documents, two of them in the top five (example from the implementation plan).
    gold = {"a": 1, "b": 1, "c": 1, "d": 1}
    ranked = ["a", "x", "b", "y", "z", "c"]
    metrics = compute_ranking_metrics(ranked, gold)

    assert metrics["hit@5"] == 1.0
    assert metrics["recall@5"] == 0.5
    assert metrics["recall@20"] == 0.75
    assert metrics["mrr@10"] == 1.0
    dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(7)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, 5))
    assert metrics["ndcg@10"] == pytest.approx(dcg / idcg)


def test_ranking_metrics_without_relevant_results() -> None:
    metrics = compute_ranking_metrics(["x", "y"], {"a": 1})
    assert all(value == 0.0 for value in metrics.values())
    with pytest.raises(ValueError, match="no positive relevance"):
        compute_ranking_metrics(["x"], {})


def test_aggregate_metrics_averages_and_handles_empty_groups() -> None:
    rows = [compute_ranking_metrics(["a"], {"a": 1}), compute_ranking_metrics(["x"], {"a": 1})]
    assert aggregate_metrics(rows)["hit@5"] == 0.5
    assert aggregate_metrics([]) == {}


def test_select_eval_queries_is_reproducible_and_filters_unindexed() -> None:
    queries = {str(i): f"question {i}" for i in range(10)}
    qrels = {str(i): {f"doc_{i}": 1} for i in range(10)}
    indexed = {f"doc_{i}" for i in range(8)}

    first = select_eval_queries(queries, qrels, num_queries=4, seed=1, indexed_doc_ids=indexed)
    assert first == select_eval_queries(queries, qrels, num_queries=4, seed=1, indexed_doc_ids=indexed)
    assert len(first) == 4
    assert all(q["query_id"] not in {"8", "9"} for q in first)
    assert len(select_eval_queries(queries, qrels, num_queries=None, seed=1, indexed_doc_ids=indexed)) == 8

    with pytest.raises(ValueError, match="No judged queries"):
        select_eval_queries(queries, qrels, num_queries=None, seed=1, indexed_doc_ids=set())


class KeywordOllamaClient:
    """Answer with a citation only if the question's first word appears in the evidence."""

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence, question = messages[1]["content"].split("Question: ")
        if question.split()[0].lower() in evidence.lower():
            return {"message": {"content": "Grounded answer [1] [7]."}}
        return {"message": {"content": "Insufficient evidence: the passages do not cover this."}}


def chunk(chunk_id: str, doc_id: str, text: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "parent_document_id": doc_id,
        "original_document_ids": [doc_id],
        "title": "",
        "text": text,
        "rerank_score": 2.0,
    }


def test_evaluate_generation_computes_grounding_checks_and_withholding() -> None:
    sample = [
        {"query_id": "q1", "question": "alpha question?", "gold": {"d1": 1}},
        {"query_id": "q2", "question": "beta question?", "gold": {"d9": 1}},
    ]
    contexts = {
        "q1": [chunk("c1", "d1", "alpha facts")],
        "q2": [chunk("c2", "d2", "beta facts")],
    }
    client: Any = KeywordOllamaClient()
    summary, records = evaluate_generation(sample, contexts, ollama_client=client, withholding_check=True)

    assert summary["abstention_rate"] == 0.0
    assert summary["citation_validity_rate"] == 0.0  # every answer also cites the nonexistent [7]
    assert summary["citation_presence_rate"] == 1.0
    assert summary["gold_in_context_rate"] == 0.5
    assert summary["gold_citation_rate"] == 1.0  # only q1 had gold evidence in context, and it was cited
    assert summary["withheld_context_abstention_rate"] == 1.0
    assert records[0]["context"][0]["is_gold"] is True
    assert records[0]["manual_review"]["correctness_0_2"] is None
