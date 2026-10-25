from typing import Any

import numpy as np
import pytest

from rag_fiqa.reranking import rerank_chunks
from rag_fiqa.retrieval import deduplicate_to_parent_documents


class FakeCrossEncoder:
    """Deterministic fake CrossEncoder that scores passages containing 'exact answer' highest."""

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        scores: list[float] = []
        for question, passage in pairs:
            if "exact answer" in passage.lower():
                scores.append(4.5)
            elif "partial" in passage.lower():
                scores.append(1.2)
            else:
                scores.append(-2.0 + 0.01 * len(question))
        return np.array(scores, dtype=np.float32)


def sample_candidates() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "c1",
            "parent_document_id": "doc_1",
            "original_document_ids": ["doc_1"],
            "title": "",
            "text": "General background information.",
            "retrieval_score": 0.91,
            "retrieval_rank": 1,
        },
        {
            "chunk_id": "c2",
            "parent_document_id": "doc_2",
            "original_document_ids": ["doc_2", "doc_2_alias"],
            "title": "Tax Guide",
            "text": "Partial details about tax deductions.",
            "retrieval_score": 0.85,
            "retrieval_rank": 2,
        },
        {
            "chunk_id": "c3",
            "parent_document_id": "doc_3",
            "original_document_ids": ["doc_3"],
            "title": "",
            "text": "Here is the exact answer to your financial question.",
            "retrieval_score": 0.72,
            "retrieval_rank": 3,
        },
    ]


def test_rerank_chunks_reorders_and_preserves_retrieval_metadata() -> None:
    reranker: Any = FakeCrossEncoder()
    candidates = sample_candidates()

    reranked = rerank_chunks(
        question="What is the tax rule?",
        chunks=candidates,
        reranker=reranker,
        top_k=2,
    )

    assert len(reranked) == 2
    assert [c["chunk_id"] for c in reranked] == ["c3", "c2"]
    assert [c["rerank_rank"] for c in reranked] == [1, 2]
    assert reranked[0]["rerank_score"] == pytest.approx(4.5)
    assert reranked[0]["retrieval_rank"] == 3
    assert reranked[0]["retrieval_score"] == pytest.approx(0.72)


def test_rerank_chunks_integrates_with_parent_deduplication() -> None:
    reranker: Any = FakeCrossEncoder()
    candidates = sample_candidates()

    reranked = rerank_chunks(
        question="What is the tax rule?",
        chunks=candidates,
        reranker=reranker,
        top_k=None,
    )
    parent_scores = deduplicate_to_parent_documents(reranked, score_key="rerank_score", top_k=10)

    assert list(parent_scores.keys()) == ["doc_3", "doc_2", "doc_2_alias", "doc_1"]
    assert parent_scores["doc_3"] == pytest.approx(4.5)
    assert parent_scores["doc_2_alias"] == pytest.approx(1.2)


def test_rerank_chunks_empty_and_validation() -> None:
    reranker: Any = FakeCrossEncoder()
    assert rerank_chunks("Valid question?", [], reranker=reranker) == []

    with pytest.raises(ValueError, match="must not be empty"):
        rerank_chunks("   ", sample_candidates(), reranker=reranker)

    with pytest.raises(ValueError, match="top_k must be positive"):
        rerank_chunks("Valid question?", sample_candidates(), reranker=reranker, top_k=0)
