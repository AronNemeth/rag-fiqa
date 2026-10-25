"""Rerank retrieved candidate chunks using a Cross-Encoder model."""

from typing import Any

from sentence_transformers import CrossEncoder

from rag_fiqa.config import (
    RERANK_TOP_K_CHUNKS,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
)
from rag_fiqa.indexing import embedding_text
from rag_fiqa.retrieval import normalize_question


def load_reranker_model(
    model_id: str = RERANKER_MODEL_ID,
    revision: str = RERANKER_MODEL_REVISION,
    *,
    local_files_only: bool = True,
) -> CrossEncoder:
    """Load the pinned CrossEncoder reranker model from the local cache."""
    return CrossEncoder(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
    top_k: int | None = RERANK_TOP_K_CHUNKS,
) -> list[dict[str, Any]]:
    """Score and sort candidate chunks jointly with the question using a Cross-Encoder.

    Preserves existing retrieval metadata (such as `retrieval_score` and
    `retrieval_rank`) and attaches `rerank_score` and `rerank_rank`.

    Raises:
        ValueError: If the question is empty.
    """
    normalized_question = normalize_question(question)
    if not chunks:
        return []

    pairs = [(normalized_question, embedding_text(chunk.get("title", ""), chunk["text"])) for chunk in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored_chunks = [
        {
            **chunk,
            "rerank_score": float(score),
        }
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    scored_chunks.sort(key=lambda item: item["rerank_score"], reverse=True)

    if top_k is not None:
        scored_chunks = scored_chunks[:top_k]

    return [
        {
            **chunk,
            "rerank_rank": rank,
        }
        for rank, chunk in enumerate(scored_chunks, start=1)
    ]
