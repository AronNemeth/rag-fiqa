"""Retrieve candidate chunks from Qdrant and aggregate scores to parent documents."""

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_fiqa.config import (
    EVAL_TOP_K_DOCUMENTS,
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
    RETRIEVAL_TOP_K_CHUNKS,
)
from rag_fiqa.utils import get_logger

logger = get_logger(__name__)


def normalize_question(question: str) -> str:
    """Strip surrounding whitespace and reject an empty question.

    Raises:
        ValueError: If the question contains only whitespace.
    """
    normalized = question.strip()
    if not normalized:
        raise ValueError("Question must not be empty.")
    return normalized


def open_vector_store(
    qdrant_path: Path = QDRANT_PATH,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> QdrantClient:
    """Open the persistent local Qdrant store and verify the collection is populated.

    Raises:
        RuntimeError: If the index directory or collection is missing or empty.
    """
    if not qdrant_path.exists():
        raise RuntimeError(f"Vector store directory not found at {qdrant_path}. Run `scripts/build_index.py` first.")

    client = QdrantClient(path=str(qdrant_path))
    try:
        if not client.collection_exists(collection_name):
            client.close()
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' not found at {qdrant_path}. Run `scripts/build_index.py` first."
            )
        count = client.count(collection_name=collection_name).count
        if count == 0:
            client.close()
            raise RuntimeError(f"Qdrant collection '{collection_name}' at {qdrant_path} is empty.")
    except Exception:
        client.close()
        raise

    logger.info("Opened Qdrant collection '%s' (%d chunks) at %s", collection_name, count, qdrant_path)
    return client


def retrieve_chunks(
    question: str,
    client: QdrantClient,
    embedding_model: SentenceTransformer,
    top_k: int = RETRIEVAL_TOP_K_CHUNKS,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """Embed a question and retrieve the top-k candidate chunks from Qdrant.

    Raises:
        ValueError: If the question is empty.
    """
    normalized_question = normalize_question(question)
    query_vector = embedding_model.encode(
        normalized_question,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    )

    candidates: list[dict[str, Any]] = []
    for rank, scored_point in enumerate(response.points, start=1):
        payload = dict(scored_point.payload or {})
        candidates.append(
            {
                **payload,
                "retrieval_score": float(scored_point.score),
                "retrieval_rank": rank,
            }
        )
    return candidates


def deduplicate_to_parent_documents(
    chunks: list[dict[str, Any]],
    score_key: str = "retrieval_score",
    top_k: int | None = EVAL_TOP_K_DOCUMENTS,
) -> dict[str, float]:
    """Aggregate chunk scores to parent document IDs using maximum chunk score.

    Expands `original_document_ids` so exact-duplicate documents in the original
    corpus receive their shared chunk's score for BEIR qrels evaluation.

    Raises:
        ValueError: If a chunk lacks score_key.
    """
    doc_scores: dict[str, float] = {}
    for chunk in chunks:
        if score_key not in chunk:
            raise ValueError(f"Chunk is missing required score field '{score_key}'.")
        score = float(chunk[score_key])
        parent_id = chunk["parent_document_id"]
        doc_ids: list[str] = chunk.get("original_document_ids") or [parent_id]
        for doc_id in doc_ids:
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

    sorted_items = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        sorted_items = sorted_items[:top_k]
    return dict(sorted_items)
