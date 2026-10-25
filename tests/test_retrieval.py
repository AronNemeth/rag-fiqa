from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_fiqa.indexing import chunk_id_to_uuid
from rag_fiqa.retrieval import (
    deduplicate_to_parent_documents,
    normalize_question,
    open_vector_store,
    retrieve_chunks,
)


class FakeQueryEncoder:
    """Fake bi-encoder that maps query strings to deterministic 4-D vectors."""

    def encode(
        self,
        question: str,
        *,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "tax" in question.lower():
            vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return vec


def populate_sample_qdrant(qdrant_path: Path, collection_name: str = "test_chunks") -> None:
    client = QdrantClient(path=str(qdrant_path))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        chunks = [
            {
                "chunk_id": "c1",
                "parent_document_id": "doc_1",
                "original_document_ids": ["doc_1"],
                "chunk_index": 0,
                "title": "",
                "text": "General investment advice.",
                "token_count": 10,
                "start_char": 0,
                "end_char": 26,
                "vec": [0.9, 0.1, 0.0, 0.0],
            },
            {
                "chunk_id": "c2",
                "parent_document_id": "doc_2",
                "original_document_ids": ["doc_2", "doc_2_dup"],
                "chunk_index": 0,
                "title": "",
                "text": "Capital gains tax rates part 1.",
                "token_count": 10,
                "start_char": 0,
                "end_char": 31,
                "vec": [0.1, 0.95, 0.0, 0.0],
            },
            {
                "chunk_id": "c3",
                "parent_document_id": "doc_2",
                "original_document_ids": ["doc_2", "doc_2_dup"],
                "chunk_index": 1,
                "title": "",
                "text": "Capital gains tax rates part 2.",
                "token_count": 10,
                "start_char": 25,
                "end_char": 56,
                "vec": [0.2, 0.8, 0.0, 0.0],
            },
        ]
        points = [
            PointStruct(
                id=chunk_id_to_uuid(c["chunk_id"]),
                vector=c["vec"],
                payload={k: v for k, v in c.items() if k != "vec"},
            )
            for c in chunks
        ]
        client.upsert(collection_name=collection_name, points=points)
    finally:
        client.close()


def test_normalize_question_strips_and_rejects_empty() -> None:
    assert normalize_question("  What is an ETF? \n") == "What is an ETF?"
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_question("   \t\n ")


def test_open_vector_store_errors_on_missing_or_empty(tmp_path: Path) -> None:
    missing_path = tmp_path / "nonexistent"
    with pytest.raises(RuntimeError, match="not found"):
        open_vector_store(missing_path, "test_chunks")

    empty_path = tmp_path / "empty_qdrant"
    client = QdrantClient(path=str(empty_path))
    client.create_collection("empty_col", vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.close()

    with pytest.raises(RuntimeError, match="not found"):
        open_vector_store(empty_path, "missing_col")

    with pytest.raises(RuntimeError, match="is empty"):
        open_vector_store(empty_path, "empty_col")


def test_retrieve_chunks_orders_by_similarity(tmp_path: Path) -> None:
    qdrant_path = tmp_path / "qdrant"
    populate_sample_qdrant(qdrant_path)

    client = open_vector_store(qdrant_path, "test_chunks")
    encoder: Any = FakeQueryEncoder()
    try:
        results = retrieve_chunks(
            question="How does capital gains tax work?",
            client=client,
            embedding_model=encoder,
            top_k=2,
            collection_name="test_chunks",
        )
    finally:
        client.close()

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["retrieval_rank"] == 1
    assert results[1]["chunk_id"] == "c3"
    assert results[1]["retrieval_rank"] == 2
    assert results[0]["retrieval_score"] > results[1]["retrieval_score"]


def test_deduplicate_to_parent_documents_takes_max_score_and_expands_aliases() -> None:
    chunks = [
        {
            "chunk_id": "c3",
            "parent_document_id": "doc_2",
            "original_document_ids": ["doc_2", "doc_2_dup"],
            "retrieval_score": 0.75,
        },
        {
            "chunk_id": "c1",
            "parent_document_id": "doc_1",
            "original_document_ids": ["doc_1"],
            "retrieval_score": 0.82,
        },
        {
            "chunk_id": "c2",
            "parent_document_id": "doc_2",
            "original_document_ids": ["doc_2", "doc_2_dup"],
            "retrieval_score": 0.95,
        },
    ]
    doc_scores = deduplicate_to_parent_documents(chunks, score_key="retrieval_score", top_k=10)
    assert list(doc_scores.keys()) == ["doc_2", "doc_2_dup", "doc_1"]
    assert doc_scores["doc_2"] == 0.95
    assert doc_scores["doc_2_dup"] == 0.95
    assert doc_scores["doc_1"] == 0.82

    top_one = deduplicate_to_parent_documents(chunks, score_key="retrieval_score", top_k=1)
    assert len(top_one) == 1
    assert top_one["doc_2"] == 0.95
