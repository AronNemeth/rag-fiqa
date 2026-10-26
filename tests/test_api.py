from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_fiqa.api import create_app
from rag_fiqa.config import GENERATION_MODEL_NAME
from rag_fiqa.indexing import chunk_id_to_uuid


class FakeQueryEncoder:
    """Deterministic fake bi-encoder."""

    def encode(
        self,
        question: str,
        *,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeReranker:
    """Deterministic fake cross-encoder."""

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        scores = []
        for question, _ in pairs:
            if "low_score" in question:
                scores.append(-10.0)
            else:
                scores.append(3.5)
        return np.array(scores, dtype=np.float32)


class FakeOllamaClient:
    """Fake Ollama client for testing chat and list endpoints."""

    def __init__(self, fail_generation: bool = False, model_missing: bool = False) -> None:
        self.fail_generation = fail_generation
        self.model_missing = model_missing

    def list(self) -> dict[str, Any]:
        if self.model_missing:
            return {"models": [{"name": "other-model:latest"}]}
        return {"models": [{"name": GENERATION_MODEL_NAME}]}

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.fail_generation:
            raise RuntimeError("Ollama connection timed out.")
        return {"message": {"content": "Roth IRA contributions are not tax-deductible [1]."}}


@pytest.fixture
def test_client(tmp_path: Path) -> TestClient:
    """Build a TestClient connected to an in-memory vector store and fake models."""
    collection_name = "api_test_collection"
    qdrant = QdrantClient(path=str(tmp_path / "qdrant"))
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    chunks = [
        {
            "chunk_id": "c1",
            "parent_document_id": "doc_101",
            "original_document_ids": ["doc_101"],
            "title": "IRA Guidelines",
            "text": "Roth IRA contributions are made with after-tax money.",
            "vec": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "chunk_id": "c2",
            "parent_document_id": "doc_102",
            "original_document_ids": ["doc_102"],
            "title": "401k Plans",
            "text": "Traditional 401k contributions reduce taxable income.",
            "vec": [0.9, 0.1, 0.0, 0.0],
        },
    ]
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=chunk_id_to_uuid(c["chunk_id"]),
                vector=c["vec"],
                payload={k: v for k, v in c.items() if k != "vec"},
            )
            for c in chunks
        ],
    )

    fake_encoder: Any = FakeQueryEncoder()
    fake_reranker: Any = FakeReranker()
    fake_ollama: Any = FakeOllamaClient()

    app = create_app(
        qdrant_client=qdrant,
        embedding_model=fake_encoder,
        reranker=fake_reranker,
        ollama_client=fake_ollama,
        qdrant_path=tmp_path / "qdrant",
        collection_name=collection_name,
    )

    with TestClient(app) as client:
        yield client


def test_health_endpoint_healthy(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vector_store"]["points_count"] == 2
    assert data["ollama"]["model_available"] is True


def test_health_endpoint_degraded_when_model_missing(tmp_path: Path) -> None:
    collection_name = "test_degraded"
    qdrant = QdrantClient(path=str(tmp_path / "qdrant"))
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    fake_encoder: Any = FakeQueryEncoder()
    fake_reranker: Any = FakeReranker()
    fake_ollama: Any = FakeOllamaClient(model_missing=True)

    app = create_app(
        qdrant_client=qdrant,
        embedding_model=fake_encoder,
        reranker=fake_reranker,
        ollama_client=fake_ollama,
        qdrant_path=tmp_path / "qdrant",
        collection_name=collection_name,
    )

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama"]["model_available"] is False


def test_info_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "fiqa"
    assert "MiniLM" in data["embedding_model"]
    assert "MiniLM" in data["reranker_model"]
    assert data["retrieval_top_k"] == 30


def test_ask_endpoint_returns_grounded_answer(test_client: TestClient) -> None:
    response = test_client.post("/ask", json={"question": "What are Roth IRA rules?"})
    assert response.status_code == 200
    data = response.json()

    assert data["abstained"] is False
    assert data["abstention_reason"] is None
    assert "Roth IRA" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation_id"] == 1
    assert data["citations"][0]["parent_document_id"] == "doc_101"
    assert data["citations"][0]["rerank_score"] == pytest.approx(3.5)
    assert len(data["context_chunks"]) == 2
    assert data["timings"]["total_ms"] > 0
    assert data["timings"]["retrieval_ms"] >= 0


def test_ask_endpoint_abstains_on_low_rerank_score(test_client: TestClient) -> None:
    response = test_client.post("/ask", json={"question": "low_score out of domain sourdough?"})
    assert response.status_code == 200
    data = response.json()

    assert data["abstained"] is True
    assert data["abstention_reason"] == "below_rerank_threshold"
    assert data["answer"].startswith("Insufficient evidence:")
    assert data["citations"] == []


def test_ask_endpoint_rejects_empty_question(test_client: TestClient) -> None:
    response = test_client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_endpoint_handles_ollama_runtime_error(tmp_path: Path) -> None:
    collection_name = "test_err"
    qdrant = QdrantClient(path=str(tmp_path / "qdrant"))
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    chunks = [
        {
            "chunk_id": "c1",
            "parent_document_id": "doc_1",
            "original_document_ids": ["doc_1"],
            "title": "",
            "text": "Valid text passage.",
            "vec": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=chunk_id_to_uuid(c["chunk_id"]),
                vector=c["vec"],
                payload={k: v for k, v in c.items() if k != "vec"},
            )
            for c in chunks
        ],
    )
    fake_encoder: Any = FakeQueryEncoder()
    fake_reranker: Any = FakeReranker()
    fake_ollama: Any = FakeOllamaClient(fail_generation=True)

    app = create_app(
        qdrant_client=qdrant,
        embedding_model=fake_encoder,
        reranker=fake_reranker,
        ollama_client=fake_ollama,
        qdrant_path=tmp_path / "qdrant",
        collection_name=collection_name,
    )

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Valid financial question?"})
        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]
