"""FastAPI application exposing the financial Q&A RAG pipeline."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from ollama import Client
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_fiqa.config import (
    API_TITLE,
    API_VERSION,
    DATASET_NAME,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    GENERATION_MODEL_NAME,
    MIN_RERANK_SCORE,
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
    RERANK_TOP_K_CHUNKS,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    RETRIEVAL_TOP_K_CHUNKS,
)
from rag_fiqa.generation import create_ollama_client, generate_answer
from rag_fiqa.indexing import load_embedding_model
from rag_fiqa.reranking import load_reranker_model, rerank_chunks
from rag_fiqa.retrieval import normalize_question, open_vector_store, retrieve_chunks
from rag_fiqa.utils import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------


class AskRequest(BaseModel):
    """User query request for the financial Q&A RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=1000, description="Financial question to answer.")
    retrieval_top_k: int | None = Field(
        default=None, ge=1, le=100, description="Optional override for dense retrieval candidate count."
    )
    rerank_top_k: int | None = Field(
        default=None, ge=1, le=20, description="Optional override for reranked evidence count passed to generator."
    )
    min_rerank_score: float | None = Field(
        default=None, description="Optional override for minimum reranking relevance threshold."
    )
    model_name: str | None = Field(default=None, description="Optional override for Ollama generation model.")


class CitationEvidence(BaseModel):
    """Retrieved evidence passage referenced in the answer."""

    citation_id: int = Field(..., description="1-based citation index corresponding to inline [N] tags.")
    chunk_id: str = Field(..., description="Unique identifier for the chunk.")
    parent_document_id: str = Field(..., description="Original FiQA parent document ID.")
    title: str = Field(default="", description="Parent document title if available.")
    text: str = Field(..., description="Passage text.")
    rerank_score: float | None = Field(default=None, description="Cross-encoder relevance score.")
    retrieval_score: float | None = Field(default=None, description="First-stage bi-encoder cosine similarity.")


class StageLatencies(BaseModel):
    """Latency breakdown for each pipeline stage in milliseconds."""

    retrieval_ms: float = Field(..., description="Dense retrieval time in ms.")
    rerank_ms: float = Field(..., description="Cross-encoder reranking time in ms.")
    generation_ms: float = Field(..., description="LLM answer generation time in ms.")
    total_ms: float = Field(..., description="Total end-to-end processing time in ms.")


class AskResponse(BaseModel):
    """Full pipeline answer and grounding metadata."""

    question: str = Field(..., description="Normalized question string.")
    answer: str = Field(..., description="Generated answer or insufficient-evidence message.")
    abstained: bool = Field(..., description="True if the pipeline safely refrained from answering.")
    abstention_reason: str | None = Field(
        default=None, description="Reason for abstention (e.g. below_rerank_threshold, model_abstained)."
    )
    citations: list[CitationEvidence] = Field(
        default_factory=list, description="Validated evidence passages cited inline in the answer."
    )
    invalid_citations: list[int] = Field(
        default_factory=list, description="Out-of-range citation IDs emitted by the generator."
    )
    context_chunks: list[CitationEvidence] = Field(
        default_factory=list, description="Top reranked evidence passages provided in prompt context."
    )
    model: str = Field(..., description="Name of the generation model used.")
    timings: StageLatencies = Field(..., description="Latency breakdown in milliseconds.")


class HealthResponse(BaseModel):
    """Health and readiness status of the API and its underlying services."""

    status: str = Field(..., description="'ok' when fully operational, 'degraded' when dependencies are unhealthy.")
    vector_store: dict[str, Any] = Field(..., description="Qdrant connection and index stats.")
    ollama: dict[str, Any] = Field(..., description="Ollama server connectivity and active model status.")
    version: str = Field(..., description="API software version.")


class InfoResponse(BaseModel):
    """Static pipeline configuration and model metadata."""

    version: str
    dataset: str
    embedding_model: str
    reranker_model: str
    default_generation_model: str
    retrieval_top_k: int
    rerank_top_k: int
    min_rerank_score: float
    index_path: str


# -----------------------------------------------------------------------------
# Application Factory & Lifespan
# -----------------------------------------------------------------------------


def create_app(
    qdrant_client: QdrantClient | None = None,
    embedding_model: SentenceTransformer | None = None,
    reranker: CrossEncoder | None = None,
    ollama_client: Client | None = None,
    qdrant_path: Path = QDRANT_PATH,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Supports dependency injection of clients and models for offline testing.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Initialize pipeline resources on startup and clean up on shutdown."""
        owns_client = qdrant_client is None
        client = qdrant_client if qdrant_client is not None else open_vector_store(qdrant_path, collection_name)
        embed = embedding_model if embedding_model is not None else load_embedding_model()
        rank = reranker if reranker is not None else load_reranker_model()
        ollama = ollama_client if ollama_client is not None else create_ollama_client()

        app.state.qdrant_client = client
        app.state.embedding_model = embed
        app.state.reranker = rank
        app.state.ollama_client = ollama
        app.state.collection_name = collection_name
        app.state.vector_store_path = qdrant_path

        logger.info("FastAPI pipeline initialized with collection '%s'", collection_name)
        yield
        if owns_client:
            client.close()
            logger.info("Closed vector store connection")

    app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health_endpoint(request: Request, response: Response) -> HealthResponse:
        """Check availability of Qdrant vector store and Ollama service."""
        is_healthy = True
        vector_status: dict[str, Any] = {}
        try:
            collection = request.app.state.qdrant_client.get_collection(request.app.state.collection_name)
            vector_status = {
                "status": "connected",
                "collection": request.app.state.collection_name,
                "points_count": collection.points_count,
            }
        except Exception as exc:
            is_healthy = False
            vector_status = {"status": "error", "error": str(exc)}

        ollama_status: dict[str, Any] = {}
        try:
            available_models = request.app.state.ollama_client.list()
            raw_models = getattr(available_models, "models", None) or (
                available_models.get("models", []) if isinstance(available_models, dict) else []
            )
            model_names: list[str] = []
            for item in raw_models:
                name = getattr(item, "model", None) or (
                    item.get("name") or item.get("model") if isinstance(item, dict) else str(item)
                )
                if name:
                    model_names.append(str(name))

            has_model = any(GENERATION_MODEL_NAME in name for name in model_names)
            ollama_status = {
                "status": "connected",
                "target_model": GENERATION_MODEL_NAME,
                "model_available": has_model,
            }
            if not has_model:
                is_healthy = False
        except Exception as exc:
            is_healthy = False
            ollama_status = {"status": "error", "error": str(exc)}

        if not is_healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return HealthResponse(
            status="ok" if is_healthy else "degraded",
            vector_store=vector_status,
            ollama=ollama_status,
            version=API_VERSION,
        )

    @app.get("/info", response_model=InfoResponse)
    def info_endpoint(request: Request) -> InfoResponse:
        """Return static model and configuration metadata."""
        return InfoResponse(
            version=API_VERSION,
            dataset=DATASET_NAME,
            embedding_model=f"{EMBEDDING_MODEL_ID}@{EMBEDDING_MODEL_REVISION}",
            reranker_model=f"{RERANKER_MODEL_ID}@{RERANKER_MODEL_REVISION}",
            default_generation_model=GENERATION_MODEL_NAME,
            retrieval_top_k=RETRIEVAL_TOP_K_CHUNKS,
            rerank_top_k=RERANK_TOP_K_CHUNKS,
            min_rerank_score=MIN_RERANK_SCORE,
            index_path=str(request.app.state.vector_store_path),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(payload: AskRequest, request: Request) -> AskResponse:
        """Execute retrieve-rerank-generate pipeline and return grounded answer."""
        try:
            normalized = normalize_question(payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

        retrieval_k = payload.retrieval_top_k or RETRIEVAL_TOP_K_CHUNKS
        rerank_k = payload.rerank_top_k or RERANK_TOP_K_CHUNKS
        min_score = payload.min_rerank_score if payload.min_rerank_score is not None else MIN_RERANK_SCORE
        model_name = payload.model_name or GENERATION_MODEL_NAME

        t0 = time.perf_counter()
        try:
            candidates = retrieve_chunks(
                normalized,
                request.app.state.qdrant_client,
                request.app.state.embedding_model,
                top_k=retrieval_k,
                collection_name=request.app.state.collection_name,
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Vector retrieval failed."
            ) from exc
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        try:
            reranked = rerank_chunks(normalized, candidates, request.app.state.reranker, top_k=None)
        except Exception as exc:
            logger.exception("Reranking failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cross-encoder reranking failed."
            ) from exc
        rerank_ms = (time.perf_counter() - t1) * 1000

        top_chunks = reranked[:rerank_k]

        t2 = time.perf_counter()
        try:
            gen_result = generate_answer(
                normalized,
                top_chunks,
                model_name=model_name,
                client=request.app.state.ollama_client,
                min_rerank_score=min_score,
            )
        except RuntimeError as exc:
            logger.exception("Ollama generation failed")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected generation failure")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Generation failed.") from exc
        generation_ms = (time.perf_counter() - t2) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        # Build context records with scores
        context_evidences: list[CitationEvidence] = []
        for idx, chunk in enumerate(top_chunks, start=1):
            context_evidences.append(
                CitationEvidence(
                    citation_id=idx,
                    chunk_id=chunk["chunk_id"],
                    parent_document_id=chunk["parent_document_id"],
                    title=chunk.get("title", ""),
                    text=chunk["text"],
                    rerank_score=chunk.get("rerank_score"),
                    retrieval_score=chunk.get("retrieval_score"),
                )
            )

        # Build citation records matching cited IDs
        citations: list[CitationEvidence] = []
        for c in gen_result["citations"]:
            cid = c["citation_id"]
            if 1 <= cid <= len(top_chunks):
                source_chunk = top_chunks[cid - 1]
                citations.append(
                    CitationEvidence(
                        citation_id=cid,
                        chunk_id=c["chunk_id"],
                        parent_document_id=c["parent_document_id"],
                        title=c.get("title", ""),
                        text=c["text"],
                        rerank_score=source_chunk.get("rerank_score"),
                        retrieval_score=source_chunk.get("retrieval_score"),
                    )
                )

        return AskResponse(
            question=normalized,
            answer=gen_result["answer"],
            abstained=gen_result["abstained"],
            abstention_reason=gen_result["abstention_reason"],
            citations=citations,
            invalid_citations=gen_result["invalid_citations"],
            context_chunks=context_evidences,
            model=gen_result["model"],
            timings=StageLatencies(
                retrieval_ms=round(retrieval_ms, 1),
                rerank_ms=round(rerank_ms, 1),
                generation_ms=round(generation_ms, 1),
                total_ms=round(total_ms, 1),
            ),
        )

    return app


# Default app instance for ASGI servers (e.g. uvicorn rag_fiqa.api:app)
app = create_app()
