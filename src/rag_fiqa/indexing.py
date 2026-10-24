"""Chunk documents, compute embeddings, and populate the local Qdrant vector store."""

import csv
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from rag_fiqa.chunking import chunk_document, load_embedding_tokenizer
from rag_fiqa.config import (
    CHUNK_CORPUS_PATH,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DATASET_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    INDEX_MANIFEST_PATH,
    PREPARED_DATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
    SUBSET_RANDOM_SEED,
)
from rag_fiqa.data import file_checksum
from rag_fiqa.utils import get_logger

logger = get_logger(__name__)


def load_embedding_model(
    model_id: str = EMBEDDING_MODEL_ID,
    revision: str = EMBEDDING_MODEL_REVISION,
    *,
    local_files_only: bool = True,
) -> SentenceTransformer:
    """Load the pinned SentenceTransformer embedding model from the local cache."""
    return SentenceTransformer(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )


def embedding_text(title: str, text: str) -> str:
    """Build the complete text input for embedding, including an optional title."""
    return f"{title}\n\n{text}" if title else text


def load_relevant_document_ids(
    qrels_dir: Path = PREPARED_DATA_DIR / "qrels",
    splits: tuple[str, ...] = ("dev", "test"),
) -> set[str]:
    """Return all document IDs with positive relevance in the specified qrels splits."""
    relevant_ids: set[str] = set()
    for split in splits:
        tsv_path = qrels_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                if int(row["score"]) > 0:
                    relevant_ids.add(row["corpus-id"])
    return relevant_ids


def select_corpus_documents(
    corpus_path: Path = CHUNK_CORPUS_PATH,
    subset_size: int | None = None,
    seed: int = SUBSET_RANDOM_SEED,
    relevant_doc_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load corpus documents, optionally selecting a reproducible subset.

    When `subset_size` is set and `relevant_doc_ids` are provided, documents
    referenced by those qrels are prioritized first, and remaining slots are
    filled via a seeded random sample of other documents.

    Raises:
        ValueError: If subset_size is not positive or no documents are found.
    """
    if subset_size is not None and subset_size < 1:
        raise ValueError("subset_size must be positive when specified.")

    documents: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            documents.append(json.loads(line))

    if not documents:
        raise ValueError(f"No documents found in {corpus_path}")

    if subset_size is None or subset_size >= len(documents):
        return documents

    if not relevant_doc_ids:
        rng = random.Random(seed)
        return rng.sample(documents, subset_size)

    prioritized: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for doc in documents:
        aliases = set(doc.get("original_document_ids", [doc["_id"]]))
        if aliases.intersection(relevant_doc_ids):
            prioritized.append(doc)
        else:
            others.append(doc)

    rng = random.Random(seed)
    if len(prioritized) >= subset_size:
        return rng.sample(prioritized, subset_size)

    needed = subset_size - len(prioritized)
    sampled_others = rng.sample(others, needed)
    return prioritized + sampled_others


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a deterministic hex chunk_id into a deterministic UUIDv5 string for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def build_index(
    corpus_path: Path = CHUNK_CORPUS_PATH,
    qdrant_path: Path = QDRANT_PATH,
    collection_name: str = QDRANT_COLLECTION_NAME,
    manifest_path: Path = INDEX_MANIFEST_PATH,
    subset_size: int | None = None,
    seed: int = SUBSET_RANDOM_SEED,
    max_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    model: SentenceTransformer | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Chunk selected corpus documents, embed them, and index them in local Qdrant.

    Returns:
        The index manifest dictionary written to `manifest_path`.
    """
    start_time = time.perf_counter()
    input_checksum = file_checksum(corpus_path)

    relevant_ids = load_relevant_document_ids(corpus_path.parent / "qrels") if subset_size is not None else None
    documents = select_corpus_documents(
        corpus_path=corpus_path,
        subset_size=subset_size,
        seed=seed,
        relevant_doc_ids=relevant_ids,
    )
    logger.info("Selected %d documents from %s (subset_size=%s)", len(documents), corpus_path.name, subset_size)

    active_tokenizer = tokenizer if tokenizer is not None else load_embedding_tokenizer()
    chunk_start = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    split_documents = 0
    for doc in documents:
        doc_chunks = chunk_document(
            doc,
            tokenizer=active_tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        if len(doc_chunks) > 1:
            split_documents += 1
        chunks.extend(doc_chunks)
    chunk_seconds = time.perf_counter() - chunk_start

    if not chunks:
        raise ValueError("Selected documents produced no usable chunks.")

    logger.info(
        "Chunked %d documents (%d split) into %d chunks in %.2fs",
        len(documents),
        split_documents,
        len(chunks),
        chunk_seconds,
    )

    active_model = model if model is not None else load_embedding_model()
    texts = [embedding_text(chunk["title"], chunk["text"]) for chunk in chunks]

    embed_start = time.perf_counter()
    logger.info("Embedding %d chunks with %s (batch_size=%d)", len(chunks), EMBEDDING_MODEL_ID, batch_size)
    vectors = active_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embed_seconds = time.perf_counter() - embed_start
    vector_dim = int(vectors.shape[1])

    upsert_start = time.perf_counter()
    qdrant_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

        upsert_batch_size = 500
        for offset in range(0, len(chunks), upsert_batch_size):
            batch_chunks = chunks[offset : offset + upsert_batch_size]
            batch_vectors = vectors[offset : offset + upsert_batch_size]
            points = [
                PointStruct(
                    id=chunk_id_to_uuid(chunk["chunk_id"]),
                    vector=vec.tolist(),
                    payload=chunk,
                )
                for chunk, vec in zip(batch_chunks, batch_vectors, strict=True)
            ]
            client.upsert(collection_name=collection_name, points=points)
    finally:
        client.close()

    upsert_seconds = time.perf_counter() - upsert_start
    total_seconds = time.perf_counter() - start_time

    manifest: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "input_file": corpus_path.name,
        "input_sha256": input_checksum,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "vector_dimension": vector_dim,
        "qdrant_collection": collection_name,
        "subset_size": subset_size,
        "subset_seed": seed if subset_size is not None else None,
        "chunking_settings": {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
        "documents_indexed": len(documents),
        "split_documents": split_documents,
        "chunks_indexed": len(chunks),
        "max_chunk_tokens": max(c["token_count"] for c in chunks),
        "timings_seconds": {
            "chunking": round(chunk_seconds, 3),
            "embedding": round(embed_seconds, 3),
            "indexing": round(upsert_seconds, 3),
            "total": round(total_seconds, 3),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Indexed %d chunks in %.2fs (embed: %.2fs, qdrant: %.2fs). Manifest saved to %s",
        len(chunks),
        total_seconds,
        embed_seconds,
        upsert_seconds,
        manifest_path,
    )
    return manifest
