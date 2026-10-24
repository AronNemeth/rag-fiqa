"""Split documents into character-bounded chunks while preserving source offsets."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rag_fiqa.config import (
    CHUNK_CORPUS_PATH,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    CHUNKS_DIR,
    DATASET_NAME,
)
from rag_fiqa.data import file_checksum
from rag_fiqa.utils import get_logger

logger = get_logger(__name__)

CHUNKING_VERSION = 3


def embedding_text(title: str, text: str) -> str:
    """Build the complete text input for embedding, including an optional title."""
    return f"{title}\n\n{text}" if title else text


def _window_end(text: str, start: int, minimum_end: int, size: int) -> int:
    """Prefer paragraph, sentence, then word boundaries in the latter half of a window."""
    end = min(start + size, len(text))
    if end == len(text):
        return end
    minimum = max(start + size // 2, minimum_end)
    window = text[start:end]
    for pattern in (r"\n[ \t]*\r?\n", r"[.!?][\"')\]]*(?:\s+|$)", r"\s+"):
        boundaries = [start + match.end() for match in re.finditer(pattern, window)]
        if boundaries and boundaries[-1] >= minimum:
            return boundaries[-1]
    return end


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    """Split a document into character windows, keeping short documents whole.

    Raises:
        ValueError: If chunk_size or overlap are invalid.
    """
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be nonnegative and smaller than the chunk size.")

    doc_id: str = document["_id"]
    title: str = document.get("title", "")
    text: str = document["text"]
    aliases: list[str] = list(document.get("original_document_ids", [doc_id]))

    if not text.strip():
        return []

    parent_digest = hashlib.sha256(json.dumps([doc_id, title, text], ensure_ascii=False).encode()).hexdigest()
    chunks: list[dict[str, Any]] = []
    start = 0
    previous_end = 0

    while start < len(text):
        end = _window_end(text, start, previous_end + 1, chunk_size)
        body = text[start:end]
        identity = f"{CHUNKING_VERSION}:{parent_digest}:{start}:{end}"
        chunks.append(
            {
                "chunk_id": hashlib.sha256(identity.encode()).hexdigest(),
                "parent_document_id": doc_id,
                "original_document_ids": aliases,
                "chunk_index": len(chunks),
                "title": title,
                "text": body,
                "start_char": start,
                "end_char": end,
            }
        )
        if end == len(text):
            break
        previous_end = end
        start = end - min(overlap, (end - start) // 2)

    return chunks


def chunk_corpus(
    corpus_path: Path = CHUNK_CORPUS_PATH,
    output_dir: Path = CHUNKS_DIR,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> dict[str, Any]:
    """Chunk all documents in a prepared JSONL corpus and write chunks and manifest.

    Raises:
        ValueError: If the corpus produces no usable chunks.
    """
    input_checksum = file_checksum(corpus_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "chunking_manifest.json"

    doc_count = split_count = chunk_count = max_chars = 0
    logger.info("Chunking corpus %s", corpus_path)

    with corpus_path.open(encoding="utf-8") as source, chunks_path.open("w", encoding="utf-8") as target:
        for line in source:
            document = json.loads(line)
            doc_count += 1
            chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
            if len(chunks) > 1:
                split_count += 1
            for chunk in chunks:
                target.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1
                max_chars = max(max_chars, len(chunk["text"]))

    if chunk_count == 0:
        chunks_path.unlink(missing_ok=True)
        raise ValueError("Corpus produced no usable chunks.")

    manifest: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "input_file": corpus_path.name,
        "input_sha256": input_checksum,
        "chunking_version": CHUNKING_VERSION,
        "settings": {
            "chunk_size_chars": chunk_size,
            "overlap_chars": overlap,
        },
        "documents": doc_count,
        "split_documents": split_count,
        "chunks": chunk_count,
        "max_chunk_chars": max_chars,
        "chunks_sha256": file_checksum(chunks_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Saved %d chunks from %d documents (%d split) to %s",
        chunk_count,
        doc_count,
        split_count,
        output_dir,
    )
    return manifest
