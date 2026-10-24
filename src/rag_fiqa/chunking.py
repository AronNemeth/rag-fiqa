"""Split documents into token-bounded chunks while preserving source character offsets."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rag_fiqa.config import (
    CHUNK_CORPUS_PATH,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)


def load_embedding_tokenizer(
    model_id: str = EMBEDDING_MODEL_ID,
    revision: str = EMBEDDING_MODEL_REVISION,
    *,
    local_files_only: bool = True,
) -> PreTrainedTokenizerBase:
    """Load the cached Hugging Face tokenizer for the embedding model."""
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )


def _window_end_char(
    text: str,
    offsets: list[tuple[int, int]],
    start_tok: int,
    max_end_tok: int,
    minimum_end_tok: int,
) -> int:
    """Prefer paragraph, sentence, then word boundaries in the latter half of a token window."""
    if max_end_tok >= len(offsets):
        return len(text)

    start_char = offsets[start_tok][0]
    limit_char = offsets[max_end_tok - 1][1]
    min_tok_idx = max(start_tok + (max_end_tok - start_tok) // 2, minimum_end_tok) - 1
    minimum_char = offsets[min_tok_idx][1]

    window = text[start_char:limit_char]
    for pattern in (r"\n[ \t]*\r?\n", r"[.!?][\"')\]]*(?:\s+|$)", r"\s+"):
        boundaries = [start_char + match.end() for match in re.finditer(pattern, window)]
        if boundaries and boundaries[-1] >= minimum_char:
            return boundaries[-1]
    return limit_char


def chunk_document(
    document: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Split a document into token-bounded windows, keeping short documents whole.

    Reserves 2 tokens (`[CLS]` and `[SEP]`) from `max_tokens` so every emitted
    chunk's complete token length fits within `max_tokens`.

    Raises:
        ValueError: If max_tokens or overlap_tokens are invalid.
    """
    body_budget = max_tokens - 2
    if body_budget < 1:
        raise ValueError("max_tokens must be at least 3 to accommodate [CLS] and [SEP] tokens.")
    if not 0 <= overlap_tokens < body_budget:
        raise ValueError("overlap_tokens must be nonnegative and smaller than max_tokens - 2.")

    doc_id: str = document["_id"]
    title: str = document.get("title", "")
    text: str = document["text"]
    aliases: list[str] = list(document.get("original_document_ids", [doc_id]))

    if not text.strip():
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    raw_offsets: list[tuple[int, int]] = encoded["offset_mapping"]  # type: ignore[assignment]
    offsets: list[tuple[int, int]] = [(int(start), int(end)) for start, end in raw_offsets if end > start]
    if not offsets:
        return []

    parent_digest = hashlib.sha256(json.dumps([doc_id, title, text], ensure_ascii=False).encode()).hexdigest()

    # Keep short documents completely whole (preserving leading/trailing whitespace).
    if len(offsets) <= body_budget:
        identity = f"{parent_digest}:0:{len(text)}"
        return [
            {
                "chunk_id": hashlib.sha256(identity.encode()).hexdigest(),
                "parent_document_id": doc_id,
                "original_document_ids": aliases,
                "chunk_index": 0,
                "title": title,
                "text": text,
                "token_count": len(offsets) + 2,
                "start_char": 0,
                "end_char": len(text),
            }
        ]

    chunks: list[dict[str, Any]] = []
    start_tok = 0
    previous_end_tok = 0
    previous_end_char = 0

    while start_tok < len(offsets):
        max_end_tok = min(start_tok + body_budget, len(offsets))
        start_char = (
            0 if not chunks else (previous_end_char if start_tok == previous_end_tok else offsets[start_tok][0])
        )
        end_char = _window_end_char(text, offsets, start_tok, max_end_tok, previous_end_tok + 1)

        # Count how many tokens fall inside [start_char, end_char).
        end_tok = start_tok
        while end_tok < len(offsets) and offsets[end_tok][1] <= end_char:
            end_tok += 1
        if end_tok <= start_tok:
            end_tok = max_end_tok
        if end_tok == len(offsets):
            end_char = len(text)
        elif end_char < offsets[end_tok - 1][1]:
            end_char = offsets[end_tok - 1][1]

        body = text[start_char:end_char]
        body_tokens = end_tok - start_tok
        identity = f"{parent_digest}:{start_char}:{end_char}"
        chunks.append(
            {
                "chunk_id": hashlib.sha256(identity.encode()).hexdigest(),
                "parent_document_id": doc_id,
                "original_document_ids": aliases,
                "chunk_index": len(chunks),
                "title": title,
                "text": body,
                "token_count": body_tokens + 2,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        if end_tok >= len(offsets):
            break
        previous_end_tok = end_tok
        previous_end_char = end_char
        actual_overlap = min(overlap_tokens, body_tokens // 2)
        start_tok = end_tok - actual_overlap

    return chunks


def chunk_corpus(
    corpus_path: Path = CHUNK_CORPUS_PATH,
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Read a prepared JSONL corpus and return all token-bounded chunks in memory."""
    active_tokenizer = tokenizer if tokenizer is not None else load_embedding_tokenizer()
    all_chunks: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            document = json.loads(line)
            all_chunks.extend(
                chunk_document(
                    document,
                    tokenizer=active_tokenizer,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
    return all_chunks
