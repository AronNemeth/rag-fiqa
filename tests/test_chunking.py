import json
import re
from pathlib import Path
from typing import Any

import pytest

from rag_fiqa.chunking import chunk_corpus, chunk_document


class FakeWordTokenizer:
    """Lightweight deterministic word/character tokenizer for fast offline unit tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = True,
        truncation: bool = False,
    ) -> dict[str, Any]:
        # Split into words of at most 8 characters so long spaceless strings also tokenize.
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S{1,8}", text)]
        return {"offset_mapping": offsets}


TOKENIZER: Any = FakeWordTokenizer()


def document(text: str, title: str = "") -> dict[str, Any]:
    return {"_id": "a", "title": title, "text": text, "original_document_ids": ["a", "duplicate"]}


def assert_valid_chunks(doc: dict[str, Any], max_tokens: int, overlap_tokens: int) -> None:
    chunks = chunk_document(doc, tokenizer=TOKENIZER, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    assert chunks[0]["start_char"] == 0
    assert chunks[-1]["end_char"] == len(doc["text"])
    covered_until = 0
    for index, chunk in enumerate(chunks):
        assert chunk["start_char"] <= covered_until < chunk["end_char"]
        covered_until = chunk["end_char"]
        assert chunk["text"] == doc["text"][chunk["start_char"] : chunk["end_char"]]
        assert chunk["chunk_index"] == index
        assert chunk["original_document_ids"] == ["a", "duplicate"]
        assert 2 < chunk["token_count"] <= max_tokens


def test_short_document_stays_whole() -> None:
    doc = document("  hello café world!\n", "title")
    chunks = chunk_document(doc, tokenizer=TOKENIZER, max_tokens=256, overlap_tokens=32)
    assert len(chunks) == 1
    assert chunks[0]["text"] == doc["text"]
    assert chunks[0]["token_count"] == 3 + 2


@pytest.mark.parametrize("max_tokens,overlap", [(3, 0), (10, 0), (10, 2), (10, 7), (256, 32)])
@pytest.mark.parametrize("text", ["hello world " * 40, "finance" * 200, "café! hello\r\n\r\nworld. 🙂 " * 25])
def test_long_documents_preserve_all_characters_and_token_budget(max_tokens: int, overlap: int, text: str) -> None:
    assert_valid_chunks(document(text, "title"), max_tokens, overlap)


def test_paragraph_boundary_is_preferred() -> None:
    doc = document("one two three four\n\nfive six seven eight nine ten")
    # max_tokens=8 -> body_budget=6 tokens ("one two three four five six"). Paragraph break is after "four".
    chunks = chunk_document(doc, tokenizer=TOKENIZER, max_tokens=8, overlap_tokens=0)
    assert chunks[0]["text"] == "one two three four\n\n"


def test_sentence_boundary_is_preferred() -> None:
    doc = document("one two three. four five six seven eight")
    # max_tokens=7 -> body_budget=5 tokens ("one two three. four five"). Sentence break is after "three. ".
    chunks = chunk_document(doc, tokenizer=TOKENIZER, max_tokens=7, overlap_tokens=0)
    assert chunks[0]["text"] == "one two three. "


def test_token_overlap_and_stable_ids() -> None:
    doc = document(" ".join(f"w{i}" for i in range(20)))
    first = chunk_document(doc, tokenizer=TOKENIZER, max_tokens=10, overlap_tokens=2)
    assert first == chunk_document(doc, tokenizer=TOKENIZER, max_tokens=10, overlap_tokens=2)
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)
    # First chunk ends at the whitespace boundary after w6 (7 words: w0..w6); with overlap=2, second chunk starts at w5.
    assert first[0]["text"] == "w0 w1 w2 w3 w4 w5 w6 "
    assert first[1]["text"].startswith("w5 w6")
    changed = chunk_document(document(doc["text"] + " extra"), tokenizer=TOKENIZER, max_tokens=10, overlap_tokens=2)
    assert changed[0]["chunk_id"] != first[0]["chunk_id"]


def test_empty_document_and_exact_token_limit() -> None:
    assert chunk_document(document(" \n\t"), tokenizer=TOKENIZER) == []
    # 254 words + 2 special tokens == 256 tokens -> 1 chunk
    assert len(chunk_document(document("word " * 254), tokenizer=TOKENIZER, max_tokens=256, overlap_tokens=32)) == 1
    # 255 words + 2 special tokens == 257 tokens -> 2 chunks
    assert len(chunk_document(document("word " * 255), tokenizer=TOKENIZER, max_tokens=256, overlap_tokens=32)) == 2


@pytest.mark.parametrize("max_tokens,overlap", [(2, 0), (0, 0), (-1, 0), (10, -1), (10, 8), (10, 10)])
def test_invalid_settings(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(document("hello"), tokenizer=TOKENIZER, max_tokens=max_tokens, overlap_tokens=overlap)


def test_chunk_corpus_in_memory(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    docs = [document("one two three four five"), {**document("   "), "_id": "empty"}]
    source.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    chunks = chunk_corpus(source, tokenizer=TOKENIZER, max_tokens=6, overlap_tokens=1)
    assert len(chunks) == 2
    assert all(c["parent_document_id"] == "a" for c in chunks)
