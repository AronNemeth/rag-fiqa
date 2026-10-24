import json
from pathlib import Path
from typing import Any

import pytest

from rag_fiqa.chunking import chunk_corpus, chunk_document, embedding_text
from rag_fiqa.data import file_checksum


def document(text: str, title: str = "") -> dict[str, Any]:
    return {"_id": "a", "title": title, "text": text, "original_document_ids": ["a", "duplicate"]}


def assert_valid_chunks(doc: dict[str, Any], chunk_size: int, overlap: int) -> None:
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
    assert chunks[0]["start_char"] == 0
    assert chunks[-1]["end_char"] == len(doc["text"])
    covered_until = 0
    for index, chunk in enumerate(chunks):
        assert chunk["start_char"] <= covered_until < chunk["end_char"]
        covered_until = chunk["end_char"]
        assert chunk["text"] == doc["text"][chunk["start_char"] : chunk["end_char"]]
        assert chunk["chunk_index"] == index
        assert chunk["original_document_ids"] == ["a", "duplicate"]
        assert 0 < len(chunk["text"]) <= chunk_size


def test_short_document_stays_whole() -> None:
    doc = document("  hello café world!\n", "title")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0]["text"] == doc["text"]
    assert embedding_text(chunks[0]["title"], chunks[0]["text"]) == "title\n\n" + doc["text"]
    assert embedding_text("", chunks[0]["text"]) == doc["text"]


@pytest.mark.parametrize("size,overlap", [(1, 0), (30, 0), (30, 4), (30, 29), (800, 120)])
@pytest.mark.parametrize("text", ["hello world " * 100, "finance" * 3000, "café! hello\r\n\r\nworld. 🙂 " * 50])
def test_long_documents_preserve_all_characters(size: int, overlap: int, text: str) -> None:
    assert_valid_chunks(document(text, "title"), size, overlap)


def test_paragraph_boundary_is_preferred() -> None:
    doc = document("hello world hello world\n\nhello world hello world hello world")
    chunks = chunk_document(doc, chunk_size=40, overlap=0)
    assert chunks[0]["text"] == "hello world hello world\n\n"


def test_sentence_boundary_is_preferred() -> None:
    chunks = chunk_document(
        document("hello world hello. world hello world hello world hello"), chunk_size=32, overlap=0
    )
    assert chunks[0]["text"] == "hello world hello. "


def test_word_boundary_before_character_fallback() -> None:
    chunks = chunk_document(document("hello world hello world"), chunk_size=15, overlap=0)
    assert chunks[0]["text"] == "hello world "
    assert chunk_document(document("x" * 31), chunk_size=15, overlap=0)[0]["text"] == "x" * 15


def test_actual_overlap_and_stable_ids() -> None:
    doc = document("hello world " * 20)
    first = chunk_document(doc, chunk_size=30, overlap=4)
    assert first == chunk_document(doc, chunk_size=30, overlap=4)
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)
    assert first[0]["end_char"] - first[1]["start_char"] == 4
    changed = chunk_document(document(doc["text"] + "hello"), chunk_size=30, overlap=4)
    assert changed[0]["chunk_id"] != first[0]["chunk_id"]


def test_empty_document_and_exact_limit() -> None:
    assert chunk_document(document(" \n\t")) == []
    assert len(chunk_document(document("x" * 800))) == 1
    assert len(chunk_document(document("x" * 801))) == 2


def test_title_does_not_change_character_boundaries() -> None:
    text = "hello world " * 100
    plain = chunk_document(document(text), chunk_size=50, overlap=5)
    titled = chunk_document(document(text, "Long title " * 100), chunk_size=50, overlap=5)
    assert [chunk["text"] for chunk in plain] == [chunk["text"] for chunk in titled]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (8, -1), (8, 8)])
def test_invalid_settings(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(document("hello"), chunk_size=size, overlap=overlap)


def test_artifacts_are_reproducible_and_manifest_matches(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text(json.dumps(document("hello world " * 200)) + "\n", encoding="utf-8")
    output = tmp_path / "chunks"
    manifest = chunk_corpus(source, output)
    assert manifest["input_sha256"] == file_checksum(source)
    assert manifest["chunks_sha256"] == file_checksum(output / "chunks.jsonl")
    assert manifest["settings"] == {"chunk_size_chars": 800, "overlap_chars": 120}
    original_manifest = (output / "chunking_manifest.json").read_bytes()
    original_chunks = (output / "chunks.jsonl").read_bytes()
    chunk_corpus(source, output)
    assert (output / "chunking_manifest.json").read_bytes() == original_manifest
    assert (output / "chunks.jsonl").read_bytes() == original_chunks


def test_no_usable_chunks_raises(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text(json.dumps(document("  \n")) + "\n")
    output = tmp_path / "chunks"
    with pytest.raises(ValueError, match="no usable chunks"):
        chunk_corpus(source, output)
    assert not (output / "chunks.jsonl").exists()
