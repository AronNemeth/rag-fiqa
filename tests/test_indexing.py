import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_fiqa.indexing import build_index, select_corpus_documents


class FakeWordTokenizer:
    """Lightweight deterministic word tokenizer for fast offline unit tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = True,
        truncation: bool = False,
    ) -> dict[str, Any]:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S{1,8}", text)]
        return {"offset_mapping": offsets}


class FakeEmbeddingModel:
    """Fake embedding model returning deterministic normalized 4-D vectors."""

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for idx, text in enumerate(texts):
            vectors[idx, 0] = float(len(text))
            vectors[idx, 1] = float(idx + 1)
            norm = np.linalg.norm(vectors[idx])
            if norm > 0:
                vectors[idx] /= norm
        return vectors


def write_sample_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "processed" / "fiqa"
    qrels_dir = corpus_dir / "qrels"
    qrels_dir.mkdir(parents=True)

    docs = [
        {
            "_id": f"doc_{i}",
            "title": "",
            "text": f"Financial passage number {i} with several words.",
            "original_document_ids": [f"doc_{i}"],
        }
        for i in range(10)
    ]
    corpus_path = corpus_dir / "corpus_deduplicated.jsonl"
    corpus_path.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")

    (qrels_dir / "dev.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tdoc_2\t1\n", encoding="utf-8")
    (qrels_dir / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq2\tdoc_7\t1\n", encoding="utf-8")
    return corpus_path


def test_select_corpus_documents_prioritizes_relevant_ids_and_is_reproducible(tmp_path: Path) -> None:
    corpus_path = write_sample_corpus(tmp_path)
    selected_a = select_corpus_documents(
        corpus_path=corpus_path,
        subset_size=4,
        seed=42,
        relevant_doc_ids={"doc_2", "doc_7"},
    )
    selected_b = select_corpus_documents(
        corpus_path=corpus_path,
        subset_size=4,
        seed=42,
        relevant_doc_ids={"doc_2", "doc_7"},
    )
    assert selected_a == selected_b
    assert len(selected_a) == 4
    selected_ids = {doc["_id"] for doc in selected_a}
    assert {"doc_2", "doc_7"}.issubset(selected_ids)


def test_build_index_populates_qdrant_and_writes_manifest(tmp_path: Path) -> None:
    corpus_path = write_sample_corpus(tmp_path)
    qdrant_path = tmp_path / "index" / "qdrant"
    manifest_path = tmp_path / "index" / "index_manifest.json"

    fake_model: Any = FakeEmbeddingModel()
    fake_tokenizer: Any = FakeWordTokenizer()
    manifest = build_index(
        corpus_path=corpus_path,
        qdrant_path=qdrant_path,
        collection_name="test_chunks",
        manifest_path=manifest_path,
        subset_size=5,
        seed=42,
        max_tokens=8,
        overlap_tokens=1,
        model=fake_model,
        tokenizer=fake_tokenizer,
    )

    assert manifest["documents_indexed"] == 5
    assert manifest["chunks_indexed"] >= 5
    assert manifest["vector_dimension"] == 4
    assert manifest_path.exists()

    client = QdrantClient(path=str(qdrant_path))
    try:
        assert client.count(collection_name="test_chunks").count == manifest["chunks_indexed"]
    finally:
        client.close()


def test_invalid_subset_size_raises(tmp_path: Path) -> None:
    corpus_path = write_sample_corpus(tmp_path)
    with pytest.raises(ValueError, match="subset_size must be positive"):
        select_corpus_documents(corpus_path=corpus_path, subset_size=0)
