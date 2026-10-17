import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from rag_squad import prepare_data
from rag_squad.prepare_data import deduplicate_corpus, file_checksum, prepare_dataset, read_records


def make_archive(path: Path, *, missing_document: bool = False) -> Path:
    corpus = [
        {"_id": "a", "title": "", "text": "Original\ntext"},
        {"_id": "b", "title": "", "text": "Original\ntext"},
        {"_id": "empty", "title": "", "text": "  "},
    ]
    with ZipFile(path, "w") as archive:
        archive.writestr("fiqa/corpus.jsonl", "".join(json.dumps(row) + "\n" for row in corpus))
        archive.writestr("fiqa/queries.jsonl", json.dumps({"_id": "q", "text": "Question?"}) + "\n")
        for split in ("train", "dev", "test"):
            document_id = "unknown" if missing_document else "b"
            archive.writestr(f"fiqa/qrels/{split}.tsv", f"query-id\tcorpus-id\tscore\nq\t{document_id}\t1\n")
    return path


def test_deduplication_preserves_aliases_and_original_text() -> None:
    corpus = [
        {"_id": "a", "text": " text\n", "title": ""},
        {"_id": "b", "text": " text\n", "title": ""},
        {"_id": "c", "text": "text", "title": ""},
        {"_id": "d", "text": " \n", "title": ""},
    ]
    records, empty = deduplicate_corpus(corpus)
    assert len(records) == 2
    assert records[0]["original_document_ids"] == ["a", "b"]
    assert records[0]["text"] == " text\n"
    assert empty == ["d"]


def test_preparation_preserves_qrels_and_is_reproducible(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "fiqa.zip")
    output = tmp_path / "prepared"
    prepare_dataset(archive, output)
    assert len(read_records(output / "corpus.jsonl")) == 3
    deduplicated = json.loads((output / "corpus_deduplicated.jsonl").read_text())
    assert deduplicated["original_document_ids"] == ["a", "b"]
    assert "q\tb\t1" in (output / "qrels/test.tsv").read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["empty_document_ids"] == ["empty"]
    for filename, checksum in manifest["output_sha256"].items():
        assert file_checksum(output / filename) == checksum
    first_manifest = (output / "manifest.json").read_bytes()
    prepare_dataset(archive, output)
    assert (output / "manifest.json").read_bytes() == first_manifest


def test_unknown_qrels_do_not_replace_existing_outputs(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "fiqa.zip", missing_document=True)
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "corpus.jsonl").write_text("existing data")
    with pytest.raises(ValueError, match="Unknown query/document"):
        prepare_dataset(archive, output)
    assert (output / "corpus.jsonl").read_text() == "existing data"


def test_conflicting_record_ids_fail(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text('{"_id":"a","text":"one"}\n{"_id":"a","text":"two"}\n')
    with pytest.raises(ValueError, match="Conflicting records"):
        read_records(source)


def test_identical_repeated_ids_are_removed(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text('{"_id":"a","text":"one"}\n' * 2)
    assert read_records(source) == [{"_id": "a", "text": "one"}]


def test_corrupt_cached_download_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "fiqa.zip"
    archive.write_bytes(b"corrupt archive")

    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("A cached corrupt archive should fail without a download")

    monkeypatch.setattr(prepare_data, "urlopen", unexpected_download)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare_data.download_dataset(archive)


def test_incomplete_download_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "fiqa.zip"

    def interrupted_download(*args: object, **kwargs: object) -> None:
        raise TimeoutError("download interrupted")

    monkeypatch.setattr(prepare_data, "urlopen", interrupted_download)
    with pytest.raises(TimeoutError):
        prepare_data.download_dataset(archive)
    assert not archive.exists()
    assert not archive.with_suffix(".zip.part").exists()
