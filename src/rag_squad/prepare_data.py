"""Download and prepare FiQA without loading models or a vector store."""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from rag_squad.config import (
    DATASET_ARCHIVE,
    DATASET_MD5,
    DATASET_NAME,
    DATASET_URL,
    DOWNLOAD_TIMEOUT_SECONDS,
    PREPARED_DATA_DIR,
    QRELS_SPLITS,
)
from rag_squad.utils import get_logger, setup_logging

logger = get_logger(__name__)


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute a file checksum without reading the entire file into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def download_dataset(archive: Path = DATASET_ARCHIVE) -> Path:
    """Download the pinned archive, reusing an existing verified download."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if file_checksum(archive, "md5") != DATASET_MD5:
            raise ValueError(f"Checksum mismatch for cached archive: {archive}")
        logger.info("Reusing verified archive %s", archive)
        return archive

    temporary = archive.with_suffix(".zip.part")
    logger.info("Downloading %s", DATASET_URL)
    try:
        with urlopen(DATASET_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if file_checksum(temporary, "md5") != DATASET_MD5:
            raise ValueError("Downloaded archive does not match the expected BEIR checksum")
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Saved verified archive to %s", archive)
    return archive


def read_records(path: Path) -> list[dict[str, str]]:
    """Validate BEIR JSONL records and remove identical repeated IDs.

    Raises:
        ValueError: If a record is malformed or an ID has conflicting records.
    """
    records: dict[str, dict[str, str]] = {}
    repeated = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if not isinstance(record.get("_id"), str) or not record["_id"]:
                raise ValueError(f"Missing string ID at {path}:{line_number}")
            if not isinstance(record.get("text"), str) or not isinstance(record.get("title", ""), str):
                raise ValueError(f"Invalid text or title at {path}:{line_number}")
            identifier = record["_id"]
            if identifier in records:
                if records[identifier] != record:
                    raise ValueError(f"Conflicting records for ID {identifier} in {path}")
                repeated += 1
            else:
                records[identifier] = record
    if not records:
        raise ValueError(f"No records found in {path}")
    logger.info("Loaded %s: %d unique IDs, %d repeated records removed", path.name, len(records), repeated)
    return list(records.values())


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as UTF-8 JSON Lines."""
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def deduplicate_corpus(corpus: list[dict[str, str]]) -> tuple[list[dict], list[str]]:
    """Group exact title/text duplicates while retaining their original IDs.

    Preserve original text so future chunk offsets remain meaningful. Empty text
    is excluded only from the embedding copy and reported for inspection.
    """
    groups: dict[tuple[str, str], dict] = {}
    empty_ids: list[str] = []
    for document in corpus:
        if not document["text"].strip():
            empty_ids.append(document["_id"])
            continue
        key = (document.get("title", ""), document["text"])
        if key not in groups:
            groups[key] = {**document, "original_document_ids": []}
        groups[key]["original_document_ids"].append(document["_id"])
    return list(groups.values()), empty_ids


def prepare_qrels(source: Path, target: Path, document_ids: set[str], query_ids: set[str]) -> int:
    """Validate relevance judgments and remove identical repeated judgments."""
    judgments: dict[tuple[str, str], int] = {}
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrels columns in {source}")
        for row in reader:
            query_id, document_id = row["query-id"], row["corpus-id"]
            if query_id not in query_ids or document_id not in document_ids:
                raise ValueError(f"Unknown query/document reference in {source}: {query_id}/{document_id}")
            score = int(row["score"])
            key = (query_id, document_id)
            if key in judgments and judgments[key] != score:
                raise ValueError(f"Conflicting relevance scores in {source}: {key}")
            judgments[key] = score
    if not judgments:
        raise ValueError(f"No relevance judgments in {source}")
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows((query, document, score) for (query, document), score in judgments.items())
    logger.info("Validated %s: %d judgments", source.name, len(judgments))
    return len(judgments)


def prepare_dataset(archive: Path, output_dir: Path = PREPARED_DATA_DIR) -> None:
    """Prepare corpus, queries, qrels, and a manifest from a BEIR archive.

    Validate all inputs in a staging directory before publishing prepared files.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        raw = staging / "raw"
        prepared = staging / "prepared"
        prepared.mkdir()
        (prepared / "qrels").mkdir()
        members = ["corpus.jsonl", "queries.jsonl", *(f"qrels/{split}.tsv" for split in QRELS_SPLITS)]
        # Extract only known members rather than arbitrary paths from the ZIP.
        with ZipFile(archive) as bundle:
            for member in members:
                destination = raw / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(f"{DATASET_NAME}/{member}") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

        corpus = read_records(raw / "corpus.jsonl")
        queries = read_records(raw / "queries.jsonl")
        unique_corpus, empty_ids = deduplicate_corpus(corpus)
        write_jsonl(prepared / "corpus.jsonl", corpus)
        write_jsonl(prepared / "corpus_deduplicated.jsonl", unique_corpus)
        write_jsonl(prepared / "queries.jsonl", queries)
        document_ids = {record["_id"] for record in corpus}
        query_ids = {record["_id"] for record in queries}
        qrels_counts = {
            split: prepare_qrels(raw / f"qrels/{split}.tsv", prepared / f"qrels/{split}.tsv", document_ids, query_ids)
            for split in QRELS_SPLITS
        }
        manifest = {
            "dataset": DATASET_NAME,
            "source_url": DATASET_URL,
            "archive_sha256": file_checksum(archive),
            "archive_md5": file_checksum(archive, "md5"),
            "preprocessing_version": 1,
            "deduplication": "exact title and text; preserve all original IDs; no text normalization",
            "documents": len(corpus),
            "embedding_documents": len(unique_corpus),
            "duplicate_text_records": len(corpus) - len(empty_ids) - len(unique_corpus),
            "empty_document_ids": empty_ids,
            "queries": len(queries),
            "qrels": qrels_counts,
            "output_sha256": {
                str(path.relative_to(prepared)): file_checksum(path)
                for path in sorted(prepared.rglob("*"))
                if path.is_file()
            },
        }
        (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        output_dir.mkdir(exist_ok=True)
        for path in prepared.rglob("*"):
            if path.is_file():
                destination = output_dir / path.relative_to(prepared)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
        logger.info(
            "Prepared %d documents (%d embedding records, %d empty), %d queries in %s",
            len(corpus),
            len(unique_corpus),
            len(empty_ids),
            len(queries),
            output_dir,
        )
        if empty_ids:
            logger.warning("Excluded %d empty documents from embedding copy; IDs retained in manifest", len(empty_ids))


def main() -> None:
    """Configure logging and prepare the downloaded dataset."""
    setup_logging()
    try:
        prepare_dataset(download_dataset())
    except Exception:
        logger.exception("Dataset preparation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
