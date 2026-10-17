# rag-squad

A small financial Q&A RAG application using FiQA-2018. See
[the implementation plan](plans/rag_implementation_plan.md) for the planned pipeline.

## Setup

From the repository root, install the project and its development dependencies:

```bash
uv sync
```

## Prepare the dataset

From the repository root, run:

```bash
uv run python -m rag_squad.prepare_data
```

This downloads the official [BEIR FiQA archive](https://github.com/beir-cellar/beir/wiki/Datasets-available)
into `data/raw/fiqa.zip`, verifies BEIR's published MD5 checksum, and writes
`data/processed/fiqa/`:

- `corpus.jsonl`: original document IDs and unchanged text, with identical repeated ID records removed.
- `corpus_deduplicated.jsonl`: one record per exact title/text pair, with all source IDs in
  `original_document_ids`. Whitespace-only documents are excluded from this embedding copy.
- `queries.jsonl` and `qrels/{train,dev,test}.tsv`: validated queries and relevance judgments,
  kept separate from searchable documents. Duplicate identical judgments are removed.
- `manifest.json`: archive/output checksums, preprocessing policy/version, counts, and empty document IDs.

Repeated runs reuse the verified archive and regenerate prepared outputs. Conflicting IDs or
judgments and unknown qrels references fail preparation. Original text is retained without
HTML stripping or whitespace normalization so future chunk offsets refer to the source text.
Do not index queries or qrels. When indexing the deduplicated copy, retain and expand
`original_document_ids` for document-level evaluation against the original qrels.

`prepare_data.py` handles download and preprocessing; vector-store ingestion will be a separate step.
Logs show progress and record counts without printing source text. Dataset artifacts are ignored by Git.
Data paths are anchored to the repository root through `rag_squad.config`, independently of
the working directory when using the editable installation.

## Logging

Application entry points configure logging to both the console and `logs/app.log`.
The directory is created automatically under the repository root. Logs append across runs
and rotate at 5 MiB, retaining three backups (`app.log.1` through `app.log.3`).
The entire `logs/` directory is ignored by Git.

Logging level, format, file path, and rotation limits are defined in `rag_squad.config`.
Call `setup_logging()` once at each application entry point; importing modules does not
create log files. Repeated setup replaces existing root handlers.

## Checks

```bash
uv run pytest
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
```

Tests use small local fixtures and do not download datasets or load models.
