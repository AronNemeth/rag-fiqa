# rag-fiqa

A small financial Q&A RAG application using FiQA-2018.

## Project Structure

- `src/rag_fiqa/`: Reusable library modules (`config`, `utils`, `data`, `chunking`, etc.) with no CLI side effects on import.
- `scripts/`: Orchestrator entry-point scripts for running individual pipeline stages (`prepare_data.py`, `chunk_corpus.py`, etc.).
- `tests/`: Unit tests using fast local fixtures.

## Setup

From the repository root, install the project and its development dependencies:

```bash
uv sync
```

## Prepare the dataset

From the repository root, run:

```bash
uv run python scripts/prepare_data.py
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
HTML stripping or whitespace normalization so chunk offsets refer to the source text.
Do not index queries or qrels. When indexing the deduplicated copy, retain and expand
`original_document_ids` for document-level evaluation against the original qrels.

## Chunk the corpus

Run the separate chunking step after dataset preparation:

```bash
uv run python scripts/chunk_corpus.py
```

Use `--corpus PATH`, `--output-dir PATH`, `--chunk-size INT`, and `--overlap INT` to override defaults.

The default input is `data/processed/fiqa/corpus_deduplicated.jsonl`. Outputs go into
`data/chunks/fiqa/chunks.jsonl` and `chunking_manifest.json`.

Settings in `rag_fiqa.config` specify a default maximum body length of 800 characters and
up to 120 characters of overlap. Short documents stay whole. Longer documents prefer paragraph,
sentence, then word boundaries in the latter half of a window, falling back to the exact character
limit. Overlap is capped at half the emitted chunk length to ensure progress.

Each chunk stores a deterministic `chunk_id`, `parent_document_id`, `original_document_ids`,
`chunk_index`, `title`, `text`, and half-open character offsets (`start_char`, `end_char`)
where `parent_text[start_char:end_char] == text`. During indexing, use `rag_fiqa.chunking.embedding_text(title, text)`
to form the model input and validate against the model's token limit.

## Logging

Application entry points in `scripts/` configure logging to both the console and `logs/app.log`.
The directory is created automatically under the repository root. Logs append across runs
and rotate at 5 MiB, retaining three backups (`app.log.1` through `app.log.3`).
The entire `logs/` directory is ignored by Git.

## Checks

```bash
uv run pytest
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
```

Tests use small local fixtures and do not download datasets or load models.
