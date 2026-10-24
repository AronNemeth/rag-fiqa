# rag-fiqa

A small financial Q&A RAG application using FiQA-2018.

## Project Structure

- `src/rag_fiqa/`: Reusable library modules (`config`, `utils`, `data`, `chunking`, `models`, etc.) with no CLI side effects on import.
- `scripts/`: Orchestrator entry-point scripts for running individual pipeline stages (`prepare_data.py`, `download_models.py`, `chunk_corpus.py`, etc.).
- `tests/`: Unit tests using fast local fixtures.

## Setup

From the repository root, install the project and its development dependencies:

```bash
uv sync
```

## Download Models

Download the pinned Hugging Face embedding and reranker models and pull the default Ollama generation model:

```bash
uv run python scripts/download_models.py
```

By default, this script:

- Downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) and reranker model (`cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`) via the Hugging Face Hub API (`snapshot_download`), skipping redundant non-PyTorch weights.
- Stores Hugging Face snapshots in the standard Hugging Face cache (`~/.cache/huggingface/hub` by default, or configurable via `HF_HOME`, `HF_HUB_CACHE`, or `--cache-dir PATH`). Model weights are never copied into the repository.
- Pulls the default generation model (`qwen3:4b`) through the Ollama CLI (`ollama pull qwen3:4b`). Requires the `ollama` CLI and a running Ollama daemon (`ollama serve`).
- Writes a reproducibility manifest to `data/models/models_manifest.json` recording the pinned model IDs, Git revisions, resolved snapshot paths, and Ollama model digest.

Useful options for selective or constrained environments:

```bash
# Download only the embedding model and skip Ollama
uv run python scripts/download_models.py --hf-role embedding --skip-ollama

# Use a custom Hugging Face cache directory
uv run python scripts/download_models.py --cache-dir /path/to/hf_cache

# Pull the smaller fallback generation model explicitly
uv run python scripts/download_models.py --generation-model qwen2.5:3b
```

Repeated runs reuse cached Hugging Face snapshots and Ollama layers without re-downloading.

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
