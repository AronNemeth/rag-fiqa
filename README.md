# rag-fiqa

A small financial Q&A RAG application using FiQA-2018.

## Project Structure

- `src/rag_fiqa/`: Reusable library modules (`config`, `utils`, `data`, `chunking`, `models`, `indexing`) with no CLI side effects on import.
- `scripts/`: Orchestrator entry-point scripts for running individual pipeline stages (`download_models.py`, `prepare_data.py`, `build_index.py`).
- `tests/`: Unit tests using fast local fixtures.

## Setup

From the repository root, install the project and its development dependencies:

```bash
uv sync
```

## 1. Download Models

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

## 2. Prepare the Dataset

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

## 3. Build the Vector Index (Chunking + Embedding + Qdrant)

After downloading the embedding model and preparing the dataset, build the local Qdrant vector store:

```bash
# Test indexing speed on a reproducible 500-document subset
uv run python scripts/build_index.py --subset-size 500

# Build the index over the full deduplicated corpus (57,598 documents)
uv run python scripts/build_index.py
```

How `scripts/build_index.py` works:

- **Optional Reproducible Subset (`--subset-size N`):** When specified, retains known positively judged documents from `qrels/{dev,test}.tsv` first and fills the remaining slots with a seeded random sample (`--seed 42`) of other corpus documents.
- **In-Memory Token-Based Chunking (`rag_fiqa.chunking`):** Uses the cached `sentence-transformers/all-MiniLM-L6-v2` tokenizer (`CHUNK_SIZE_TOKENS = 256`, `CHUNK_OVERLAP_TOKENS = 32`). Short documents (`~80.8%` of FiQA) stay whole; longer documents split at paragraph, sentence, then word boundaries while preserving exact character offsets (`start_char`, `end_char`).
- **Embedding & Qdrant Storage (`rag_fiqa.indexing`):** Computes normalized 384-dimensional cosine embeddings and writes chunks and metadata payloads (`chunk_id`, `parent_document_id`, `original_document_ids`, `chunk_index`, `title`, `text`, `token_count`, `start_char`, `end_char`) into persistent local Qdrant at `data/index/fiqa/qdrant` and saves timings and counts to `data/index/fiqa/index_manifest.json`.

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
