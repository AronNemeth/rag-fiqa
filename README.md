# rag-fiqa

A small financial Q&A RAG application using FiQA-2018.

## Project Structure

- `src/rag_fiqa/`: Reusable library modules (`config`, `utils`, `data`, `chunking`, `models`, `indexing`, `retrieval`, `reranking`, `generation`, `evaluation`) with no CLI side effects on import.
- `scripts/`: Orchestrator entry-point scripts for running individual pipeline stages (`download_models.py`, `prepare_data.py`, `build_index.py`, `evaluate.py`).
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
- Pulls the default generation model (`qwen3:1.7b`) through the Ollama CLI (`ollama pull qwen3:1.7b`). Requires the `ollama` CLI and a running Ollama daemon (`ollama serve`).
- Writes a reproducibility manifest to `data/models/models_manifest.json` recording the pinned model IDs, Git revisions, resolved snapshot paths, and Ollama model digest.

Useful options for selective or constrained environments:

```bash
# Download only the embedding model and skip Ollama
uv run python scripts/download_models.py --hf-role embedding --skip-ollama

# Use a custom Hugging Face cache directory
uv run python scripts/download_models.py --cache-dir /path/to/hf_cache
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

## 4. Retrieval and Reranking (`rag_fiqa.retrieval`, `rag_fiqa.reranking`)

Query-time candidate selection uses a two-stage retrieve-and-rerank pipeline:

- **Stage 1 — Dense Bi-Encoder Retrieval (`rag_fiqa.retrieval`):**
  - `open_vector_store()` opens the local Qdrant collection and verifies it is populated.
  - `retrieve_chunks()` normalizes the user question, encodes it with `sentence-transformers/all-MiniLM-L6-v2`, and queries Qdrant for the top `RETRIEVAL_TOP_K_CHUNKS` (`30`) candidate chunks by cosine similarity.
  - `deduplicate_to_parent_documents()` maps chunks back to `original_document_ids` and assigns each parent document its maximum chunk score (`EVAL_TOP_K_DOCUMENTS = 20`) for document-level BEIR evaluation.
- **Stage 2 — Cross-Encoder Reranking (`rag_fiqa.reranking`):**
  - `rerank_chunks()` pairs the question with each retrieved chunk and scores them jointly using `cross-encoder/ms-marco-MiniLM-L6-v2` (`load_reranker_model()`), preserving first-stage `retrieval_score`/`retrieval_rank` while sorting by `rerank_score` to select the top `RERANK_TOP_K_CHUNKS` (`5`) evidence chunks for answer generation.

## 5. Grounded Answer Generation (`rag_fiqa.generation`)

- **Prompting & Evidence Formatting:** `format_evidence_context()` numbers the top reranked chunks (`[1]`, `[2]`, ...) with their parent document IDs. `SYSTEM_PROMPT` instructs the LLM (`qwen3:1.7b` via Ollama, with `think=False`, `temperature=0.0`, `num_ctx=2048`, `num_predict=250`) to treat passages strictly as untrusted evidence, preserve caveats/disagreements, and cite every claim inline.
- **Two-Layer Abstention:** `generate_answer()` abstains before calling Ollama if no chunks are retrieved or if the highest `rerank_score` falls below `MIN_RERANK_SCORE` (`-5.0`). When passages are supplied, the model is instructed to reply with `"Insufficient evidence: ..."` if the context does not answer the question (`is_abstention()`).
- **Citation Validation:** `extract_and_validate_citations()` parses bracketed citations (`[1]..[N]`), maps valid citation IDs back to source chunk metadata and character offsets, and flags any out-of-range citation IDs in `invalid_citations`.

## 6. Evaluation (`rag_fiqa.evaluation`, `scripts/evaluate.py`)

Run evaluation to benchmark retrieval, assess answer generation quality, or test safe abstention:

```bash
# 1. Fast retrieval & reranking evaluation on 100 dev queries (no LLM required)
uv run python scripts/evaluate.py --split dev --num-queries 100

# 2. End-to-end evaluation with Ollama answer generation on a smaller sample
uv run python scripts/evaluate.py --split dev --num-queries 20 --with-generation --generation-sample 10

# 3. Dedicated abstention evaluation on 20 out-of-domain and unanswerable queries
uv run python scripts/evaluate.py --abstention-only
```

How `scripts/evaluate.py` works:

- **Terminal Progress (`tqdm`):** Interactive runs display real-time progress bars with per-query throughput and generation latency (disable with `--no-progress` in CI/batch scripts).
- **Run Timestamping & History:** Outputs are saved to a unique timestamped directory `data/eval/fiqa/<split>/<YYYYMMDD_HHMMSS>/` (or with an optional `--run-name label`), and simultaneously updated in `latest/` to preserve historical run comparisons.
- **Query Selection & Reproducibility:** Samples judged queries with a fixed seed (`--seed 42`) from the specified split (`--split dev` or `--split test`). Queries whose relevant documents are missing from the index are skipped to avoid penalizing unindexed subsets.
- **Retrieval & Reranking Comparison:** Computes standard IR metrics at the parent-document level (`Hit@5`, `Recall@5`, `Recall@20`, `nDCG@10`, `MRR@10`) using maximum chunk score deduplication. Both dense retrieval alone and dense retrieval plus reranking are evaluated on the exact same candidate pool, providing a direct ablation and measuring stage latency in milliseconds per query.
- **Automated Generation & Grounding Checks:** When `--with-generation` is enabled, the generator runs on the top reranked chunks and computes automated quality metrics without requiring a paid LLM judge:
  - `citation_validity_rate`: Proportion of answers where all cited IDs exist among the provided evidence chunks.
  - `citation_presence_rate`: Proportion of non-abstained answers containing at least one inline citation.
  - `gold_citation_rate`: When a known relevant document was retrieved in the context, whether the model actually cited it.
  - `abstention_rate`: Frequency of insufficient-evidence responses.
  - `withholding_check` (optional via `--withholding-check`): Tests whether the model abstains when provided with mismatched context.
- **Abstention Suite (`--abstention-only`):** Evaluates a curated set of 20 unanswerable queries across 4 distinct failure modes (`out_of_domain`, `future_or_post_corpus`, `false_premise`, and `missing_information`). Reports:
  - `pre_llm_abstention_rate`: Queries safely rejected before calling Ollama because candidate chunk scores fell below `MIN_RERANK_SCORE`.
  - `llm_abstention_rate`: Queries reaching the LLM where the model explicitly refused to answer due to insufficient evidence.
  - `total_abstention_rate`: Combined proportion of unanswerable queries safely handled without guessing.
  - `hallucination_rate`: Percentage of ungrounded responses produced despite absent evidence.
- **Artifacts:** Writes `eval_summary.json` (or `eval_abstention_summary.json`), `eval_per_query.json` (or `eval_abstention_records.json`), and `eval_generations.json` (answers, citations, and manual scoring rubric template for `correctness`, `support`, and `clarity`).

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
