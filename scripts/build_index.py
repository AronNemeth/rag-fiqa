"""Orchestrator script to chunk the FiQA corpus, compute embeddings, and build the Qdrant index."""

import argparse
import sys
from pathlib import Path

from rag_fiqa.config import (
    CHUNK_CORPUS_PATH,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_BATCH_SIZE,
    INDEX_MANIFEST_PATH,
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
    SUBSET_RANDOM_SEED,
)
from rag_fiqa.indexing import build_index
from rag_fiqa.utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for building the vector index."""
    parser = argparse.ArgumentParser(
        description="Chunk the prepared FiQA corpus, compute embeddings, and populate the local Qdrant vector store."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CHUNK_CORPUS_PATH,
        help=f"Path to the prepared JSONL corpus (default: {CHUNK_CORPUS_PATH}).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=QDRANT_PATH,
        help=f"Directory for the persistent local Qdrant store (default: {QDRANT_PATH}).",
    )
    parser.add_argument(
        "--collection-name",
        default=QDRANT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {QDRANT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=INDEX_MANIFEST_PATH,
        help=f"Path to write the indexing manifest (default: {INDEX_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help=(
            "Optional number of corpus documents to index for fast benchmarking or subset evaluation. "
            "Retains known dev/test relevant documents first and samples the rest using --seed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SUBSET_RANDOM_SEED,
        help=f"Random seed for reproducible corpus subset sampling (default: {SUBSET_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CHUNK_SIZE_TOKENS,
        help=f"Maximum tokens per chunk including [CLS] and [SEP] (default: {CHUNK_SIZE_TOKENS}).",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=CHUNK_OVERLAP_TOKENS,
        help=f"Token overlap between consecutive chunks (default: {CHUNK_OVERLAP_TOKENS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Batch size for embedding generation (default: {EMBEDDING_BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging and build the Qdrant index."""
    args = parse_args()
    setup_logging()
    try:
        build_index(
            corpus_path=args.corpus,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
            manifest_path=args.manifest_path,
            subset_size=args.subset_size,
            seed=args.seed,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            batch_size=args.batch_size,
        )
    except Exception:
        logger.exception("Index build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
