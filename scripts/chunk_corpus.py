"""Orchestrator script to split the prepared FiQA corpus into character chunks."""

import argparse
import sys
from pathlib import Path

from rag_fiqa.chunking import chunk_corpus
from rag_fiqa.config import CHUNK_CORPUS_PATH, CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, CHUNKS_DIR
from rag_fiqa.utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Parse CLI arguments, configure logging, and chunk the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CHUNK_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_CHARS)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_CHARS)
    args = parser.parse_args()

    setup_logging()
    try:
        chunk_corpus(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except Exception:
        logger.exception("Corpus chunking failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
