"""Orchestrator script to download and prepare the FiQA dataset."""

import sys

from rag_fiqa.data import download_dataset, prepare_dataset
from rag_fiqa.utils import get_logger, setup_logging

logger = get_logger(__name__)


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
