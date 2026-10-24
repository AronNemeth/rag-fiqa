"""Orchestrator script to download pinned Hugging Face and Ollama models."""

import argparse
import sys
from pathlib import Path

from rag_fiqa.config import (
    GENERATION_MODEL_NAME,
    HF_MODELS,
    MODELS_MANIFEST_PATH,
    SUPPORTED_GENERATION_MODELS,
)
from rag_fiqa.models import setup_models
from rag_fiqa.utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for model setup."""
    parser = argparse.ArgumentParser(
        description="Download pinned Hugging Face embedding/reranker models and pull the Ollama generation model."
    )
    parser.add_argument(
        "--hf-role",
        dest="hf_roles",
        action="append",
        choices=tuple(HF_MODELS.keys()),
        help="Hugging Face model role to download (repeatable). Defaults to both 'embedding' and 'reranker'.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Explicit Hugging Face cache directory (defaults to HF_HOME / HF_HUB_CACHE / ~/.cache/huggingface).",
    )
    parser.add_argument(
        "--generation-model",
        choices=SUPPORTED_GENERATION_MODELS,
        default=GENERATION_MODEL_NAME,
        help=f"Ollama generation model to pull (default: {GENERATION_MODEL_NAME}).",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip pulling the Ollama generation model and only download Hugging Face models.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MODELS_MANIFEST_PATH,
        help=f"Path to write the model setup manifest (default: {MODELS_MANIFEST_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging and execute model downloads."""
    args = parse_args()
    setup_logging()
    roles = tuple(args.hf_roles) if args.hf_roles else tuple(HF_MODELS.keys())
    try:
        setup_models(
            roles=roles,
            cache_dir=args.cache_dir,
            generation_model=args.generation_model,
            skip_ollama=args.skip_ollama,
            manifest_path=args.manifest_path,
        )
    except Exception as exc:
        logger.error("Model setup failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
