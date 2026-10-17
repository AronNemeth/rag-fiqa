"""Shared utilities."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag_fiqa.config import LOG_BACKUP_COUNT, LOG_FILE, LOG_FORMAT, LOG_LEVEL, LOG_MAX_BYTES


def setup_logging(level: str = LOG_LEVEL, log_file: Path = LOG_FILE) -> None:
    """Configure console and rotating file logging at an application entry point.

    Replace existing root handlers so repeated setup does not duplicate messages.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(), file_handler],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger without configuring global logging."""
    return logging.getLogger(name)
