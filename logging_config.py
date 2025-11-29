"""Centralized logging configuration for the project."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

_LOGGING_INITIALIZED = False
_DEFAULT_LOG_FILE = Path("log/app.log")


def _resolve_level(level: Optional[Union[str, int]]) -> int:
    """Convert user-provided level into a logging level constant."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    env_level = os.getenv("LOG_LEVEL", "INFO")
    return logging._nameToLevel.get(env_level.upper(), logging.INFO)


def setup_logging(level: Optional[Union[str, int]] = None) -> None:
    """Configure root logging once."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    log_level = _resolve_level(level)

    log_dir = _DEFAULT_LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        _DEFAULT_LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    setup_logging()
    return logging.getLogger(name)
