"""Structured logging setup using the rich library."""

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure structured logging with console (rich) and optional file output."""
    logger = logging.getLogger("tabular_benchmark")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "tabular_benchmark") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
