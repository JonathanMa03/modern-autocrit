"""
Logging utilities for Modern AutoCrit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.utils.paths import LOG_DIR


def get_logger(
    name: str = "modern_autocrit",
    log_file: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_file or LOG_DIR / "modern_autocrit.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger