"""Dropbox- and Windows-resilient atomic file persistence helpers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4


_RETRYABLE_WINDOWS_ERRORS = {5, 32, 33}


def _retryable_replace_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or (
        getattr(exc, "winerror", None) in _RETRYABLE_WINDOWS_ERRORS
    )


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    max_attempts: int = 10,
    initial_delay_seconds: float = 0.025,
) -> None:
    """Write text through a unique sibling and retry transient replacements."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    temporary.write_text(text, encoding=encoding)
    delay = initial_delay_seconds
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                os.replace(temporary, destination)
                return
            except OSError as exc:
                if (
                    not _retryable_replace_error(exc)
                    or attempt == max_attempts
                ):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.4)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A synchronizer may still hold an abandoned unique temporary
            # file. It is harmless and must not mask the original result.
            pass


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def atomic_write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )
