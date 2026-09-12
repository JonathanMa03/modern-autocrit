"""Load statistical condition policies without dashboard dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_condition_catalog(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Condition catalog must be a JSON object.")
    builder = payload.get("condition_builder")
    if not isinstance(builder, dict):
        raise ValueError("Condition catalog must define condition_builder.")
    return payload
