"""Adapt Modern AutoCrit's LLM provider to EligCrit's JSON boundary."""

from __future__ import annotations

import json
import re
from typing import Any

from backend.services.llm.base import BaseLLMProvider


class StructuredJsonAdapter:
    def __init__(self, provider: BaseLLMProvider, *, model: str) -> None:
        self.provider = provider
        self.model = model
        self.reasoning_effort = ""
        self.service_tier = None

    def generate_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        schema_instruction = (
            "\n\nReturn one JSON object only. It must satisfy this JSON Schema:\n"
            + json.dumps(output_schema, separators=(",", ":"))
        )
        response = self.provider.generate(
            prompt + schema_instruction,
            model=self.model,
            temperature=0.0,
        )
        return _parse_json_object(response.text)

    def close(self) -> None:
        return None


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    result = json.loads(candidate)
    if not isinstance(result, dict):
        raise ValueError("Structured LLM response must be a JSON object.")
    return result
