"""
Modern OpenAI SDK client wrapper.
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from backend.schemas.settings_schema import OpenAISettings
from backend.services.cost_monitor import CostMonitor
from backend.utils.logging_utils import get_logger


logger = get_logger(__name__)


class OpenAIClient:
    def __init__(
        self,
        settings: OpenAISettings | None = None,
        cost_monitor: CostMonitor | None = None,
        max_retries: int = 3,
    ):
        load_dotenv()

        self.settings = settings or OpenAISettings()
        self.cost_monitor = cost_monitor
        self.max_retries = max_retries

        self.api_key = self.settings.api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY was not found. Add it to .env or application settings."
            )

        self.client = OpenAI(api_key=self.api_key)

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        selected_model = model or self.settings.model

        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=selected_model,
                    input=prompt,
                    temperature=(
                        self.settings.temperature
                        if temperature is None
                        else temperature
                    ),
                    max_output_tokens=max_output_tokens
                    or self.settings.max_output_tokens,
                )

                self._record_usage(response, selected_model)

                return response.output_text

            except Exception as exc:
                error_text = str(exc).lower()

                if "insufficient_quota" in error_text or "exceeded your current quota" in error_text:   
                    logger.error(
                        "OpenAI request failed due to insufficient quota: %s", exc
                    )
                    raise RuntimeError(
                        "OpenAI request failed due to insufficient quota."
                    ) from exc
                logger.warning(
                    "OpenAI request failed (%d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

                if attempt == self.max_retries - 1:
                    raise

                time.sleep(2**attempt)

        raise RuntimeError("OpenAI request failed.")

    def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        return self.generate_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _record_usage(self, response: Any, model: str) -> None:
        if self.cost_monitor is None:
            return

        usage = getattr(response, "usage", None)

        if usage is None:
            return

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

        self.cost_monitor.record_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            label="openai_request",
        )