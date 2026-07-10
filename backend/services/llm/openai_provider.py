from __future__ import annotations

import os
import time

from openai import OpenAI
from pathlib import Path

from dotenv import load_dotenv
from backend.services.cost_monitor import CostMonitor
from backend.services.llm.base import BaseLLMProvider, LLMResponse
from backend.utils.logging_utils import get_logger

# from dotenv import load_dotenv

# load_dotenv()

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class OpenAIProvider(BaseLLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        cost_monitor: CostMonitor | None = None,
        max_retries: int = 3,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(api_key=resolved_key)
        self.cost_monitor = cost_monitor
        self.max_retries = max_retries

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

                if self.cost_monitor is not None:
                    self.cost_monitor.record_usage(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        label="openai_request",
                    )

                return LLMResponse(
                    text=response.output_text,
                    provider=self.provider_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except Exception as exc:
                error_text = str(exc).lower()

                if (
                    "insufficient_quota" in error_text
                    or "exceeded your current quota" in error_text
                ):
                    raise RuntimeError(
                        "OpenAI quota exhausted. Check billing or API credits."
                    ) from exc

                if attempt == self.max_retries - 1:
                    raise

                logger.warning(
                    "OpenAI request failed (%d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                time.sleep(2**attempt)

        raise RuntimeError("OpenAI request failed.")