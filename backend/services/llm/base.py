from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLMProvider(ABC):
    provider_name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        raise NotImplementedError

    def test_connection(self) -> bool:
        response = self.generate(
            "Reply with exactly: connected",
            model=self.default_model,
            temperature=0.0,
            max_output_tokens=20,
        )
        return response.text.strip().lower() == "connected"

    @property
    @abstractmethod
    def default_model(self) -> str:
        raise NotImplementedError