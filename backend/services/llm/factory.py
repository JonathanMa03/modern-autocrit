from __future__ import annotations

from backend.schemas.settings_schema import LLMSettings
from backend.services.cost_monitor import CostMonitor
from backend.services.llm.base import BaseLLMProvider
from backend.services.llm.openai_provider import OpenAIProvider


def create_llm_provider(
    settings: LLMSettings,
    cost_monitor: CostMonitor | None = None,
) -> BaseLLMProvider:
    provider = settings.provider.lower().strip()

    if provider == "openai":
        return OpenAIProvider(
            api_key=settings.api_key,
            cost_monitor=cost_monitor,
        )

    if provider == "anthropic":
        from backend.services.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.api_key,
            cost_monitor=cost_monitor,
        )

    if provider == "gemini":
        from backend.services.llm.gemini_provider import GeminiProvider

        return GeminiProvider(
            api_key=settings.api_key,
            cost_monitor=cost_monitor,
        )

    if provider == "ollama":
        from backend.services.llm.ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=settings.base_url,
            cost_monitor=cost_monitor,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.provider}")