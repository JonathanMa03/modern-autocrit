"""Configuration used by the browser extraction application."""

from typing import Optional

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = Field(default=4096, ge=1)


class CostSettings(BaseModel):
    enable_tracking: bool = True
    warning_threshold_usd: float = 5.0
    hard_limit_usd: Optional[float] = None


class AppSettings(BaseModel):
    llm: LLMSettings = LLMSettings()
    cost: CostSettings = CostSettings()
