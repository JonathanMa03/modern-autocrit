"""
Cost monitoring utilities.

Tracks estimated API usage cost during extraction runs.
Pricing values are intentionally configurable so they can be updated
without changing the extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelPricing:
    input_per_1m_tokens: float = 0.0
    output_per_1m_tokens: float = 0.0


DEFAULT_PRICING = {
    "gpt-5": ModelPricing(input_per_1m_tokens=0.0, output_per_1m_tokens=0.0),
    "gpt-5-mini": ModelPricing(input_per_1m_tokens=0.0, output_per_1m_tokens=0.0),
    "gpt-4o": ModelPricing(input_per_1m_tokens=5.00, output_per_1m_tokens=15.00),
    "gpt-4o-mini": ModelPricing(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60),
}


@dataclass
class CostEvent:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    label: str | None = None


@dataclass
class CostMonitor:
    warning_threshold_usd: float = 5.00
    hard_limit_usd: float | None = None
    pricing: dict[str, ModelPricing] = field(default_factory=lambda: DEFAULT_PRICING.copy())
    events: list[CostEvent] = field(default_factory=list)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        price = self.pricing.get(model, ModelPricing())

        input_cost = (input_tokens / 1_000_000) * price.input_per_1m_tokens
        output_cost = (output_tokens / 1_000_000) * price.output_per_1m_tokens

        return input_cost + output_cost

    def record_usage(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        label: str | None = None,
    ) -> CostEvent:
        cost = self.estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        event = CostEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            label=label,
        )

        self.events.append(event)

        if self.hard_limit_usd is not None and self.total_cost > self.hard_limit_usd:
            raise RuntimeError(
                f"Estimated cost ${self.total_cost:.2f} exceeded hard limit "
                f"${self.hard_limit_usd:.2f}."
            )

        return event

    @property
    def total_cost(self) -> float:
        return sum(event.estimated_cost_usd for event in self.events)

    @property
    def total_input_tokens(self) -> int:
        return sum(event.input_tokens for event in self.events)

    @property
    def total_output_tokens(self) -> int:
        return sum(event.output_tokens for event in self.events)

    def warning_reached(self) -> bool:
        return self.total_cost >= self.warning_threshold_usd

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_events": len(self.events),
            "warning_reached": self.warning_reached(),
        }

    def reset(self) -> None:
        self.events.clear()