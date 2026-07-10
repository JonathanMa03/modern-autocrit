from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AnomalySeverity = Literal[
    "info",
    "warning",
    "high",
]

AnomalyCategory = Literal[
    "numeric_range",
    "unit_mismatch",
    "dosage",
    "terminology",
    "missing_value",
    "attribute_value_mismatch",
    "cross_row_inconsistency",
    "other",
]


class AnomalyFinding(BaseModel):
    trial_id: str
    row_index: int | None = None

    category: AnomalyCategory
    severity: AnomalySeverity

    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    unit: str | None = None

    message: str
    suggested_action: str | None = None

    rule_id: str
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )