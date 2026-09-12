"""Structural input contracts for criteria processors."""

from __future__ import annotations

from typing import Protocol, Sequence


class OutcomeInput(Protocol):
    outcome_type: str
    measure: str
    description: str


class EligibilityCandidateInput(Protocol):
    nct_id: str
    sex: str
    minimum_age: str
    maximum_age: str
    conditions: Sequence[str]
    eligibility_text: str
    outcomes: Sequence[OutcomeInput]


class ExtractedCriterionInput(Protocol):
    trial_id: str
    criteria_type: str
    entity: str
    attribute: str
    value: str | None
    condition: str | None
    source_text: str
    source_path: str | None
    source_index: int | None
    extraction_method: str
    confidence: float
