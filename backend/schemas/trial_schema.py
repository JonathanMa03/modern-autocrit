"""
Trial-level data models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TrialMetadata(BaseModel):
    nct_id: str
    title: Optional[str] = None
    phase: Optional[str] = None
    conditions: list[str] = Field(default_factory=list)
    url: Optional[str] = None

    study_type: Optional[str] = None
    allocation: Optional[str] = None
    intervention_model: Optional[str] = None
    masking: Optional[str] = None
    primary_purpose: Optional[str] = None

    randomized: Optional[bool] = None
    placebo_controlled: Optional[bool] = None

    lead_sponsor: Optional[str] = None
    sponsor_class: Optional[str] = None


class TrialRecord(BaseModel):
    metadata: TrialMetadata
    eligibility_text: str = ""
    inclusion_text: str = ""
    exclusion_text: str = ""