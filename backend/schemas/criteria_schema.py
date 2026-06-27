"""
Data models for extracted eligibility criteria.

Every extraction produced by the backend should be represented
as one EligibilityCriterion object before exporting to Excel,
CSV, or the GUI.
"""

from typing import Optional

from pydantic import BaseModel


class EligibilityCriterion(BaseModel):
    trial_id: str
    criteria_type: str

    entity: str
    attribute: str
    value: Optional[str] = None

    temporal: Optional[str] = None
    modifier: Optional[str] = None

    source_sentence: Optional[str] = None

    phase: Optional[str] = None
    url: Optional[str] = None


class TrialMetadata(BaseModel):
    nct_id: str

    title: Optional[str] = None

    disease: Optional[str] = None

    indication: Optional[str] = None

    phase: Optional[str] = None

    randomized: Optional[bool] = None

    placebo_controlled: Optional[bool] = None

    url: Optional[str] = None