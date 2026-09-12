"""General ClinicalTrials.gov data contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClinicalTrialsSearchQuery(BaseModel):
    """One externally configured ClinicalTrials.gov search query."""

    query_id: str
    query_condition: str | None = None
    query_term: str
    rationale: str = ""


class ClinicalTrialsOutcome(BaseModel):
    """One parsed ClinicalTrials.gov outcome row."""

    outcome_type: Literal["primary", "secondary", "other"]
    measure: str
    description: str = ""
    time_frame: str = ""


class ClinicalTrialsIntervention(BaseModel):
    """One parsed intervention with arm linkage preserved."""

    intervention_type: str = ""
    name: str = ""
    description: str = ""
    arm_labels: list[str] = Field(default_factory=list)


class ClinicalTrialsCandidate(BaseModel):
    """Target-agnostic parsed ClinicalTrials.gov study record."""

    nct_id: str
    brief_title: str = ""
    official_title: str = ""
    acronym: str = ""
    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    interventions: list[ClinicalTrialsIntervention] = Field(
        default_factory=list
    )
    arm_groups: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    overall_status: str = ""
    study_type: str = ""
    allocation: str = ""
    intervention_model: str = ""
    primary_purpose: str = ""
    masking: str = ""
    enrollment_count: int | None = None
    enrollment_type: str = ""
    sex: str = ""
    minimum_age: str = ""
    maximum_age: str = ""
    healthy_volunteers: bool | None = None
    eligibility_text: str = ""
    outcomes: list[ClinicalTrialsOutcome] = Field(default_factory=list)
    lead_sponsor: str = ""
    start_date: str = ""
    completion_date: str = ""
    last_update_post_date: str = ""
    countries: list[str] = Field(default_factory=list)
    source_url: str
    api_url: str
    matched_query_ids: list[str] = Field(default_factory=list)
