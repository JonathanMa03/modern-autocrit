from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CoverageStatus = Literal[
    "covered",
    "partially_covered",
    "uncovered",
]

SupportStatus = Literal[
    "supported",
    "weakly_supported",
    "unsupported",
]


class ClauseValidation(BaseModel):
    trial_id: str
    criteria_type: str

    source_clause: str
    coverage_status: CoverageStatus
    matched_row_indices: list[int] = []

    missing_concepts: list[str] = []
    explanation: str = ""

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class ExtractionValidation(BaseModel):
    trial_id: str
    row_index: int

    support_status: SupportStatus
    source_clause: str | None = None
    explanation: str = ""

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class ValidationMetrics(BaseModel):
    trial_id: str | None = None

    total_source_clauses: int = 0
    covered_clauses: int = 0
    partially_covered_clauses: int = 0
    uncovered_clauses: int = 0

    total_extracted_rows: int = 0
    supported_rows: int = 0
    weakly_supported_rows: int = 0
    unsupported_rows: int = 0

    clause_coverage_rate: float = 0.0
    full_coverage_rate: float = 0.0
    unsupported_extraction_rate: float = 0.0
    missing_value_rate: float = 0.0
    atomicity_issue_rate: float = 0.0