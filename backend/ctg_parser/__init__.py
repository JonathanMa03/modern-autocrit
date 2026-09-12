"""General ClinicalTrials.gov parser package."""

from backend.ctg_parser.models import (
    ClinicalTrialsCandidate,
    ClinicalTrialsIntervention,
    ClinicalTrialsOutcome,
    ClinicalTrialsSearchQuery,
)
from backend.ctg_parser.parser import CTG_API_BASE, parse_ctgov_study

__all__ = [
    "CTG_API_BASE",
    "ClinicalTrialsCandidate",
    "ClinicalTrialsIntervention",
    "ClinicalTrialsOutcome",
    "ClinicalTrialsSearchQuery",
    "parse_ctgov_study",
]
