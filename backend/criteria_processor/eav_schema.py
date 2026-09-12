"""Controlled EAV schema shared by extraction, mapping, and normalization."""

from __future__ import annotations

from typing import Literal

from backend.criteria_processor.text import clean_attribute_key


AllowedEntity = Literal[
    "Demographic",
    "Vital",
    "Score",
    "Contraceptive",
    "Biomarker",
    "Diagnosis",
    "Comorbidity",
    "Previous Treatment",
    "Lab test",
    "Survival",
]

ALLOWED_ENTITIES: tuple[str, ...] = (
    "Demographic",
    "Vital",
    "Score",
    "Contraceptive",
    "Biomarker",
    "Diagnosis",
    "Comorbidity",
    "Previous Treatment",
    "Lab test",
    "Survival",
)

_ENTITY_ALIASES = {
    clean_attribute_key(entity): entity for entity in ALLOWED_ENTITIES
}
_ENTITY_ALIASES.update(
    {
        "demographics": "Demographic",
        "vitals": "Vital",
        "vital sign": "Vital",
        "vital signs": "Vital",
        "performance status": "Score",
        "clinical score": "Score",
        "contraception": "Contraceptive",
        "reproductive status": "Contraceptive",
        "biomarkers": "Biomarker",
        "diagnoses": "Diagnosis",
        "disease": "Diagnosis",
        "condition": "Diagnosis",
        "comorbidities": "Comorbidity",
        "treatment history": "Previous Treatment",
        "previous therapy": "Previous Treatment",
        "prior treatment": "Previous Treatment",
        "medication": "Previous Treatment",
        "drug": "Previous Treatment",
        "procedure": "Previous Treatment",
        "surgery": "Previous Treatment",
        "device": "Previous Treatment",
        "lab": "Lab test",
        "laboratory": "Lab test",
        "laboratory test": "Lab test",
        "prognosis": "Survival",
        "life expectancy": "Survival",
    }
)

SELECTION_STATES: tuple[str, str] = ("included", "excluded")


def canonical_entity(value: str | None) -> str | None:
    """Return an allowed canonical Entity or ``None`` for an unknown label."""

    key = clean_attribute_key(value or "")
    return _ENTITY_ALIASES.get(key)


def is_allowed_entity(value: str | None) -> bool:
    return value in ALLOWED_ENTITIES
