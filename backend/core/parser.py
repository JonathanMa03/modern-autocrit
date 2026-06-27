"""
Parser utilities for LLM extraction responses.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.schemas.criteria_schema import EligibilityCriterion
from backend.utils.text_utils import clean_text


REQUIRED_KEYS = {
    "Entity",
    "Attribute",
    "Value",
    "Condition",
    "Sentence",
}


BAD_ATTRIBUTE_PATTERNS = [
    "the sentence does not mention",
    "the sentence does not provide",
    "the text does not mention",
    "no specific diseases",
    "no diseases are mentioned",
    "no treatment names",
]


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    Parse a JSON array from a model response.

    Handles:
    - raw JSON arrays
    - markdown fenced JSON
    - extra text surrounding JSON
    """

    text = text.strip()

    if not text:
        return []

    fenced = re.findall(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced:
        text = fenced[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")

        if start == -1 or end == -1 or end <= start:
            return []

        data = json.loads(text[start : end + 1])

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return []

    return [x for x in data if isinstance(x, dict)]


def is_bad_attribute(attribute: str | None) -> bool:
    if not attribute:
        return True

    attr = attribute.lower().strip()

    if attr in {"", "na", "n/a", "none", "not specified", "not mentioned"}:
        return True

    return any(pattern in attr for pattern in BAD_ATTRIBUTE_PATTERNS)


def normalize_entity(entity: str | None) -> str:
    if not entity:
        return "Other"

    entity = clean_text(entity)

    entity_map = {
        "lab test": "Lab Test",
        "lab": "Lab Test",
        "previous treatment": "Treatment History",
        "treatment history": "Treatment History",
        "contraception": "Contraceptive",
        "contraceptive": "Contraceptive",
        "vitals": "Vital",
        "vital": "Vital",
        "score": "Score",
        "survival": "Survival",
        "diagnosis": "Diagnosis",
        "comorbidity": "Comorbidity",
        "demographic": "Demographic",
        "biomarker": "Biomarker",
        "procedure": "Procedure",
        "medication": "Medication",
        "consent": "Consent",
    }

    return entity_map.get(entity.lower(), entity)


def parse_extraction_response(
    response_text: str,
    trial_id: str,
    criteria_type: str,
    phase: str | None = None,
    url: str | None = None,
) -> list[EligibilityCriterion]:
    """
    Convert an LLM response into EligibilityCriterion objects.
    """

    raw_items = extract_json_array(response_text)

    criteria: list[EligibilityCriterion] = []

    for item in raw_items:
        if not REQUIRED_KEYS.issubset(set(item.keys())):
            continue

        entity = normalize_entity(item.get("Entity"))
        attribute = clean_text(item.get("Attribute"))
        value = clean_text(item.get("Value"))
        condition = clean_text(item.get("Condition"))
        sentence = clean_text(item.get("Sentence"))

        if is_bad_attribute(attribute):
            continue

        if not sentence:
            continue

        criterion = EligibilityCriterion(
            trial_id=trial_id,
            criteria_type=criteria_type,
            entity=entity,
            attribute=attribute,
            value=value or None,
            temporal=None,
            modifier=condition or None,
            source_sentence=sentence,
            phase=phase,
            url=url,
        )

        criteria.append(criterion)

    return criteria