from __future__ import annotations

import re


GARBAGE_PATTERNS = [
    r"\bthe sentence does not mention\b",
    r"\bthe sentence does not provide\b",
    r"\bthe text does not mention\b",
    r"\bno specific disease\b",
    r"\bno treatment name\b",
    r"\bnot enough information\b",
    r"\bunknown\b",
    r"\bunspecified\b",
    r"\bn/?a\b",
]

PLACEHOLDER_TERMS = {
    "",
    "none",
    "null",
    "nan",
    "not applicable",
    "not specified",
    "other",
    "criteria",
    "condition",
    "disease",
    "treatment",
    "medication",
    "lab test",
    "test",
}

SUSPICIOUS_PHRASES = [
    "the patient",
    "the participant",
    "the subject",
    "the sentence",
    "this criterion",
    "the text",
]


def clean_term(term: str | None) -> str:
    if term is None:
        return ""

    value = str(term).lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


def check_bad_term(
    attribute: str | None,
    entity: str | None = None,
) -> list[dict]:
    attribute_clean = clean_term(attribute)
    entity_clean = clean_term(entity)

    findings: list[dict] = []

    def add(
        rule_id: str,
        message: str,
        severity: str = "warning",
        confidence: float = 1.0,
    ) -> None:
        findings.append(
            {
                "rule_id": rule_id,
                "severity": severity,
                "category": "terminology",
                "message": message,
                "suggested_action": "Review or normalize the extracted terminology.",
                "unit": None,
                "confidence": confidence,
            }
        )

    if attribute_clean in PLACEHOLDER_TERMS:
        add(
            "TERM_PLACEHOLDER_ATTRIBUTE",
            f"Attribute is too generic or empty: '{attribute}'.",
            severity="high",
        )

    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, attribute_clean):
            add(
                "TERM_GARBAGE_MODEL_RESPONSE",
                f"Attribute appears to contain a model fallback response: '{attribute}'.",
                severity="high",
            )
            break

    if len(attribute_clean) > 160:
        add(
            "TERM_ATTRIBUTE_TOO_LONG",
            (
                f"Attribute is unusually long ({len(attribute_clean)} characters) "
                "and may contain an entire source clause."
            ),
            severity="warning",
            confidence=0.95,
        )

    if len(attribute_clean.split()) > 20:
        add(
            "TERM_ATTRIBUTE_TOO_MANY_WORDS",
            (
                f"Attribute contains {len(attribute_clean.split())} words "
                "and may not be atomic."
            ),
            severity="warning",
            confidence=0.9,
        )

    if any(
        phrase in attribute_clean
        for phrase in SUSPICIOUS_PHRASES
    ):
        add(
            "TERM_SENTENCE_LIKE_ATTRIBUTE",
            f"Attribute appears sentence-like rather than atomic: '{attribute}'.",
            severity="warning",
            confidence=0.85,
        )

    if attribute_clean.endswith((".", ";", ":")):
        add(
            "TERM_TRAILING_SENTENCE_PUNCTUATION",
            f"Attribute ends with sentence punctuation: '{attribute}'.",
            severity="info",
            confidence=0.8,
        )

    if entity_clean in {"", "unknown", "none"}:
        add(
            "TERM_MISSING_OR_UNKNOWN_ENTITY",
            f"Entity is missing or unknown: '{entity}'.",
            severity="warning",
            confidence=0.95,
        )
    
    if entity_clean == "other":
        likely_specific_terms = {
            "pregnancy",
            "pregnancy test",
            "contraception",
            "medication",
            "drug",
            "therapy",
            "chemotherapy",
            "radiotherapy",
            "surgery",
            "infection",
            "malignancy",
            "transplant",
            "consent",
        }

        if any(
            term in attribute_clean
            for term in likely_specific_terms
        ):
            add(
                "TERM_OTHER_ENTITY_MAY_BE_SPECIFIC",
                (
                    f"Entity 'Other' may be too generic for "
                    f"attribute '{attribute}'."
                ),
                severity="info",
                confidence=0.75,
            )

    return findings


def check_value_term(
    value: str | None,
) -> list[dict]:
    value_clean = clean_term(value)

    findings: list[dict] = []

    if value_clean in {
        "",
        "none",
        "null",
        "nan",
        "unknown",
        "not provided",
    }:
        findings.append(
            {
                "rule_id": "TERM_MISSING_VALUE",
                "severity": "warning",
                "category": "missing_value",
                "message": f"Criterion has a missing or uninformative value: '{value}'.",
                "suggested_action": "Review the source sentence for a threshold or status.",
                "unit": None,
                "confidence": 1.0,
            }
        )

    return findings