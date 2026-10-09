"""Validation and downstream-export readiness checks for structured criteria."""

from __future__ import annotations

import re
from typing import Any


COMPARISON_LANGUAGE = re.compile(
    r"(?:>=|<=|>|<|=|≥|≤|\b(?:greater|less|more|younger|older|above|below|between)\b)",
    re.I,
)
NEGATION_LANGUAGE = re.compile(
    r"\b(?:no|not|without|absence of|negative for|free of|never)\b", re.I
)
TEMPORAL_LANGUAGE = re.compile(
    r"\b(?:within|prior to|before|after|during|for at least|past|previous|last|history of)\b",
    re.I,
)
REPETITION_LANGUAGE = re.compile(
    r"\b(?:once|twice|\d+|one|two|three|four|five)\s+"
    r"(?:occasions?|times?|measurements?|visits?)\b",
    re.I,
)
UNIT_LANGUAGE = re.compile(
    r"\b(?:years?|months?|weeks?|days?|hours?|mmhg|mg/dl|mmol/l|umol/l|µmol/l|"
    r"kg/m2|kg/m²|iu/l|u/l|ml/min|cells?/ul|seconds?|sec|msec|ms|%)\b",
    re.I,
)
COMPOUND_LANGUAGE = re.compile(r"\b(?:and|or|except|unless)\b|;", re.I)


def _issue(code: str, severity: str, message: str, field: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "field": field, "message": message}


def validate_criterion(row: dict[str, Any]) -> list[dict[str, str]]:
    """Return machine-readable structural and source-consistency findings."""
    issues: list[dict[str, str]] = []
    source = str(row.get("source") or "")
    comparator = row.get("comparator") or None
    value = row.get("value")
    lower, upper = row.get("lower_bound"), row.get("upper_bound")

    if not str(row.get("entity") or "").strip():
        issues.append(_issue("ENTITY_MISSING", "error", "A normalized entity is required.", "entity"))
    if not source.strip():
        issues.append(_issue("SOURCE_MISSING", "error", "Verbatim source evidence is required.", "source"))
    if row.get("criterion_type") not in {"inclusion", "exclusion"}:
        issues.append(_issue("CRITERION_TYPE_INVALID", "error", "Criterion type must be inclusion or exclusion.", "criterion_type"))

    if comparator and value in {None, ""}:
        issues.append(_issue("COMPARATOR_WITHOUT_VALUE", "error", "Comparator requires a value.", "value"))
    if COMPARISON_LANGUAGE.search(source) and not comparator and lower is None and upper is None:
        issues.append(_issue("COMPARISON_NOT_STRUCTURED", "error", "Source contains comparison language but no comparator or interval.", "comparator"))
    if lower is not None and upper is not None and float(lower) > float(upper):
        issues.append(_issue("INTERVAL_REVERSED", "error", "Lower interval bound exceeds upper bound.", "lower_bound"))
    if lower is None and row.get("lower_inclusive") is not None:
        issues.append(_issue("LOWER_INCLUSIVITY_WITHOUT_BOUND", "error", "Lower inclusivity requires a lower bound.", "lower_inclusive"))
    if upper is None and row.get("upper_inclusive") is not None:
        issues.append(_issue("UPPER_INCLUSIVITY_WITHOUT_BOUND", "error", "Upper inclusivity requires an upper bound.", "upper_inclusive"))

    if NEGATION_LANGUAGE.search(source) and not row.get("negated"):
        # "not due to" modifies etiology and must not negate the main entity.
        if not re.search(r"\bnot\s+due\s+to\b", source, re.I):
            issues.append(_issue("NEGATION_UNCAPTURED", "error", "Source contains negation that is not represented.", "negated"))
    if row.get("negated") and str(value).casefold() not in {"absent", "0", "false", "no"}:
        issues.append(_issue("NEGATION_VALUE_CONFLICT", "error", "Negated categorical criteria should encode absence.", "value"))

    if comparator and UNIT_LANGUAGE.search(source) and not str(row.get("unit") or "").strip():
        issues.append(_issue("UNIT_UNCAPTURED", "error", "Source contains a measurement unit that is not represented.", "unit"))
    if TEMPORAL_LANGUAGE.search(source) and not row.get("temporal"):
        issues.append(_issue("TEMPORAL_UNCAPTURED", "error", "Source contains a temporal restriction that is not represented.", "temporal"))
    if REPETITION_LANGUAGE.search(source) and not row.get("repetition"):
        issues.append(_issue("REPETITION_UNCAPTURED", "error", "Source contains repetition that is not represented.", "repetition"))

    parent_size = int(row.get("parent_atom_count") or 1)
    operator = row.get("logical_operator") or "standalone"
    if parent_size > 1 and operator not in {"and", "or"}:
        issues.append(_issue("COMPOUND_SCOPE_MISSING", "error", "Multiple atoms require an AND or OR scope.", "logical_operator"))
    if parent_size == 1 and COMPOUND_LANGUAGE.search(source) and operator == "standalone":
        issues.append(_issue("POSSIBLE_UNSPLIT_COMPOUND", "warning", "Source may contain more than one clinical requirement.", "source"))
    return issues


def readiness(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a criterion is approved for validated export and why not."""
    reasons = [issue["code"] for issue in row.get("validation_issues", []) if issue.get("severity") == "error"]
    if row.get("human_review_status") not in {"accepted", "corrected"}:
        reasons.append("HUMAN_REVIEW_REQUIRED")
    if (
        not row.get("protocol_id")
        or not row.get("source_version")
        or not row.get("source_retrieved_at")
        or not row.get("protocol_hash")
    ):
        reasons.append("PROVENANCE_INCOMPLETE")
    return not reasons, list(dict.fromkeys(reasons))
