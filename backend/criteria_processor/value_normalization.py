"""Deterministic normalization for attribute-bound EAV value schemas."""

from __future__ import annotations

import html
import re
from typing import Any

from backend.criteria_processor.numeric_intervals import (
    canonicalize_intervals,
    complement_intervals,
)
from backend.criteria_processor.text import clean_attribute_key


SUPPORTED_VALUE_SCHEMAS = {
    "categorical",
    "numeric_range",
    "duration_range",
    "selection_state",
    "boolean",
    "free_text",
}

_NUMBER = r"-?\d+(?:\.\d+)?"
_UNIT_TOKEN = (
    r"years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?|hrs?|points?"
)
_TIME_UNIT_ALIASES = {
    "year": "year",
    "years": "year",
    "yr": "year",
    "yrs": "year",
    "month": "month",
    "months": "month",
    "mo": "month",
    "mos": "month",
    "week": "week",
    "weeks": "week",
    "wk": "week",
    "wks": "week",
    "day": "day",
    "days": "day",
    "d": "day",
    "hour": "hour",
    "hours": "hour",
    "hr": "hour",
    "hrs": "hour",
}
_DAYS_PER_UNIT = {
    "year": 365.25,
    "month": 30.4375,
    "week": 7.0,
    "day": 1.0,
    "hour": 1.0 / 24.0,
}


def normalize_structured_value(
    *,
    raw_value: str | None,
    attribute_id: str | None,
    canonical_attribute: str | None,
    value_schema: str | None,
    canonical_unit: str | None = None,
    canonical_value: str | None = None,
    context_text: str | None = None,
    criterion_type: str | None = None,
) -> dict[str, Any]:
    """Return a stable, JSON-compatible value representation.

    Attribute mapping decides which schema applies. This function only parses
    the value according to that schema; it never changes the attribute target.
    """

    original = "" if raw_value is None else str(raw_value).strip()
    schema = (value_schema or "free_text").strip().casefold()
    if schema not in SUPPORTED_VALUE_SCHEMAS:
        schema = "free_text"

    base: dict[str, Any] = {
        "schema": schema,
        "original": original,
        "parse_status": "preserved",
        "category": None,
        "lower": None,
        "upper": None,
        "lower_inclusive": None,
        "upper_inclusive": None,
        "unit": canonical_unit or None,
        "lower_unit": canonical_unit or None,
        "upper_unit": canonical_unit or None,
        "normalized_lower": None,
        "normalized_upper": None,
        "normalized_unit": canonical_unit or None,
        "selection_effect": None,
        "assertion": None,
        "eligible_values": [],
        "excluded_values": [],
        "eligible_ranges": [],
        "excluded_ranges": [],
        "intervals": [],
    }

    if schema == "selection_state":
        return _normalize_selection_state(
            base,
            context_text=context_text,
            criterion_type=criterion_type,
        )

    if schema == "categorical":
        category = _normalize_category(
            original=original,
            canonical_value=canonical_value,
            attribute_id=attribute_id,
            canonical_attribute=canonical_attribute,
        )
        base["category"] = category
        base["parse_status"] = "parsed" if category else "needs_value_review"
        _apply_categorical_selection_logic(base, criterion_type)
        return base

    if schema in {"numeric_range", "duration_range"}:
        return _normalize_range(
            base,
            schema=schema,
            canonical_unit=canonical_unit,
            context_text=context_text,
            criterion_type=criterion_type,
        )

    if schema == "boolean":
        key = clean_attribute_key(canonical_value or original)
        if key in {"yes", "true", "present", "required", "allowed"}:
            base["category"] = True
            base["parse_status"] = "parsed"
        elif key in {"no", "false", "absent", "not required", "not allowed"}:
            base["category"] = False
            base["parse_status"] = "parsed"
        else:
            base["parse_status"] = "needs_value_review"
        return base

    return base


def _apply_categorical_selection_logic(
    base: dict[str, Any],
    criterion_type: str | None,
) -> None:
    """Record whether the named category is selected in or out."""

    category = base["category"]
    criterion = clean_attribute_key(criterion_type or "")
    if category is None or criterion not in {"inclusion", "exclusion"}:
        return
    if criterion == "inclusion":
        base["selection_effect"] = "included"
        base["eligible_values"] = [category]
    else:
        base["selection_effect"] = "excluded"
        base["excluded_values"] = [category]


def _normalize_selection_state(
    base: dict[str, Any],
    *,
    context_text: str | None,
    criterion_type: str | None,
) -> dict[str, Any]:
    """Resolve a binary criterion to the eligible population state.

    ``included`` means the Attribute must be present in the eligible
    population. ``excluded`` means the Attribute must be absent. Criterion
    type and assertion polarity are combined so that both "Inclusion:
    non-pregnant" and "Exclusion: pregnancy" normalize to ``excluded`` for the
    Pregnancy Attribute.
    """

    text = "; ".join(
        value
        for value in [str(base["original"]), str(context_text or "")]
        if value.strip()
    )
    assertion = _assertion_polarity(text)
    criterion = clean_attribute_key(criterion_type or "")
    base["assertion"] = assertion

    if assertion == "unknown" or criterion not in {"inclusion", "exclusion"}:
        base["parse_status"] = "needs_value_review"
        return base

    if criterion == "inclusion":
        state = "included" if assertion == "present" else "excluded"
    else:
        state = "excluded" if assertion == "present" else "included"
    base["category"] = state
    base["selection_effect"] = state
    base["parse_status"] = "parsed"
    return base


def _assertion_polarity(value: str) -> str:
    text = _comparison_text(value)
    if not text or text in {"na", "n/a", "unknown", "unclear"}:
        return "unknown"
    negative_patterns = [
        r"\bnon[-\s]?[a-z]+",
        r"\bnot\b",
        r"\bno\b",
        r"\bwithout\b",
        r"\babsence\s+of\b",
        r"\babsent\b",
        r"\bfree\s+(?:from|of)\b",
        r"\bnegative\b",
        r"\bnever\b",
        r"\black\s+of\b",
        r"\bdoes\s+not\s+have\b",
        r"\bdo\s+not\s+have\b",
        r"\bfirst[-\s]ever\b",
        r"\bfirst\s+stroke\s+only\b",
    ]
    if any(re.search(pattern, text) for pattern in negative_patterns):
        return "absent"
    return "present"


def _normalize_category(
    *,
    original: str,
    canonical_value: str | None,
    attribute_id: str | None,
    canonical_attribute: str | None,
) -> str | None:
    attribute_key = clean_attribute_key(
        attribute_id or canonical_attribute or ""
    )
    value_key = clean_attribute_key(canonical_value or original)
    if attribute_key in {"sex", "eligible sex", "gender"}:
        if value_key in {
            "male",
            "males",
            "man",
            "men",
        }:
            return "male"
        if value_key in {
            "female",
            "females",
            "woman",
            "women",
        }:
            return "female"
        if value_key in {
            "all",
            "all gender",
            "all genders",
            "all sex",
            "all sexes",
            "both",
            "both gender",
            "both genders",
            "both sex",
            "both sexes",
            "male or female",
            "female or male",
            "male and female",
            "female and male",
            "men and women",
            "women and men",
            "males and females",
            "females and males",
        }:
            return "all gender"
        return None
    return (canonical_value or original).strip() or None


def _normalize_range(
    base: dict[str, Any],
    *,
    schema: str,
    canonical_unit: str | None,
    context_text: str | None,
    criterion_type: str | None,
) -> dict[str, Any]:
    raw_text = _comparison_text(str(base["original"]))
    contextual_text = raw_text
    if context_text:
        contextual_text = _comparison_text(
            f"{base['original']}; {context_text}"
        )
    text = (
        raw_text
        if _has_explicit_range_expression(raw_text)
        else contextual_text
    )
    detected_unit = _detect_unit(text)
    canonical = _canonical_unit(canonical_unit)
    if not detected_unit and not canonical and schema == "duration_range":
        base["parse_status"] = "needs_value_review"
        return base

    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool | None = None
    upper_inclusive: bool | None = None
    lower_unit: str | None = None
    upper_unit: str | None = None

    discrete_intervals = _discrete_point_intervals(
        _comparison_text(str(base["original"])),
        detected_unit=detected_unit,
        canonical_unit=canonical,
    )
    pair = _range_pair(text) if discrete_intervals is None else None
    if discrete_intervals is not None:
        normalized_unit = canonical or detected_unit
        base.update(
            {
                "parse_status": "parsed",
                "unit": normalized_unit,
                "lower_unit": detected_unit or canonical,
                "upper_unit": detected_unit or canonical,
                "normalized_unit": normalized_unit,
            }
        )
        return _apply_range_selection_logic(
            base,
            criterion_type,
            context_text=context_text,
            asserted_intervals=discrete_intervals,
        )
    if pair is not None:
        lower, upper, lower_unit, upper_unit = pair
        lower_inclusive = True
        upper_inclusive = True
    else:
        lower_match = re.search(
            rf"(?:>=|=>|≥|at\s+least|minimum(?:\s+of)?|min(?:imum)?|"
            rf"no\s+less\s+than)(?:\s+[a-z-]+){{0,2}}\s*({_NUMBER})",
            text,
        )
        strict_lower_match = re.search(
            rf"(?:>|greater\s+than|older\s+than|more\s+than|over)\s*"
            rf"({_NUMBER})",
            text,
        )
        suffix_lower_match = re.search(
            rf"({_NUMBER})(?:\s+[a-z]+)?\s+or\s+older",
            text,
        )
        suffix_minimum_match = re.search(
            rf"({_NUMBER})\s*({_UNIT_TOKEN})?\s+or\s+"
            rf"(?:more|greater|higher|above)",
            text,
        )
        upper_match = re.search(
            rf"(?:<=|=<|≤|at\s+most|maximum(?:\s+of)?|max(?:imum)?|"
            rf"no\s+more\s+than|not\s+more\s+than)"
            rf"(?:\s+[a-z-]+){{0,2}}\s*({_NUMBER})",
            text,
        )
        strict_upper_match = re.search(
            rf"(?:<|less\s+than|younger\s+than|under)\s*({_NUMBER})",
            text,
        )
        suffix_upper_match = re.search(
            rf"({_NUMBER})(?:\s+[a-z]+)?\s+or\s+younger",
            text,
        )
        suffix_maximum_match = re.search(
            rf"({_NUMBER})\s*({_UNIT_TOKEN})?\s+or\s+"
            rf"(?:less|fewer|lower|below)",
            text,
        )
        recent_window_match = re.search(
            rf"(?:within|during|in)?\s*(?:the\s+)?"
            rf"(?:last|past|previous)\s+({_NUMBER})\s*({_UNIT_TOKEN})",
            text,
        )
        survival_upper_match = re.search(
            rf"(?:not\s+(?:expected|likely)\s+to\s+survive|"
            rf"unlikely\s+to\s+survive)\s+(?:for\s+)?"
            rf"({_NUMBER})\s*({_UNIT_TOKEN})",
            text,
        )
        if lower_match:
            lower = float(lower_match.group(1))
            lower_inclusive = True
            lower_unit = _unit_after(text, lower_match.end(1))
        elif strict_lower_match:
            lower = float(strict_lower_match.group(1))
            lower_inclusive = False
            lower_unit = _unit_after(text, strict_lower_match.end(1))
        elif suffix_lower_match:
            lower = float(suffix_lower_match.group(1))
            lower_inclusive = True
            lower_unit = _unit_after(text, suffix_lower_match.end(1))
        elif suffix_minimum_match:
            lower = float(suffix_minimum_match.group(1))
            lower_inclusive = True
            lower_unit = _canonical_detected_unit(
                suffix_minimum_match.group(2)
            )
        if upper_match:
            upper = float(upper_match.group(1))
            upper_inclusive = True
            upper_unit = _unit_after(text, upper_match.end(1))
        elif strict_upper_match:
            upper = float(strict_upper_match.group(1))
            upper_inclusive = False
            upper_unit = _unit_after(text, strict_upper_match.end(1))
        elif suffix_upper_match:
            upper = float(suffix_upper_match.group(1))
            upper_inclusive = True
            upper_unit = _unit_after(text, suffix_upper_match.end(1))
        elif suffix_maximum_match:
            upper = float(suffix_maximum_match.group(1))
            upper_inclusive = True
            upper_unit = _canonical_detected_unit(
                suffix_maximum_match.group(2)
            )
        elif recent_window_match:
            upper = float(recent_window_match.group(1))
            upper_inclusive = True
            upper_unit = _canonical_detected_unit(
                recent_window_match.group(2)
            )
        elif survival_upper_match:
            upper = float(survival_upper_match.group(1))
            upper_inclusive = False
            upper_unit = _canonical_detected_unit(
                survival_upper_match.group(2)
            )

    if lower is None and upper is None:
        base["parse_status"] = "needs_value_review"
        return base

    lower_unit = lower_unit or upper_unit or detected_unit or canonical
    upper_unit = upper_unit or lower_unit or detected_unit or canonical
    normalized_unit = canonical or (
        "day" if schema == "duration_range" else lower_unit
    )
    normalized_lower = _convert_unit(
        lower,
        lower_unit,
        normalized_unit,
    )
    normalized_upper = _convert_unit(
        upper,
        upper_unit,
        normalized_unit,
    )
    if schema == "duration_range" and not canonical:
        normalized_unit = "day"
        normalized_lower = _convert_unit(
            lower,
            lower_unit,
            normalized_unit,
        )
        normalized_upper = _convert_unit(
            upper,
            upper_unit,
            normalized_unit,
        )

    base.update(
        {
            "parse_status": "parsed",
            "lower": lower,
            "upper": upper,
            "lower_inclusive": lower_inclusive,
            "upper_inclusive": upper_inclusive,
            "unit": normalized_unit,
            "lower_unit": lower_unit,
            "upper_unit": upper_unit,
            "normalized_lower": normalized_lower,
            "normalized_upper": normalized_upper,
            "normalized_unit": normalized_unit,
        }
    )
    return _apply_range_selection_logic(
        base,
        criterion_type,
        context_text=context_text,
    )


def _apply_range_selection_logic(
    base: dict[str, Any],
    criterion_type: str | None,
    *,
    context_text: str | None,
    asserted_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    unit = base["normalized_unit"]
    if asserted_intervals is None:
        asserted_intervals = [
            {
                "lower": base["normalized_lower"],
                "upper": base["normalized_upper"],
                "lower_inclusive": base["lower_inclusive"],
                "upper_inclusive": base["upper_inclusive"],
                "unit": unit,
            }
        ]
    asserted_intervals = canonicalize_intervals(
        asserted_intervals,
        unit=unit,
    )
    criterion = clean_attribute_key(criterion_type or "")
    if criterion not in {"inclusion", "exclusion"}:
        base["intervals"] = canonicalize_intervals(
            asserted_intervals,
            unit=unit,
            include_unit=False,
        )
        return base

    complement = complement_intervals(asserted_intervals, unit=unit)
    selection_text = _comparison_text(
        f"{base['original']}; {context_text or ''}"
    )
    outside_interval = bool(
        re.search(
            r"\boutside\b|\bnot\s+(?:between|within)\b",
            selection_text,
        )
    )
    asserted_ranges = complement if outside_interval else asserted_intervals
    opposite_ranges = asserted_intervals if outside_interval else complement
    if criterion == "inclusion":
        base["selection_effect"] = "included"
        base["eligible_ranges"] = asserted_ranges
        base["excluded_ranges"] = opposite_ranges
    else:
        base["selection_effect"] = "excluded"
        base["eligible_ranges"] = opposite_ranges
        base["excluded_ranges"] = asserted_ranges
    base["intervals"] = canonicalize_intervals(
        base["eligible_ranges"],
        unit=unit,
        include_unit=False,
    )
    return base


def _discrete_point_intervals(
    text: str,
    *,
    detected_unit: str | None,
    canonical_unit: str | None,
) -> list[dict[str, Any]] | None:
    """Parse disjoint exact values such as ``=0 or 6`` as point intervals."""

    if not re.search(r"\bor\b|,", text):
        return None
    if re.search(
        r"\b(?:less|more|greater|higher|lower|older|younger|under|over|"
        r"above|below|between|from|to|through|outside|within)\b",
        text,
    ):
        return None
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|\bor\b)\s*", text)
        if part.strip()
    ]
    if len(parts) < 2:
        return None

    parsed: list[tuple[float, str | None]] = []
    for part in parts:
        match = re.fullmatch(
            rf"(?:=|equals?(?:\s+to)?)?\s*({_NUMBER})"
            rf"\s*({_UNIT_TOKEN})?",
            part,
        )
        if match is None:
            return None
        parsed.append(
            (
                float(match.group(1)),
                _canonical_detected_unit(match.group(2)),
            )
        )

    normalized_unit = canonical_unit or detected_unit
    intervals = []
    for value, source_unit in parsed:
        normalized_value = _convert_unit(
            value,
            source_unit or detected_unit or canonical_unit,
            normalized_unit,
        )
        intervals.append(
            {
                "lower": normalized_value,
                "upper": normalized_value,
                "lower_inclusive": True,
                "upper_inclusive": True,
                "unit": normalized_unit,
            }
        )
    return canonicalize_intervals(intervals, unit=normalized_unit)


def _comparison_text(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"\\([<>])", r"\1", text)
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _has_explicit_range_expression(text: str) -> bool:
    """Whether the extracted Value should be parsed without source context."""

    if re.search(_NUMBER, text):
        return True
    if _range_pair(text) is not None:
        return True
    return bool(
        re.search(
            r"(?:[<>≤≥]=?|(?:^|\s)=\s*-?\d)|"
            r"\b(?:at\s+least|at\s+most|minimum|maximum|"
            r"no\s+less\s+than|no\s+more\s+than|not\s+more\s+than|"
            r"greater\s+than|less\s+than|older\s+than|younger\s+than|"
            r"under|over|above|below|outside|within|"
            r"last|past|previous)\b|"
            rf"\b{_NUMBER}\s*(?:{_UNIT_TOKEN}\s*)?"
            r"\bor\s+(?:older|younger|more|less|greater|higher|"
            r"lower|above|below)\b",
            text,
        )
    )


def _range_pair(
    text: str,
) -> tuple[float, float, str | None, str | None] | None:
    prefixed_units = re.search(
        rf"\bfrom\s+({_UNIT_TOKEN})\s+({_NUMBER})\s+"
        rf"(?:to|through)\s+({_UNIT_TOKEN})\s+({_NUMBER})",
        text,
    )
    if prefixed_units:
        return (
            float(prefixed_units.group(2)),
            float(prefixed_units.group(4)),
            _canonical_detected_unit(prefixed_units.group(1)),
            _canonical_detected_unit(prefixed_units.group(3)),
        )
    patterns = [
        rf"\bfrom\s+({_NUMBER})\s*({_UNIT_TOKEN})?\s+"
        rf"(?:to|through)\s+({_NUMBER})\s*({_UNIT_TOKEN})?",
        rf"\bbetween\s+({_NUMBER})\s*({_UNIT_TOKEN})?\s+"
        rf"and\s+({_NUMBER})\s*({_UNIT_TOKEN})?",
        rf"(?<![a-z0-9.])({_NUMBER})\s*({_UNIT_TOKEN})?\s*"
        rf"(?:-|to|through)\s*({_NUMBER})\s*({_UNIT_TOKEN})?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            lower_unit = _canonical_detected_unit(match.group(2))
            upper_unit = _canonical_detected_unit(match.group(4))
            lower_unit = lower_unit or upper_unit
            upper_unit = upper_unit or lower_unit
            return (
                float(match.group(1)),
                float(match.group(3)),
                lower_unit,
                upper_unit,
            )
    return None


def _unit_after(text: str, position: int) -> str | None:
    match = re.match(rf"\s*({_UNIT_TOKEN})", text[position:])
    return _canonical_detected_unit(match.group(1) if match else None)


def _canonical_detected_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return _TIME_UNIT_ALIASES.get(unit.casefold(), "point")


def _detect_unit(text: str) -> str | None:
    for raw_unit, canonical in _TIME_UNIT_ALIASES.items():
        if re.search(rf"(?<![a-z]){re.escape(raw_unit)}(?![a-z])", text):
            return canonical
    if re.search(r"\b(?:point|points|score|scores)\b", text):
        return "point"
    return None


def _canonical_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    key = clean_attribute_key(unit)
    return _TIME_UNIT_ALIASES.get(key, key or None)


def _convert_unit(
    value: float | None,
    source_unit: str | None,
    target_unit: str | None,
) -> float | None:
    if value is None:
        return None
    if not source_unit or not target_unit or source_unit == target_unit:
        return value
    if source_unit not in _DAYS_PER_UNIT or target_unit not in _DAYS_PER_UNIT:
        return value
    converted = (
        value * _DAYS_PER_UNIT[source_unit] / _DAYS_PER_UNIT[target_unit]
    )
    return round(converted, 6)
