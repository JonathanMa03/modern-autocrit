"""Canonical interval-set operations for numerical EAV values."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


Interval = dict[str, Any]


def canonicalize_intervals(
    intervals: Iterable[Mapping[str, Any]],
    *,
    unit: str | None = None,
    include_unit: bool = True,
) -> list[Interval]:
    """Validate, sort, and merge a union of numerical intervals."""

    cleaned: list[Interval] = []
    for candidate in intervals:
        lower = _number(candidate.get("lower"))
        upper = _number(candidate.get("upper"))
        lower_inclusive = (
            bool(candidate.get("lower_inclusive"))
            if lower is not None
            else None
        )
        upper_inclusive = (
            bool(candidate.get("upper_inclusive"))
            if upper is not None
            else None
        )
        if lower is not None and upper is not None:
            if lower > upper:
                continue
            if lower == upper and not (
                lower_inclusive and upper_inclusive
            ):
                continue
        interval_unit = _text(candidate.get("unit")) or unit
        interval = {
            "lower": lower,
            "upper": upper,
            "lower_inclusive": lower_inclusive,
            "upper_inclusive": upper_inclusive,
        }
        if include_unit:
            interval["unit"] = interval_unit
        cleaned.append(interval)

    cleaned.sort(key=_interval_sort_key)
    merged: list[Interval] = []
    for interval in cleaned:
        if not merged or not _can_merge(merged[-1], interval):
            merged.append(interval)
            continue
        merged[-1] = _merge_pair(merged[-1], interval)
    return merged


def complement_intervals(
    intervals: Iterable[Mapping[str, Any]],
    *,
    unit: str | None = None,
) -> list[Interval]:
    """Return the complement of an interval union over the real line."""

    normalized = canonicalize_intervals(intervals, unit=unit)
    if not normalized:
        return [
            {
                "lower": None,
                "upper": None,
                "lower_inclusive": None,
                "upper_inclusive": None,
                "unit": unit,
            }
        ]

    complement: list[Interval] = []
    first = normalized[0]
    if first["lower"] is not None:
        complement.append(
            {
                "lower": None,
                "upper": first["lower"],
                "lower_inclusive": None,
                "upper_inclusive": not bool(first["lower_inclusive"]),
                "unit": unit or first.get("unit"),
            }
        )

    for left, right in zip(normalized, normalized[1:]):
        complement.append(
            {
                "lower": left["upper"],
                "upper": right["lower"],
                "lower_inclusive": (
                    not bool(left["upper_inclusive"])
                    if left["upper"] is not None
                    else None
                ),
                "upper_inclusive": (
                    not bool(right["lower_inclusive"])
                    if right["lower"] is not None
                    else None
                ),
                "unit": unit or left.get("unit") or right.get("unit"),
            }
        )

    last = normalized[-1]
    if last["upper"] is not None:
        complement.append(
            {
                "lower": last["upper"],
                "upper": None,
                "lower_inclusive": not bool(last["upper_inclusive"]),
                "upper_inclusive": None,
                "unit": unit or last.get("unit"),
            }
        )
    return canonicalize_intervals(complement, unit=unit)


def intersect_interval_sets(
    left: Iterable[Mapping[str, Any]],
    right: Iterable[Mapping[str, Any]],
    *,
    unit: str | None = None,
) -> list[Interval]:
    """Return the mathematical intersection of two interval unions."""

    left_set = canonicalize_intervals(left, unit=unit)
    right_set = canonicalize_intervals(right, unit=unit)
    intersections: list[Interval] = []
    for left_interval in left_set:
        for right_interval in right_set:
            intersection = _intersect_pair(
                left_interval,
                right_interval,
                unit=unit,
            )
            if intersection is not None:
                intersections.append(intersection)
    return canonicalize_intervals(intersections, unit=unit)


def compact_interval_json(
    *,
    schema: str,
    intervals: Iterable[Mapping[str, Any]],
    unit: str | None,
    aggregation_status: str = "parsed",
) -> dict[str, Any]:
    """Return the public JSON contract for a numerical eligible set."""

    normalized = canonicalize_intervals(
        intervals,
        unit=unit,
        include_unit=False,
    )
    return {
        "schema": schema,
        "unit": unit,
        "intervals": normalized,
        "aggregation_status": aggregation_status,
    }


def aggregate_numeric_constraints(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Intersect eligible intervals for one trial and canonical Attribute."""

    source_rows = list(rows)
    schemas = {
        _text(row.get("value_schema"))
        for row in source_rows
        if _text(row.get("value_schema"))
    }
    schema = next(iter(schemas), "numeric_range")
    if len(schemas) > 1:
        return compact_interval_json(
            schema=schema,
            intervals=[],
            unit=None,
            aggregation_status="schema_conflict",
        )

    units = {
        _text(
            row.get("normalized_value_unit")
            or row.get("canonical_unit")
            or row.get("value_unit")
        )
        for row in source_rows
        if _text(
            row.get("normalized_value_unit")
            or row.get("canonical_unit")
            or row.get("value_unit")
        )
    }
    unit = next(iter(units), None)
    if len(units) > 1:
        return compact_interval_json(
            schema=schema,
            intervals=[],
            unit=None,
            aggregation_status="unit_conflict",
        )

    parsed_sets: list[list[Interval]] = []
    has_review_item = False
    for row in source_rows:
        parse_status = _text(row.get("value_parse_status"))
        intervals = _eligible_intervals(row, unit=unit)
        if parse_status and parse_status != "parsed":
            has_review_item = True
        if intervals:
            parsed_sets.append(intervals)
        elif parse_status == "parsed":
            has_review_item = True

    if not parsed_sets:
        return compact_interval_json(
            schema=schema,
            intervals=[],
            unit=unit,
            aggregation_status="needs_value_review",
        )

    aggregate = parsed_sets[0]
    for interval_set in parsed_sets[1:]:
        aggregate = intersect_interval_sets(
            aggregate,
            interval_set,
            unit=unit,
        )
    if not aggregate:
        status = "empty_intersection"
    elif has_review_item:
        status = "partial_needs_value_review"
    elif len(parsed_sets) > 1:
        status = "aggregated"
    else:
        status = "parsed"
    return compact_interval_json(
        schema=schema,
        intervals=aggregate,
        unit=unit,
        aggregation_status=status,
    )


def _eligible_intervals(
    row: Mapping[str, Any],
    *,
    unit: str | None,
) -> list[Interval]:
    serialized = row.get("eligible_ranges_json")
    parsed = _json_list(serialized)
    if parsed:
        return canonicalize_intervals(parsed, unit=unit)

    structured = _json_object(row.get("normalized_value_json"))
    compact = structured.get("intervals")
    if isinstance(compact, list):
        return canonicalize_intervals(compact, unit=unit)
    eligible = structured.get("eligible_ranges")
    if isinstance(eligible, list):
        return canonicalize_intervals(eligible, unit=unit)
    return []


def _json_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, Mapping)]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _intersect_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    unit: str | None,
) -> Interval | None:
    lower, lower_inclusive = _max_lower(left, right)
    upper, upper_inclusive = _min_upper(left, right)
    if lower is not None and upper is not None:
        if lower > upper:
            return None
        if lower == upper and not (
            lower_inclusive and upper_inclusive
        ):
            return None
    return {
        "lower": lower,
        "upper": upper,
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": upper_inclusive,
        "unit": unit or _text(left.get("unit")) or _text(right.get("unit")),
    }


def _max_lower(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[float | None, bool | None]:
    left_value = _number(left.get("lower"))
    right_value = _number(right.get("lower"))
    if left_value is None:
        return right_value, _bound_bool(right, "lower", right_value)
    if right_value is None:
        return left_value, _bound_bool(left, "lower", left_value)
    if left_value > right_value:
        return left_value, _bound_bool(left, "lower", left_value)
    if right_value > left_value:
        return right_value, _bound_bool(right, "lower", right_value)
    return left_value, bool(left.get("lower_inclusive")) and bool(
        right.get("lower_inclusive")
    )


def _min_upper(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[float | None, bool | None]:
    left_value = _number(left.get("upper"))
    right_value = _number(right.get("upper"))
    if left_value is None:
        return right_value, _bound_bool(right, "upper", right_value)
    if right_value is None:
        return left_value, _bound_bool(left, "upper", left_value)
    if left_value < right_value:
        return left_value, _bound_bool(left, "upper", left_value)
    if right_value < left_value:
        return right_value, _bound_bool(right, "upper", right_value)
    return left_value, bool(left.get("upper_inclusive")) and bool(
        right.get("upper_inclusive")
    )


def _bound_bool(
    interval: Mapping[str, Any],
    side: str,
    value: float | None,
) -> bool | None:
    if value is None:
        return None
    return bool(interval.get(f"{side}_inclusive"))


def _interval_sort_key(interval: Mapping[str, Any]) -> tuple[float, int]:
    lower = _number(interval.get("lower"))
    return (
        float("-inf") if lower is None else lower,
        0 if bool(interval.get("lower_inclusive")) else 1,
    )


def _can_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_upper = _number(left.get("upper"))
    right_lower = _number(right.get("lower"))
    if left_upper is None or right_lower is None:
        return True
    if right_lower < left_upper:
        return True
    if right_lower > left_upper:
        return False
    return bool(left.get("upper_inclusive")) or bool(
        right.get("lower_inclusive")
    )


def _merge_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> Interval:
    left_upper = _number(left.get("upper"))
    right_upper = _number(right.get("upper"))
    if left_upper is None or right_upper is None:
        upper = None
        upper_inclusive = None
    elif right_upper > left_upper:
        upper = right_upper
        upper_inclusive = bool(right.get("upper_inclusive"))
    elif right_upper < left_upper:
        upper = left_upper
        upper_inclusive = bool(left.get("upper_inclusive"))
    else:
        upper = left_upper
        upper_inclusive = bool(left.get("upper_inclusive")) or bool(
            right.get("upper_inclusive")
        )
    merged = {
        "lower": _number(left.get("lower")),
        "upper": upper,
        "lower_inclusive": _bound_bool(
            left,
            "lower",
            _number(left.get("lower")),
        ),
        "upper_inclusive": upper_inclusive,
    }
    if "unit" in left or "unit" in right:
        merged["unit"] = _text(left.get("unit")) or _text(right.get("unit"))
    return merged


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
