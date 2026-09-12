import json

from backend.criteria_processor.numeric_intervals import (
    aggregate_numeric_constraints,
)
from backend.criteria_processor.value_normalization import normalize_structured_value


def _row(raw_value: str, criterion_type: str) -> dict[str, str]:
    normalized = normalize_structured_value(
        raw_value=raw_value,
        attribute_id="example_score",
        canonical_attribute="Example Score",
        value_schema="numeric_range",
        canonical_unit="point",
        criterion_type=criterion_type,
    )
    return {
        "value_schema": "numeric_range",
        "canonical_unit": "point",
        "normalized_value_unit": "point",
        "value_parse_status": normalized["parse_status"],
        "eligible_ranges_json": json.dumps(
            normalized["eligible_ranges"]
        ),
        "normalized_value_json": json.dumps(normalized),
    }


def test_aggregate_numeric_constraints_supports_disjoint_intervals():
    result = aggregate_numeric_constraints(
        [
            _row(">25 and <=65", "Inclusion"),
            _row(">45 and <50", "Exclusion"),
        ]
    )

    assert result == {
        "schema": "numeric_range",
        "unit": "point",
        "intervals": [
            {
                "lower": 25.0,
                "upper": 45.0,
                "lower_inclusive": False,
                "upper_inclusive": True,
            },
            {
                "lower": 50.0,
                "upper": 65.0,
                "lower_inclusive": True,
                "upper_inclusive": True,
            },
        ],
        "aggregation_status": "aggregated",
    }


def test_aggregate_numeric_constraints_flags_empty_intersection():
    result = aggregate_numeric_constraints(
        [
            _row("<=3", "Inclusion"),
            _row(">3", "Inclusion"),
        ]
    )

    assert result["intervals"] == []
    assert result["aggregation_status"] == "empty_intersection"
