from pathlib import Path

import pytest
from openpyxl import load_workbook

from backend.criteria_processor.models import TerminologyAuditRecord
from backend.criteria_processor.terminology import TerminologyRepository
from backend.criteria_processor.terminology_updates import (
    apply_audit_records_to_workbook,
)
from backend.criteria_processor.value_normalization import normalize_structured_value


def test_age_range_schema_preserves_strict_bounds():
    value = normalize_structured_value(
        raw_value=r"\>22 years and \<80 years",
        attribute_id="age",
        canonical_attribute="Age",
        value_schema="numeric_range",
        canonical_unit="year",
        criterion_type="Inclusion",
    )

    assert value["lower"] == 22
    assert value["upper"] == 80
    assert value["lower_inclusive"] is False
    assert value["upper_inclusive"] is False
    assert value["normalized_unit"] == "year"
    assert value["unit"] == "year"
    assert value["intervals"] == [
        {
            "lower": 22.0,
            "upper": 80.0,
            "lower_inclusive": False,
            "upper_inclusive": False,
        }
    ]
    assert value["eligible_ranges"] == [
        {
            "lower": 22.0,
            "upper": 80.0,
            "lower_inclusive": False,
            "upper_inclusive": False,
            "unit": "year",
        }
    ]


def test_duration_range_converts_weeks_to_days():
    value = normalize_structured_value(
        raw_value="between 2 and 6 weeks",
        attribute_id="time_since_stroke",
        canonical_attribute="Time Since Stroke",
        value_schema="duration_range",
        canonical_unit="day",
    )

    assert value["lower"] == 2
    assert value["upper"] == 6
    assert value["normalized_lower"] == 14
    assert value["normalized_upper"] == 42
    assert value["normalized_unit"] == "day"


def test_duration_range_uses_condition_context_and_mixed_units():
    value = normalize_structured_value(
        raw_value="unilateral supratentorial ischemic stroke",
        context_text=(
            "occurred at least 9 months but not more than ten 10 years "
            "prior to enrollment"
        ),
        attribute_id="time_since_stroke",
        canonical_attribute="Time Since Stroke",
        value_schema="duration_range",
        canonical_unit="day",
    )

    assert value["lower"] == 9
    assert value["upper"] == 10
    assert value["lower_unit"] == "month"
    assert value["upper_unit"] == "year"
    assert value["normalized_lower"] == 273.9375
    assert value["normalized_upper"] == 3652.5


def test_duration_range_parses_units_before_bounds():
    value = normalize_structured_value(
        raw_value="from day 5 to day 10 after stroke",
        attribute_id="time_since_stroke",
        canonical_attribute="Time Since Stroke",
        value_schema="duration_range",
        canonical_unit="day",
    )

    assert value["lower"] == 5
    assert value["upper"] == 10
    assert value["normalized_lower"] == 5
    assert value["normalized_upper"] == 10


@pytest.mark.parametrize(
    ("raw_value", "schema", "unit", "lower", "upper", "inclusive"),
    [
        ("12 or more", "numeric_range", "point", 12, None, True),
        ("3 or less", "numeric_range", "point", None, 3, True),
        (
            "taken in the last 5 weeks",
            "duration_range",
            "week",
            None,
            5,
            True,
        ),
        (
            "not expected to survive 1 year",
            "duration_range",
            "year",
            None,
            1,
            False,
        ),
    ],
)
def test_range_schema_parses_common_natural_language_bounds(
    raw_value,
    schema,
    unit,
    lower,
    upper,
    inclusive,
):
    value = normalize_structured_value(
        raw_value=raw_value,
        attribute_id="test_attribute",
        canonical_attribute="Test Attribute",
        value_schema=schema,
        canonical_unit=unit,
        criterion_type="Inclusion",
    )

    assert value["parse_status"] == "parsed"
    assert value["lower"] == lower
    assert value["upper"] == upper
    if lower is not None:
        assert value["lower_inclusive"] is inclusive
    if upper is not None:
        assert value["upper_inclusive"] is inclusive


def test_attribute_mapping_always_inherits_fixed_parent_entity():
    repository = TerminologyRepository(Path("config/attribute_library.xlsx"))

    mapping = repository.map_criterion(
        attribute="age",
        value="18 to 80 years",
        entity="Demographic",
    )

    assert mapping.attribute_id == "age"
    assert mapping.canonical_entity == "Demographic"


def test_unique_attribute_lookup_repairs_uncontrolled_source_entity():
    repository = TerminologyRepository(Path("config/attribute_library.xlsx"))

    mapping = repository.map_criterion(
        attribute="age",
        value="18 to 80 years",
        entity="Clinical",
    )

    assert mapping.attribute_id == "age"
    assert mapping.canonical_entity == "Demographic"
    assert mapping.mapping_status == "mapped"


def test_scoped_alias_can_target_attribute_under_different_fixed_parent(
    tmp_path,
):
    output = tmp_path / "working.xlsx"
    record = TerminologyAuditRecord(
        change_id="change_functional_fma",
        target_object="attribute_alias",
        action="add",
        object_id="aliases_baseline_fma_ue",
        source_entity="Functional Status",
        attribute_id="baseline_fma_ue",
        aliases=["FMA-UE score"],
    )
    apply_audit_records_to_workbook(
        source_workbook="config/attribute_library.xlsx",
        output_workbook=output,
        records=[record],
        allowed_review_statuses={"proposed"},
        metadata_status="working",
    )

    repository = TerminologyRepository(output)
    mapping = repository.map_criterion(
        attribute="FMA-UE score",
        value="20 to 50",
        entity="Functional Status",
    )

    assert mapping.attribute_id == "baseline_fma_ue"
    assert mapping.canonical_entity == "Score"


def test_pregnancy_uses_selection_state_without_library_value_rows():
    mapping = TerminologyRepository(
        Path("config/attribute_library.xlsx")
    ).map_criterion(
        attribute="pregnancy",
        value="non-pregnant",
        entity="Contraceptive",
    )

    assert mapping.attribute_id == "pregnancy"
    assert mapping.value_schema == "selection_state"
    assert mapping.value_id is None


def test_existing_attribute_cannot_be_reparented(tmp_path):
    output = tmp_path / "working.xlsx"
    record = TerminologyAuditRecord(
        change_id="change_reparent_age",
        target_object="attribute",
        action="add",
        object_id="age",
        attribute_id="age",
        canonical_name="Age",
        parent_entity="Diagnosis",
        value_schema="numeric_range",
        canonical_unit="year",
    )

    with pytest.raises(ValueError, match="fixed parent entity 'Demographic'"):
        apply_audit_records_to_workbook(
            source_workbook="config/attribute_library.xlsx",
            output_workbook=output,
            records=[record],
            allowed_review_statuses={"proposed"},
            metadata_status="working",
        )


@pytest.mark.parametrize(
    ("criterion_type", "source_text", "expected_state", "expected_assertion"),
    [
        (
            "Inclusion",
            "Confirmed diagnosis of Parkinson's disease.",
            "included",
            "present",
        ),
        (
            "Exclusion",
            "Diagnosis of Parkinson's disease.",
            "excluded",
            "present",
        ),
        (
            "Inclusion",
            "Participants must be non-pregnant.",
            "excluded",
            "absent",
        ),
        (
            "Exclusion",
            "Patients without Parkinson's disease.",
            "included",
            "absent",
        ),
    ],
)
def test_selection_state_combines_criterion_type_and_assertion(
    criterion_type,
    source_text,
    expected_state,
    expected_assertion,
):
    value = normalize_structured_value(
        raw_value=source_text,
        context_text=source_text,
        attribute_id="parkinsons_disease",
        canonical_attribute="Parkinson's Disease",
        value_schema="selection_state",
        criterion_type=criterion_type,
    )

    assert value["category"] == expected_state
    assert value["selection_effect"] == expected_state
    assert value["assertion"] == expected_assertion
    assert value["parse_status"] == "parsed"


def test_exclusion_numeric_range_becomes_disallowed_interval():
    value = normalize_structured_value(
        raw_value="age under 40 years",
        attribute_id="age",
        canonical_attribute="Age",
        value_schema="numeric_range",
        canonical_unit="year",
        criterion_type="Exclusion",
    )

    assert value["excluded_ranges"][0]["upper"] == 40
    assert value["excluded_ranges"][0]["upper_inclusive"] is False
    assert value["eligible_ranges"][0]["lower"] == 40
    assert value["eligible_ranges"][0]["lower_inclusive"] is True


def test_inclusion_outside_range_becomes_two_eligible_intervals():
    value = normalize_structured_value(
        raw_value="age outside 40 to 60 years",
        attribute_id="age",
        canonical_attribute="Age",
        value_schema="numeric_range",
        canonical_unit="year",
        criterion_type="Inclusion",
    )

    assert len(value["eligible_ranges"]) == 2
    assert value["eligible_ranges"][0]["upper"] == 40
    assert value["eligible_ranges"][1]["lower"] == 60
    assert value["excluded_ranges"][0]["lower"] == 40
    assert value["excluded_ranges"][0]["upper"] == 60


def test_disjoint_exact_values_are_point_intervals():
    value = normalize_structured_value(
        raw_value="=0 or 6",
        attribute_id="modified_rankin_scale",
        canonical_attribute="Modified Rankin Scale",
        value_schema="numeric_range",
        canonical_unit="point",
        criterion_type="Exclusion",
    )

    assert value["parse_status"] == "parsed"
    assert value["unit"] == "point"
    assert value["excluded_ranges"] == [
        {
            "lower": 0.0,
            "upper": 0.0,
            "lower_inclusive": True,
            "upper_inclusive": True,
            "unit": "point",
        },
        {
            "lower": 6.0,
            "upper": 6.0,
            "lower_inclusive": True,
            "upper_inclusive": True,
            "unit": "point",
        },
    ]
    assert value["intervals"] == [
        {
            "lower": None,
            "upper": 0.0,
            "lower_inclusive": None,
            "upper_inclusive": False,
        },
        {
            "lower": 0.0,
            "upper": 6.0,
            "lower_inclusive": False,
            "upper_inclusive": False,
        },
        {
            "lower": 6.0,
            "upper": None,
            "lower_inclusive": False,
            "upper_inclusive": None,
        },
    ]


def test_raw_numeric_value_is_not_contaminated_by_other_context_scores():
    value = normalize_structured_value(
        raw_value=">1",
        context_text=(
            "not independent before stroke, defined as scores <95 on the "
            "Barthel Index or >1 on the Modified Rankin Scale"
        ),
        attribute_id="modified_rankin_scale",
        canonical_attribute="Modified Rankin Scale",
        value_schema="numeric_range",
        canonical_unit="point",
        criterion_type="Exclusion",
    )

    assert value["intervals"] == [
        {
            "lower": None,
            "upper": 1.0,
            "lower_inclusive": None,
            "upper_inclusive": True,
        }
    ]


def test_ambiguous_raw_number_is_not_replaced_by_context_range():
    value = normalize_structured_value(
        raw_value="1 year post injury",
        context_text="3 to 9 months post infarct or 1 year post injury",
        attribute_id="time_since_injury",
        canonical_attribute="Time Since Injury",
        value_schema="duration_range",
        canonical_unit="year",
        criterion_type="Inclusion",
    )

    assert value["parse_status"] == "needs_value_review"
    assert value["intervals"] == []


def test_categorical_exclusion_records_disallowed_value():
    value = normalize_structured_value(
        raw_value="hemorrhagic",
        attribute_id="stroke_type",
        canonical_attribute="Stroke Type",
        value_schema="categorical",
        canonical_value="Hemorrhagic",
        criterion_type="Exclusion",
    )

    assert value["selection_effect"] == "excluded"
    assert value["eligible_values"] == []
    assert value["excluded_values"] == ["Hemorrhagic"]


def test_library_rejects_uncontrolled_parent_entity(tmp_path):
    invalid = tmp_path / "invalid_entity.xlsx"
    workbook = load_workbook("config/attribute_library.xlsx")
    workbook["Attributes"]["C2"] = "Safety"
    workbook.save(invalid)

    with pytest.raises(ValueError, match="uncontrolled parent entity 'Safety'"):
        TerminologyRepository(invalid)
