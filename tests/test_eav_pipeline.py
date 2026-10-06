import io
import json
from pathlib import Path

from openpyxl import load_workbook

from backend.eav_pipeline import canonical_term, deterministic_criterion, normalize_rows, segment_criteria
from backend.services.eav_workflow import EavWorkflowService, Job


def test_numerical_criterion_separates_fields_and_repetition():
    row = deterministic_criterion(
        source="Blood pressure >145/95 on three occasions",
        criterion_type="exclusion", entity="blood pressure >145/95 on three occasions",
    )
    assert row == {
        "criterion_type": "exclusion", "domain": "measurement",
        "entity": "Blood Pressure", "comparator": ">", "value": "145/95",
        "unit": None, "value_kind": "numerical", "negated": False,
        "lower_bound": None, "lower_inclusive": None,
        "upper_bound": None, "upper_inclusive": None, "interval": None,
        "temporal": None, "repetition": "on three occasions", "qualifier": None,
        "logical_operator": "standalone",
        "relations": ["HAS_VALUE", "HAS_MULTIPLIER"],
        "source": "Blood pressure >145/95 on three occasions",
    }


def test_unit_is_separate_from_value():
    row = deterministic_criterion(
        source=r"dRVVT \>37 sec", criterion_type="exclusion", entity="dRVVT >37 sec"
    )
    assert (row["entity"], row["comparator"], row["value"], row["unit"]) == (
        "dRVVT", ">", "37", "sec"
    )


def test_categorical_negation_is_explicit_and_informative():
    row = deterministic_criterion(
        source="no history of menopause", criterion_type="inclusion", entity="menopause"
    )
    assert row["entity"] == "menopause"
    assert row["comparator"] is None
    assert row["value"] == "absent"
    assert row["negated"] is True


def test_embedded_not_due_to_does_not_negate_entity():
    row = deterministic_criterion(
        source="Diabetes mellitus (NOT due to steroids) with vascular disease",
        criterion_type="exclusion", entity="diabetes mellitus with vascular disease",
    )
    assert row["value"] == "present"
    assert row["negated"] is False


def test_temporal_construct_is_separate():
    row = deterministic_criterion(
        source="Myocardial infarction within the past 60 days",
        criterion_type="exclusion", entity="myocardial infarction",
    )
    assert row["temporal"] == "within the past 60 days"
    assert row["entity"] == "myocardial infarction"


def test_segmentation_has_stable_source_ids_and_splits_semicolons():
    rows = segment_criteria("Exclusion Criteria:\nGPL >40; MPL >40; APL >50; dRVVT >37 sec")
    assert [row["source"] for row in rows] == ["GPL >40", "MPL >40", "APL >50", "dRVVT >37 sec"]
    assert all(row["criterion_type"] == "exclusion" for row in rows)
    assert all(row["source_id"] == "source-0001" for row in rows)


def test_multiple_entities_share_source_and_retain_boolean_logic():
    source = "deep vein thrombosis or pulmonary embolus"
    rows = normalize_rows({"rows": [
        _model_row(source, "DVT", "condition"),
        _model_row(source, "pulmonary embolus", "condition"),
    ]}, [{"criterion_type": "exclusion", "source": source, "source_id": "source-0007"}])
    assert len(rows) == 2
    assert {row["logical_operator"] for row in rows} == {"or"}
    assert all("OR" in row["relations"] for row in rows)
    assert {row["source_id"] for row in rows} == {"source-0007"}


def test_entity_relative_comparator_handles_two_measurements_in_one_source():
    source = "GPL>50 and blood pressure >145/95"
    rows = normalize_rows({"rows": [
        _model_row(source, "GPL", "measurement", comparator=">", value="50", logic="and"),
        _model_row(source, "blood pressure", "measurement", comparator=">", value="145/95", logic="and"),
    ]}, [{"criterion_type": "exclusion", "source": source, "source_id": "source-0001"}])
    assert [(row["entity"], row["value"]) for row in rows] == [("GPL", "50"), ("Blood Pressure", "145/95")]


def test_age_range_phrasings_normalize_to_same_closed_interval():
    sources = ["aged 18 to 65", "between 18 and 65 years", "18–65 years of age", "age ≥18 and ≤65 years"]
    rows = [deterministic_criterion(source=source, criterion_type="inclusion", entity="Age") for source in sources]
    for row in rows:
        assert row["entity"] == "Age"
        assert row["domain"] == "person"
        assert row["value_kind"] == "interval"
        assert (row["lower_bound"], row["upper_bound"]) == (18, 65)
        assert row["lower_inclusive"] is True and row["upper_inclusive"] is True
        assert row["interval"] == "[18, 65]"
        assert "HAS_VALUE" in row["relations"]


def test_configured_aliases_normalize_abbreviations_and_full_names():
    assert canonical_term("kidney impairment") == "renal impairment"
    assert canonical_term("acquired immunodeficiency syndrome") == "AIDS"
    assert canonical_term("Unequivocal diagnosis of systemic lupus erythematosus") == "unequivocal diagnosis of SLE"
    assert canonical_term("IgG phospholipid unit") == "GPL"
    assert canonical_term("NSCLC") == "Non-Small Cell Lung Cancer"
    assert canonical_term("ECOG") == "ECOG Performance Status"
    assert canonical_term("ALL") == "Acute Lymphoblastic Leukemia"
    assert canonical_term("all patients") == "all patients"


def test_legacy_library_rows_are_upgraded_in_memory(tmp_path: Path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"immutable": True, "entries": []}), encoding="utf-8")
    legacy = tmp_path / "eavLibrary_20200101_000000_000000.json"
    legacy.write_text(json.dumps({"entries": [{
        "row_id": "old", "trial_id": "NCT1", "criterion_type": "exclusion",
        "entity": "age", "attribute": ">", "value": "65", "source": "Age >65",
    }]}), encoding="utf-8")
    service = EavWorkflowService(object(), tmp_path / "runtime", base)
    row = service.library()["entries"][0]
    assert (row["domain"], row["comparator"], row["value_kind"]) == ("person", ">", "numerical")


def test_review_creates_versioned_structured_json_and_csv(tmp_path: Path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"immutable": True, "entries": []}), encoding="utf-8")
    service = EavWorkflowService(object(), tmp_path / "runtime", base)
    row = normalize_rows({"rows": [_model_row("GPL>50", "GPL", "measurement", comparator=">", value="50")]})[0]
    job = Job(job_id="job", trial_id="NCT1", input_mode="manual", status="completed", rows=[row])
    service._save(job)
    result = service.apply_review("job", [{**row, "review_status": "accepted"}])
    assert result["json"].startswith("criteriaLibrary_")
    saved = json.loads((service.libraries / result["json"]).read_text())
    assert saved["schema_version"] == "modern-autocrit.structured-criteria.v3"
    assert saved["entries"][0]["domain"] == "measurement"
    assert saved["entries"][0]["interval"] == "(50, +inf)"
    assert (base.parent / result["csv"]).is_file()


def test_library_excel_export_contains_base_and_reviewed_rows(tmp_path: Path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"immutable": True, "entries": [{
        "row_id": "base", "trial_id": "BASE", "criterion_type": "inclusion",
        "domain": "person", "entity": "Age", "comparator": ">=", "value": "18",
        "unit": "years", "source": "Age >=18", "relations": ["HAS_VALUE"],
    }]}), encoding="utf-8")
    snapshot = tmp_path / "criteriaLibrary_20200101_000000_000000.json"
    snapshot.write_text(json.dumps({"entries": [{
        "row_id": "reviewed", "trial_id": "NCT1", "criterion_type": "exclusion",
        "domain": "condition", "entity": "diabetes", "comparator": None,
        "value": "present", "unit": None, "source": "Diabetes", "relations": [],
    }]}), encoding="utf-8")
    service = EavWorkflowService(object(), tmp_path / "runtime", base)
    filename, data = service.library_workbook()
    workbook = load_workbook(io.BytesIO(data))
    sheet = workbook["Criteria Library"]
    assert filename.startswith("criteriaLibrary_") and filename.endswith(".xlsx")
    assert sheet.auto_filter.ref == sheet.dimensions
    assert sheet.freeze_panes == "A2"
    assert sheet.max_row == 3
    assert {sheet.cell(2, 1).value, sheet.cell(3, 1).value} == {"immutable_base", "reviewed"}
    assert "Field Guide" in workbook.sheetnames


def test_reset_library_removes_snapshots_and_preserves_examples(tmp_path: Path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"immutable": True, "entries": [{"trial_id": "EXAMPLE"}]}), encoding="utf-8")
    service = EavWorkflowService(object(), tmp_path / "runtime", base)
    config_snapshot = tmp_path / "criteriaLibrary_20200101_000000_000000.json"
    runtime_snapshot = service.libraries / "criteriaLibrary_20200101_000000_000000.csv"
    legacy_snapshot = tmp_path / "eavLibrary_20200101_000000_000000.json"
    unrelated = tmp_path / "keep.json"
    for path in (config_snapshot, runtime_snapshot, legacy_snapshot, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    result = service.reset_library()
    assert result["removed"] == 3
    assert result["base_entries"] == 1
    assert base.is_file() and unrelated.is_file()
    assert not config_snapshot.exists() and not runtime_snapshot.exists() and not legacy_snapshot.exists()


def _model_row(source, entity, domain, *, comparator=None, value="present", logic="standalone"):
    return {
        "criterion_type": "inclusion", "domain": domain, "entity": entity,
        "comparator": comparator, "value": value, "unit": None, "negated": False,
        "temporal": None, "repetition": None, "qualifier": None,
        "logical_operator": logic, "source": source,
        "lower_bound": None, "lower_inclusive": None,
        "upper_bound": None, "upper_inclusive": None,
    }
