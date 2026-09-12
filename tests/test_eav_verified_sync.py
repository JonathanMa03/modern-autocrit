import csv
import hashlib
import json
from pathlib import Path

import pytest

from backend.criteria_processor.eva_library import stable_id
from backend.criteria_processor.eva_verified_sync import (
    apply_imported_audits_to_library,
    import_verified_audits,
)


def _write_library(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "eligcrit.eligibility_eva_library.v1",
                "library_name": "Test eligibility EVA",
                "revision": 0,
                "updated_at": "2026-01-01T00:00:00+00:00",
                "value_definitions": {
                    "Categorical": ["Included", "Excluded"],
                    "SexGender": ["male", "female", "all"],
                    "Numerical": ["Range", "Point"],
                },
                "entities": [
                    {
                        "entity_id": "diagnosis",
                        "canonical_name": "Diagnosis",
                        "description": "Qualifying diagnoses.",
                        "aliases": [],
                        "active": True,
                    }
                ],
                "attributes": [],
                "audit_history": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_points(path: Path) -> list[dict[str, str]]:
    rows = [
        {
            "criteria": "inclusion",
            "item": "Ischemic stroke",
            "context": "Adults with ischemic stroke at least 3 months ago",
        },
        {
            "criteria": "inclusion",
            "item": "at least 3 months after stroke",
            "context": "Adults with ischemic stroke at least 3 months ago",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("criteria", "item", "context"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_verified(path: Path, points: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "eva_id": stable_id(
                "eva",
                "TRIAL-1",
                "TRIAL-1:0001",
                points[0]["item"],
            ),
            "trial_key": "TRIAL-1",
            "trial_id": "TRIAL-1",
            "criteria": points[0]["criteria"],
            "source_item": points[0]["item"],
            "context": points[0]["context"],
            "entity_id": "diagnosis",
            "entity": "Diagnosis",
            "attribute_id": "ischemic_stroke",
            "attribute": "ischemic_stroke",
            "value_type": "Categorical",
            "categorical_value": "Included",
            "numerical_type": None,
            "numerical_value": None,
            "unit": None,
            "confidence": 0.99,
            "rationale": "The verified inclusion requires ischemic stroke.",
            "review_status": "approved",
            "library_revision": 7,
            "library_schema_version": (
                "eligcrit.eligibility_eva_library.v1"
            ),
        },
        {
            "eva_id": stable_id(
                "eva",
                "TRIAL-1",
                "TRIAL-1:0002",
                points[1]["item"],
            ),
            "trial_key": "TRIAL-1",
            "trial_id": "TRIAL-1",
            "criteria": points[1]["criteria"],
            "source_item": points[1]["item"],
            "context": points[1]["context"],
            "entity_id": "diagnosis",
            "entity": "Diagnosis",
            "attribute_id": "time_since_stroke_at_enrollment",
            "attribute": "time_since_stroke_at_enrollment",
            "value_type": "Numerical",
            "categorical_value": None,
            "numerical_type": "Range",
            "numerical_value": "[3, +inf)",
            "unit": "months",
            "confidence": 0.98,
            "rationale": "The verified eligible interval begins at 3 months.",
            "review_status": "approved",
            "library_revision": 7,
            "library_schema_version": (
                "eligcrit.eligibility_eva_library.v1"
            ),
        },
    ]
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_sync_populates_audit_before_staged_library_update(
    tmp_path: Path,
):
    library_path = tmp_path / "config" / "library.json"
    points_path = (
        tmp_path
        / "points"
        / "TRIAL-1"
        / "elig_breakdown_points.csv"
    )
    verified_path = (
        tmp_path
        / "verified"
        / "TRIAL-1"
        / "eligibility_eva.jsonl"
    )
    work_directory = tmp_path / "work"
    _write_library(library_path)
    points = _write_points(points_path)
    _write_verified(verified_path, points)
    original_library = library_path.read_text(encoding="utf-8")
    original_verified_hash = _file_sha256(verified_path)

    imported = import_verified_audits(
        verified_directory=verified_path.parents[1],
        points_directory=points_path.parents[1],
        library_path=library_path,
        work_directory=work_directory,
    )

    assert imported["trial_count"] == 1
    assert imported["row_count"] == 2
    assert imported["library_revision_unchanged"] == 0
    assert library_path.read_text(encoding="utf-8") == original_library
    audit_path = (
        work_directory / "audits" / "TRIAL-1.eva_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["review_status"] == "approved"
    assert audit["schema_version"] == "eligcrit.eligibility_eva_audit.v3"
    assert all(
        item["mapping_review"]["status"] == "legacy_unreviewed"
        for item in audit["items"]
    )
    assert [item["source_item_id"] for item in audit["items"]] == [
        "TRIAL-1:0001",
        "TRIAL-1:0002",
    ]
    assert all(
        item["review_notes"].startswith("Imported from reviewer-verified")
        for item in audit["items"]
    )

    applied = apply_imported_audits_to_library(
        verified_directory=verified_path.parents[1],
        library_path=library_path,
        work_directory=work_directory,
    )

    assert applied["previous_library_revision"] == 0
    assert applied["new_library_revision"] == 1
    assert applied["attribute_count"] == 2
    assert applied["applied_trials"] == ["TRIAL-1"]
    assert _file_sha256(verified_path) == original_verified_hash
    library = json.loads(library_path.read_text(encoding="utf-8"))
    assert {row["attribute_id"] for row in library["attributes"]} == {
        "ischemic_stroke",
        "time_since_stroke_at_enrollment",
    }
    assert library["audit_history"][0]["trial_key"] == "TRIAL-1"
    normalized_path = (
        work_directory
        / "normalized"
        / "TRIAL-1"
        / "eligibility_eva.jsonl"
    )
    normalized = [
        json.loads(line)
        for line in normalized_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["library_revision"] for row in normalized} == {1}

    repeated = apply_imported_audits_to_library(
        verified_directory=verified_path.parents[1],
        library_path=library_path,
        work_directory=work_directory,
    )
    assert repeated["new_library_revision"] == 1
    assert repeated["applied_trials"] == []
    assert repeated["already_applied_trials"] == ["TRIAL-1"]


def test_verified_import_rejects_evidence_mismatch_before_writing_audit(
    tmp_path: Path,
):
    library_path = tmp_path / "config" / "library.json"
    points_path = (
        tmp_path
        / "points"
        / "TRIAL-1"
        / "elig_breakdown_points.csv"
    )
    verified_path = (
        tmp_path
        / "verified"
        / "TRIAL-1"
        / "eligibility_eva.jsonl"
    )
    work_directory = tmp_path / "work"
    _write_library(library_path)
    points = _write_points(points_path)
    _write_verified(verified_path, points)
    verified_rows = [
        json.loads(line)
        for line in verified_path.read_text(encoding="utf-8").splitlines()
    ]
    verified_rows[0]["context"] = "A changed context"
    verified_path.write_text(
        "".join(json.dumps(row) + "\n" for row in verified_rows),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="does not match its authoritative",
    ):
        import_verified_audits(
            verified_directory=verified_path.parents[1],
            points_directory=points_path.parents[1],
            library_path=library_path,
            work_directory=work_directory,
        )

    assert not (
        work_directory / "audits" / "TRIAL-1.eva_audit.json"
    ).exists()


def test_library_sync_can_retire_superseded_unreferenced_attributes(
    tmp_path: Path,
):
    library_path = tmp_path / "config" / "library.json"
    points_path = (
        tmp_path
        / "points"
        / "TRIAL-1"
        / "elig_breakdown_points.csv"
    )
    verified_path = (
        tmp_path
        / "verified"
        / "TRIAL-1"
        / "eligibility_eva.jsonl"
    )
    work_directory = tmp_path / "work"
    _write_library(library_path)
    library = json.loads(library_path.read_text(encoding="utf-8"))
    library["attributes"].append(
        {
            "attribute_id": "superseded_term",
            "canonical_name": "superseded_term",
            "entity_id": "diagnosis",
            "description": "An obsolete pre-verification term.",
            "aliases": [],
            "value_type": "Categorical",
            "numerical_type": None,
            "canonical_unit": None,
            "active": True,
        }
    )
    library_path.write_text(
        json.dumps(library, indent=2),
        encoding="utf-8",
    )
    points = _write_points(points_path)
    _write_verified(verified_path, points)
    import_verified_audits(
        verified_directory=verified_path.parents[1],
        points_directory=points_path.parents[1],
        library_path=library_path,
        work_directory=work_directory,
    )

    applied = apply_imported_audits_to_library(
        verified_directory=verified_path.parents[1],
        library_path=library_path,
        work_directory=work_directory,
        retire_unreferenced=True,
    )

    assert applied["new_library_revision"] == 2
    assert applied["attribute_count"] == 3
    assert applied["active_attribute_count"] == 2
    assert applied["retired_attribute_ids"] == ["superseded_term"]
    updated = json.loads(library_path.read_text(encoding="utf-8"))
    retired = next(
        row
        for row in updated["attributes"]
        if row["attribute_id"] == "superseded_term"
    )
    assert retired["active"] is False
    assert updated["retirement_history"][0]["attribute_ids"] == [
        "superseded_term"
    ]

    repeated = apply_imported_audits_to_library(
        verified_directory=verified_path.parents[1],
        library_path=library_path,
        work_directory=work_directory,
        retire_unreferenced=True,
    )
    assert repeated["new_library_revision"] == 2
    assert repeated["retired_attribute_ids"] == []
