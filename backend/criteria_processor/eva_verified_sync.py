"""Synchronize reviewer-verified eligibility EVA rows into audit state.

The verified JSONL files are authoritative normalized outputs.  This module
imports them into the reviewer-facing audit documents first, without mutating
the terminology library, and applies those imported audits to a staged copy of
the library in a separate second phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.criteria_processor.atomic_io import (
    atomic_write_json,
    atomic_write_jsonl,
)
from backend.criteria_processor.eva_library import (
    EvaLibraryRepository,
    stable_id,
)
from backend.criteria_processor.eva_models import (
    EvaAttributeCandidate,
    EvaAuditDocument,
    EvaAuditItem,
    EvaSearchExpansion,
)
from backend.criteria_processor.eva_audit_migration import load_eva_audit
from backend.criteria_processor.eva_models import (
    EvaAttributeIdReviewTrace,
    EvaMappingReviewTrace,
)
from backend.criteria_processor.eva_pipeline import load_breakdown_points


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERIFIED_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "selected_stroke"
    / "normalized_elig"
)
DEFAULT_POINTS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "selected_stroke"
    / "elig_points_breakdown"
)
DEFAULT_LIBRARY_PATH = (
    PROJECT_ROOT / "config" / "eligibility_eva_library.json"
)
DEFAULT_WORK_DIRECTORY = PROJECT_ROOT / "tmp" / "eligibility_eva"
VERIFIED_FILENAME = "eligibility_eva.jsonl"
IMPORTED_REVIEW_NOTE = (
    "Imported from reviewer-verified normalized eligibility EVA output."
)
NORMALIZED_FIELDS = (
    "eva_id",
    "trial_key",
    "trial_id",
    "criteria",
    "source_item",
    "context",
    "entity_id",
    "entity",
    "attribute_id",
    "attribute",
    "value_type",
    "categorical_value",
    "numerical_type",
    "numerical_value",
    "unit",
    "confidence",
    "rationale",
    "review_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}: line {line_number} must be a JSON object."
                )
            rows.append(payload)
    if not rows:
        raise ValueError(f"{path}: no verified EVA rows found.")
    return rows


def load_verified_trials(
    verified_directory: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load and cross-trial validate verified per-trial JSONL files."""

    root = Path(verified_directory).resolve()
    if not root.is_dir():
        raise ValueError(
            f"Verified eligibility EVA directory not found: {root}"
        )
    trials: dict[str, dict[str, Any]] = {}
    entity_definitions: dict[str, str] = {}
    attribute_definitions: dict[
        str,
        tuple[str, str, str, str | None],
    ] = {}
    for path in sorted(
        root.glob(f"*/{VERIFIED_FILENAME}"),
        key=lambda value: value.parent.name.casefold(),
    ):
        trial_key = path.parent.name
        if not trial_key or Path(trial_key).name != trial_key:
            raise ValueError(f"Invalid verified trial key: {trial_key!r}")
        rows = _read_jsonl(path)
        seen_eva_ids: set[str] = set()
        for line_number, row in enumerate(rows, start=1):
            _validate_verified_row(
                row,
                path=path,
                line_number=line_number,
                trial_key=trial_key,
            )
            eva_id = _text(row.get("eva_id"))
            if eva_id in seen_eva_ids:
                raise ValueError(
                    f"{path}: duplicate eva_id '{eva_id}'."
                )
            seen_eva_ids.add(eva_id)

            entity_id = _text(row.get("entity_id"))
            entity_name = _text(row.get("entity"))
            prior_entity_name = entity_definitions.setdefault(
                entity_id,
                entity_name,
            )
            if prior_entity_name != entity_name:
                raise ValueError(
                    f"{path}: Entity '{entity_id}' has conflicting names "
                    f"'{prior_entity_name}' and '{entity_name}'."
                )

            attribute_id = _text(row.get("attribute_id"))
            definition = (
                entity_id,
                _text(row.get("attribute")),
                _text(row.get("value_type")),
                _text(row.get("numerical_type")) or None,
            )
            prior_definition = attribute_definitions.setdefault(
                attribute_id,
                definition,
            )
            if prior_definition != definition:
                raise ValueError(
                    f"{path}: Attribute '{attribute_id}' has conflicting "
                    "verified definitions."
                )
        trials[trial_key] = {
            "trial_key": trial_key,
            "path": path,
            "rows": rows,
            "sha256": _json_sha256(rows),
        }
    if not trials:
        raise ValueError(
            f"No */{VERIFIED_FILENAME} files found under {root}."
        )
    return trials


def _validate_verified_row(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    trial_key: str,
) -> None:
    prefix = f"{path}: line {line_number}"
    required_text = (
        "eva_id",
        "trial_key",
        "trial_id",
        "criteria",
        "source_item",
        "context",
        "entity_id",
        "entity",
        "attribute_id",
        "attribute",
        "value_type",
        "rationale",
        "review_status",
    )
    missing = [key for key in required_text if not _text(row.get(key))]
    if missing:
        raise ValueError(
            f"{prefix} is missing required values: {', '.join(missing)}."
        )
    if _text(row.get("trial_key")) != trial_key:
        raise ValueError(
            f"{prefix} trial_key does not match folder '{trial_key}'."
        )
    if _text(row.get("criteria")).casefold() not in {
        "inclusion",
        "exclusion",
    }:
        raise ValueError(f"{prefix} has invalid criteria.")
    if _text(row.get("review_status")).casefold() != "approved":
        raise ValueError(
            f"{prefix} is not reviewer-approved and cannot be imported."
        )
    value_type = _text(row.get("value_type"))
    if value_type == "Categorical":
        if _text(row.get("categorical_value")) not in {
            "Included",
            "Excluded",
        }:
            raise ValueError(
                f"{prefix} has an invalid Categorical value."
            )
    elif value_type == "SexGender":
        if _text(row.get("categorical_value")) not in {
            "male",
            "female",
            "all",
        }:
            raise ValueError(f"{prefix} has an invalid SexGender value.")
    elif value_type == "Numerical":
        if _text(row.get("numerical_type")) not in {"Range", "Point"}:
            raise ValueError(f"{prefix} has an invalid numerical_type.")
        if not _text(row.get("numerical_value")):
            raise ValueError(f"{prefix} is missing numerical_value.")
    else:
        raise ValueError(f"{prefix} has invalid value_type '{value_type}'.")
    try:
        confidence = float(row.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix} has invalid confidence.") from exc
    if not 0 <= confidence <= 1:
        raise ValueError(f"{prefix} confidence must be between 0 and 1.")


def _audit_item_projection(item: EvaAuditItem) -> dict[str, Any]:
    return {
        "eva_id": item.eva_id,
        "criteria": item.criteria,
        "source_item": item.item,
        "context": item.context,
        "entity_id": item.entity_id,
        "entity": item.entity_name,
        "attribute_id": item.attribute_id,
        "attribute": item.attribute_name,
        "value_type": item.value_type,
        "categorical_value": item.categorical_value,
        "numerical_type": item.numerical_type,
        "numerical_value": item.numerical_value,
        "unit": item.unit,
        "confidence": item.confidence,
        "rationale": item.rationale,
        "review_status": "approved",
    }


def _normalized_core(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in NORMALIZED_FIELDS}


def _definitions_satisfied(
    library: EvaLibraryRepository,
    audit: EvaAuditDocument,
) -> bool:
    for item in audit.items:
        entity = library.entities.get(item.entity_id)
        attribute = library.attributes.get(item.attribute_id)
        if (
            entity is None
            or entity.canonical_name != item.entity_name
            or attribute is None
            or attribute.entity_id != item.entity_id
            or attribute.canonical_name != item.attribute_name
            or attribute.value_type != item.value_type
            or attribute.numerical_type != item.numerical_type
        ):
            return False
    return True


def _candidate_from_library(
    row: Mapping[str, Any],
    library: EvaLibraryRepository,
) -> list[EvaAttributeCandidate]:
    attribute_id = _text(row.get("attribute_id"))
    attribute = library.attributes.get(attribute_id)
    if attribute is None:
        return []
    entity = library.entities[attribute.entity_id]
    return [
        EvaAttributeCandidate(
            attribute_id=attribute.attribute_id,
            canonical_name=attribute.canonical_name,
            entity_id=attribute.entity_id,
            entity_name=entity.canonical_name,
            description=attribute.description,
            aliases=attribute.aliases,
            value_type=attribute.value_type,
            numerical_type=attribute.numerical_type,
            canonical_unit=attribute.canonical_unit,
            score=100.0,
            matched_term=_text(row.get("source_item")),
            matched_text=attribute.canonical_name,
            retrieval_methods=["verified_import:exact_attribute_id"],
        )
    ]


def _generic_attribute_description(row: Mapping[str, Any]) -> str:
    attribute_name = _text(row.get("attribute")).replace("_", " ")
    entity_name = _text(row.get("entity"))
    return (
        f"Reviewer-verified eligibility Attribute for {attribute_name} "
        f"under {entity_name}."
    )


def _build_imported_audit(
    *,
    trial_key: str,
    verified: Mapping[str, Any],
    points_path: Path,
    library: EvaLibraryRepository,
    existing_payload: Mapping[str, Any] | None,
    now: str,
) -> EvaAuditDocument:
    rows = list(verified["rows"])
    points = load_breakdown_points(points_path)
    if len(rows) != len(points):
        raise ValueError(
            f"{trial_key}: verified row count {len(rows)} does not match "
            f"atomic point count {len(points)}."
        )
    existing = (
        load_eva_audit(existing_payload)
        if existing_payload is not None
        else None
    )
    existing_items = (
        {item.eva_id: item for item in existing.items}
        if existing is not None
        else {}
    )
    items: list[EvaAuditItem] = []
    for index, (row, point) in enumerate(zip(rows, points), start=1):
        evidence = {
            "criteria": _text(row.get("criteria")).casefold(),
            "item": _text(row.get("source_item")),
            "context": _text(row.get("context")),
        }
        if evidence != point:
            raise ValueError(
                f"{trial_key}: verified row {index} does not match its "
                "authoritative criteria/item/context point."
            )
        source_item_id = f"{trial_key}:{index:04d}"
        expected_eva_id = stable_id(
            "eva",
            _text(row.get("trial_id")),
            source_item_id,
            point["item"],
        )
        if _text(row.get("eva_id")) != expected_eva_id:
            raise ValueError(
                f"{trial_key}: verified row {index} has unexpected eva_id."
            )
        prior = existing_items.get(expected_eva_id)
        attribute = library.attributes.get(
            _text(row.get("attribute_id"))
        )
        same_prior_attribute = bool(
            prior
            and prior.attribute_id == _text(row.get("attribute_id"))
            and prior.entity_id == _text(row.get("entity_id"))
        )
        search_expansion = (
            prior.search_expansion
            if prior is not None
            else EvaSearchExpansion(
                lexical_terms=[
                    point["item"],
                    _text(row.get("attribute")),
                ],
                semantic_terms=[
                    f"{_text(row.get('entity'))} "
                    f"{_text(row.get('attribute'))}"
                ],
                entity_hints=[_text(row.get("entity"))],
                concept_summary=(
                    "Imported from reviewer-verified normalized EVA output; "
                    "the original model search expansion was not retained."
                ),
            )
        )
        candidates = (
            prior.candidates
            if prior is not None
            else _candidate_from_library(row, library)
        )
        attribute_description = (
            attribute.description
            if attribute is not None
            else (
                prior.attribute_description
                if same_prior_attribute
                else _generic_attribute_description(row)
            )
        )
        attribute_aliases = (
            list(prior.attribute_aliases)
            if same_prior_attribute
            else []
        )
        item = EvaAuditItem(
            eva_id=expected_eva_id,
            source_item_id=source_item_id,
            criteria=point["criteria"],
            item=point["item"],
            context=point["context"],
            search_expansion=search_expansion,
            candidates=candidates,
            mapping_decision=(
                "existing_attribute"
                if attribute is not None
                else "new_attribute"
            ),
            attribute_id=_text(row.get("attribute_id")),
            entity_id=_text(row.get("entity_id")),
            entity_name=_text(row.get("entity")),
            new_entity=(
                _text(row.get("entity_id")) not in library.entities
            ),
            attribute_name=_text(row.get("attribute")),
            attribute_description=attribute_description,
            attribute_aliases=attribute_aliases,
            value_type=_text(row.get("value_type")),
            categorical_value=(
                _text(row.get("categorical_value")) or None
            ),
            numerical_type=_text(row.get("numerical_type")) or None,
            numerical_value=_text(row.get("numerical_value")) or None,
            unit=_text(row.get("unit")) or None,
            confidence=float(row.get("confidence")),
            rationale=_text(row.get("rationale")),
            review_status="approved",
            review_notes=IMPORTED_REVIEW_NOTE,
            mapping_review=EvaMappingReviewTrace(
                status="legacy_unreviewed",
                initial_mapping_decision=(
                    "existing_attribute" if attribute is not None else "new_attribute"
                ),
                initial_attribute_id=_text(row.get("attribute_id")),
                final_mapping_decision=(
                    "existing_attribute" if attribute is not None else "new_attribute"
                ),
                final_attribute_id=_text(row.get("attribute_id")),
            ),
            attribute_id_review=EvaAttributeIdReviewTrace(
                status=("not_required" if attribute is not None else "legacy_unreviewed"),
                current_attribute_id=_text(row.get("attribute_id")),
                reviewed_attribute_id=_text(row.get("attribute_id")),
            ),
        )
        EvaLibraryRepository._validate_item_value(item)
        items.append(item)

    audit_id = (
        existing.audit_id
        if existing is not None
        else stable_id(
            "eva_audit_verified",
            trial_key,
            verified["sha256"],
        )
    )
    audit = EvaAuditDocument(
        audit_id=audit_id,
        trial_key=trial_key,
        trial_id=_text(rows[0].get("trial_id")),
        source_path=str(points_path),
        library_revision=library.revision,
        library_sha256=library.sha256,
        created_at=existing.created_at if existing is not None else now,
        updated_at=now,
        review_status="approved",
        search_model=(
            existing.search_model
            if existing is not None
            else "reviewer-verified-import"
        ),
        search_reasoning_effort=(
            existing.search_reasoning_effort
            if existing is not None
            else "not_applicable"
        ),
        reasoner_model=(
            existing.reasoner_model
            if existing is not None
            else "reviewer-verified-import"
        ),
        reasoner_reasoning_effort=(
            existing.reasoner_reasoning_effort
            if existing is not None
            else "not_applicable"
        ),
        review_mode="off",
        items=items,
        approved_at=now,
        approved_library_revision=(
            existing.approved_library_revision
            if existing is not None
            and _definitions_satisfied(library, existing)
            else None
        ),
    )
    for verified_row, item in zip(rows, audit.items):
        projection = {
            **_audit_item_projection(item),
            "trial_key": audit.trial_key,
            "trial_id": audit.trial_id,
        }
        if _normalized_core(projection) != _normalized_core(verified_row):
            raise ValueError(
                f"{trial_key}: imported audit projection differs from "
                f"verified row '{item.eva_id}'."
            )
    return audit


def import_verified_audits(
    *,
    verified_directory: str | Path = DEFAULT_VERIFIED_DIRECTORY,
    points_directory: str | Path = DEFAULT_POINTS_DIRECTORY,
    library_path: str | Path = DEFAULT_LIBRARY_PATH,
    work_directory: str | Path = DEFAULT_WORK_DIRECTORY,
) -> dict[str, Any]:
    """Populate eligibility-audit files without changing the EVA library."""

    verified_trials = load_verified_trials(verified_directory)
    points_root = Path(points_directory).resolve()
    work_root = Path(work_directory).resolve()
    audit_directory = work_root / "audits"
    audit_directory.mkdir(parents=True, exist_ok=True)
    library = EvaLibraryRepository(library_path)
    now = _utc_now()
    prepared: dict[str, EvaAuditDocument] = {}
    existing_payloads: dict[str, dict[str, Any]] = {}
    for trial_key, verified in verified_trials.items():
        points_path = (
            points_root / trial_key / "elig_breakdown_points.csv"
        ).resolve()
        if not points_path.is_file():
            raise ValueError(
                f"{trial_key}: atomic eligibility points not found at "
                f"{points_path}."
            )
        audit_path = audit_directory / f"{trial_key}.eva_audit.json"
        existing_payload = None
        if audit_path.is_file():
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Existing audit must be a JSON object: {audit_path}"
                )
            existing_payload = payload
            existing_payloads[trial_key] = payload
        prepared[trial_key] = _build_imported_audit(
            trial_key=trial_key,
            verified=verified,
            points_path=points_path,
            library=library,
            existing_payload=existing_payload,
            now=now,
        )

    run_directory = (
        work_root / "verified_sync" / f"{_run_stamp()}_import"
    )
    backup_directory = run_directory / "audit_backups"
    for trial_key, payload in existing_payloads.items():
        atomic_write_json(
            backup_directory / f"{trial_key}.eva_audit.json",
            payload,
        )
    for trial_key, audit in prepared.items():
        atomic_write_json(
            audit_directory / f"{trial_key}.eva_audit.json",
            audit.model_dump(mode="json"),
        )
    summary = {
        "phase": "import_audits",
        "completed_at": _utc_now(),
        "verified_directory": str(Path(verified_directory).resolve()),
        "points_directory": str(points_root),
        "library_path": str(Path(library_path).resolve()),
        "library_revision_unchanged": library.revision,
        "audit_directory": str(audit_directory),
        "trial_count": len(prepared),
        "row_count": sum(
            len(audit.items) for audit in prepared.values()
        ),
        "replaced_audit_count": len(existing_payloads),
        "trial_keys": list(prepared),
        "verified_sources": {
            trial_key: {
                "path": str(verified["path"]),
                "sha256": verified["sha256"],
                "row_count": len(verified["rows"]),
            }
            for trial_key, verified in verified_trials.items()
        },
        "run_directory": str(run_directory),
    }
    atomic_write_json(run_directory / "import_summary.json", summary)
    return summary


def _load_matching_imported_audits(
    *,
    verified_trials: Mapping[str, Mapping[str, Any]],
    audit_directory: Path,
) -> dict[str, EvaAuditDocument]:
    audits: dict[str, EvaAuditDocument] = {}
    for trial_key, verified in verified_trials.items():
        path = audit_directory / f"{trial_key}.eva_audit.json"
        if not path.is_file():
            raise ValueError(
                f"{trial_key}: imported audit not found. Run the audit "
                f"import phase first: {path}"
            )
        audit = load_eva_audit(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if audit.trial_key != trial_key:
            raise ValueError(f"{path}: audit trial identity mismatch.")
        if len(audit.items) != len(verified["rows"]):
            raise ValueError(
                f"{path}: audit and verified row counts differ."
            )
        for verified_row, item in zip(verified["rows"], audit.items):
            projection = {
                **_audit_item_projection(item),
                "trial_key": audit.trial_key,
                "trial_id": audit.trial_id,
            }
            if _normalized_core(projection) != _normalized_core(
                verified_row
            ):
                raise ValueError(
                    f"{path}: audit item '{item.eva_id}' no longer matches "
                    "the verified source. Re-run the audit import phase."
                )
        audits[trial_key] = audit
    return audits


def apply_imported_audits_to_library(
    *,
    verified_directory: str | Path = DEFAULT_VERIFIED_DIRECTORY,
    library_path: str | Path = DEFAULT_LIBRARY_PATH,
    work_directory: str | Path = DEFAULT_WORK_DIRECTORY,
    retire_unreferenced: bool = False,
) -> dict[str, Any]:
    """Apply imported audits through a staged, all-or-nothing library update."""

    verified_trials = load_verified_trials(verified_directory)
    library_file = Path(library_path).resolve()
    work_root = Path(work_directory).resolve()
    audit_directory = work_root / "audits"
    normalized_directory = work_root / "normalized"
    audits = _load_matching_imported_audits(
        verified_trials=verified_trials,
        audit_directory=audit_directory,
    )
    current_library = EvaLibraryRepository(library_file)
    previous_payload = current_library.payload
    previous_revision = current_library.revision
    run_directory = (
        work_root / "verified_sync" / f"{_run_stamp()}_apply"
    )
    backup_path = (
        run_directory
        / "library_backup"
        / f"eligibility_eva_library.r{previous_revision}.json"
    )
    staged_path = run_directory / "staging" / library_file.name
    atomic_write_json(backup_path, previous_payload)
    atomic_write_json(staged_path, previous_payload)
    staged = EvaLibraryRepository(staged_path)
    history = staged.payload.get("audit_history") or []
    applied_history = {
        _text(entry.get("audit_id")): entry
        for entry in history
        if isinstance(entry, dict) and _text(entry.get("audit_id"))
    }
    approved_audits: dict[str, EvaAuditDocument] = {}
    working_rows: dict[str, list[dict[str, Any]]] = {}
    applied_trials: list[str] = []
    already_applied_trials: list[str] = []

    for trial_key, audit in audits.items():
        history_entry = applied_history.get(audit.audit_id)
        if history_entry is not None and _definitions_satisfied(
            staged,
            audit,
        ):
            audit.review_status = "approved"
            audit.approved_at = (
                audit.approved_at or _text(history_entry.get("applied_at"))
            )
            audit.approved_library_revision = int(
                history_entry["new_revision"]
            )
            approved = audit
            normalized = list(verified_trials[trial_key]["rows"])
            already_applied_trials.append(trial_key)
        else:
            audit.library_revision = staged.revision
            audit.library_sha256 = staged.sha256
            approved, normalized = staged.apply_audit(audit)
            applied_trials.append(trial_key)
            for generated, verified_row in zip(
                normalized,
                verified_trials[trial_key]["rows"],
            ):
                if _normalized_core(generated) != _normalized_core(
                    verified_row
                ):
                    raise ValueError(
                        f"{trial_key}: applying imported audit changed "
                        f"verified row '{verified_row.get('eva_id')}'."
                    )
        approval_revision = approved.approved_library_revision
        rows = []
        for verified_row in verified_trials[trial_key]["rows"]:
            row = dict(verified_row)
            row["library_revision"] = approval_revision
            row["library_schema_version"] = staged.payload[
                "schema_version"
            ]
            rows.append(row)
        approved_audits[trial_key] = approved
        working_rows[trial_key] = rows

    retired_attribute_ids: list[str] = []
    if retire_unreferenced:
        verified_attribute_ids = {
            _text(row.get("attribute_id"))
            for verified in verified_trials.values()
            for row in verified["rows"]
        }
        staged_payload = staged.payload
        for attribute in staged_payload.get("attributes") or []:
            attribute_id = _text(attribute.get("attribute_id"))
            if (
                attribute.get("active", True)
                and attribute_id not in verified_attribute_ids
            ):
                attribute["active"] = False
                retired_attribute_ids.append(attribute_id)
        if retired_attribute_ids:
            prior_revision = int(staged_payload["revision"])
            staged_payload["revision"] = prior_revision + 1
            staged_payload["updated_at"] = _utc_now()
            staged_payload.setdefault("retirement_history", []).append(
                {
                    "retired_at": staged_payload["updated_at"],
                    "previous_revision": prior_revision,
                    "new_revision": staged_payload["revision"],
                    "reason": (
                        "Not referenced by the authoritative reviewer-"
                        "verified eligibility EVA dataset."
                    ),
                    "verified_directory": str(
                        Path(verified_directory).resolve()
                    ),
                    "attribute_ids": sorted(retired_attribute_ids),
                }
            )
            atomic_write_json(staged_path, staged_payload)

    final_library = EvaLibraryRepository(staged_path)
    atomic_write_json(library_file, final_library.payload)
    for trial_key, audit in approved_audits.items():
        atomic_write_json(
            audit_directory / f"{trial_key}.eva_audit.json",
            audit.model_dump(mode="json"),
        )
        atomic_write_jsonl(
            normalized_directory / trial_key / VERIFIED_FILENAME,
            working_rows[trial_key],
        )
    summary = {
        "phase": "apply_library",
        "completed_at": _utc_now(),
        "verified_directory": str(Path(verified_directory).resolve()),
        "library_path": str(library_file),
        "previous_library_revision": previous_revision,
        "new_library_revision": final_library.revision,
        "entity_count": len(final_library.entities),
        "attribute_count": len(final_library.attributes),
        "active_attribute_count": sum(
            attribute.active
            for attribute in final_library.attributes.values()
        ),
        "retired_attribute_ids": sorted(retired_attribute_ids),
        "trial_count": len(audits),
        "row_count": sum(len(audit.items) for audit in audits.values()),
        "applied_trials": applied_trials,
        "already_applied_trials": already_applied_trials,
        "library_backup_path": str(backup_path),
        "normalized_directory": str(normalized_directory),
        "run_directory": str(run_directory),
    }
    atomic_write_json(run_directory / "apply_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import reviewer-verified eligibility EVA rows into audit state, "
            "then apply those imported audits to the terminology library."
        )
    )
    parser.add_argument(
        "phase",
        choices=("import-audits", "apply-library"),
        help=(
            "Run import-audits first. apply-library refuses audits that no "
            "longer exactly match the verified JSONL source."
        ),
    )
    parser.add_argument(
        "--verified-directory",
        default=str(DEFAULT_VERIFIED_DIRECTORY),
    )
    parser.add_argument(
        "--points-directory",
        default=str(DEFAULT_POINTS_DIRECTORY),
    )
    parser.add_argument(
        "--library",
        default=str(DEFAULT_LIBRARY_PATH),
    )
    parser.add_argument(
        "--work-directory",
        default=str(DEFAULT_WORK_DIRECTORY),
    )
    parser.add_argument(
        "--retire-unreferenced",
        action="store_true",
        help=(
            "During apply-library, mark active library Attributes absent "
            "from the complete verified directory inactive while retaining "
            "their revision history."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "import-audits":
        result = import_verified_audits(
            verified_directory=args.verified_directory,
            points_directory=args.points_directory,
            library_path=args.library,
            work_directory=args.work_directory,
        )
    else:
        result = apply_imported_audits_to_library(
            verified_directory=args.verified_directory,
            library_path=args.library,
            work_directory=args.work_directory,
            retire_unreferenced=args.retire_unreferenced,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
