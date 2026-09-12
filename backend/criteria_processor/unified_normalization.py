"""Unified terminology matching, working-library update, and normalization."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from pydantic import ValidationError

from backend.criteria_processor.eav_schema import canonical_entity
from backend.criteria_processor.mapping_suggestions import (
    JsonLLMProvider,
    MappingSuggestionEngine,
)
from backend.criteria_processor.models import (
    ExtractedCriterion,
    MappingEvidence,
    MappingProposal,
    MappingSelection,
    TerminologyAuditRecord,
    TerminologyMapping,
)
from backend.criteria_processor.normalizer import CriteriaNormalizer
from backend.criteria_processor.terminology import TerminologyRepository
from backend.criteria_processor.terminology_updates import (
    apply_audit_records_to_workbook,
    audit_records_from_proposals,
    create_working_library,
    write_audit_csv,
)
from backend.criteria_processor.text import clean_attribute_key
from backend.criteria_processor.value_normalization import normalize_structured_value


CriterionLike = ExtractedCriterion | Mapping[str, Any]


def _row(criterion: CriterionLike) -> dict[str, Any]:
    if isinstance(criterion, Mapping):
        return dict(criterion)
    return criterion.model_dump(mode="json")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _cluster_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    entity = _text(row.get("entity") or row.get("raw_entity") or "Other")
    attribute = _text(row.get("attribute") or row.get("raw_attribute"))
    value = _text(row.get("value") or row.get("raw_value"))
    return (
        entity,
        clean_attribute_key(attribute),
        clean_attribute_key(value),
    )


def build_mapping_evidence_clusters(
    criteria: Iterable[CriterionLike],
    terminology: TerminologyRepository,
) -> dict[tuple[str, str, str], MappingEvidence]:
    """Group unmapped extracted criteria into term-level proposal evidence."""

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for criterion in criteria:
        row = _row(criterion)
        mapping = terminology.map_criterion(
            attribute=row.get("attribute") or row.get("raw_attribute"),
            value=row.get("value") or row.get("raw_value"),
            entity=row.get("entity") or row.get("raw_entity"),
        )
        if mapping.mapping_status == "mapped":
            continue
        grouped[_cluster_key(row)].append(row)

    evidence: dict[tuple[str, str, str], MappingEvidence] = {}
    for key, rows in grouped.items():
        entity, _attribute_key, _value_key = key
        attribute_counts = Counter(
            _text(row.get("attribute") or row.get("raw_attribute"))
            for row in rows
        )
        value_counts = Counter(
            _text(row.get("value") or row.get("raw_value"))
            for row in rows
            if _text(row.get("value") or row.get("raw_value"))
        )
        raw_attribute = attribute_counts.most_common(1)[0][0]
        raw_value = value_counts.most_common(1)[0][0] if value_counts else None
        trial_ids = []
        examples = []
        for row in rows:
            trial_id = _text(row.get("trial_id"))
            source_text = _text(row.get("source_text"))
            if trial_id and trial_id not in trial_ids:
                trial_ids.append(trial_id)
            if source_text and source_text not in examples:
                examples.append(source_text)
        first = rows[0]
        evidence[key] = MappingEvidence(
            raw_entity=entity,
            raw_attribute=raw_attribute,
            raw_value=raw_value,
            source_text=_text(first.get("source_text")),
            source_path=_text(first.get("source_path")) or None,
            trial_id=_text(first.get("trial_id")) or None,
            criterion_type=_text(first.get("criterion_type")) or None,
            occurrence_count=len(rows),
            example_trial_ids=trial_ids[:10],
            example_source_texts=examples[:10],
        )
    return evidence


def mapping_from_selection(
    *,
    selection: MappingSelection,
    row: Mapping[str, Any],
    terminology: TerminologyRepository,
) -> TerminologyMapping:
    """Convert an LLM selector decision into the row's current mapping."""

    raw_entity = _text(row.get("entity") or row.get("raw_entity")) or None
    raw_canonical_entity = canonical_entity(raw_entity)
    raw_attribute = _text(row.get("attribute") or row.get("raw_attribute"))
    raw_value = _text(row.get("value") or row.get("raw_value")) or None
    method = f"working_library_proposal:{selection.decision}"
    confidence = selection.confidence

    if selection.decision in {"alias_existing_attribute", "alias_existing_value"}:
        definition = (
            terminology.attributes.get(selection.attribute_id or "")
            if selection.attribute_id
            else None
        )
        value = (
            terminology.values.get(selection.value_id or "")
            if selection.value_id
            else None
        )
        return TerminologyMapping(
            raw_attribute=raw_attribute,
            raw_value=raw_value,
            attribute_id=selection.attribute_id,
            canonical_entity=(
                definition.parent_entity
                if definition
                else raw_canonical_entity
            ),
            canonical_attribute=(
                definition.canonical_name if definition else raw_attribute
            ),
            value_schema=(
                definition.value_schema if definition else "free_text"
            ),
            canonical_unit=(
                definition.canonical_unit if definition else None
            ),
            value_id=selection.value_id,
            canonical_value=value.canonical_value if value else raw_value,
            mapping_method=method,
            mapping_status="proposed",
            confidence=confidence,
            terminology_version=terminology.terminology_version,
        )

    if selection.decision == "new_value_for_existing_attribute":
        definition = terminology.attributes.get(selection.attribute_id or "")
        return TerminologyMapping(
            raw_attribute=raw_attribute,
            raw_value=raw_value,
            attribute_id=selection.attribute_id,
            canonical_entity=(
                definition.parent_entity
                if definition
                else raw_canonical_entity
            ),
            canonical_attribute=(
                definition.canonical_name if definition else raw_attribute
            ),
            value_schema=(
                definition.value_schema if definition else "categorical"
            ),
            canonical_unit=(
                definition.canonical_unit if definition else None
            ),
            value_id=selection.proposed_value_id,
            canonical_value=selection.proposed_canonical_value or raw_value,
            mapping_method=method,
            mapping_status="proposed",
            confidence=confidence,
            terminology_version=terminology.terminology_version,
        )

    if selection.decision == "new_attribute":
        return TerminologyMapping(
            raw_attribute=raw_attribute,
            raw_value=raw_value,
            attribute_id=selection.proposed_attribute_id,
            canonical_entity=selection.proposed_parent_entity,
            canonical_attribute=selection.proposed_canonical_name
            or raw_attribute,
            value_schema=selection.proposed_value_schema or "free_text",
            canonical_unit=selection.proposed_canonical_unit,
            value_id=selection.proposed_value_id,
            canonical_value=(
                selection.proposed_canonical_value or raw_value
            ),
            mapping_method=method,
            mapping_status="proposed",
            confidence=confidence,
            terminology_version=terminology.terminology_version,
        )

    return TerminologyMapping(
        raw_attribute=raw_attribute,
        raw_value=raw_value,
        canonical_entity=raw_canonical_entity,
        canonical_attribute=raw_attribute,
        canonical_value=raw_value,
        mapping_method=method,
        mapping_status=selection.decision,
        confidence=confidence,
        terminology_version=terminology.terminology_version,
    )


def _apply_mapping_to_normalized_row(
    *,
    normalized: dict[str, Any],
    mapping: TerminologyMapping,
) -> None:
    normalized["canonical_entity"] = mapping.canonical_entity
    normalized["canonical_attribute"] = mapping.canonical_attribute
    normalized["canonical_attribute_id"] = mapping.attribute_id
    normalized["canonical_value_id"] = mapping.value_id
    normalized["canonical_value"] = mapping.canonical_value
    normalized["terminology_version"] = mapping.terminology_version
    normalized["mapping_method"] = mapping.mapping_method
    normalized["mapping_status"] = mapping.mapping_status
    normalized["mapping_confidence"] = mapping.confidence
    structured = normalize_structured_value(
        raw_value=normalized.get("raw_value"),
        attribute_id=mapping.attribute_id,
        canonical_attribute=mapping.canonical_attribute,
        value_schema=mapping.value_schema,
        canonical_unit=mapping.canonical_unit,
        canonical_value=mapping.canonical_value,
        context_text="; ".join(
            str(item).strip()
            for item in [
                normalized.get("condition"),
                normalized.get("source_text"),
            ]
            if item is not None and str(item).strip()
        ),
        criterion_type=normalized.get("criterion_type"),
    )
    normalized["value_schema"] = structured["schema"]
    normalized["canonical_unit"] = mapping.canonical_unit
    normalized["normalized_value_json"] = json.dumps(
        structured,
        ensure_ascii=False,
        sort_keys=True,
    )
    normalized["value_category"] = structured["category"]
    normalized["value_lower"] = structured["lower"]
    normalized["value_upper"] = structured["upper"]
    normalized["value_lower_inclusive"] = structured["lower_inclusive"]
    normalized["value_upper_inclusive"] = structured["upper_inclusive"]
    normalized["value_unit"] = structured["unit"]
    normalized["value_lower_unit"] = structured["lower_unit"]
    normalized["value_upper_unit"] = structured["upper_unit"]
    normalized["normalized_value_lower"] = structured["normalized_lower"]
    normalized["normalized_value_upper"] = structured["normalized_upper"]
    normalized["normalized_value_unit"] = structured["normalized_unit"]
    normalized["selection_effect"] = structured["selection_effect"]
    normalized["value_assertion"] = structured["assertion"]
    normalized["eligible_values_json"] = json.dumps(
        structured["eligible_values"],
        ensure_ascii=False,
    )
    normalized["excluded_values_json"] = json.dumps(
        structured["excluded_values"],
        ensure_ascii=False,
    )
    normalized["eligible_ranges_json"] = json.dumps(
        structured["eligible_ranges"],
        ensure_ascii=False,
        sort_keys=True,
    )
    normalized["excluded_ranges_json"] = json.dumps(
        structured["excluded_ranges"],
        ensure_ascii=False,
        sort_keys=True,
    )
    normalized["value_parse_status"] = structured["parse_status"]
    if structured["category"] is not None:
        normalized["canonical_value"] = structured["category"]
    normalized["canonical_attribute_key"] = clean_attribute_key(
        normalized["canonical_attribute"]
    )
    normalized["canonical_value_key"] = clean_attribute_key(
        normalized["canonical_value"]
    )


def normalize_with_mapping_proposals(
    *,
    criteria: Iterable[CriterionLike],
    verified_library_path: str | Path,
    working_library_path: str | Path,
    audit_file_path: str | Path,
    llm_provider: JsonLLMProvider,
    fuzzy_threshold: int = 85,
    top_k: int = 5,
    proposal_checkpoint_path: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, list[MappingProposal]]:
    """Normalize criteria while simultaneously proposing library updates.

    Exact approved mappings are used when available. Unmapped term clusters are
    sent through the LLM expansion + fuzzy retrieval + LLM selector layer. The
    selector's target term is written directly into the returned normalized
    output with ``mapping_status='proposed'``. Proposed library changes are
    also applied to the working library and documented in the audit file.
    """

    criteria_rows = [_row(criterion) for criterion in criteria]
    verified_terminology = TerminologyRepository(verified_library_path)
    evidence_by_key = build_mapping_evidence_clusters(
        criteria_rows,
        verified_terminology,
    )
    create_working_library(
        verified_library_path=verified_library_path,
        working_library_path=working_library_path,
        audit_file_path=audit_file_path,
        proposals=[],
    )
    checkpoint_by_key = _load_proposal_checkpoint(proposal_checkpoint_path)
    # Preserve all durable checkpoint entries if a later proposal fails. Each
    # encountered cluster still revalidates its saved selection against the
    # current working library before use.
    applied_proposals: dict[tuple[str, str, str], MappingProposal] = dict(
        checkpoint_by_key
    )
    audit_records: list[TerminologyAuditRecord] = []
    proposal_applied_keys: set[tuple[str, str, str]] = set()
    evidence_items = list(evidence_by_key.items())
    total = len(evidence_items)
    for index, (key, evidence) in enumerate(evidence_items, start=1):
        working_terminology = TerminologyRepository(working_library_path)
        current_mapping = working_terminology.map_criterion(
            attribute=evidence.raw_attribute,
            value=evidence.raw_value,
            entity=evidence.raw_entity,
        )
        if current_mapping.mapping_status == "mapped":
            _progress(
                progress_callback,
                f"[{index}/{total}] matched earlier working-library update: "
                f"{evidence.raw_entity} | {evidence.raw_attribute}",
            )
            continue

        engine = MappingSuggestionEngine(
            working_terminology,
            llm_provider,
            fuzzy_threshold=fuzzy_threshold,
            top_k=top_k,
        )
        if key in checkpoint_by_key:
            proposal = checkpoint_by_key[key]
            proposal.selection = engine._guard_selection(
                evidence=evidence,
                selection=proposal.selection,
                attribute_candidates=proposal.attribute_candidates,
                value_candidates=proposal.value_candidates,
            )
            _progress(
                progress_callback,
                f"[{index}/{total}] reused checkpoint: "
                f"{evidence.raw_entity} | {evidence.raw_attribute}",
            )
        else:
            _progress(
                progress_callback,
                f"[{index}/{total}] proposing mapping: "
                f"{evidence.raw_entity} | {evidence.raw_attribute}",
            )
            proposal = engine.propose_mapping(evidence)
        applied_proposals[key] = proposal
        records = audit_records_from_proposals([proposal])
        working_path = Path(working_library_path)
        candidate_path = working_path.with_name(
            f".{working_path.stem}.{uuid4().hex}.candidate"
            f"{working_path.suffix}"
        )
        shutil.copy2(working_path, candidate_path)
        try:
            apply_audit_records_to_workbook(
                source_workbook=candidate_path,
                output_workbook=candidate_path,
                records=records,
                allowed_review_statuses={"proposed", "approved", "revised"},
                metadata_status="working",
            )
            # Loading the candidate performs the full schema and alias-conflict
            # validation before it is allowed to replace the working library.
            TerminologyRepository(candidate_path)
        except ValueError as exc:
            candidate_path.unlink(missing_ok=True)
            for record in records:
                record.review_status = "conflict"
                record.notes = (
                    f"{record.notes} Working-library conflict: {exc}"
                ).strip()
            _progress(
                progress_callback,
                f"[{index}/{total}] recorded conflict: "
                f"{evidence.raw_entity} | {evidence.raw_attribute}",
            )
        else:
            candidate_path.replace(working_path)
            proposal_applied_keys.add(key)
            _progress(
                progress_callback,
                f"[{index}/{total}] updated working library: "
                f"{evidence.raw_entity} | {evidence.raw_attribute}",
            )
        audit_records.extend(records)
        write_audit_csv(audit_records, audit_file_path)
        _write_proposal_checkpoint(
            proposal_checkpoint_path,
            applied_proposals.values(),
        )
    _write_proposal_checkpoint(
        proposal_checkpoint_path,
        applied_proposals.values(),
    )
    proposals = list(applied_proposals.values())
    all_audit_records = audit_records
    write_audit_csv(all_audit_records, audit_file_path)

    working_terminology = TerminologyRepository(working_library_path)
    exact_normalizer = CriteriaNormalizer(
        terminology_repository=working_terminology
    )
    changed_attribute_ids = {
        record.attribute_id
        for record in all_audit_records
        if record.target_object in {"attribute", "attribute_alias"}
        and record.attribute_id
        and record.review_status in {"proposed", "approved", "revised"}
    }
    changed_value_ids = {
        record.value_id
        for record in all_audit_records
        if record.target_object in {"value", "value_alias"}
        and record.value_id
        and record.review_status in {"proposed", "approved", "revised"}
    }
    normalized_rows: list[dict[str, Any]] = []
    for row in criteria_rows:
        verified_mapping = verified_terminology.map_criterion(
            attribute=row.get("attribute") or row.get("raw_attribute"),
            value=row.get("value") or row.get("raw_value"),
            entity=row.get("entity") or row.get("raw_entity"),
        )
        normalized = exact_normalizer.normalize_criterion(row)
        attribute_id = normalized.get("canonical_attribute_id")
        value_id = normalized.get("canonical_value_id")
        if verified_mapping.mapping_status != "mapped" and (
            attribute_id in changed_attribute_ids
            or value_id in changed_value_ids
        ):
            normalized["mapping_method"] = (
                f"working_library:{normalized['mapping_method']}"
            )
            normalized["mapping_status"] = "proposed"
        elif attribute_id is None:
            row_key = _cluster_key(row)
            proposal = applied_proposals.get(row_key)
            if proposal is not None and row_key in proposal_applied_keys:
                fallback = mapping_from_selection(
                    selection=proposal.selection,
                    row=row,
                    terminology=working_terminology,
                )
                _apply_mapping_to_normalized_row(
                    normalized=normalized,
                    mapping=fallback,
                )
            elif proposal is not None:
                normalized["mapping_status"] = "conflict"
                normalized["mapping_method"] = "working_library_conflict"
        normalized_rows.append(normalized)

    return pd.DataFrame(normalized_rows), proposals


def _proposal_key(proposal: MappingProposal) -> tuple[str, str, str]:
    return (
        _text(proposal.evidence.raw_entity or "Other"),
        clean_attribute_key(proposal.evidence.raw_attribute),
        clean_attribute_key(proposal.evidence.raw_value),
    )


def _load_proposal_checkpoint(
    path: str | Path | None,
) -> dict[tuple[str, str, str], MappingProposal]:
    if path is None:
        return {}
    checkpoint = Path(path)
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return {}
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    try:
        proposals = [MappingProposal.model_validate(item) for item in payload]
    except ValidationError:
        return {}
    return {_proposal_key(proposal): proposal for proposal in proposals}


def _write_proposal_checkpoint(
    path: str | Path | None,
    proposals: Iterable[MappingProposal],
) -> None:
    if path is None:
        return
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            [proposal.model_dump(mode="json") for proposal in proposals],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)
