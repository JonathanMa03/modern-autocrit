"""Losslessly reconstruct EVA audits from completed runtime call records."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from backend.criteria_processor.eva_library import EvaLibraryRepository, stable_id
from backend.criteria_processor.eva_models import (
    EvaAttributeCandidate,
    EvaAttributeIdReviewTrace,
    EvaAuditDocument,
    EvaMappingReviewTrace,
    EvaReasoningSelection,
    EvaSearchExpansion,
)
from backend.criteria_processor.eva_pipeline import (
    REASONER_PROMPT_VERSION,
    build_eva_audit_item,
    finalize_trial_attribute_id_conflicts,
    load_breakdown_points,
    normalize_search_expansion,
    validate_reasoning_selection,
)
from backend.criteria_processor.eva_review import (
    ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
    ID_REVIEW_NOT_REQUIRED_EXPLANATION,
    MAPPING_CORRECTION_PROMPT_VERSION,
    MAPPING_REVIEW_PROMPT_VERSION,
    review_eva_mapping,
    review_new_attribute_id,
)
from backend.criteria_processor.llm_providers import (
    _prompt_input_payload,
    _record_stage,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _review_record_contract(
    record: Mapping[str, Any],
) -> Literal["v2", "v3"]:
    """Identify the recorded review provider contract without using its stage."""

    properties = (record.get("output_schema") or {}).get("properties") or {}
    if "explanation" in properties:
        return "v3"
    if "rationale_zh" in properties or "notes_zh" in properties:
        return "v2"
    raise ValueError("Unknown Eligibility EVA review prompt contract.")


_LEGACY_REPLAY_RELATIONSHIP = {
    "CONFIRM_EXISTING": "exact_equivalent",
    "CONFIRM_NEW": "exact_equivalent",
    "SWITCH_EXISTING": "exact_equivalent",
    "REQUIRE_NEW": "related_not_equivalent",
    "RETRIEVE_AGAIN": "no_supported_target",
    "SEGMENTATION_ISSUE": "segmentation_issue",
    "NEEDS_HUMAN_REVIEW": "related_not_equivalent",
}


def _adapt_v2_review_response(record: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt v2 output only long enough to replay its recorded transition."""

    response = dict(record.get("response_json") or {})
    stage = str(record.get("stage") or "")
    if stage == "eligibility_eva_mapping_reviewer":
        decision = str(response["decision"])
        response["semantic_relationship"] = _LEGACY_REPLAY_RELATIONSHIP[
            decision
        ]
        response["attribute_id_policy_status"] = (
            "not_applicable"
            if decision in {"RETRIEVE_AGAIN", "SEGMENTATION_ISSUE"}
            else "compliant"
        )
        response["explanation"] = (
            "Legacy v2 evidence is being replayed without semantic inference."
        )
        response.pop("rationale_zh", None)
    elif stage == "eligibility_eva_attribute_id_reviewer":
        response["explanation"] = (
            "Legacy v2 ID-review evidence is being replayed without translation."
        )
        response.pop("notes_zh", None)
    return response


def _assert_single_review_contract(
    records: Sequence[Mapping[str, Any]],
) -> Literal["v2", "v3"]:
    """Reject runtime chains that mix current and legacy review schemas."""

    reviewer_records = [
        record
        for record in records
        if record.get("stage")
        in {
            "eligibility_eva_mapping_reviewer",
            "eligibility_eva_attribute_id_reviewer",
        }
    ]
    contracts = {_review_record_contract(record) for record in reviewer_records}
    if len(contracts) > 1:
        raise ValueError(
            "Mixed Eligibility EVA review prompt contracts cannot be replayed."
        )
    return next(iter(contracts), "v3")


def _completed_records_by_source(
    run_directory: Path,
    *,
    expected_stage: str,
) -> dict[str, list[dict[str, Any]]]:
    calls_directory = run_directory / "calls"
    if not calls_directory.is_dir():
        raise ValueError(
            f"Runtime calls directory was not found: {calls_directory}"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in calls_directory.glob("*.completed.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("status") != "completed"
            or payload.get("stage") != expected_stage
        ):
            continue
        input_payload = payload.get("input_payload")
        response_payload = payload.get("response_json")
        if not isinstance(input_payload, dict) or not isinstance(
            response_payload,
            dict,
        ):
            continue
        source_item_id = str(
            input_payload.get("source_item_id") or ""
        ).strip()
        if source_item_id:
            grouped[source_item_id].append(payload)
    for records in grouped.values():
        records.sort(
            key=lambda record: (
                str(record.get("completed_at") or ""),
                str(record.get("call_id") or ""),
            )
        )
    return dict(grouped)


def _review_records_by_source(
    run_directory: Path,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls_directory = run_directory / "calls"
    if not calls_directory.is_dir():
        return {}
    accepted_stages = {
        "eligibility_eva_mapping_reviewer",
        "eligibility_eva_mapping_corrector",
        "eligibility_eva_attribute_id_reviewer",
        "eligibility_eva_reasoner",
    }
    for pattern in ("*.completed.json", "*.failed.json"):
        for path in calls_directory.glob(pattern):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("stage") not in accepted_stages:
                continue
            input_payload = payload.get("input_payload")
            if not isinstance(input_payload, dict):
                continue
            source = str(input_payload.get("source_item_id") or "").strip()
            if source:
                grouped[source].append(payload)
    for records in grouped.values():
        records.sort(
            key=lambda record: (
                str(record.get("started_at") or ""),
                str(record.get("call_id") or ""),
                0 if record.get("status") == "failed" else 1,
            )
        )
    return dict(grouped)


class _RuntimeReplayProvider:
    """Replay recorded structured calls while enforcing lineage equality."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.review_contract = _assert_single_review_contract(records)
        self.used: set[str] = set()
        self.model = ""
        self.reasoning_effort = ""
        self.last_completed_at: str | None = None

    def generate_json(self, prompt: str, *, output_schema: dict[str, Any]):
        stage = _record_stage(output_schema)
        generated = _prompt_input_payload(prompt, stage)
        if not isinstance(generated, dict):
            raise ValueError(f"Recovery cannot parse {stage} input payload.")
        source = str(generated.get("source_item_id") or "")
        lineage_keys = {
            "review_cycle",
            "candidate_set_fingerprint",
            "selection_fingerprint",
            "locked_mapping_decision",
            "locked_attribute_id",
        }
        saw_stage = False
        for record in self.records:
            call_id = str(record.get("call_id") or "")
            if call_id in self.used or record.get("stage") != stage:
                continue
            recorded = record.get("input_payload") or {}
            if str(recorded.get("source_item_id") or "") != source:
                continue
            saw_stage = True
            mismatched = [
                key
                for key in lineage_keys
                if key in generated or key in recorded
                if generated.get(key) != recorded.get(key)
            ]
            if stage == "eligibility_eva_reasoner":
                for key in ("search_expansion", "top_attribute_candidates"):
                    if generated.get(key) != recorded.get(key):
                        mismatched.append(key)
            if mismatched:
                continue
            self.used.add(call_id)
            self.model = str(record.get("model") or "")
            self.reasoning_effort = str(record.get("reasoning_effort") or "")
            if record.get("status") == "failed":
                error = record.get("error") or {}
                message = str(error.get("message") or "recorded call failed")
                exc_type = ValueError if error.get("type") == "ValueError" else RuntimeError
                raise exc_type(message)
            self.last_completed_at = str(record.get("completed_at") or "")
            response = record.get("response_json")
            if not isinstance(response, dict):
                raise ValueError(f"{stage} recorded response is invalid.")
            if (
                self.review_contract == "v2"
                and stage
                in {
                    "eligibility_eva_mapping_reviewer",
                    "eligibility_eva_attribute_id_reviewer",
                }
            ):
                return _adapt_v2_review_response(record)
            return response
        detail = "fingerprint/lineage mismatch" if saw_stage else "record missing"
        raise ValueError(f"{source}: {stage} {detail}.")


def _assert_source_matches(
    *,
    source_item_id: str,
    expected_row: Mapping[str, str],
    record_input: Mapping[str, Any],
) -> None:
    for field in ("criteria", "item", "context"):
        expected = str(expected_row[field])
        actual = str(record_input.get(field) or "")
        if actual != expected:
            raise ValueError(
                f"{source_item_id}: runtime {field} does not match the "
                "current eligibility-breakdown source."
            )


def recover_trial_eva_from_runtime_records(
    *,
    trial_key: str,
    trial_id: str,
    breakdown_path: str | Path,
    library: EvaLibraryRepository,
    search_run_directory: str | Path,
    reasoner_run_directory: str | Path,
    search_model: str,
    search_reasoning_effort: str,
    reasoner_model: str,
    reasoner_reasoning_effort: str,
    created_at: str | None = None,
) -> EvaAuditDocument:
    """Rebuild a complete pending audit without making any LLM calls."""

    rows = load_breakdown_points(breakdown_path)
    search_by_source = _completed_records_by_source(
        Path(search_run_directory),
        expected_stage="eligibility_eva_search_expander",
    )
    reasoner_by_source = _completed_records_by_source(
        Path(reasoner_run_directory),
        expected_stage="eligibility_eva_reasoner",
    )
    all_review_records = _review_records_by_source(
        Path(reasoner_run_directory)
    )
    _assert_single_review_contract(
        [
            record
            for records in all_review_records.values()
            for record in records
        ]
    )

    items = []
    for index, row in enumerate(rows, start=1):
        source_item_id = f"{trial_key}:{index:04d}"
        search_records = search_by_source.get(source_item_id, [])
        reasoner_records = reasoner_by_source.get(source_item_id, [])
        if not search_records:
            raise ValueError(
                f"{source_item_id}: completed search record is missing."
            )
        if not reasoner_records:
            raise ValueError(
                f"{source_item_id}: completed reasoner record is missing."
            )

        source_review_records = all_review_records.get(source_item_id, [])
        mapping_records = [
            record
            for record in source_review_records
            if record.get("stage") == "eligibility_eva_mapping_reviewer"
        ]
        first_mapping_started = min(
            (str(record.get("started_at") or "") for record in mapping_records),
            default="",
        )
        initial_reasoner_records = [
            record
            for record in reasoner_records
            if not first_mapping_started
            or str(record.get("started_at") or "") <= first_mapping_started
        ]
        valid_item = None
        base_result = None
        validation_errors: list[str] = []
        for reasoner_record in reversed(initial_reasoner_records):
            reasoner_input = reasoner_record["input_payload"]
            try:
                _assert_source_matches(
                    source_item_id=source_item_id,
                    expected_row=row,
                    record_input=reasoner_input,
                )
                expansion = normalize_search_expansion(
                    EvaSearchExpansion.model_validate(
                        reasoner_input.get("search_expansion")
                    )
                )
                matching_search = False
                for search_record in search_records:
                    _assert_source_matches(
                        source_item_id=source_item_id,
                        expected_row=row,
                        record_input=search_record["input_payload"],
                    )
                    recorded_expansion = normalize_search_expansion(
                        EvaSearchExpansion.model_validate(
                            search_record["response_json"]
                        )
                    )
                    if recorded_expansion == expansion:
                        matching_search = True
                        break
                if not matching_search:
                    raise ValueError(
                        "reasoner search expansion has no matching "
                        "completed search record."
                    )
                candidates = [
                    EvaAttributeCandidate.model_validate(candidate)
                    for candidate in (
                        reasoner_input.get("top_attribute_candidates")
                        or []
                    )
                ]
                selection = validate_reasoning_selection(
                    selection=EvaReasoningSelection.model_validate(
                        reasoner_record["response_json"]
                    ),
                    candidates=candidates,
                    library=library,
                    source_item=row["item"],
                )
                base_result = (expansion, candidates, selection)
                break
            except (TypeError, ValueError) as exc:
                validation_errors.append(str(exc))
        if base_result is None:
            detail = "; ".join(validation_errors) or "no valid response"
            raise ValueError(
                f"{source_item_id}: runtime result cannot be recovered: "
                f"{detail}"
            )
        expansion, candidates, selection = base_result
        if mapping_records:
            replay_records = [
                record
                for record in source_review_records
                if record.get("stage") != "eligibility_eva_reasoner"
                or str(record.get("started_at") or "") > first_mapping_started
            ]
            replay = _RuntimeReplayProvider(replay_records)
            selection, expansion, candidates, mapping_trace = review_eva_mapping(
                library=library,
                reasoner_provider=replay,
                trial_id=trial_id,
                source_item_id=source_item_id,
                row=row,
                expansion=expansion,
                candidates=candidates,
                selection=selection,
                top_k=10,
            )
            if (
                mapping_trace.status == "failed"
                and "mismatch" in str(mapping_trace.error or {}).casefold()
            ):
                raise ValueError(
                    f"{source_item_id}: review fingerprint/lineage mismatch."
                )
            if (
                mapping_trace.status == "completed"
                and selection.mapping_decision == "new_attribute"
            ):
                selection, id_trace = review_new_attribute_id(
                    reasoner_provider=replay,
                    source_item_id=source_item_id,
                    atomic_criterion_title=row["item"],
                    selection=selection,
                    library=library,
                )
                if (
                    id_trace.status == "failed"
                    and "mismatch" in str(id_trace.error or {}).casefold()
                ):
                    raise ValueError(
                        f"{source_item_id}: ID review fingerprint/lineage mismatch."
                    )
            elif selection.mapping_decision == "existing_attribute":
                _selection, id_trace = review_new_attribute_id(
                    reasoner_provider=replay,
                    source_item_id=source_item_id,
                    atomic_criterion_title=row["item"],
                    selection=selection,
                    library=library,
                )
            else:
                id_trace = EvaAttributeIdReviewTrace(
                    status="not_run",
                    current_attribute_id=selection.attribute_id or "",
                    reviewed_attribute_id=selection.attribute_id or "",
                    explanation=(
                        "Attribute ID Review was not run because Mapping Review "
                        "did not produce a completed new-Attribute decision."
                    ),
                    explanation_language="en",
                )
            if replay.review_contract == "v2":
                successful_mapping_records = [
                    record
                    for record in replay_records
                    if record.get("stage")
                    == "eligibility_eva_mapping_reviewer"
                    and record.get("status") == "completed"
                    and str(record.get("call_id") or "") in replay.used
                ]
                for attempt, record in zip(
                    mapping_trace.attempts,
                    successful_mapping_records,
                ):
                    original = dict(record.get("response_json") or {})
                    attempt.semantic_relationship = "legacy_unverified"
                    attempt.legacy_semantic_relationship = str(
                        original.get("semantic_relationship") or ""
                    )
                    attempt.attribute_id_policy_status = "legacy_unverified"
                    attempt.explanation = str(original.get("rationale_zh") or "")
                    attempt.explanation_language = "legacy_unverified"
                successful_id_records = [
                    record
                    for record in replay_records
                    if record.get("stage")
                    == "eligibility_eva_attribute_id_reviewer"
                    and record.get("status") == "completed"
                    and str(record.get("call_id") or "") in replay.used
                ]
                if successful_id_records:
                    original_id = dict(
                        successful_id_records[-1].get("response_json") or {}
                    )
                    id_trace.explanation = str(
                        original_id.get("notes_zh") or ""
                    )
                    id_trace.explanation_language = "legacy_unverified"
        else:
            mapping_trace = EvaMappingReviewTrace(
                status="legacy_unreviewed",
                initial_mapping_decision=selection.mapping_decision,
                initial_attribute_id=selection.attribute_id or "",
                final_mapping_decision=selection.mapping_decision,
                final_attribute_id=selection.attribute_id or "",
            )
            id_trace = EvaAttributeIdReviewTrace(
                status=(
                    "not_required"
                    if selection.mapping_decision == "existing_attribute"
                    else "legacy_unreviewed"
                ),
                current_attribute_id=selection.attribute_id or "",
                reviewed_attribute_id=selection.attribute_id or "",
                explanation=(
                    ID_REVIEW_NOT_REQUIRED_EXPLANATION
                    if selection.mapping_decision == "existing_attribute"
                    else "Attribute ID Review was not run for legacy evidence."
                ),
                explanation_language="en",
            )
        items.append(
            build_eva_audit_item(
                trial_id=trial_id,
                source_item_id=source_item_id,
                row=row,
                expansion=expansion,
                candidates=candidates,
                selection=selection,
                mapping_review=mapping_trace,
                attribute_id_review=id_trace,
            )
        )

    now = _utc_now()
    return EvaAuditDocument(
        audit_id=stable_id(
            "eva_audit",
            trial_key,
            library.revision,
            library.sha256,
        ),
        trial_key=trial_key,
        trial_id=trial_id,
        source_path=str(Path(breakdown_path).resolve()),
        library_revision=library.revision,
        library_sha256=library.sha256,
        created_at=created_at or now,
        updated_at=now,
        search_model=search_model,
        search_reasoning_effort=search_reasoning_effort,
        reasoner_model=reasoner_model,
        reasoner_reasoning_effort=reasoner_reasoning_effort,
        review_mode=(
            "full"
            if any(item.mapping_review.status != "legacy_unreviewed" for item in items)
            else "off"
        ),
        review_prompt_versions={
            "reasoner": REASONER_PROMPT_VERSION,
            "mapping_review": MAPPING_REVIEW_PROMPT_VERSION,
            "mapping_correction": MAPPING_CORRECTION_PROMPT_VERSION,
            "attribute_id_review": ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
        },
        items=finalize_trial_attribute_id_conflicts(items, library),
    )
