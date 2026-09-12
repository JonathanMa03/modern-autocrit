"""Load Eligibility EVA audit documents across supported schema versions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from backend.criteria_processor.eva_models import EvaAuditDocument


AUDIT_SCHEMA_V1 = "eligcrit.eligibility_eva_audit.v1"
AUDIT_SCHEMA_V2 = "eligcrit.eligibility_eva_audit.v2"
AUDIT_SCHEMA_V3 = "eligcrit.eligibility_eva_audit.v3"
_KNOWN_V2_RELATIONSHIPS = {
    "exact": "exact_equivalent",
    "equivalent": "exact_equivalent",
    "exact_equivalent": "exact_equivalent",
    "broader": "broader_than_source",
    "broader_than_source": "broader_than_source",
    "narrower": "narrower_than_source",
    "narrower_than_source": "narrower_than_source",
    "related": "related_not_equivalent",
    "related_not_equivalent": "related_not_equivalent",
}


def migrate_eva_audit_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic in-memory v2 representation of a v1 audit."""

    migrated = deepcopy(dict(payload))
    if migrated.get("schema_version") != AUDIT_SCHEMA_V1:
        raise ValueError("Expected an Eligibility EVA audit v1 payload.")

    migrated["schema_version"] = AUDIT_SCHEMA_V2
    migrated["review_mode"] = "off"
    migrated.setdefault("review_prompt_versions", {})
    migrated.setdefault("approved_library_sha256", None)
    for item in migrated.get("items", []):
        mapping_decision = item.get("mapping_decision", "new_attribute")
        attribute_id = str(item.get("attribute_id") or "")
        item["mapping_review"] = {
            "status": "legacy_unreviewed",
            "initial_mapping_decision": mapping_decision,
            "initial_attribute_id": attribute_id,
            "final_mapping_decision": mapping_decision,
            "final_attribute_id": attribute_id,
            "final_decision": None,
            "attempts": [],
            "error": None,
        }
        id_status = (
            "not_required"
            if mapping_decision == "existing_attribute"
            else "legacy_unreviewed"
        )
        item["attribute_id_review"] = {
            "status": id_status,
            "current_attribute_id": attribute_id,
            "reviewed_attribute_id": attribute_id,
            "decision": None,
            "notes_zh": "",
            "confidence": None,
            "selection_fingerprint": "",
            "model": "",
            "reasoning_effort": "",
            "timestamp": None,
            "error": None,
        }
    return migrated


def migrate_eva_audit_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lossless in-memory v3 representation of a v2 audit."""

    migrated = deepcopy(dict(payload))
    if migrated.get("schema_version") != AUDIT_SCHEMA_V2:
        raise ValueError("Expected an Eligibility EVA audit v2 payload.")

    migrated["schema_version"] = AUDIT_SCHEMA_V3
    for item in migrated.get("items", []):
        mapping_trace = item.get("mapping_review") or {}
        for attempt in mapping_trace.get("attempts", []):
            raw_relationship = str(
                attempt.pop("semantic_relationship", "") or ""
            ).strip()
            normalized = _KNOWN_V2_RELATIONSHIPS.get(
                raw_relationship.casefold()
            )
            attempt["semantic_relationship"] = (
                normalized or "legacy_unverified"
            )
            attempt["legacy_semantic_relationship"] = (
                "" if normalized else raw_relationship
            )
            attempt["attribute_id_policy_status"] = "legacy_unverified"
            attempt["explanation"] = str(
                attempt.pop("rationale_zh", "") or ""
            )
            attempt["explanation_language"] = "legacy_unverified"

        id_trace = item.get("attribute_id_review") or {}
        id_trace["explanation"] = str(
            id_trace.pop("notes_zh", "") or ""
        )
        id_trace["explanation_language"] = "legacy_unverified"
    return migrated


def load_eva_audit(payload: Mapping[str, Any]) -> EvaAuditDocument:
    """Validate v3 or migrate a v1/v2 payload without writing it."""

    version = payload.get("schema_version")
    if version == AUDIT_SCHEMA_V3:
        return EvaAuditDocument.model_validate(payload)
    if version == AUDIT_SCHEMA_V2:
        return EvaAuditDocument.model_validate(migrate_eva_audit_v2(payload))
    if version == AUDIT_SCHEMA_V1:
        return EvaAuditDocument.model_validate(
            migrate_eva_audit_v2(migrate_eva_audit_v1(payload))
        )
    raise ValueError(
        f"Unsupported Eligibility EVA audit schema_version: {version!r}."
    )
