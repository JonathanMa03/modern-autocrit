"""Deterministic integrity reconciliation for Eligibility EVA audits.

Client input contains decisions only. Candidate evidence is always rebuilt from
the locked Library. Plan B may pass a server-loaded trusted semantic preview to
the single final validator; request parsing must never accept that mapping.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from backend.criteria_processor.eva_models import (
    EvaApplyEnvelope,
    EvaAuditDocument,
    EvaAuditItem,
    EvaEntityDefinition,
    EvaLibraryReconciliationEvidence,
    EvaReconciliationCandidateEvidence,
    EvaReconciliationDecisionInput,
    EvaReconciliationSummary,
    EvaSemanticProposalReview,
)
from backend.criteria_processor.eva_retrieval import eva_schema_conflicts
from backend.criteria_processor.eva_retrieval import (
    SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
)
from backend.criteria_processor.text import clean_attribute_key

if TYPE_CHECKING:
    from backend.criteria_processor.eva_library import EvaLibraryRepository


class EvaReconciliationError(ValueError):
    """Base class for stable reconciliation-domain failures."""

    code = "reconciliation_invalid"

    def __init__(self, message: str, *, eva_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.eva_ids = list(dict.fromkeys(eva_ids))


class ReconciliationRequiredError(EvaReconciliationError):
    code = "reconciliation_required"


class ReconciliationConflictError(EvaReconciliationError):
    code = "reconciliation_conflict"


class SemanticRetrievalVersionStaleError(EvaReconciliationError):
    code = "semantic_retrieval_stale"


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def proposal_fingerprint(item: EvaAuditItem) -> str:
    """Hash identity/schema fields while ignoring values and review prose."""

    return _canonical_json_sha256(
        {
            "eva_id": item.eva_id,
            "mapping_decision": item.mapping_decision,
            "attribute_id": item.attribute_id,
            "attribute_name": item.attribute_name,
            "attribute_aliases": item.attribute_aliases,
            "entity_id": item.entity_id,
            "entity_name": item.entity_name,
            "new_entity": item.new_entity,
            "value_type": item.value_type,
            "numerical_type": item.numerical_type,
            "unit": item.unit,
        }
    )


def resolve_active_entity(
    item: EvaAuditItem,
    library: EvaLibraryRepository,
) -> tuple[EvaEntityDefinition | None, list[str]]:
    direct = library.entities.get(item.entity_id)
    if direct is not None and direct.active:
        return direct, []
    key = clean_attribute_key(item.entity_name)
    matches = [
        entity
        for entity in library.entities.values()
        if entity.active
        and key
        and any(
            clean_attribute_key(name) == key
            for name in [entity.canonical_name, *entity.aliases]
        )
    ]
    matches.sort(key=lambda entity: entity.entity_id)
    if len(matches) == 1:
        return matches[0], []
    if len(matches) > 1:
        return None, ["ambiguous_entity"]
    return None, []


def _candidate_evidence(item, entity, attribute):
    proposal_names = [item.attribute_name, *item.attribute_aliases]
    target_names = [attribute.canonical_name, *attribute.aliases]
    proposal_keys = {
        clean_attribute_key(name) for name in proposal_names if str(name).strip()
    }
    matched_names = sorted(
        {
            name
            for name in target_names
            if clean_attribute_key(name) in proposal_keys
        },
        key=lambda value: (clean_attribute_key(value), value),
    )
    conflicts = list(
        eva_schema_conflicts(
            value_type=item.value_type,
            numerical_type=item.numerical_type,
            canonical_unit=item.unit,
            candidate=attribute,
        )
    )
    return EvaReconciliationCandidateEvidence(
        attribute_id=attribute.attribute_id,
        canonical_name=attribute.canonical_name,
        entity_id=entity.entity_id,
        entity_name=entity.canonical_name,
        value_type=attribute.value_type,
        numerical_type=attribute.numerical_type,
        canonical_unit=attribute.canonical_unit,
        compatible=not conflicts,
        matched_names=matched_names,
        conflict_reasons=conflicts,
    )


def _evidence_for_item(
    item: EvaAuditItem,
    library: EvaLibraryRepository,
) -> EvaLibraryReconciliationEvidence:
    fingerprint = proposal_fingerprint(item)
    base = {
        "library_revision": library.revision,
        "library_sha256": library.sha256,
        "proposal_fingerprint": fingerprint,
    }
    if item.review_status == "rejected" or item.mapping_decision != "new_attribute":
        return EvaLibraryReconciliationEvidence(**base)
    entity, entity_conflicts = resolve_active_entity(item, library)
    if entity_conflicts:
        return EvaLibraryReconciliationEvidence(status="blocked", **base)
    if entity is None:
        return EvaLibraryReconciliationEvidence(**base)
    proposal_keys = {
        clean_attribute_key(name)
        for name in [item.attribute_name, *item.attribute_aliases]
        if str(name).strip()
    }
    candidates = []
    for attribute in library.attributes.values():
        if not attribute.active or attribute.entity_id != entity.entity_id:
            continue
        target_keys = {
            clean_attribute_key(name)
            for name in [attribute.canonical_name, *attribute.aliases]
            if str(name).strip()
        }
        if proposal_keys & target_keys:
            candidates.append(_candidate_evidence(item, entity, attribute))
    candidates.sort(key=lambda candidate: candidate.attribute_id)
    if not candidates:
        return EvaLibraryReconciliationEvidence(**base)
    status = "needs_review" if all(c.compatible for c in candidates) else "blocked"
    return EvaLibraryReconciliationEvidence(
        status=status,
        candidates=candidates,
        **base,
    )


def build_integrity_preview(
    *,
    audit: EvaAuditDocument,
    library: EvaLibraryRepository,
) -> EvaAuditDocument:
    """Return a read-only deep copy with exact-match evidence attached."""

    preview = audit.model_copy(deep=True)
    for item in preview.items:
        item.library_reconciliation = _evidence_for_item(item, library)
    return preview


def validate_reconciliation_decisions_locked(
    *,
    audit: EvaAuditDocument,
    decisions: Sequence[EvaReconciliationDecisionInput],
    library: EvaLibraryRepository,
    trusted_semantic_preview: Mapping[str, Any] | None = None,
) -> tuple[EvaAuditDocument, EvaReconciliationSummary]:
    """Rebuild exact evidence and apply trusted Plan A/Plan B decisions."""

    preview = build_integrity_preview(audit=audit, library=library)
    semantic_reviews: dict[str, EvaSemanticProposalReview] = {}
    if trusted_semantic_preview is not None:
        # Delayed to avoid eva_library -> reconciliation -> review -> library.
        from backend.criteria_processor.eva_review import candidate_set_sha256

        semantic_payload = (
            trusted_semantic_preview.model_dump(mode="json")
            if hasattr(trusted_semantic_preview, "model_dump")
            else dict(trusted_semantic_preview)
        )
        preview_retrieval_version = semantic_payload.get(
            "retrieval_version", "legacy-unversioned"
        )
        if (
            preview_retrieval_version
            != SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION
        ):
            raise SemanticRetrievalVersionStaleError(
                "Trusted semantic preview uses an obsolete retrieval algorithm; "
                "run a fresh reconciliation check."
            )
        if (
            semantic_payload.get("trial_key") != audit.trial_key
            or semantic_payload.get("audit_id") != audit.audit_id
        ):
            raise EvaReconciliationError(
                "Trusted semantic preview does not own this audit."
            )
        if (
            semantic_payload.get("library_revision") != library.revision
            or semantic_payload.get("library_sha256") != library.sha256
        ):
            raise EvaReconciliationError(
                "Trusted semantic preview Library fingerprint is stale."
            )
        for raw_review in semantic_payload.get("reviews", []):
            review = EvaSemanticProposalReview.model_validate(raw_review)
            if review.retrieval_version != preview_retrieval_version:
                raise SemanticRetrievalVersionStaleError(
                    "Trusted semantic review retrieval version does not match "
                    "its preview."
                )
            if review.proposal_fingerprint in semantic_reviews:
                raise EvaReconciliationError(
                    "Trusted semantic preview contains duplicate proposals."
                )
            if review.candidates and (
                candidate_set_sha256(review.candidates)
                != review.candidate_fingerprint
            ):
                raise EvaReconciliationError(
                    "Trusted semantic candidate fingerprint is invalid."
                )
            semantic_reviews[review.proposal_fingerprint] = review
    decision_by_id: dict[str, EvaReconciliationDecisionInput] = {}
    for raw in decisions:
        decision = (
            raw
            if isinstance(raw, EvaReconciliationDecisionInput)
            else EvaReconciliationDecisionInput.model_validate(raw)
        )
        if decision.eva_id in decision_by_id:
            raise EvaReconciliationError(
                f"{decision.eva_id}: duplicate reconciliation decision.",
                eva_ids=[decision.eva_id],
            )
        decision_by_id[decision.eva_id] = decision

    known_ids = {item.eva_id for item in preview.items}
    extras = sorted(set(decision_by_id) - known_ids)
    if extras:
        raise EvaReconciliationError(
            "Reconciliation decisions reference unknown audit rows.",
            eva_ids=extras,
        )
    blocked = [
        item.eva_id
        for item in preview.items
        if item.library_reconciliation.status == "blocked"
    ]
    if blocked:
        raise ReconciliationConflictError(
            "Exact-name Entity, schema, or unit conflicts must be edited before apply.",
            eva_ids=blocked,
        )
    exact_required = {
        item.eva_id
        for item in preview.items
        if item.library_reconciliation.status == "needs_review"
    }
    semantic_items: dict[str, EvaSemanticProposalReview] = {}
    if trusted_semantic_preview is not None:
        for item in preview.items:
            if (
                item.review_status == "rejected"
                or item.mapping_decision != "new_attribute"
                or item.library_reconciliation.status != "not_required"
            ):
                continue
            fingerprint = item.library_reconciliation.proposal_fingerprint
            review = semantic_reviews.get(fingerprint)
            if review is None:
                raise ReconciliationRequiredError(
                    "Trusted semantic preview is missing a proposal review.",
                    eva_ids=[item.eva_id],
                )
            if review.proposal_fingerprint != proposal_fingerprint(item):
                raise EvaReconciliationError(
                    f"{item.eva_id}: semantic proposal fingerprint is stale.",
                    eva_ids=[item.eva_id],
                )
            semantic_items[item.eva_id] = review
    required = exact_required | set(semantic_items)
    missing = sorted(required - set(decision_by_id))
    if missing:
        raise ReconciliationRequiredError(
            "Library reconciliation requires reviewer confirmation.",
            eva_ids=missing,
        )
    unnecessary = sorted(set(decision_by_id) - required)
    if unnecessary:
        raise EvaReconciliationError(
            "Reconciliation decisions were supplied for rows that need no decision.",
            eva_ids=unnecessary,
        )

    now = datetime.now(timezone.utc).isoformat()
    reused = 0
    for item in preview.items:
        evidence = item.library_reconciliation
        if evidence.status != "needs_review":
            continue
        decision = decision_by_id[item.eva_id]
        if decision.proposal_fingerprint != evidence.proposal_fingerprint:
            raise EvaReconciliationError(
                f"{item.eva_id}: proposal fingerprint is stale or invalid.",
                eva_ids=[item.eva_id],
            )
        if decision.decision == "BYPASS_KEEP_NEW":
            raise EvaReconciliationError(
                f"{item.eva_id}: BYPASS_KEEP_NEW is unavailable in Plan A.",
                eva_ids=[item.eva_id],
            )
        if decision.decision == "KEEP_NEW":
            if decision.target_attribute_id is not None:
                raise EvaReconciliationError(
                    f"{item.eva_id}: KEEP_NEW cannot name a target Attribute.",
                    eva_ids=[item.eva_id],
                )
            if not decision.reason.strip():
                raise EvaReconciliationError(
                    f"{item.eva_id}: keeping an exact match as new requires "
                    "a reviewer reason.",
                    eva_ids=[item.eva_id],
                )
            canonical_key = clean_attribute_key(item.attribute_name)
            if any(
                candidate.compatible
                and clean_attribute_key(candidate.canonical_name)
                == canonical_key
                for candidate in evidence.candidates
            ):
                raise ReconciliationConflictError(
                    f"{item.eva_id}: exact canonical name must be renamed "
                    "before KEEP_NEW.",
                    eva_ids=[item.eva_id],
                )
            evidence.status = "resolved"
            evidence.decision = "KEEP_NEW"
            evidence.target_attribute_id = None
            evidence.reason = decision.reason.strip()
            evidence.reviewed_at = now
            continue
        if decision.decision != "REUSE_EXISTING":
            raise ReconciliationConflictError(
                f"{item.eva_id}: an exact compatible Attribute must be reused.",
                eva_ids=[item.eva_id],
            )
        target = next(
            (
                candidate
                for candidate in evidence.candidates
                if candidate.attribute_id == decision.target_attribute_id
                and candidate.compatible
            ),
            None,
        )
        if target is None:
            raise EvaReconciliationError(
                f"{item.eva_id}: reconciliation target is invalid.",
                eva_ids=[item.eva_id],
            )
        attribute = library.attributes[target.attribute_id]
        entity = library.entities[attribute.entity_id]
        item.mapping_decision = "existing_attribute"
        item.attribute_id = attribute.attribute_id
        item.attribute_name = attribute.canonical_name
        item.attribute_description = attribute.description
        item.attribute_aliases = list(attribute.aliases)
        item.entity_id = entity.entity_id
        item.entity_name = entity.canonical_name
        item.new_entity = False
        evidence.status = "resolved"
        evidence.decision = decision.decision
        evidence.target_attribute_id = attribute.attribute_id
        evidence.reason = decision.reason
        evidence.reviewed_at = now
        reused += 1

    for item in preview.items:
        review = semantic_items.get(item.eva_id)
        if review is None:
            continue
        evidence = item.library_reconciliation
        decision = decision_by_id[item.eva_id]
        if decision.proposal_fingerprint != evidence.proposal_fingerprint:
            raise EvaReconciliationError(
                f"{item.eva_id}: proposal fingerprint is stale or invalid.",
                eva_ids=[item.eva_id],
            )
        evidence.semantic_review = review.model_copy(deep=True)
        evidence.status = "resolved"
        evidence.decision = decision.decision
        evidence.reason = decision.reason
        evidence.reviewed_at = now

        if review.status == "failed":
            if decision.decision != "BYPASS_KEEP_NEW" or not decision.reason.strip():
                raise EvaReconciliationError(
                    f"{item.eva_id}: failed semantic review requires a reasoned BYPASS_KEEP_NEW.",
                    eva_ids=[item.eva_id],
                )
            evidence.target_attribute_id = None
            evidence.semantic_review.status = "bypassed"
            continue

        if decision.decision == "BYPASS_KEEP_NEW":
            raise EvaReconciliationError(
                f"{item.eva_id}: completed semantic review cannot be bypassed.",
                eva_ids=[item.eva_id],
            )
        exact_assessments = {
            assessment.attribute_id: assessment
            for assessment in review.assessments
            if assessment.relationship == "exact_equivalent"
        }
        if decision.decision == "KEEP_NEW":
            if decision.target_attribute_id is not None:
                raise EvaReconciliationError(
                    f"{item.eva_id}: KEEP_NEW cannot name a target Attribute.",
                    eva_ids=[item.eva_id],
                )
            if exact_assessments and not decision.reason.strip():
                raise EvaReconciliationError(
                    f"{item.eva_id}: keeping a semantic equivalent requires a reason.",
                    eva_ids=[item.eva_id],
                )
            evidence.target_attribute_id = None
            continue
        if decision.decision != "REUSE_EXISTING":
            raise EvaReconciliationError(
                f"{item.eva_id}: invalid semantic reconciliation decision.",
                eva_ids=[item.eva_id],
            )
        target_id = str(decision.target_attribute_id or "")
        if target_id not in exact_assessments:
            raise EvaReconciliationError(
                f"{item.eva_id}: semantic reuse requires trusted exact_equivalent evidence.",
                eva_ids=[item.eva_id],
            )
        candidate = next(
            (
                row for row in review.candidates
                if row.attribute_id == target_id
            ),
            None,
        )
        attribute = library.attributes.get(target_id)
        if candidate is None or attribute is None or not attribute.active:
            raise EvaReconciliationError(
                f"{item.eva_id}: semantic target is missing or inactive.",
                eva_ids=[item.eva_id],
            )
        entity = library.entities.get(attribute.entity_id)
        if (
            entity is None
            or not entity.active
            or candidate.canonical_name != attribute.canonical_name
            or candidate.entity_id != attribute.entity_id
            or candidate.value_type != attribute.value_type
            or candidate.numerical_type != attribute.numerical_type
            or candidate.canonical_unit != attribute.canonical_unit
            or bool(
                eva_schema_conflicts(
                    value_type=item.value_type,
                    numerical_type=item.numerical_type,
                    canonical_unit=item.unit,
                    candidate=attribute,
                )
            )
        ):
            raise EvaReconciliationError(
                f"{item.eva_id}: semantic target metadata or schema is stale.",
                eva_ids=[item.eva_id],
            )
        item.mapping_decision = "existing_attribute"
        item.attribute_id = attribute.attribute_id
        item.attribute_name = attribute.canonical_name
        item.attribute_description = attribute.description
        item.attribute_aliases = list(attribute.aliases)
        item.entity_id = entity.entity_id
        item.entity_name = entity.canonical_name
        item.new_entity = False
        evidence.target_attribute_id = attribute.attribute_id
        reused += 1

    summary = EvaReconciliationSummary(
        reused_count=reused,
        new_count=sum(
            item.review_status != "rejected"
            and item.mapping_decision == "new_attribute"
            for item in preview.items
        ),
        existing_count=sum(
            item.review_status != "rejected"
            and item.mapping_decision == "existing_attribute"
            for item in preview.items
        )
        - reused,
        rejected_count=sum(
            item.review_status == "rejected" for item in preview.items
        ),
    )
    return preview, summary


__all__ = [
    "EvaApplyEnvelope",
    "EvaReconciliationError",
    "ReconciliationConflictError",
    "ReconciliationRequiredError",
    "build_integrity_preview",
    "proposal_fingerprint",
    "validate_reconciliation_decisions_locked",
]
