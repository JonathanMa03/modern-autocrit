import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import backend.criteria_processor.eva_jobs as eva_jobs_module
from backend.criteria_processor.eva_library import (
    EvaAttributeIdConflictError,
    EvaLibraryRepository,
)
from backend.criteria_processor.eva_library import EvaApplyReceiptConflictError
from backend.criteria_processor.eva_jobs import EligibilityEvaJobManager
from backend.criteria_processor.eva_models import (
    EvaApplyEnvelope,
    EvaAuditDocument,
    EvaAuditItem,
    EvaEntityDefinition,
    EvaReconciliationDecisionInput,
    EvaSemanticAssessment,
    EvaSemanticCandidateEvidence,
    EvaSemanticProposalReview,
)
from backend.criteria_processor.eva_reconciliation_jobs import EvaReconciliationPreview
from backend.criteria_processor.eva_review import candidate_set_sha256
from backend.criteria_processor.eva_reconciliation import (
    EvaReconciliationError,
    ReconciliationConflictError,
    ReconciliationRequiredError,
    SemanticRetrievalVersionStaleError,
    build_integrity_preview,
    _candidate_evidence,
    proposal_fingerprint,
    resolve_active_entity,
    validate_reconciliation_decisions_locked,
)
from backend.criteria_processor.eva_retrieval import (
    SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
    eva_schema_conflicts,
)


def _item(**updates):
    payload = {
        "eva_id": "eva_1",
        "source_item_id": "TEST:0001",
        "criteria": "inclusion",
        "item": "Adults aged 18 years or older",
        "context": "Adults aged 18 years or older are eligible.",
        "search_expansion": {},
        "candidates": [],
        "mapping_decision": "new_attribute",
        "attribute_id": "age_at_enrollment",
        "entity_id": "edited-demographics",
        "entity_name": "Demographics",
        "new_entity": True,
        "attribute_name": "Participant age",
        "attribute_description": "Age at enrollment.",
        "attribute_aliases": ["Age"],
        "value_type": "Numerical",
        "categorical_value": None,
        "numerical_type": "Range",
        "numerical_value": "[18,+inf)",
        "unit": "years",
        "confidence": 0.9,
        "rationale": "Age criterion.",
        "review_status": "revised",
        "review_notes": "",
    }
    payload.update(updates)
    return EvaAuditItem.model_validate(payload)


def _audit(*items):
    return EvaAuditDocument(
        audit_id="audit-1",
        trial_key="TEST",
        trial_id="TEST",
        source_path="fixture.csv",
        library_revision=0,
        library_sha256="",
        created_at="2026-08-11T00:00:00+00:00",
        updated_at="2026-08-11T00:00:00+00:00",
        search_model="search",
        search_reasoning_effort="low",
        reasoner_model="reasoner",
        reasoner_reasoning_effort="medium",
        items=list(items or [_item()]),
    )


@pytest.fixture
def library(tmp_path: Path) -> EvaLibraryRepository:
    payload = {
        "schema_version": "eligcrit.eligibility_eva_library.v1",
        "library_name": "fixture",
        "revision": 0,
        "updated_at": "2026-08-11T00:00:00+00:00",
        "value_definitions": {
            "Categorical": ["Included", "Excluded"],
            "SexGender": ["male", "female", "all"],
            "Numerical": ["Range", "Point"],
        },
        "entities": [
            {
                "entity_id": "demographic",
                "canonical_name": "Demographic",
                "description": "",
                "aliases": ["Demographics"],
                "active": True,
            }
        ],
        "attributes": [
            {
                "attribute_id": "participant_age",
                "canonical_name": "Age",
                "entity_id": "demographic",
                "description": "Participant age.",
                "aliases": ["Participant age"],
                "value_type": "Numerical",
                "numerical_type": "Range",
                "canonical_unit": "years",
                "active": True,
            }
        ],
        "audit_history": [],
    }
    path = tmp_path / "library.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return EvaLibraryRepository(path)


def test_legacy_item_defaults_to_not_required():
    item = _item()
    assert item.library_reconciliation.status == "not_required"
    assert item.library_reconciliation.candidates == []


def test_client_cannot_submit_candidate_evidence():
    payload = {
        "audit": _audit().model_dump(mode="json"),
        "reconciliation_decisions": [
            {
                "eva_id": "eva_1",
                "proposal_fingerprint": "a" * 64,
                "decision": "REUSE_EXISTING",
                "target_attribute_id": "participant_age",
                "reason": "",
                "candidates": [{"attribute_id": "forged"}],
            }
        ],
    }
    with pytest.raises(ValidationError):
        EvaApplyEnvelope.model_validate(payload)


def test_proposal_fingerprint_ignores_value_and_review_prose():
    before = _item()
    after = before.model_copy(
        update={
            "numerical_value": "[21,+inf)",
            "rationale": "Changed rationale.",
            "review_notes": "Changed notes.",
        },
        deep=True,
    )
    assert proposal_fingerprint(after) == proposal_fingerprint(before)


def test_preview_resolves_entity_alias_before_attribute_match(library):
    evidence = build_integrity_preview(
        audit=_audit(), library=library
    ).items[0].library_reconciliation
    assert evidence.status == "needs_review"
    assert [candidate.attribute_id for candidate in evidence.candidates] == [
        "participant_age"
    ]


def test_active_entity_resolver_rejects_inactive_and_ambiguous(library):
    item = _item()
    library.entities["demographic"].active = False
    entity, conflicts = resolve_active_entity(item, library)
    assert entity is None
    assert conflicts == []

    library.entities["demographic"].active = True
    ambiguous = item.model_copy(
        update={"entity_id": "missing", "entity_name": "Shared alias"}
    )
    for entity_id in ("entity_one", "entity_two"):
        library.entities[entity_id] = EvaEntityDefinition(
            entity_id=entity_id,
            canonical_name=entity_id,
            description="",
            aliases=["Shared alias"],
            active=True,
        )
    entity, conflicts = resolve_active_entity(ambiguous, library)
    assert entity is None
    assert conflicts == ["ambiguous_entity"]


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, ()),
        ({"numerical_type": "Point"}, ("numerical_type",)),
        ({"unit": "Years"}, ()),
        ({"unit": "months"}, ("canonical_unit",)),
        (
            {"value_type": "Categorical", "numerical_type": None, "unit": None},
            ("value_type", "numerical_type", "canonical_unit"),
        ),
    ],
)
def test_plan_a_schema_evidence_uses_shared_conflicts(library, updates, expected):
    item = _item(**updates)
    attribute = library.get_attribute("participant_age")
    entity = library.entities[attribute.entity_id]
    evidence = _candidate_evidence(item, entity, attribute)
    shared = eva_schema_conflicts(
        value_type=item.value_type,
        numerical_type=item.numerical_type,
        canonical_unit=item.unit,
        candidate=attribute,
    )
    assert shared == expected
    assert tuple(evidence.conflict_reasons) == shared
    assert evidence.compatible is (not shared)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("unit", "days", "canonical_unit"),
        ("numerical_type", "Point", "numerical_type"),
        ("value_type", "Categorical", "value_type"),
    ],
)
def test_same_name_schema_conflict_blocks_keep_new(
    library, field, value, reason
):
    item = _item(**{field: value})
    if field == "value_type":
        item.numerical_type = None
        item.unit = None
        item.numerical_value = None
        item.categorical_value = "Included"
    evidence = build_integrity_preview(
        audit=_audit(item), library=library
    ).items[0].library_reconciliation
    assert evidence.status == "blocked"
    assert reason in evidence.candidates[0].conflict_reasons


def test_preview_ignores_inactive_attribute(library):
    library.attributes["participant_age"].active = False
    evidence = build_integrity_preview(
        audit=_audit(), library=library
    ).items[0].library_reconciliation
    assert evidence.status == "not_required"


def _reuse_decision(item):
    return EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="REUSE_EXISTING",
        target_attribute_id="participant_age",
        reason="Confirmed exact match.",
    )


def _keep_new_decision(item, reason="Distinct reviewed concept"):
    return EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="KEEP_NEW",
        target_attribute_id=None,
        reason=reason,
    )


def test_locked_validator_reuses_exact_candidate(library):
    audit = _audit()
    resolved, summary = validate_reconciliation_decisions_locked(
        audit=audit,
        decisions=[_reuse_decision(audit.items[0])],
        library=library,
        trusted_semantic_preview=None,
    )
    assert resolved.items[0].attribute_id == "participant_age"
    assert resolved.items[0].mapping_decision == "existing_attribute"
    assert resolved.items[0].library_reconciliation.status == "resolved"
    assert summary.reused_count == 1
    assert summary.new_count == 0


def test_locked_validator_requires_decision_for_exact_candidate(library):
    with pytest.raises(ReconciliationRequiredError) as error:
        validate_reconciliation_decisions_locked(
            audit=_audit(), decisions=[], library=library
        )
    assert error.value.eva_ids == ["eva_1"]


def test_locked_validator_rejects_extra_or_unnecessary_decisions(library):
    exact = _audit()
    unknown = _reuse_decision(exact.items[0]).model_copy(
        update={"eva_id": "eva_unknown"}
    )
    with pytest.raises(EvaReconciliationError, match="unknown audit rows"):
        validate_reconciliation_decisions_locked(
            audit=exact, decisions=[unknown], library=library
        )

    novel = _audit(
        _item(attribute_name="Novel measurement", attribute_aliases=[])
    )
    unnecessary = _keep_new_decision(novel.items[0])
    with pytest.raises(EvaReconciliationError, match="need no decision"):
        validate_reconciliation_decisions_locked(
            audit=novel, decisions=[unnecessary], library=library
        )


def test_alias_only_exact_match_allows_reasoned_keep_new(library):
    audit = _audit(
        _item(
            attribute_id="distinct_age_concept",
            attribute_name="distinct_age_concept",
            attribute_aliases=["Participant age"],
        )
    )
    resolved, summary = validate_reconciliation_decisions_locked(
        audit=audit,
        decisions=[_keep_new_decision(audit.items[0])],
        library=library,
    )
    item = resolved.items[0]
    assert item.mapping_decision == "new_attribute"
    assert item.library_reconciliation.decision == "KEEP_NEW"
    assert item.library_reconciliation.target_attribute_id is None
    assert item.library_reconciliation.reason == "Distinct reviewed concept"
    assert summary.reused_count == 0
    assert summary.new_count == 1


@pytest.mark.parametrize("reason", ["", "   "])
def test_exact_keep_new_requires_reason(library, reason):
    audit = _audit(
        _item(
            attribute_name="distinct_age_concept",
            attribute_aliases=["Participant age"],
        )
    )
    with pytest.raises(EvaReconciliationError, match="reviewer reason"):
        validate_reconciliation_decisions_locked(
            audit=audit,
            decisions=[_keep_new_decision(audit.items[0], reason)],
            library=library,
        )


def test_exact_keep_new_rejects_target_and_bypass(library):
    audit = _audit(
        _item(
            attribute_name="distinct_age_concept",
            attribute_aliases=["Participant age"],
        )
    )
    targeted = _keep_new_decision(audit.items[0]).model_copy(
        update={"target_attribute_id": "participant_age"}
    )
    with pytest.raises(EvaReconciliationError, match="cannot name a target"):
        validate_reconciliation_decisions_locked(
            audit=audit, decisions=[targeted], library=library
        )
    bypass = targeted.model_copy(
        update={"decision": "BYPASS_KEEP_NEW", "target_attribute_id": None}
    )
    with pytest.raises(EvaReconciliationError, match="unavailable in Plan A"):
        validate_reconciliation_decisions_locked(
            audit=audit, decisions=[bypass], library=library
        )


def test_keep_new_rejects_exact_canonical_name(library):
    audit = _audit(
        _item(
            attribute_id="distinct_id",
            attribute_name="Age",
            attribute_aliases=[],
        )
    )
    with pytest.raises(ReconciliationConflictError, match="canonical name"):
        validate_reconciliation_decisions_locked(
            audit=audit,
            decisions=[_keep_new_decision(audit.items[0])],
            library=library,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"unit": "months"},
        {"numerical_type": "Point", "numerical_value": "18"},
        {
            "value_type": "Categorical",
            "categorical_value": "Included",
            "numerical_type": None,
            "numerical_value": None,
            "unit": None,
        },
        {"entity_id": "missing", "entity_name": "Shared alias"},
    ],
)
@pytest.mark.parametrize("decision_name", ["KEEP_NEW", "BYPASS_KEEP_NEW"])
def test_blocked_exact_conflicts_cannot_be_bypassed(
    library, updates, decision_name
):
    if updates.get("entity_name") == "Shared alias":
        for entity_id in ("entity_one", "entity_two"):
            library.entities[entity_id] = EvaEntityDefinition(
                entity_id=entity_id,
                canonical_name=entity_id,
                description="",
                aliases=["Shared alias"],
                active=True,
            )
    audit = _audit(_item(**updates))
    decision = _keep_new_decision(audit.items[0]).model_copy(
        update={"decision": decision_name}
    )
    with pytest.raises(ReconciliationConflictError):
        validate_reconciliation_decisions_locked(
            audit=audit, decisions=[decision], library=library
        )


def test_locked_validator_allows_empty_decisions_for_true_new(library):
    audit = _audit(_item(attribute_name="Novel measurement", attribute_aliases=[]))
    resolved, summary = validate_reconciliation_decisions_locked(
        audit=audit, decisions=[], library=library
    )
    assert resolved.items[0].attribute_id == "age_at_enrollment"
    assert summary.new_count == 1


def test_locked_validator_rejects_forged_fingerprint(library):
    audit = _audit()
    decision = _reuse_decision(audit.items[0]).model_copy(
        update={"proposal_fingerprint": "a" * 64}
    )
    with pytest.raises(EvaReconciliationError, match="fingerprint"):
        validate_reconciliation_decisions_locked(
            audit=audit, decisions=[decision], library=library
        )


def _repository_audit(library):
    audit = _audit()
    audit.library_revision = library.revision
    audit.library_sha256 = library.sha256
    return audit


def test_reconciliation_reuse_is_snapshotted_alias_safe_and_replayable(
    library, tmp_path
):
    repository = EvaLibraryRepository(
        library.path,
        snapshot_directory=tmp_path / "snapshots",
    )
    audit = _repository_audit(repository)
    decision = _reuse_decision(audit.items[0])
    aliases_before = list(repository.get_attribute("participant_age").aliases)

    first = repository.apply_audit(
        audit, reconciliation_decisions=[decision]
    )
    bytes_after_first = repository.read_raw_bytes()
    second = repository.apply_audit(
        audit, reconciliation_decisions=[decision]
    )

    assert first.replayed is False
    assert second.replayed is True
    assert second.receipt == first.receipt
    assert repository.revision == 1
    assert first.receipt.new_library_sha256 == repository.sha256
    assert repository.get_attribute("participant_age").aliases == aliases_before
    assert repository.read_raw_bytes() == bytes_after_first
    assert len(list((tmp_path / "snapshots").glob("revision-0-*.json"))) == 1
    receipts = [
        row
        for row in repository.payload["audit_history"]
        if row.get("receipt_schema_version")
    ]
    assert len(receipts) == 1


def test_reasoned_keep_new_creates_distinct_attribute_and_replays(
    library, tmp_path
):
    repository = EvaLibraryRepository(
        library.path,
        snapshot_directory=tmp_path / "snapshots",
    )
    audit = _repository_audit(repository)
    audit.items[0].attribute_id = "distinct_manual_id"
    audit.items[0].attribute_name = "Distinct age concept"
    audit.items[0].attribute_aliases = ["Participant age"]
    decision = _keep_new_decision(audit.items[0], "Clinically distinct cohort age")
    aliases_before = list(repository.get_attribute("participant_age").aliases)

    first = repository.apply_audit(
        audit, reconciliation_decisions=[decision]
    )
    bytes_after_first = repository.read_raw_bytes()
    second = repository.apply_audit(
        audit, reconciliation_decisions=[decision]
    )

    assert first.replayed is False
    assert second.replayed is True
    assert repository.revision == 1
    assert first.audit.items[0].attribute_id == "distinct_manual_id"
    assert first.audit.items[0].library_reconciliation.reason == (
        "Clinically distinct cohort age"
    )
    assert repository.get_attribute("distinct_manual_id") is not None
    assert repository.get_attribute("participant_age").aliases == aliases_before
    assert repository.read_raw_bytes() == bytes_after_first
    assert len(list((tmp_path / "snapshots").glob("revision-0-*.json"))) == 1

    repository.reload()
    later = _audit(
        _item(
            eva_id="eva_later",
            attribute_id="third_age",
            attribute_name="Third age concept",
            attribute_aliases=["Participant age"],
        )
    )
    candidates = build_integrity_preview(
        audit=later, library=repository
    ).items[0].library_reconciliation.candidates
    assert [row.attribute_id for row in candidates] == sorted(
        ["participant_age", "distinct_manual_id"]
    )


@pytest.mark.parametrize("attribute_id", ["participant_age", "Participant Age"])
def test_keep_new_rejects_existing_or_slug_colliding_attribute_id(
    library, attribute_id
):
    audit = _repository_audit(library)
    audit.items[0].attribute_id = attribute_id
    audit.items[0].attribute_name = "Distinct age concept"
    audit.items[0].attribute_aliases = ["Participant age"]
    with pytest.raises(EvaAttributeIdConflictError):
        library.apply_audit(
            audit,
            reconciliation_decisions=[_keep_new_decision(audit.items[0])],
        )


def test_snapshot_failure_aborts_library_mutation(library, tmp_path, monkeypatch):
    repository = EvaLibraryRepository(
        library.path,
        snapshot_directory=tmp_path / "snapshots",
    )
    audit = _repository_audit(repository)
    original = repository.read_raw_bytes()

    def fail_snapshot(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(repository, "_write_snapshot", fail_snapshot)
    with pytest.raises(OSError, match="disk full"):
        repository.apply_audit(
            audit,
            reconciliation_decisions=[_reuse_decision(audit.items[0])],
        )
    assert repository.read_raw_bytes() == original


def test_same_audit_id_with_changed_input_conflicts_before_stale(library):
    audit = _repository_audit(library)
    decision = _reuse_decision(audit.items[0])
    library.apply_audit(audit, reconciliation_decisions=[decision])
    changed = audit.model_copy(deep=True)
    changed.items[0].numerical_value = "[21,+inf)"
    with pytest.raises(EvaApplyReceiptConflictError):
        library.apply_audit(changed, reconciliation_decisions=[decision])


def test_manager_clears_client_evidence_and_requires_legacy_decision(
    library, tmp_path
):
    points = tmp_path / "points"
    points.mkdir()
    manager = EligibilityEvaJobManager(
        points_directory=points,
        library_path=library.path,
        work_directory=tmp_path / "work",
    )
    try:
        audit = _repository_audit(library)
        manager._audit_path("TEST").write_text(
            json.dumps(audit.model_dump(mode="json")), encoding="utf-8"
        )
        forged = build_integrity_preview(audit=audit, library=library)
        saved = manager.save_audit(
            "TEST", forged.model_dump(mode="json")
        )
        assert (
            saved["items"][0]["library_reconciliation"]["status"]
            == "not_required"
        )
        with pytest.raises(ReconciliationRequiredError):
            manager.apply_audit("TEST", audit.model_dump(mode="json"))
        persisted = manager.get_audit("TEST")
        assert (
            persisted["items"][0]["library_reconciliation"]["status"]
            == "not_required"
        )
    finally:
        manager.close()


def test_manager_envelope_applies_and_receipt_replays_downstream(
    library, tmp_path
):
    points = tmp_path / "points"
    points.mkdir()
    manager = EligibilityEvaJobManager(
        points_directory=points,
        library_path=library.path,
        work_directory=tmp_path / "work",
        verified_output_directory=tmp_path / "verified",
    )
    try:
        audit = _repository_audit(library)
        manager._audit_path("TEST").write_text(
            json.dumps(audit.model_dump(mode="json")), encoding="utf-8"
        )
        envelope = {
            "audit": audit.model_dump(mode="json"),
            "reconciliation_decisions": [
                _reuse_decision(audit.items[0]).model_dump(mode="json")
            ],
        }
        first = manager.apply_audit("TEST", envelope)
        manager._normalized_path("TEST").unlink()
        verified = manager._verified_normalized_path("TEST")
        assert verified is not None
        verified.unlink()
        second = manager.apply_audit("TEST", envelope)
    finally:
        manager.close()

    assert first["replayed"] is False
    assert second["replayed"] is True
    assert second["reconciliation_summary"]["reused_count"] == 1
    assert manager._normalized_path("TEST").is_file()
    assert verified.is_file()


def test_full_plan_a_repository_integration_processes_each_row_once(
    library, tmp_path
):
    exact = _item()
    new = _item(
        eva_id="eva_2",
        source_item_id="TEST:0002",
        attribute_id="smoking_status",
        attribute_name="Smoking status",
        attribute_aliases=[],
        value_type="Categorical",
        categorical_value="Included",
        numerical_type=None,
        numerical_value=None,
        unit=None,
    )
    existing = _item(
        eva_id="eva_3",
        source_item_id="TEST:0003",
        mapping_decision="existing_attribute",
        attribute_id="participant_age",
        entity_id="demographic",
        entity_name="Demographic",
        new_entity=False,
        attribute_name="Participant age",
        attribute_aliases=[],
    )
    rejected = _item(
        eva_id="eva_4",
        source_item_id="TEST:0004",
        attribute_name="Rejected proposal",
        attribute_aliases=[],
        review_status="rejected",
    )
    audit = _audit(exact, new, existing, rejected)
    audit.library_revision = library.revision
    audit.library_sha256 = library.sha256
    repository = EvaLibraryRepository(
        library.path, snapshot_directory=tmp_path / "snapshots"
    )
    before_aliases = list(repository.attributes["participant_age"].aliases)
    before_count = len(repository.attributes)

    result = repository.apply_audit(
        audit, reconciliation_decisions=[_reuse_decision(exact)]
    )

    assert result.reconciliation_summary.model_dump() == {
        "reused_count": 1,
        "new_count": 1,
        "existing_count": 1,
        "rejected_count": 1,
    }
    assert len(result.normalized_rows) == 3
    assert len(repository.attributes) == before_count + 1
    assert "smoking_status" in repository.attributes
    assert repository.attributes["participant_age"].aliases == before_aliases
    assert len(list((tmp_path / "snapshots").glob("*.json"))) == 1
    assert sum(
        bool(row.get("receipt_schema_version"))
        for row in repository.payload["audit_history"]
    ) == 1


@pytest.mark.parametrize(
    "failure_point", ["after_library", "after_working", "after_verified"]
)
def test_identical_retry_repairs_each_downstream_failure_point(
    library, tmp_path, monkeypatch, failure_point
):
    points = tmp_path / "points"
    points.mkdir()
    manager = EligibilityEvaJobManager(
        points_directory=points,
        library_path=library.path,
        work_directory=tmp_path / "work",
        verified_output_directory=tmp_path / "verified",
    )
    audit = _repository_audit(library)
    manager._audit_path("TEST").write_text(
        json.dumps(audit.model_dump(mode="json")), encoding="utf-8"
    )
    envelope = {
        "audit": audit.model_dump(mode="json"),
        "reconciliation_decisions": [
            _reuse_decision(audit.items[0]).model_dump(mode="json")
        ],
    }
    real_jsonl = eva_jobs_module.atomic_write_jsonl
    real_json = eva_jobs_module.atomic_write_json
    calls = {"jsonl": 0, "json": 0}

    def flaky_jsonl(path, rows):
        calls["jsonl"] += 1
        fail_at = 1 if failure_point == "after_library" else 2
        if failure_point in {"after_library", "after_working"} and calls["jsonl"] == fail_at:
            raise OSError(failure_point)
        return real_jsonl(path, rows)

    def flaky_json(path, payload):
        calls["json"] += 1
        if failure_point == "after_verified" and calls["json"] == 2:
            raise OSError(failure_point)
        return real_json(path, payload)

    monkeypatch.setattr(eva_jobs_module, "atomic_write_jsonl", flaky_jsonl)
    monkeypatch.setattr(eva_jobs_module, "atomic_write_json", flaky_json)
    try:
        with pytest.raises(OSError, match=failure_point):
            manager.apply_audit("TEST", envelope)
        retried = manager.apply_audit("TEST", envelope)
    finally:
        manager.close()

    repository = EvaLibraryRepository(library.path)
    assert retried["replayed"] is True
    assert repository.revision == 1
    assert sum(
        bool(row.get("receipt_schema_version"))
        for row in repository.payload["audit_history"]
    ) == 1
    assert len(list((tmp_path / "work" / "library_snapshots").glob("*.json"))) == 1
    assert manager._normalized_path("TEST").is_file()
    verified = manager._verified_normalized_path("TEST")
    assert verified is not None and verified.is_file()
    assert manager.get_audit("TEST")["review_status"] == "approved"


def _semantic_preview(
    library,
    item,
    relationship="exact_equivalent",
    *,
    status="completed",
    retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
):
    candidate = EvaSemanticCandidateEvidence(
        attribute_id="participant_age",
        canonical_name="Age",
        entity_id="demographic",
        entity_name="Demographic",
        value_type="Numerical",
        numerical_type="Range",
        canonical_unit="years",
        retrieval_score=88,
        matched_term="Enrollment age",
        matched_text="Age",
        retrieval_methods=["llm_semantic:fuzzy_wratio"],
    )
    fingerprint = candidate_set_sha256([candidate])
    assessments = (
        [
            EvaSemanticAssessment(
                attribute_id="participant_age",
                candidate_fingerprint=fingerprint,
                relationship=relationship,
                confidence=0.9,
                explanation="The concepts were reviewed for reusable equivalence.",
            )
        ]
        if status == "completed"
        else []
    )
    review = EvaSemanticProposalReview(
        retrieval_version=retrieval_version,
        status=status,
        proposal_fingerprint=proposal_fingerprint(item),
        candidate_fingerprint=fingerprint,
        candidates=[candidate],
        assessments=assessments,
        prompt_version="semantic-v1",
        retryable=status == "failed",
        error_type="ProviderError" if status == "failed" else "",
        error_message="Provider failed twice." if status == "failed" else "",
    )
    return EvaReconciliationPreview(
        job_id="evarec_test",
        trial_key="TEST",
        audit_id="audit-1",
        base_input_fingerprint="c" * 64,
        library_revision=library.revision,
        library_sha256=library.sha256,
        attempt_number=1,
        retrieval_version=retrieval_version,
        reviews=[review],
        created_at="2026-08-11T00:00:00+00:00",
    )


def _semantic_item():
    return _item(attribute_name="Enrollment age", attribute_aliases=[])


def test_locked_validator_rejects_legacy_unversioned_semantic_preview(library):
    item = _semantic_item()
    preview = _semantic_preview(
        library, item, retrieval_version="legacy-unversioned"
    )
    decision = EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="REUSE_EXISTING",
        target_attribute_id="participant_age",
    )
    with pytest.raises(SemanticRetrievalVersionStaleError):
        validate_reconciliation_decisions_locked(
            audit=_audit(item),
            decisions=[decision],
            library=library,
            trusted_semantic_preview=preview,
        )


def test_current_retrieval_version_is_accepted(library):
    item = _semantic_item()
    preview = _semantic_preview(
        library,
        item,
        retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
    )
    decision = EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="REUSE_EXISTING",
        target_attribute_id="participant_age",
    )
    resolved, summary = validate_reconciliation_decisions_locked(
        audit=_audit(item),
        decisions=[decision],
        library=library,
        trusted_semantic_preview=preview,
    )
    assert summary.reused_count == 1
    assert resolved.items[0].mapping_decision == "existing_attribute"


def test_semantic_reuse_requires_trusted_exact_equivalent(library):
    item = _semantic_item()
    audit = _audit(item)
    decision = EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="REUSE_EXISTING",
        target_attribute_id="participant_age",
    )
    with pytest.raises(EvaReconciliationError, match="exact_equivalent"):
        validate_reconciliation_decisions_locked(
            audit=audit,
            decisions=[decision],
            library=library,
            trusted_semantic_preview=_semantic_preview(
                library, item, "related_not_equivalent"
            ),
        )


def test_semantic_exact_equivalent_can_be_reused_without_alias_mutation(library):
    item = _semantic_item()
    audit = _audit(item)
    decision = EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="REUSE_EXISTING",
        target_attribute_id="participant_age",
    )
    resolved, summary = validate_reconciliation_decisions_locked(
        audit=audit,
        decisions=[decision],
        library=library,
        trusted_semantic_preview=_semantic_preview(library, item),
    )
    assert resolved.items[0].attribute_id == "participant_age"
    assert resolved.items[0].library_reconciliation.semantic_review.status == "completed"
    assert summary.reused_count == 1


def test_keep_new_with_semantic_equivalent_requires_reason(library):
    item = _semantic_item()
    decision = EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="KEEP_NEW",
    )
    with pytest.raises(EvaReconciliationError, match="requires a reason"):
        validate_reconciliation_decisions_locked(
            audit=_audit(item),
            decisions=[decision],
            library=library,
            trusted_semantic_preview=_semantic_preview(library, item),
        )


def test_failed_review_allows_only_reasoned_bypass(library):
    item = _semantic_item()
    preview = _semantic_preview(library, item, status="failed")
    blank = EvaReconciliationDecisionInput(
        eva_id=item.eva_id,
        proposal_fingerprint=proposal_fingerprint(item),
        decision="BYPASS_KEEP_NEW",
        reason="",
    )
    with pytest.raises(EvaReconciliationError, match="reasoned"):
        validate_reconciliation_decisions_locked(
            audit=_audit(item), decisions=[blank], library=library,
            trusted_semantic_preview=preview,
        )
    resolved, summary = validate_reconciliation_decisions_locked(
        audit=_audit(item),
        decisions=[blank.model_copy(update={"reason": "Provider failed twice"})],
        library=library,
        trusted_semantic_preview=preview,
    )
    assert resolved.items[0].library_reconciliation.semantic_review.status == "bypassed"
    assert summary.new_count == 1
