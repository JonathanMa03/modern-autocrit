import json
import time
from pathlib import Path

import pytest

from backend.criteria_processor.eva_models import EvaAuditDocument, EvaAuditItem
from backend.criteria_processor.eva_jobs import EligibilityEvaJobManager
from backend.criteria_processor.eva_library import EvaLibraryRepository
from backend.criteria_processor.eva_reconciliation import proposal_fingerprint
from backend.criteria_processor.eva_reconciliation_jobs import (
    EligibilityEvaReconciliationJobStore,
    EvaReconciliationJobInput,
    EvaReconciliationJobRecord,
    EvaReconciliationPreview,
    InvalidReconciliationRetryParentError,
    ReconciliationPreviewNotReadyError,
    reconciliation_base_input_fingerprint,
)
from backend.criteria_processor.eva_retrieval import (
    SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
)


def _audit(name="Novel liver measure"):
    item = EvaAuditItem.model_validate(
        {
            "eva_id": "eva_1",
            "source_item_id": "TEST:0001",
            "criteria": "inclusion",
            "item": "Source",
            "context": "Context",
            "search_expansion": {},
            "mapping_decision": "new_attribute",
            "attribute_id": "novel_liver_measure",
            "entity_id": "condition",
            "entity_name": "Condition",
            "attribute_name": name,
            "attribute_description": "A reusable liver measurement.",
            "value_type": "Categorical",
            "categorical_value": "Included",
            "confidence": 0.8,
            "rationale": "Reason.",
        }
    )
    return EvaAuditDocument(
        audit_id="audit-1",
        trial_key="TEST",
        trial_id="TEST",
        source_path="fixture.csv",
        library_revision=3,
        library_sha256="b" * 64,
        created_at="2026-08-11T00:00:00+00:00",
        updated_at="2026-08-11T00:00:00+00:00",
        search_model="search",
        search_reasoning_effort="low",
        reasoner_model="reasoner",
        reasoner_reasoning_effort="medium",
        items=[item],
    )


def _input(audit=None, **updates):
    audit = audit or _audit()
    provider = {"provider": "Fake", "model": "semantic", "effort": "low"}
    fingerprint = reconciliation_base_input_fingerprint(
        trial_key="TEST",
        audit=audit,
        library_revision=3,
        library_sha256="b" * 64,
        provider_config=provider,
        prompt_version="prompt-v1",
        retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
        review_mode="full",
    )
    payload = {
        "trial_key": "TEST",
        "audit_id": audit.audit_id,
        "base_input_fingerprint": fingerprint,
        "proposal_fingerprints": [proposal_fingerprint(audit.items[0])],
        "library_revision": 3,
        "library_sha256": "b" * 64,
        "provider_name": "Fake",
        "provider_model": "semantic",
        "provider_reasoning_effort": "low",
        "prompt_version": "prompt-v1",
        "retrieval_version": SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
        "review_mode": "full",
    }
    payload.update(updates)
    return EvaReconciliationJobInput.model_validate(payload)


def test_identical_full_input_reuses_active_job(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    first = store.create_or_reuse(_input())
    second = store.create_or_reuse(_input())
    assert second.job_id == first.job_id


def test_retrieval_version_changes_base_input_fingerprint():
    common = dict(
        trial_key="TEST",
        audit=_audit(),
        library_revision=3,
        library_sha256="b" * 64,
        provider_config={
            "provider": "Fake", "model": "semantic", "effort": "low"
        },
        prompt_version="prompt-v1",
        review_mode="full",
    )
    old = reconciliation_base_input_fingerprint(
        **common, retrieval_version="legacy-unversioned"
    )
    current = reconciliation_base_input_fingerprint(
        **common,
        retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
    )
    assert old != current


def test_persisted_v1_job_without_retrieval_version_remains_readable(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    created = store.create_or_reuse(_input())
    payload = created.model_dump(mode="json")
    payload.pop("retrieval_version")
    path = tmp_path / "reconciliation_jobs" / f"{created.job_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    job = EligibilityEvaReconciliationJobStore(tmp_path).get(created.job_id)
    assert job.retrieval_version == "legacy-unversioned"


def test_old_completed_job_is_not_reused_for_current_retrieval(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    old_fingerprint = reconciliation_base_input_fingerprint(
        trial_key="TEST",
        audit=_audit(),
        library_revision=3,
        library_sha256="b" * 64,
        provider_config={
            "provider": "Fake", "model": "semantic", "effort": "low"
        },
        prompt_version="prompt-v1",
        retrieval_version="legacy-unversioned",
        review_mode="full",
    )
    old_input = _input(
        base_input_fingerprint=old_fingerprint,
        retrieval_version="legacy-unversioned",
    )
    old = store.create_or_reuse(old_input)
    store.mark_running(old.job_id)
    store.mark_completed(
        old.job_id,
        preview=EvaReconciliationPreview(
            job_id=old.job_id,
            trial_key="TEST",
            audit_id="audit-1",
            base_input_fingerprint=old.base_input_fingerprint,
            library_revision=3,
            library_sha256="b" * 64,
            attempt_number=1,
            retrieval_version="legacy-unversioned",
            reviews=[],
            created_at="2026-08-11T00:00:00+00:00",
        ),
    )
    current = store.create_or_reuse(_input())
    assert current.job_id != old.job_id


def test_preview_version_mismatch_is_rejected_at_completion(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    job = store.create_or_reuse(_input())
    store.mark_running(job.job_id)
    preview = EvaReconciliationPreview(
        job_id=job.job_id,
        trial_key="TEST",
        audit_id="audit-1",
        base_input_fingerprint=job.base_input_fingerprint,
        library_revision=job.library_revision,
        library_sha256=job.library_sha256,
        attempt_number=job.attempt_number,
        retrieval_version="legacy-unversioned",
        reviews=[],
        created_at="2026-08-11T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="retrieval version"):
        store.mark_completed(job.job_id, preview=preview)
    assert store.get(job.job_id).status == "running"


def test_completed_success_is_reused(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    first = store.create_or_reuse(_input())
    store.mark_running(first.job_id)
    store.mark_completed(
        first.job_id,
        preview=EvaReconciliationPreview(
            job_id=first.job_id,
            trial_key="TEST",
            audit_id="audit-1",
            base_input_fingerprint=first.base_input_fingerprint,
            library_revision=3,
            library_sha256="b" * 64,
            attempt_number=1,
            retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
            reviews=[],
            created_at="2026-08-11T00:00:00+00:00",
        ),
    )
    assert store.create_or_reuse(_input()).job_id == first.job_id


def test_retry_of_failed_job_creates_new_attempt(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    failed = store.create_or_reuse(_input())
    store.mark_failed(failed.job_id, retryable=True)
    with pytest.raises(InvalidReconciliationRetryParentError):
        store.create_or_reuse(_input())
    retried = store.create_or_reuse(_input(), retry_of=failed.job_id)
    assert retried.job_id != failed.job_id
    assert retried.base_input_fingerprint == failed.base_input_fingerprint
    assert retried.parent_job_id == failed.job_id
    assert retried.attempt_number == 2


def test_cross_input_retry_parent_is_rejected(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    failed = store.create_or_reuse(_input())
    store.mark_failed(failed.job_id, retryable=True)
    changed = _input(_audit("Changed proposal"))
    with pytest.raises(InvalidReconciliationRetryParentError):
        store.create_or_reuse(changed, retry_of=failed.job_id)


def test_startup_marks_orphaned_running_job_retryable(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    job = store.create_or_reuse(_input())
    store.mark_running(job.job_id)

    recovered = EligibilityEvaReconciliationJobStore(tmp_path).get(job.job_id)

    assert recovered.status == "failed"
    assert recovered.retryable is True
    assert recovered.error_type == "InterruptedReconciliationJob"


def test_preview_not_readable_before_completion(tmp_path):
    store = EligibilityEvaReconciliationJobStore(tmp_path)
    job = store.create_or_reuse(_input())
    with pytest.raises(ReconciliationPreviewNotReadyError):
        store.get_preview(job.job_id)


def test_changed_business_inputs_change_base_fingerprint():
    original = _input()
    changed_proposal = _input(_audit("Changed proposal"))
    changed_model = _input(provider_model="other")
    assert changed_proposal.base_input_fingerprint != original.base_input_fingerprint
    # The caller must recompute the base fingerprint when provider config changes.
    recomputed = reconciliation_base_input_fingerprint(
        trial_key="TEST",
        audit=_audit(),
        library_revision=3,
        library_sha256="b" * 64,
        provider_config={"provider": "Fake", "model": "other", "effort": "low"},
        prompt_version="prompt-v1",
        retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
        review_mode="full",
    )
    assert recomputed != original.base_input_fingerprint
    assert changed_model.provider_model == "other"


def test_corrupt_job_is_quarantined(tmp_path):
    jobs = tmp_path / "reconciliation_jobs"
    jobs.mkdir(parents=True)
    corrupt = jobs / "bad.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="quarantined"):
        EligibilityEvaReconciliationJobStore(tmp_path)
    assert (jobs / "bad.json.corrupt").is_file()


def _write_library(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "eligcrit.eligibility_eva_library.v1",
        "revision": 0,
        "value_definitions": {
            "Categorical": ["Included", "Excluded"],
            "SexGender": ["male", "female", "all"],
            "Numerical": ["Range", "Point"],
        },
        "entities": [
            {
                "entity_id": "condition",
                "canonical_name": "Condition",
                "description": "",
                "aliases": [],
                "active": True,
            }
        ],
        "attributes": [
            {
                "attribute_id": "liver_dysfunction",
                "canonical_name": "Liver dysfunction",
                "entity_id": "condition",
                "description": "Abnormal liver function.",
                "aliases": [],
                "value_type": "Categorical",
                "numerical_type": None,
                "canonical_unit": None,
                "active": True,
            }
        ],
        "audit_history": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class SemanticProvider:
    model = "semantic-fake"
    reasoning_effort = "low"

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0

    def generate_json(self, prompt, *, output_schema):
        self.calls += 1
        if self.fail:
            raise RuntimeError("semantic provider failed")
        raw = prompt.split("BEGIN_UNTRUSTED_EVIDENCE_JSON\n", 1)[1].split(
            "\nEND_UNTRUSTED_EVIDENCE_JSON", 1
        )[0]
        payload = json.loads(raw)
        fingerprint = payload["candidate_fingerprint"]
        return {
            "candidate_fingerprint": fingerprint,
            "assessments": [
                {
                    "attribute_id": candidate["attribute_id"],
                    "candidate_fingerprint": fingerprint,
                    "relationship": "related_not_equivalent",
                    "confidence": 0.8,
                    "explanation": "The concepts are related but not equivalent.",
                }
                for candidate in payload["untrusted_library_retrieval_evidence"]
            ],
        }

    def close(self):
        return None


def _wait_terminal(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get_reconciliation_job(job_id)
        if job and job.status in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("reconciliation job did not finish")


def _manager_fixture(tmp_path, *, review_mode="full", provider=None):
    library_path = tmp_path / "library.json"
    _write_library(library_path)
    repository = EvaLibraryRepository(library_path)
    audit = _audit("Hepatic impairment")
    audit.library_revision = repository.revision
    audit.library_sha256 = repository.sha256
    providers = []

    def builder(*args):
        chosen = provider or SemanticProvider()
        providers.append(chosen)
        return SemanticProvider(), chosen

    manager = EligibilityEvaJobManager(
        points_directory=tmp_path / "points",
        library_path=library_path,
        work_directory=tmp_path / "work",
        provider_builder=builder,
        review_mode=review_mode,
    )
    manager._audit_path("TEST").write_text(
        json.dumps(audit.model_dump(mode="json")), encoding="utf-8"
    )
    return manager, audit, providers


def test_manager_trials_metadata_exposes_current_retrieval_version(tmp_path):
    manager, _, _ = _manager_fixture(tmp_path)
    try:
        payload = manager.list_trials()
    finally:
        manager.close()
    assert (
        payload["semantic_retrieval_version"]
        == SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION
    )


def test_manager_runs_persisted_semantic_job_and_preview(tmp_path):
    manager, audit, providers = _manager_fixture(tmp_path)
    try:
        started = manager.start_reconciliation(trial_key="TEST", audit=audit)
        completed = _wait_terminal(manager, started.job_id)
        preview = manager.get_reconciliation_preview(
            started.job_id, trial_key="TEST"
        )
    finally:
        manager.close()
    assert completed.status == "completed"
    assert completed.completed_items == 1
    assert len(preview.reviews) == 1
    assert preview.reviews[0].status == "completed"
    assert providers[0].calls == 1


def test_manager_duplicate_start_reuses_active_or_successful_job(tmp_path):
    manager, audit, _ = _manager_fixture(tmp_path)
    try:
        first = manager.start_reconciliation(trial_key="TEST", audit=audit)
        second = manager.start_reconciliation(trial_key="TEST", audit=audit)
        _wait_terminal(manager, first.job_id)
        third = manager.start_reconciliation(trial_key="TEST", audit=audit)
    finally:
        manager.close()
    assert second.job_id == first.job_id
    assert third.job_id == first.job_id


def test_manager_explicit_retry_creates_child_attempt(tmp_path):
    broken = SemanticProvider(fail=True)
    manager, audit, _ = _manager_fixture(tmp_path, provider=broken)
    try:
        first = manager.start_reconciliation(trial_key="TEST", audit=audit)
        failed_items = _wait_terminal(manager, first.job_id)
        child = manager.start_reconciliation(
            trial_key="TEST", audit=audit, retry_of=first.job_id
        )
        _wait_terminal(manager, child.job_id)
    finally:
        manager.close()
    assert failed_items.retryable_item_count == 1
    assert child.parent_job_id == first.job_id
    assert child.attempt_number == 2


def test_review_mode_off_has_no_semantic_job_or_provider(tmp_path):
    manager, audit, providers = _manager_fixture(
        tmp_path, review_mode="off"
    )
    try:
        from backend.criteria_processor.eva_reconciliation_jobs import (
            SemanticReconciliationDisabledError,
        )

        with pytest.raises(SemanticReconciliationDisabledError):
            manager.start_reconciliation(trial_key="TEST", audit=audit)
    finally:
        manager.close()
    assert providers == []
    assert not (tmp_path / "work" / "reconciliation_jobs").exists()


def test_manager_apply_uses_trusted_job_without_provider_call(tmp_path):
    manager, audit, providers = _manager_fixture(tmp_path)
    try:
        started = manager.start_reconciliation(trial_key="TEST", audit=audit)
        completed = _wait_terminal(manager, started.job_id)
        calls_before_apply = providers[0].calls
        decision = {
            "eva_id": audit.items[0].eva_id,
            "proposal_fingerprint": proposal_fingerprint(audit.items[0]),
            "decision": "KEEP_NEW",
            "target_attribute_id": None,
            "reason": "The reviewed candidate is related but not equivalent.",
        }
        result = manager.apply_audit(
            "TEST",
            {
                "audit": audit.model_dump(mode="json"),
                "reconciliation_decisions": [decision],
                "reconciliation_job_id": started.job_id,
            },
        )
    finally:
        manager.close()
    assert completed.status == "completed"
    assert providers[0].calls == calls_before_apply
    assert result["reconciliation_summary"]["new_count"] == 1
    assert result["audit"]["items"][0]["library_reconciliation"][
        "semantic_review"
    ]["status"] == "completed"
