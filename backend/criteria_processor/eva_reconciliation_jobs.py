"""Persisted lifecycle for asynchronous EVA semantic reconciliation jobs."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.criteria_processor.atomic_io import atomic_write_json
from backend.criteria_processor.eva_models import (
    EvaAuditDocument,
    EvaSemanticProposalReview,
)
from backend.criteria_processor.eva_reconciliation import proposal_fingerprint


JobStatus = Literal["queued", "running", "completed", "failed"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Any) -> str:
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SemanticReconciliationDisabledError(ValueError):
    code = "semantic_reconciliation_disabled"


class InvalidReconciliationRetryParentError(ValueError):
    code = "invalid_reconciliation_retry_parent"


class ReconciliationPreviewNotReadyError(ValueError):
    code = "reconciliation_preview_not_ready"


class EvaReconciliationJobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_key: str
    audit_id: str
    base_input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprints: list[str]
    library_revision: int = Field(ge=0)
    library_sha256: str
    provider_name: str
    provider_model: str
    provider_reasoning_effort: str
    prompt_version: str
    retrieval_version: str = Field(min_length=1)
    review_mode: Literal["off", "full"]


class EvaReconciliationJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["eligcrit.eva_reconciliation_job.v1"] = (
        "eligcrit.eva_reconciliation_job.v1"
    )
    job_id: str
    trial_key: str
    audit_id: str
    status: JobStatus = "queued"
    base_input_fingerprint: str
    proposal_fingerprints: list[str]
    library_revision: int
    library_sha256: str
    provider_name: str
    provider_model: str
    provider_reasoning_effort: str
    prompt_version: str
    retrieval_version: str = "legacy-unversioned"
    review_mode: Literal["off", "full"]
    parent_job_id: str | None = None
    attempt_number: int = Field(default=1, ge=1)
    total_items: int = Field(default=0, ge=0)
    completed_items: int = Field(default=0, ge=0)
    retryable_item_count: int = Field(default=0, ge=0)
    retryable: bool = False
    error_type: str = ""
    error_message: str = ""
    preview_path: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class EvaReconciliationPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["eligcrit.eva_reconciliation_preview.v1"] = (
        "eligcrit.eva_reconciliation_preview.v1"
    )
    job_id: str
    trial_key: str
    audit_id: str
    base_input_fingerprint: str
    library_revision: int
    library_sha256: str
    attempt_number: int
    retrieval_version: str = "legacy-unversioned"
    reviews: list[EvaSemanticProposalReview] = Field(default_factory=list)
    created_at: str


def reconciliation_base_input_fingerprint(
    *,
    trial_key: str,
    audit: EvaAuditDocument,
    library_revision: int,
    library_sha256: str,
    provider_config: Mapping[str, Any],
    prompt_version: str,
    retrieval_version: str,
    review_mode: str,
) -> str:
    items = sorted(
        [
            {
                "eva_id": item.eva_id,
                "proposal_fingerprint": proposal_fingerprint(item),
                "attribute_description": item.attribute_description,
                "review_status": item.review_status,
                "source_item_id": item.source_item_id,
            }
            for item in audit.items
        ],
        key=lambda row: row["eva_id"],
    )
    return _canonical_sha256(
        {
            "trial_key": trial_key,
            "audit_id": audit.audit_id,
            "items": items,
            "library_revision": library_revision,
            "library_sha256": library_sha256,
            "provider_config": dict(provider_config),
            "prompt_version": prompt_version,
            "retrieval_version": retrieval_version,
            "review_mode": review_mode,
        }
    )


class EligibilityEvaReconciliationJobStore:
    """Atomic job/preview store with immutable terminal attempts."""

    def __init__(self, work_directory: str | Path) -> None:
        self.work_directory = Path(work_directory).resolve()
        self.job_directory = self.work_directory / "reconciliation_jobs"
        self.preview_directory = self.work_directory / "reconciliation_previews"
        self.job_directory.mkdir(parents=True, exist_ok=True)
        self.preview_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._recover_interrupted()

    def _job_path(self, job_id: str) -> Path:
        return self.job_directory / f"{job_id}.json"

    def _preview_path(self, job_id: str) -> Path:
        return self.preview_directory / f"{job_id}.json"

    def _read_job_path(self, path: Path) -> EvaReconciliationJobRecord:
        try:
            return EvaReconciliationJobRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            quarantine = path.with_suffix(path.suffix + ".corrupt")
            path.replace(quarantine)
            raise ValueError(f"Corrupt reconciliation job quarantined: {path.name}")

    def _all(self) -> list[EvaReconciliationJobRecord]:
        records = []
        for path in sorted(self.job_directory.glob("*.json")):
            records.append(self._read_job_path(path))
        return records

    def _write(self, job: EvaReconciliationJobRecord) -> None:
        atomic_write_json(self._job_path(job.job_id), job.model_dump(mode="json"))

    def _recover_interrupted(self) -> None:
        with self._lock:
            for job in self._all():
                if job.status not in {"queued", "running"}:
                    continue
                job.status = "failed"
                job.retryable = True
                job.error_type = "InterruptedReconciliationJob"
                job.error_message = (
                    "The process stopped before semantic reconciliation completed."
                )
                job.updated_at = _utc_now()
                job.completed_at = job.updated_at
                self._write(job)

    def get(self, job_id: str) -> EvaReconciliationJobRecord | None:
        path = self._job_path(job_id)
        if not path.is_file():
            return None
        with self._lock:
            return self._read_job_path(path)

    def create_or_reuse(
        self,
        input: EvaReconciliationJobInput | Mapping[str, Any],
        *,
        retry_of: str | None = None,
    ) -> EvaReconciliationJobRecord:
        validated = (
            input
            if isinstance(input, EvaReconciliationJobInput)
            else EvaReconciliationJobInput.model_validate(input)
        )
        with self._lock:
            jobs = self._all()
            if retry_of is None:
                reusable = [
                    job
                    for job in jobs
                    if job.base_input_fingerprint
                    == validated.base_input_fingerprint
                    and (
                        job.status in {"queued", "running"}
                        or (
                            job.status == "completed"
                            and job.retryable_item_count == 0
                        )
                    )
                ]
                if reusable:
                    return sorted(
                        reusable,
                        key=lambda job: (job.attempt_number, job.created_at),
                    )[-1]
                matching_failed = [
                    job
                    for job in jobs
                    if job.base_input_fingerprint
                    == validated.base_input_fingerprint
                    and (
                        job.status == "failed"
                        or job.retryable_item_count > 0
                    )
                ]
                if matching_failed:
                    raise InvalidReconciliationRetryParentError(
                        "A retryable terminal job requires explicit retry_of."
                    )
                parent = None
                attempt = 1
            else:
                parent = next((job for job in jobs if job.job_id == retry_of), None)
                if (
                    parent is None
                    or parent.base_input_fingerprint
                    != validated.base_input_fingerprint
                    or not (
                        parent.retryable
                        or parent.retryable_item_count > 0
                    )
                    or parent.status not in {"completed", "failed"}
                ):
                    raise InvalidReconciliationRetryParentError(
                        "retry_of must name a retryable terminal same-input job."
                    )
                attempt = parent.attempt_number + 1
            now = _utc_now()
            job = EvaReconciliationJobRecord(
                job_id=(
                    f"evarec_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_"
                    f"{uuid4().hex[:12]}"
                ),
                **validated.model_dump(mode="json"),
                parent_job_id=parent.job_id if parent else None,
                attempt_number=attempt,
                total_items=len(validated.proposal_fingerprints),
                created_at=now,
                updated_at=now,
            )
            self._write(job)
            return job

    def mark_running(self, job_id: str) -> EvaReconciliationJobRecord:
        return self._transition(job_id, expected="queued", status="running")

    def update_progress(
        self, job_id: str, *, completed_items: int
    ) -> EvaReconciliationJobRecord:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status != "running":
                raise ValueError("Only a running reconciliation job can progress.")
            job.completed_items = completed_items
            job.updated_at = _utc_now()
            self._write(job)
            return job

    def mark_completed(
        self,
        job_id: str,
        *,
        preview: EvaReconciliationPreview,
        retryable_item_count: int = 0,
    ) -> EvaReconciliationJobRecord:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status != "running":
                raise ValueError("Only a running job can complete.")
            if preview.job_id != job_id or preview.trial_key != job.trial_key:
                raise ValueError("Preview ownership does not match the job.")
            if preview.retrieval_version != job.retrieval_version:
                raise ValueError(
                    "Preview retrieval version does not match the job."
                )
            preview_path = self._preview_path(job_id)
            atomic_write_json(preview_path, preview.model_dump(mode="json"))
            now = _utc_now()
            job.status = "completed"
            job.completed_items = job.total_items
            job.retryable_item_count = retryable_item_count
            job.retryable = retryable_item_count > 0
            job.preview_path = str(preview_path)
            job.updated_at = now
            job.completed_at = now
            self._write(job)
            return job

    def mark_failed(
        self,
        job_id: str,
        *,
        retryable: bool,
        error_type: str = "ReconciliationJobError",
        error_message: str = "Semantic reconciliation failed.",
    ) -> EvaReconciliationJobRecord:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                raise ValueError("Only an active reconciliation job can fail.")
            now = _utc_now()
            job.status = "failed"
            job.retryable = retryable
            job.error_type = error_type
            job.error_message = error_message
            job.updated_at = now
            job.completed_at = now
            self._write(job)
            return job

    def _transition(
        self, job_id: str, *, expected: JobStatus, status: JobStatus
    ) -> EvaReconciliationJobRecord:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status != expected:
                raise ValueError(f"Expected reconciliation job status {expected}.")
            job.status = status
            job.updated_at = _utc_now()
            if status == "running":
                job.started_at = job.updated_at
            self._write(job)
            return job

    def get_preview(
        self, job_id: str, *, trial_key: str | None = None
    ) -> EvaReconciliationPreview:
        job = self.get(job_id)
        if job is None or (trial_key is not None and job.trial_key != trial_key):
            raise FileNotFoundError("Reconciliation job not found.")
        if job.status != "completed" or not job.preview_path:
            raise ReconciliationPreviewNotReadyError(
                "Reconciliation preview is not ready."
            )
        return EvaReconciliationPreview.model_validate_json(
            Path(job.preview_path).read_text(encoding="utf-8")
        )
