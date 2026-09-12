"""Persisted per-trial jobs and audit approval for eligibility EVA."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from backend.criteria_processor.atomic_io import (
    atomic_write_json,
    atomic_write_jsonl,
)
from backend.criteria_processor.eva_library import EvaLibraryRepository
from backend.criteria_processor.eva_audit_migration import load_eva_audit
from backend.criteria_processor.eva_models import (
    EvaApplyEnvelope,
    EvaAuditDocument,
    EvaLibraryReconciliationEvidence,
)
from backend.criteria_processor.eva_reconciliation import (
    EvaReconciliationError,
    ReconciliationRequiredError,
    build_integrity_preview,
    proposal_fingerprint,
)
from backend.criteria_processor.eva_reconciliation_jobs import (
    EligibilityEvaReconciliationJobStore,
    EvaReconciliationJobInput,
    EvaReconciliationPreview,
    SemanticReconciliationDisabledError,
    reconciliation_base_input_fingerprint,
)
from backend.criteria_processor.eva_semantic_reconciliation import (
    SEMANTIC_RECONCILIATION_PROMPT_VERSION,
    retrieve_semantic_candidates,
    review_semantic_proposal,
)
from backend.criteria_processor.eva_retrieval import (
    SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
)
from backend.criteria_processor.eva_pipeline import (
    extract_trial_eva,
    load_breakdown_points,
)
from backend.criteria_processor.eva_recovery import (
    recover_trial_eva_from_runtime_records,
)
from backend.criteria_processor.llm_providers import (
    CodexJsonLLMProvider,
    RuntimeRecordingJsonLLMProvider,
)


DEFAULT_SEARCH_MODEL = "gpt-5.4-mini"
DEFAULT_REASONER_MODEL = "gpt-5.5"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EligibilityEvaJobManager:
    """Run one selected trial at a time and retain auditable state."""

    def __init__(
        self,
        *,
        points_directory: str | Path,
        library_path: str | Path,
        work_directory: str | Path,
        verified_output_directory: str | Path | None = None,
        runtime_record_directory: str | Path | None = None,
        search_model: str = DEFAULT_SEARCH_MODEL,
        search_reasoning_effort: str = "low",
        reasoner_model: str = DEFAULT_REASONER_MODEL,
        reasoner_reasoning_effort: str = "medium",
        top_k: int = 10,
        item_workers: int = 4,
        max_jobs: int = 2,
        provider_builder: Callable[
            [str, str, str, str], tuple[Any, Any]
        ]
        | None = None,
        review_mode: str = "full",
    ) -> None:
        self.points_directory = Path(points_directory).resolve()
        self.library_path = Path(library_path).resolve()
        self.work_directory = Path(work_directory).resolve()
        self.audit_directory = self.work_directory / "audits"
        self.job_directory = self.work_directory / "jobs"
        self.normalized_directory = self.work_directory / "normalized"
        self.verified_output_directory = (
            Path(verified_output_directory).resolve()
            if verified_output_directory is not None
            else None
        )
        self.runtime_record_directory = (
            Path(runtime_record_directory).resolve()
            if runtime_record_directory is not None
            else None
        )
        self.search_model = search_model
        self.search_reasoning_effort = search_reasoning_effort
        self.reasoner_model = reasoner_model
        self.reasoner_reasoning_effort = reasoner_reasoning_effort
        self.top_k = top_k
        self.item_workers = item_workers
        if review_mode not in {"off", "full"}:
            raise ValueError("review_mode must be 'off' or 'full'.")
        self.review_mode = review_mode
        self.provider_builder = (
            provider_builder or self._default_provider_builder
        )
        self._executor = ThreadPoolExecutor(max_workers=max_jobs)
        self._reconciliation_executor = (
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="eva-semantic-reconciliation",
            )
            if review_mode == "full"
            else None
        )
        self.reconciliation_store = (
            EligibilityEvaReconciliationJobStore(self.work_directory)
            if review_mode == "full"
            else None
        )
        self._reconciliation_futures: dict[str, Future[Any]] = {}
        self._reconciliation_accepting = True
        self._lock = threading.Lock()
        self._job_io_lock = threading.RLock()
        self._running_by_trial: dict[str, str] = {}
        for directory in (
            self.audit_directory,
            self.job_directory,
            self.normalized_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.verified_output_directory is not None:
            self.verified_output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
        self._reconciliation_accepting = False
        executor = self._reconciliation_executor
        if executor is not None:
            futures = list(self._reconciliation_futures.values())
            for future in futures:
                future.cancel()
            wait(futures, timeout=2.0)
            executor.shutdown(wait=False, cancel_futures=True)

    def list_trials(self) -> dict[str, Any]:
        library = EvaLibraryRepository(self.library_path)
        trials = []
        for path in sorted(
            self.points_directory.glob(
                "*/elig_breakdown_points.csv"
            ),
            key=lambda value: value.parent.name.casefold(),
        ):
            trial_key = path.parent.name
            try:
                point_count = len(load_breakdown_points(path))
                input_status = "ready"
                input_error = ""
            except ValueError as exc:
                point_count = 0
                input_status = "invalid"
                input_error = str(exc)
            audit = self.get_audit(trial_key)
            trials.append(
                {
                    "trial_key": trial_key,
                    "trial_id": (
                        audit["trial_id"] if audit else trial_key
                    ),
                    "point_count": point_count,
                    "input_status": input_status,
                    "input_error": input_error,
                    "audit_status": (
                        audit["review_status"]
                        if audit
                        else "not_run"
                    ),
                    "audit_library_revision": (
                        audit["library_revision"] if audit else None
                    ),
                    "library_revision_stale": bool(
                        audit
                        and audit["review_status"] != "approved"
                        and audit["library_revision"] != library.revision
                    ),
                }
            )
        return {
            "trials": trials,
            "library_revision": library.revision,
            "search_model": self.search_model,
            "search_reasoning_effort": self.search_reasoning_effort,
            "reasoner_model": self.reasoner_model,
            "reasoner_reasoning_effort": (
                self.reasoner_reasoning_effort
            ),
            "top_k": self.top_k,
            "review_mode": self.review_mode,
            "semantic_retrieval_version": (
                SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION
            ),
        }

    def library_payload(self) -> dict[str, Any]:
        return EvaLibraryRepository(self.library_path).public_payload()

    def start(self, trial_key: str) -> dict[str, Any]:
        breakdown_path = self._breakdown_path(trial_key)
        rows = load_breakdown_points(breakdown_path)
        with self._lock:
            running_id = self._running_by_trial.get(trial_key)
            if running_id:
                job = self.get(running_id)
                if job and job["status"] in {"queued", "running"}:
                    return job
            job_id = (
                f"eva_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_"
                f"{uuid4().hex[:10]}"
            )
            job = {
                "schema_version": "eligcrit.eligibility_eva_job.v1",
                "job_id": job_id,
                "trial_key": trial_key,
                "trial_id": trial_key,
                "status": "queued",
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "completed_items": 0,
                "total_items": len(rows),
                "audit_path": None,
                "error": None,
            }
            self._write_job(job)
            self._running_by_trial[trial_key] = job_id
            self._executor.submit(
                self._run_job,
                job_id,
                trial_key,
                breakdown_path,
            )
            return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self.job_directory / f"{job_id}.json"
        with self._job_io_lock:
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def get_audit(self, trial_key: str) -> dict[str, Any] | None:
        path = self._audit_path(trial_key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            load_eva_audit(payload).model_dump(mode="json")
            if isinstance(payload, dict)
            else None
        )

    def recover_completed_job(self, job_id: str) -> dict[str, Any]:
        """Recover a complete failed run from its recorded structured calls."""

        job = self.get(job_id)
        if job is None:
            raise ValueError(f"Eligibility EVA job not found: {job_id}")
        if job.get("status") in {"queued", "running"}:
            raise ValueError("A running EVA job cannot be recovered.")
        trial_key = str(job.get("trial_key") or "").strip()
        breakdown_path = self._breakdown_path(trial_key)
        if self.runtime_record_directory is None:
            raise ValueError(
                "Runtime records are not configured for EVA recovery."
            )
        library = EvaLibraryRepository(self.library_path)
        audit = recover_trial_eva_from_runtime_records(
            trial_key=trial_key,
            trial_id=str(job.get("trial_id") or trial_key),
            breakdown_path=breakdown_path,
            library=library,
            search_run_directory=(
                self.runtime_record_directory / f"{job_id}_search"
            ),
            reasoner_run_directory=(
                self.runtime_record_directory / f"{job_id}_reasoner"
            ),
            search_model=self.search_model,
            search_reasoning_effort=self.search_reasoning_effort,
            reasoner_model=self.reasoner_model,
            reasoner_reasoning_effort=self.reasoner_reasoning_effort,
            created_at=str(job.get("created_at") or "") or None,
        )
        audit_path = self._audit_path(trial_key)
        atomic_write_json(
            audit_path,
            audit.model_dump(mode="json"),
        )
        recovered_at = _utc_now()
        original_error = job.get("error")
        job["status"] = "completed"
        job["completed_items"] = len(audit.items)
        job["total_items"] = len(audit.items)
        job["audit_path"] = str(audit_path)
        job["error"] = None
        job["updated_at"] = recovered_at
        job["completed_at"] = recovered_at
        job["recovered_at"] = recovered_at
        job["recovered_from_runtime_records"] = True
        mapping_failed = sum(
            item.mapping_review.status in {"failed", "blocked"}
            for item in audit.items
        )
        id_failed = sum(
            item.attribute_id_review.status in {"failed", "blocked"}
            for item in audit.items
        )
        job["mapping_review_failed_count"] = mapping_failed
        job["id_review_failed_count"] = id_failed
        job["review_issue_count"] = mapping_failed + id_failed
        job["requires_human_review"] = bool(mapping_failed + id_failed)
        if original_error:
            job["recovery_source_error"] = original_error
        self._write_job(job)
        return job

    def save_audit(
        self,
        trial_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        incoming = self._validated_incoming_audit(
            trial_key=trial_key,
            audit=load_eva_audit(payload),
        )
        current_payload = self.get_audit(trial_key)
        assert current_payload is not None
        current = load_eva_audit(current_payload)
        incoming.updated_at = _utc_now()
        if current.review_status == "approved" or incoming.review_status == "approved":
            incoming.approved_at = None
            incoming.approved_library_revision = None
            incoming.approved_library_sha256 = None
        incoming.review_status = "revised"
        atomic_write_json(
            self._audit_path(trial_key),
            incoming.model_dump(mode="json"),
        )
        return incoming.model_dump(mode="json")

    def _validated_incoming_audit(
        self,
        *,
        trial_key: str,
        audit: EvaAuditDocument,
    ) -> EvaAuditDocument:
        """Restore immutable source fields and discard client evidence."""

        current_payload = self.get_audit(trial_key)
        if current_payload is None:
            raise ValueError(
                f"No EVA audit exists for trial '{trial_key}'."
            )
        current = load_eva_audit(current_payload)
        incoming = audit.model_copy(deep=True)
        if (
            incoming.trial_key != trial_key
            or incoming.audit_id != current.audit_id
        ):
            approved = load_eva_audit(current)
            live = EvaLibraryRepository(self.library_path)
            if (
                approved.approved_library_revision != live.revision
                or approved.approved_library_sha256 != live.sha256
            ):
                from backend.criteria_processor.eva_library import EvaLibraryStaleError

                raise EvaLibraryStaleError(
                    "Approved audit library fingerprint is stale."
                )
            raise ValueError("EVA audit identity cannot be changed.")
        current_items = {
            item.eva_id: item for item in current.items
        }
        if {item.eva_id for item in incoming.items} != set(current_items):
            raise ValueError(
                "EVA audit source rows cannot be added or removed."
            )
        for item in incoming.items:
            source = current_items[item.eva_id]
            item.source_item_id = source.source_item_id
            item.criteria = source.criteria
            item.item = source.item
            item.context = source.context
            item.search_expansion = source.search_expansion
            item.candidates = source.candidates
            item.library_reconciliation = EvaLibraryReconciliationEvidence()

        incoming.schema_version = current.schema_version
        incoming.audit_id = current.audit_id
        incoming.trial_key = current.trial_key
        incoming.trial_id = current.trial_id
        incoming.source_path = current.source_path
        incoming.library_revision = current.library_revision
        incoming.library_sha256 = current.library_sha256
        incoming.created_at = current.created_at
        incoming.search_model = current.search_model
        incoming.search_reasoning_effort = (
            current.search_reasoning_effort
        )
        incoming.reasoner_model = current.reasoner_model
        incoming.reasoner_reasoning_effort = (
            current.reasoner_reasoning_effort
        )
        return incoming

    def preview_integrity(
        self,
        *,
        trial_key: str,
        audit: EvaAuditDocument | dict[str, Any],
    ) -> EvaAuditDocument:
        incoming = self._validated_incoming_audit(
            trial_key=trial_key,
            audit=(
                audit
                if isinstance(audit, EvaAuditDocument)
                else load_eva_audit(audit)
            ),
        )
        return build_integrity_preview(
            audit=incoming,
            library=EvaLibraryRepository(self.library_path),
        )

    def start_reconciliation(
        self,
        *,
        trial_key: str,
        audit: EvaAuditDocument | dict[str, Any],
        retry_of: str | None = None,
    ):
        if self.review_mode == "off":
            raise SemanticReconciliationDisabledError(
                "Semantic reconciliation is disabled in review_mode=off."
            )
        if not self._reconciliation_accepting:
            raise RuntimeError("Reconciliation manager is shutting down.")
        assert self.reconciliation_store is not None
        assert self._reconciliation_executor is not None
        cleaned = self._validated_incoming_audit(
            trial_key=trial_key,
            audit=(
                audit
                if isinstance(audit, EvaAuditDocument)
                else load_eva_audit(audit)
            ),
        )
        library = EvaLibraryRepository(self.library_path)
        exact_preview = build_integrity_preview(audit=cleaned, library=library)
        eligible = [
            item
            for item in exact_preview.items
            if item.review_status != "rejected"
            and item.mapping_decision == "new_attribute"
            and item.library_reconciliation.status == "not_required"
        ]
        provider_config = {
            "provider": "configured_reasoner",
            "model": self.reasoner_model,
            "reasoning_effort": self.reasoner_reasoning_effort,
        }
        base_fingerprint = reconciliation_base_input_fingerprint(
            trial_key=trial_key,
            audit=cleaned,
            library_revision=library.revision,
            library_sha256=library.sha256,
            provider_config=provider_config,
            prompt_version=SEMANTIC_RECONCILIATION_PROMPT_VERSION,
            retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
            review_mode=self.review_mode,
        )
        job_input = EvaReconciliationJobInput(
            trial_key=trial_key,
            audit_id=cleaned.audit_id,
            base_input_fingerprint=base_fingerprint,
            proposal_fingerprints=[
                item.library_reconciliation.proposal_fingerprint
                or proposal_fingerprint(item)
                for item in eligible
            ],
            library_revision=library.revision,
            library_sha256=library.sha256,
            provider_name="configured_reasoner",
            provider_model=self.reasoner_model,
            provider_reasoning_effort=self.reasoner_reasoning_effort,
            prompt_version=SEMANTIC_RECONCILIATION_PROMPT_VERSION,
            retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
            review_mode="full",
        )
        job = self.reconciliation_store.create_or_reuse(
            job_input, retry_of=retry_of
        )
        if job.status == "queued" and job.job_id not in self._reconciliation_futures:
            future = self._reconciliation_executor.submit(
                self._run_reconciliation_job,
                job.job_id,
                cleaned.model_dump(mode="json"),
            )
            self._reconciliation_futures[job.job_id] = future
        return job

    def get_reconciliation_job(self, job_id: str):
        if self.reconciliation_store is None:
            return None
        return self.reconciliation_store.get(job_id)

    def get_reconciliation_preview(
        self, job_id: str, *, trial_key: str | None = None
    ):
        if self.reconciliation_store is None:
            raise FileNotFoundError("Reconciliation job not found.")
        return self.reconciliation_store.get_preview(
            job_id, trial_key=trial_key
        )

    def _run_reconciliation_job(
        self, job_id: str, audit_payload: dict[str, Any]
    ) -> None:
        assert self.reconciliation_store is not None
        search_provider = None
        reasoner_provider = None
        try:
            job = self.reconciliation_store.mark_running(job_id)
            audit = load_eva_audit(audit_payload)
            library = EvaLibraryRepository(self.library_path)
            exact_preview = build_integrity_preview(audit=audit, library=library)
            eligible = [
                item
                for item in exact_preview.items
                if item.review_status != "rejected"
                and item.mapping_decision == "new_attribute"
                and item.library_reconciliation.status == "not_required"
            ]
            search_provider, reasoner_provider = self.provider_builder(
                job_id,
                self.search_model,
                self.reasoner_model,
                job.trial_key,
            )
            reviews = []
            for index, proposal in enumerate(eligible, start=1):
                diagnostic, review_candidates = retrieve_semantic_candidates(
                    proposal=proposal, library=library
                )
                review = review_semantic_proposal(
                    proposal=proposal,
                    candidates=review_candidates,
                    provider=reasoner_provider,
                )
                review.retrieval_version = (
                    SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION
                )
                review.diagnostic_candidates = diagnostic
                reviews.append(review)
                self.reconciliation_store.update_progress(
                    job_id, completed_items=index
                )
            preview = EvaReconciliationPreview(
                job_id=job_id,
                trial_key=job.trial_key,
                audit_id=job.audit_id,
                base_input_fingerprint=job.base_input_fingerprint,
                library_revision=job.library_revision,
                library_sha256=job.library_sha256,
                attempt_number=job.attempt_number,
                retrieval_version=SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION,
                reviews=reviews,
                created_at=_utc_now(),
            )
            self.reconciliation_store.mark_completed(
                job_id,
                preview=preview,
                retryable_item_count=sum(
                    review.status == "failed" for review in reviews
                ),
            )
        except Exception as exc:
            current = self.reconciliation_store.get(job_id)
            if current is not None and current.status in {"queued", "running"}:
                self.reconciliation_store.mark_failed(
                    job_id,
                    retryable=True,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        finally:
            for provider in (search_provider, reasoner_provider):
                close = getattr(provider, "close", None)
                if callable(close):
                    close()

    def apply_audit(
        self,
        trial_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        current = self.get_audit(trial_key)
        is_envelope = "audit" in payload
        if is_envelope:
            envelope = EvaApplyEnvelope.model_validate(payload)
            requested_audit = envelope.audit
            decisions = envelope.reconciliation_decisions
            reconciliation_job_id = envelope.reconciliation_job_id
        else:
            requested_audit = load_eva_audit(payload)
            decisions = []
            reconciliation_job_id = None
        if (
            not is_envelope
            and current is not None
            and current.get("review_status") == "approved"
            and payload.get("review_status") == "approved"
            and payload.get("audit_id") == current.get("audit_id")
        ):
            normalized_path = self._normalized_path(trial_key)
            verified_path = self._verified_normalized_path(trial_key)
            if normalized_path.is_file():
                normalized_rows = self._read_jsonl(normalized_path)
                if verified_path is not None:
                    atomic_write_jsonl(verified_path, normalized_rows)
                normalized_count = len(normalized_rows)
            else:
                normalized_count = self._jsonl_count(
                    verified_path
                    if verified_path is not None and verified_path.is_file()
                    else normalized_path
                )
            return {
                "audit": current,
                "normalized_path": str(normalized_path),
                "verified_normalized_path": (
                    str(verified_path) if verified_path is not None else None
                ),
                "normalized_count": normalized_count,
                "library": self.library_payload(),
            }
        audit = self._validated_incoming_audit(
            trial_key=trial_key,
            audit=requested_audit,
        )
        audit.updated_at = _utc_now()
        audit.review_status = "revised"
        audit.approved_at = None
        audit.approved_library_revision = None
        audit.approved_library_sha256 = None
        atomic_write_json(
            self._audit_path(trial_key),
            audit.model_dump(mode="json"),
        )
        library = EvaLibraryRepository(
            self.library_path,
            snapshot_directory=self.work_directory / "library_snapshots",
        )
        trusted_semantic_preview = None
        if reconciliation_job_id is not None:
            if self.review_mode == "off" or self.reconciliation_store is None:
                raise SemanticReconciliationDisabledError(
                    "Semantic reconciliation is disabled in review_mode=off."
                )
            job = self.reconciliation_store.get(reconciliation_job_id)
            if (
                job is None
                or job.trial_key != trial_key
                or job.audit_id != audit.audit_id
                or job.status != "completed"
            ):
                raise EvaReconciliationError(
                    "Reconciliation job is missing, incomplete, or belongs to another audit."
                )
            trusted_semantic_preview = self.reconciliation_store.get_preview(
                reconciliation_job_id,
                trial_key=trial_key,
            )
            if (
                trusted_semantic_preview.base_input_fingerprint
                != job.base_input_fingerprint
            ):
                raise EvaReconciliationError(
                    "Trusted semantic preview fingerprint does not match its job."
                )
        elif self.review_mode == "full" and is_envelope:
            exact_preview = build_integrity_preview(audit=audit, library=library)
            semantic_required = [
                item.eva_id
                for item in exact_preview.items
                if item.review_status != "rejected"
                and item.mapping_decision == "new_attribute"
                and item.library_reconciliation.status == "not_required"
            ]
            if semantic_required:
                raise ReconciliationRequiredError(
                    "Semantic reconciliation must complete before apply.",
                    eva_ids=semantic_required,
                )
        repository_result = library.apply_audit(
            audit,
            reconciliation_decisions=decisions,
            trusted_semantic_preview=trusted_semantic_preview,
        )
        approved = repository_result.audit
        normalized = repository_result.normalized_rows
        normalized_path = self._normalized_path(trial_key)
        verified_path = self._verified_normalized_path(trial_key)
        for row in normalized:
            row["library_revision"] = library.revision
            row["library_schema_version"] = (
                library.payload["schema_version"]
            )
        atomic_write_jsonl(normalized_path, normalized)
        if verified_path is not None:
            atomic_write_jsonl(verified_path, normalized)
        atomic_write_json(
            self._audit_path(trial_key),
            approved.model_dump(mode="json"),
        )
        return {
            "audit": approved.model_dump(mode="json"),
            "normalized_path": str(normalized_path),
            "verified_normalized_path": (
                str(verified_path) if verified_path is not None else None
            ),
            "normalized_count": len(normalized),
            "library": library.public_payload(),
            "reconciliation_summary": repository_result.reconciliation_summary.model_dump(
                mode="json"
            ),
            "replayed": repository_result.replayed,
            "receipt": repository_result.receipt.model_dump(mode="json"),
        }

    def _run_job(
        self,
        job_id: str,
        trial_key: str,
        breakdown_path: Path,
    ) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["updated_at"] = _utc_now()
        self._write_job(job)
        search_provider = None
        reasoner_provider = None
        try:
            search_provider, reasoner_provider = self.provider_builder(
                job_id,
                self.search_model,
                self.reasoner_model,
                trial_key,
            )

            def progress(completed: int, total: int) -> None:
                job["completed_items"] = completed
                job["total_items"] = total
                job["updated_at"] = _utc_now()
                try:
                    self._write_job(job)
                except OSError as exc:
                    job["persistence_warning"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }

            library = EvaLibraryRepository(self.library_path)
            audit = extract_trial_eva(
                trial_key=trial_key,
                trial_id=trial_key,
                breakdown_path=breakdown_path,
                library=library,
                search_provider=search_provider,
                reasoner_provider=reasoner_provider,
                search_model=self.search_model,
                search_reasoning_effort=self.search_reasoning_effort,
                reasoner_model=self.reasoner_model,
                reasoner_reasoning_effort=(
                    self.reasoner_reasoning_effort
                ),
                top_k=self.top_k,
                workers=self.item_workers,
                progress_callback=progress,
                review_mode=self.review_mode,
            )
            audit_path = self._audit_path(trial_key)
            atomic_write_json(
                audit_path,
                audit.model_dump(mode="json"),
            )
            job["status"] = "completed"
            job["completed_items"] = len(audit.items)
            job["total_items"] = len(audit.items)
            job["audit_path"] = str(audit_path)
            job["updated_at"] = _utc_now()
            job["completed_at"] = job["updated_at"]
            mapping_failed = sum(
                item.mapping_review.status in {"failed", "blocked"}
                for item in audit.items
            )
            id_failed = sum(
                item.attribute_id_review.status in {"failed", "blocked"}
                for item in audit.items
            )
            job["mapping_review_failed_count"] = mapping_failed
            job["id_review_failed_count"] = id_failed
            job["review_issue_count"] = mapping_failed + id_failed
            job["requires_human_review"] = bool(mapping_failed + id_failed)
            self._write_job(job)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            job["updated_at"] = _utc_now()
            job["completed_at"] = job["updated_at"]
            self._write_job(job)
        finally:
            for provider in (search_provider, reasoner_provider):
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
            with self._lock:
                if self._running_by_trial.get(trial_key) == job_id:
                    self._running_by_trial.pop(trial_key, None)

    def _default_provider_builder(
        self,
        job_id: str,
        search_model: str,
        reasoner_model: str,
        trial_key: str,
    ) -> tuple[Any, Any]:
        project_root = Path(__file__).resolve().parents[2]
        search: Any = CodexJsonLLMProvider(
            model=search_model,
            cwd=project_root,
            reasoning_effort=self.search_reasoning_effort,
        )
        reasoner: Any = CodexJsonLLMProvider(
            model=reasoner_model,
            cwd=project_root,
            reasoning_effort=self.reasoner_reasoning_effort,
        )
        if self.runtime_record_directory is None:
            return search, reasoner
        metadata = {
            "workflow": "eligibility_eva_extraction",
            "trial_key": trial_key,
            "top_k": self.top_k,
            "review_mode": self.review_mode,
        }
        return (
            RuntimeRecordingJsonLLMProvider(
                search,
                record_directory=self.runtime_record_directory,
                run_id=f"{job_id}_search",
                run_metadata={**metadata, "stage": "search_expansion"},
            ),
            RuntimeRecordingJsonLLMProvider(
                reasoner,
                record_directory=self.runtime_record_directory,
                run_id=f"{job_id}_reasoner",
                run_metadata={**metadata, "stage": "eva_reasoning"},
            ),
        )

    def _breakdown_path(self, trial_key: str) -> Path:
        safe_key = Path(trial_key).name
        if safe_key != trial_key or not safe_key:
            raise ValueError("Invalid trial key.")
        path = (
            self.points_directory
            / safe_key
            / "elig_breakdown_points.csv"
        ).resolve()
        if path.parent.parent != self.points_directory:
            raise ValueError("Trial key resolves outside points directory.")
        if not path.is_file():
            raise ValueError(
                f"Eligibility breakdown not found for '{trial_key}'."
            )
        return path

    def _audit_path(self, trial_key: str) -> Path:
        safe_key = Path(trial_key).name
        if safe_key != trial_key or not safe_key:
            raise ValueError("Invalid trial key.")
        return self.audit_directory / f"{safe_key}.eva_audit.json"

    def _normalized_path(self, trial_key: str) -> Path:
        safe_key = Path(trial_key).name
        if safe_key != trial_key or not safe_key:
            raise ValueError("Invalid trial key.")
        return (
            self.normalized_directory
            / safe_key
            / "eligibility_eva.jsonl"
        )

    def _verified_normalized_path(self, trial_key: str) -> Path | None:
        if self.verified_output_directory is None:
            return None
        safe_key = Path(trial_key).name
        if safe_key != trial_key or not safe_key:
            raise ValueError("Invalid trial key.")
        return (
            self.verified_output_directory
            / safe_key
            / "eligibility_eva.jsonl"
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _jsonl_count(path: Path) -> int:
        return (
            sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if path.is_file()
            else 0
        )

    def _write_job(self, payload: dict[str, Any]) -> None:
        with self._job_io_lock:
            atomic_write_json(
                self.job_directory / f"{payload['job_id']}.json",
                payload,
            )
