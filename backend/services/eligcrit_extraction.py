"""Modern AutoCrit adapter for the EligCrit extraction and EVA pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from backend.criteria_processor.atomic_io import atomic_write_json
from backend.criteria_processor.breakdown_cli import load_eligibility_trial, process_trial
from backend.criteria_processor.eva_library import EvaLibraryRepository
from backend.criteria_processor.eva_pipeline import extract_trial_eva
from backend.criteria_processor.llm_providers import RuntimeRecordingJsonLLMProvider

from backend.api.app_state import AppState
from backend.services.ctgov_v2_service import eligibility_text_from_study, fetch_study
from backend.services.llm.factory import create_llm_provider
from backend.services.llm.json_adapter import StructuredJsonAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(
    *, trial_id: str, eligibility_text: str, model: str, library_sha256: str
) -> str:
    trial_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", trial_id).strip("._") or "trial"
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "eligibility_text": eligibility_text.strip(),
                "model": model,
                "library_sha256": library_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{trial_slug}_{fingerprint}"


@dataclass
class ExtractionJob:
    job_id: str
    trial_id: str
    status: str
    created_at: str
    updated_at: str
    source_path: str
    breakdown_path: str | None = None
    audit_path: str | None = None
    error: str | None = None
    cache_hit: bool = False


class EligCritExtractionService:
    """Persist, run, resume, and inspect browser-initiated extraction jobs."""

    def __init__(self, state: AppState, work_dir: Path | None = None) -> None:
        self.state = state
        self.work_dir = (work_dir or PROJECT_ROOT / "runtime" / "eligcrit").resolve()
        self.jobs_dir = self.work_dir / "jobs"
        self.sources_dir = self.work_dir / "sources"
        self.points_dir = self.work_dir / "points"
        self.checkpoints_dir = self.work_dir / "checkpoints"
        self.audits_dir = self.work_dir / "audits"
        self.records_dir = self.work_dir / "records"
        for path in (
            self.jobs_dir,
            self.sources_dir,
            self.points_dir,
            self.checkpoints_dir,
            self.audits_dir,
            self.records_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def submit(self, *, trial_id: str, eligibility_text: str) -> dict[str, Any]:
        trial_id = trial_id.strip() or f"LOCAL-{uuid4().hex[:8].upper()}"
        source_payload: dict[str, Any]
        if not eligibility_text.strip():
            if trial_id.startswith("LOCAL-"):
                raise ValueError("Enter an NCT ID or paste eligibility criteria text.")
            source_payload = fetch_study(trial_id)
            eligibility_text = eligibility_text_from_study(source_payload)
        else:
            source_payload = {
                "protocolSection": {
                    "identificationModule": {"nctId": trial_id},
                    "eligibilityModule": {"eligibilityCriteria": eligibility_text},
                }
            }
        job_id = uuid4().hex
        source_path = self.sources_dir / f"{job_id}.json"
        atomic_write_json(source_path, source_payload)
        now = _now()
        job = ExtractionJob(
            job_id=job_id,
            trial_id=trial_id,
            status="queued",
            created_at=now,
            updated_at=now,
            source_path=str(source_path),
        )
        self._save(job)
        Thread(target=self._run, args=(job_id,), daemon=True).start()
        return asdict(job)

    def get(self, job_id: str) -> dict[str, Any]:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def resume_incomplete(self) -> int:
        resumed = 0
        for path in self.jobs_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") in {"queued", "running"}:
                payload["status"] = "queued"
                payload["updated_at"] = _now()
                atomic_write_json(path, payload)
                Thread(target=self._run, args=(payload["job_id"],), daemon=True).start()
                resumed += 1
        return resumed

    def _save(self, job: ExtractionJob) -> None:
        with self._lock:
            atomic_write_json(self.jobs_dir / f"{job.job_id}.json", asdict(job))

    def _run(self, job_id: str) -> None:
        job = ExtractionJob(**self.get(job_id))
        job.status = "running"
        job.updated_at = _now()
        self._save(job)
        provider = None
        try:
            settings = self.state.settings.llm
            source_payload = json.loads(Path(job.source_path).read_text(encoding="utf-8"))
            eligibility_text = eligibility_text_from_study(source_payload)
            library = EvaLibraryRepository(
                PROJECT_ROOT / "config" / "eligibility_eva_library.json",
                snapshot_directory=self.work_dir / "library_snapshots",
            )
            cache_key = _cache_key(
                trial_id=job.trial_id,
                eligibility_text=eligibility_text,
                model=settings.model,
                library_sha256=library.sha256,
            )
            trial = replace(load_eligibility_trial(job.source_path), trial_key=cache_key)
            cached_audit_path = self.audits_dir / f"{cache_key}.json"
            cached_breakdown_path = (
                self.points_dir / cache_key / "elig_breakdown_points.csv"
            )
            if cached_audit_path.is_file() and cached_breakdown_path.is_file():
                job.breakdown_path = str(cached_breakdown_path.resolve())
                job.audit_path = str(cached_audit_path.resolve())
                job.status = "completed"
                job.error = None
                job.cache_hit = True
                return
            base_provider = create_llm_provider(
                settings,
                cost_monitor=self.state.cost_monitor,
            )
            provider = RuntimeRecordingJsonLLMProvider(
                StructuredJsonAdapter(base_provider, model=settings.model),
                record_directory=self.records_dir,
                run_id=job.job_id,
                run_metadata={"trial_id": job.trial_id},
            )
            breakdown = process_trial(
                trial=trial,
                provider=provider,
                output_directory=self.points_dir,
                checkpoint_directory=self.checkpoints_dir,
                progress_callback=lambda _message: None,
            )
            job.breakdown_path = str(breakdown.output_path)
            audit = extract_trial_eva(
                trial_key=trial.trial_key,
                trial_id=trial.trial_id,
                breakdown_path=breakdown.output_path,
                library=library,
                search_provider=provider,
                reasoner_provider=provider,
                search_model=settings.model,
                search_reasoning_effort="configured",
                reasoner_model=settings.model,
                reasoner_reasoning_effort="configured",
                workers=1,
                review_mode="full",
            )
            audit_path = cached_audit_path
            atomic_write_json(audit_path, audit.model_dump(mode="json"))
            job.audit_path = str(audit_path)
            job.status = "completed"
            job.error = None
            job.cache_hit = False
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            if provider is not None:
                provider.close()
            job.updated_at = _now()
            self._save(job)
