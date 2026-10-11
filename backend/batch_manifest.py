"""Recoverable sequential extraction for a manifest of ClinicalTrials.gov IDs."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from backend.api.app_state import create_app_state
from backend.atomic_io import atomic_write_json
from backend.eav_pipeline import normalize_rows
from backend.services.eav_workflow import EavWorkflowService, now


def manifest_key(path: Path) -> str:
    """Return a stable state key, preferring an explicit cohort identity."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cohort_id = payload.get("cohort_id") if isinstance(payload, dict) else None
    if cohort_id:
        key = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(cohort_id)).strip("-")
        if not key:
            raise ValueError("cohort_id must contain at least one letter or number.")
        return key
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def manifest_state_path(path: Path, runtime: Path) -> Path:
    return runtime / "manifests" / f"manifest-{manifest_key(path)}.json"


def load_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("trials") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Manifest must be an array or contain a trials array.")
    trial_ids = []
    for item in items:
        trial_id = item if isinstance(item, str) else item.get("trial_id") if isinstance(item, dict) else None
        if isinstance(item, dict) and not trial_id and (
            item.get("enabled") is False or str(item.get("status") or "").upper() in {"TBD", "PLACEHOLDER"}
        ):
            continue
        if not trial_id:
            raise ValueError("Every manifest item requires trial_id.")
        trial_ids.append(str(trial_id).strip().upper())
    return list(dict.fromkeys(trial_ids))


def renormalize_manifest(path: Path) -> dict[str, Any]:
    """Rebuild completed job rows from saved model checkpoints without API calls."""
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime" / "criteria"
    state_path = manifest_state_path(path, runtime)
    if not state_path.is_file():
        raise ValueError("No cohort state exists yet. Run extraction before re-normalizing.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed_jobs = row_count = 0
    skipped: list[str] = []
    for trial_id in load_manifest(path):
        record = state.get("jobs", {}).get(trial_id) or {}
        job_id = record.get("job_id")
        job_path = runtime / "jobs" / f"{job_id}.json"
        if not job_id or not job_path.is_file():
            skipped.append(trial_id); continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("status") != "completed" or not job.get("checkpoint_rows"):
            skipped.append(trial_id); continue
        job["rows"] = normalize_rows(
            {"rows": job["checkpoint_rows"]}, job.get("source_items") or [],
            protocol_id=job.get("protocol_id") or trial_id,
            source_version=job.get("source_version") or "unversioned",
            source_uri=job.get("source_uri"),
            source_retrieved_at=job.get("source_retrieved_at"),
            protocol_hash=job.get("protocol_hash"), trial_id=job.get("trial_id") or trial_id,
        )
        job["updated_at"] = now()
        job.setdefault("events", []).append({
            "timestamp": now(), "stage": "renormalized",
            "message": f"Re-normalized {len(job['rows'])} rows from saved model checkpoints without API calls.",
        })
        atomic_write_json(job_path, job)
        completed_jobs += 1; row_count += len(job["rows"])
        record.update({"status": "completed", "stage": "renormalized", "updated_at": job["updated_at"]})
    state["renormalized_at"] = now()
    atomic_write_json(state_path, state)
    return {
        "state_file": str(state_path), "completed_jobs": completed_jobs,
        "rows": row_count, "skipped_trials": skipped,
    }


def run_manifest(
    path: Path, *, poll_seconds: float = 1.0, retry_limit: int = 2,
    force_restart: bool = False,
) -> dict[str, Any]:
    """Run or resume a manifest; job checkpoints survive process interruption."""
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime" / "criteria"
    service = EavWorkflowService(
        create_app_state(), runtime, root / "config" / "base_criterion_library.json"
    )
    trial_ids = load_manifest(path)
    manifest_payload = json.loads(path.read_text(encoding="utf-8"))
    cohort_id = manifest_payload.get("cohort_id") if isinstance(manifest_payload, dict) else None
    state_path = manifest_state_path(path, runtime)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"manifest": str(path.resolve()), "cohort_id": cohort_id, "created_at": now(), "jobs": {}}
    state["manifest"] = str(path.resolve())
    state["manifest_fingerprint"] = hashlib.sha256(path.read_bytes()).hexdigest()

    for trial_id in trial_ids:
        record = state["jobs"].get(trial_id, {})
        job_id = record.get("job_id")
        job = service.get(job_id) if job_id and (service.jobs / f"{job_id}.json").is_file() else None
        if force_restart and job:
            print(f"{trial_id}: starting fresh; prior job retained as {job_id}", flush=True)
            job = None
        if job and job.get("status") == "completed":
            print(f"{trial_id}: already completed ({job_id})", flush=True)
            continue
        if job:
            print(f"{trial_id}: resuming {job_id}", flush=True)
            job = service.resume(job_id)
        else:
            print(f"{trial_id}: starting", flush=True)
            job = service.submit(input_mode="nct", trial_id=trial_id, protocol_text="")
            state["jobs"][trial_id] = {"job_id": job["job_id"], "attempts": 0}
            atomic_write_json(state_path, state)

        last_reported: tuple[Any, ...] | None = None
        while True:
            time.sleep(poll_seconds)
            job = service.get(job["job_id"])
            record = state["jobs"][trial_id]
            record.update({"status": job["status"], "stage": job["stage"], "updated_at": now()})
            atomic_write_json(state_path, state)
            report = (job.get("status"), job.get("stage"), job.get("progress"), job.get("message"))
            if report != last_reported:
                print(f"{trial_id}: {job['progress']:3d}% {job['message']}", flush=True)
                last_reported = report
            if job["status"] == "completed":
                break
            if job["status"] == "failed":
                record["attempts"] = int(record.get("attempts", 0)) + 1
                atomic_write_json(state_path, state)
                if "No eligibility criteria could be segmented" in str(job.get("error") or job.get("message") or ""):
                    print(
                        f"{trial_id}: deterministic segmentation failure; not retrying unchanged input",
                        flush=True,
                    )
                    break
                if record["attempts"] > retry_limit:
                    print(f"{trial_id}: failed after resumable retries; continuing to next trial", flush=True)
                    break
                job = service.resume(job["job_id"])

    state["completed_at"] = now()
    atomic_write_json(state_path, state)
    return {"state_file": str(state_path), **state}
