"""Recoverable sequential extraction for a manifest of ClinicalTrials.gov IDs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.api.app_state import create_app_state
from backend.atomic_io import atomic_write_json
from backend.services.eav_workflow import EavWorkflowService, now


def load_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("trials") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Manifest must be an array or contain a trials array.")
    trial_ids = []
    for item in items:
        trial_id = item if isinstance(item, str) else item.get("trial_id") if isinstance(item, dict) else None
        if not trial_id:
            raise ValueError("Every manifest item requires trial_id.")
        trial_ids.append(str(trial_id).strip().upper())
    return list(dict.fromkeys(trial_ids))


def run_manifest(path: Path, *, poll_seconds: float = 1.0, retry_limit: int = 2) -> dict[str, Any]:
    """Run or resume a manifest; job checkpoints survive process interruption."""
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime" / "criteria"
    service = EavWorkflowService(
        create_app_state(), runtime, root / "config" / "base_criterion_library.json"
    )
    trial_ids = load_manifest(path)
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    state_path = runtime / "manifests" / f"manifest-{manifest_hash}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"manifest": str(path.resolve()), "created_at": now(), "jobs": {}}

    for trial_id in trial_ids:
        record = state["jobs"].get(trial_id, {})
        job_id = record.get("job_id")
        job = service.get(job_id) if job_id and (service.jobs / f"{job_id}.json").is_file() else None
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

        while True:
            time.sleep(poll_seconds)
            job = service.get(job["job_id"])
            record = state["jobs"][trial_id]
            record.update({"status": job["status"], "stage": job["stage"], "updated_at": now()})
            atomic_write_json(state_path, state)
            print(f"{trial_id}: {job['progress']:3d}% {job['message']}", flush=True)
            if job["status"] == "completed":
                break
            if job["status"] == "failed":
                record["attempts"] = int(record.get("attempts", 0)) + 1
                atomic_write_json(state_path, state)
                if record["attempts"] > retry_limit:
                    print(f"{trial_id}: failed after resumable retries; continuing to next trial", flush=True)
                    break
                job = service.resume(job["job_id"])

    state["completed_at"] = now()
    atomic_write_json(state_path, state)
    return {"state_file": str(state_path), **state}
