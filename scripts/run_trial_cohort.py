#!/usr/bin/env python3
"""Run or resume the full or calibration clinical-trial extraction cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    LOCAL_PYTHON.is_file()
    and Path(sys.executable).resolve() != LOCAL_PYTHON.resolve()
    and os.environ.get("MODERN_AUTOCRIT_COHORT_BOOTSTRAPPED") != "1"
):
    environment = dict(os.environ)
    environment["MODERN_AUTOCRIT_COHORT_BOOTSTRAPPED"] = "1"
    os.execve(str(LOCAL_PYTHON), [str(LOCAL_PYTHON), *sys.argv], environment)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.batch_manifest import load_manifest, run_manifest

DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "trial_cohort_100.json"
CALIBRATION_MANIFEST = PROJECT_ROOT / "config" / "trial_cohort_calibration_10.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the full cohort or its 10-trial calibration subset."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="Use a custom cohort manifest.")
    source.add_argument(
        "--calibration", action="store_true",
        help="Run the balanced 10-trial calibration subset instead of all 100 trials.",
    )
    parser.add_argument("--yes", action="store_true", help="Start without an interactive confirmation.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Validate and summarize without extracting.")
    action.add_argument(
        "--export-review", action="store_true",
        help="Export completed jobs to a root-level human-review XLSX without extracting.",
    )
    action.add_argument(
        "--apply-review", type=Path, metavar="XLSX",
        help="Validate and apply a completed human-review workbook without extracting.",
    )
    action.add_argument(
        "--renormalize", action="store_true",
        help="Rebuild completed rows from saved model checkpoints and export a new workbook without API calls.",
    )
    parser.add_argument(
        "--review-output", type=Path,
        help="Optional path for --export-review; automatic exports use a timestamped root filename.",
    )
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start new jobs even when this manifest already has completed jobs; prior job files are retained.",
    )
    args = parser.parse_args()

    manifest = (args.manifest or (CALIBRATION_MANIFEST if args.calibration else DEFAULT_MANIFEST)).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    slots = payload.get("trials") if isinstance(payload, dict) else payload
    if not isinstance(slots, list):
        parser.error("Manifest must contain a trials array.")
    configured_ids = [
        str(item.get("trial_id")).strip().upper()
        for item in slots
        if isinstance(item, dict) and item.get("trial_id")
    ]
    if len(configured_ids) != len(set(configured_ids)):
        parser.error("Duplicate NCT IDs were found.")
    trial_ids = load_manifest(manifest)
    blank = len(slots) - len(trial_ids)
    expected_count = payload.get("expected_trial_count") if isinstance(payload, dict) else None
    print(f"Manifest: {manifest}")
    print(f"Configured slots: {len(slots)}")
    print(f"Active trial IDs: {len(trial_ids)}")
    print(f"Unfilled/disabled slots: {blank}")
    if expected_count is not None and len(slots) != expected_count:
        parser.error(f"Expected exactly {expected_count} entries, found {len(slots)}.")
    if args.dry_run:
        print("Dry run complete; no API or model calls were made.")
        return
    if args.fresh and (args.export_review or args.apply_review or args.renormalize):
        parser.error("--fresh is only valid when running extraction.")
    if args.renormalize:
        from backend.batch_manifest import renormalize_manifest
        from backend.review_workbook import export_review_workbook
        normalized = renormalize_manifest(manifest)
        print(
            f"Re-normalized {normalized['rows']} rows from "
            f"{normalized['completed_jobs']} completed trials without API calls."
        )
        if normalized["skipped_trials"]:
            print("Skipped: " + ", ".join(normalized["skipped_trials"]))
        exported = export_review_workbook(manifest, PROJECT_ROOT)
        print(f"Review workbook: {exported['path']} ({exported['rows']} rows)")
        return
    if args.export_review:
        from backend.review_workbook import export_review_workbook
        exported = export_review_workbook(
            manifest, PROJECT_ROOT,
            args.review_output.resolve() if args.review_output else None,
        )
        print(
            f"Review workbook: {exported['path']} "
            f"({exported['completed_jobs']} completed trials, {exported['rows']} rows)"
        )
        return
    if args.apply_review:
        from backend.review_workbook import apply_review_workbook
        applied = apply_review_workbook(manifest, args.apply_review.resolve(), PROJECT_ROOT)
        print(f"Applied {applied['applied_rows']} rows from {applied['applied_jobs']} trials.")
        if applied["incomplete_trials"]:
            print("Not applied because review is incomplete: " + ", ".join(applied["incomplete_trials"]))
        return
    if not args.yes:
        answer = input(
            "This will make ClinicalTrials.gov and model API calls for every unfinished active trial. Continue? [y/N] "
        ).strip().casefold()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return
    result = run_manifest(manifest, retry_limit=args.retry_limit, force_restart=args.fresh)
    print(f"Cohort run finished. Recoverable state: {result['state_file']}")
    from backend.review_workbook import export_review_workbook
    exported = export_review_workbook(manifest, PROJECT_ROOT)
    print(
        f"Human-review workbook: {exported['path']} "
        f"({exported['completed_jobs']} completed trials, {exported['rows']} rows)"
    )


if __name__ == "__main__":
    main()
