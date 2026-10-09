#!/usr/bin/env python3
"""Run or resume the configured 100-slot clinical-trial extraction cohort."""

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume every populated trial in the 100-slot cohort manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--yes", action="store_true", help="Start without an interactive confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without extracting.")
    parser.add_argument("--retry-limit", type=int, default=2)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
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
    print(f"Manifest: {manifest}")
    print(f"Configured slots: {len(slots)}")
    print(f"Active trial IDs: {len(trial_ids)}")
    print(f"Unfilled/disabled slots: {blank}")
    if len(slots) != 100:
        parser.error(f"Expected exactly 100 slots, found {len(slots)}.")
    if args.dry_run:
        print("Dry run complete; no API or model calls were made.")
        return
    if not args.yes:
        answer = input(
            "This will make ClinicalTrials.gov and model API calls for every unfinished active trial. Continue? [y/N] "
        ).strip().casefold()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return
    result = run_manifest(manifest, retry_limit=args.retry_limit)
    print(f"Cohort run finished. Recoverable state: {result['state_file']}")


if __name__ == "__main__":
    main()
