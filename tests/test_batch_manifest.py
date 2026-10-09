import json
from pathlib import Path

from backend.batch_manifest import load_manifest


def test_load_manifest_skips_explicit_placeholders(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "trials": [
                    {"trial_id": "nct00000001"},
                    {"trial_id": "", "enabled": False, "status": "TBD"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_manifest(manifest) == ["NCT00000001"]


def test_packaged_cohort_has_fifty_trials_per_therapeutic_area() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "trial_cohort_100.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots = payload["trials"]
    active = load_manifest(path)

    assert payload["cohort_id"] == "metricspace-100-trial-cohort"
    assert len(slots) == 100
    assert len(active) == 100
    assert len(set(active)) == 100
    assert all(item["cohort"] == "oncology" for item in slots[:50])
    assert all(item["cohort"] == "cardiovascular" for item in slots[50:])
