import csv
import json
from pathlib import Path

import pytest

from backend.criteria_processor.breakdown_cli import (
    CSV_COLUMNS,
    EligibilityTrial,
    build_breakdown_prompt,
    load_eligibility_trial,
    process_trial,
    process_trials_parallel,
    read_breakdown_csv,
    validate_breakdown_result,
)


class FakeBreakdownProvider:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.closed = False

    def generate_json(self, prompt, *, output_schema):
        self.calls += 1
        assert "SOURCE_ROWS_JSON:" in prompt
        assert output_schema["required"] == ["points"]
        return self.result

    def close(self):
        self.closed = True


def _write_trial(path: Path, eligibility_text: str, trial_id="NCT-TEST"):
    path.write_text(
        json.dumps(
            {
                "protocolSection": {
                    "identificationModule": {"nctId": trial_id},
                    "eligibilityModule": {
                        "eligibilityCriteria": eligibility_text
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _trial(tmp_path: Path) -> EligibilityTrial:
    source = tmp_path / "NCT-TEST.json"
    _write_trial(
        source,
        """Inclusion Criteria:

* Age 18 to 80 years and able to give informed consent

Exclusion Criteria:

* renal or hepatic impairment
""",
    )
    return load_eligibility_trial(source)


def test_load_trial_preserves_headed_source_rows(tmp_path):
    trial = _trial(tmp_path)

    assert trial.trial_key == "NCT-TEST"
    assert trial.trial_id == "NCT-TEST"
    assert [row["criteria"] for row in trial.source_rows] == [
        "inclusion",
        "exclusion",
    ]
    assert trial.source_rows[0]["context"] == (
        "Age 18 to 80 years and able to give informed consent"
    )
    assert trial.source_rows[1]["context"] == (
        "renal or hepatic impairment"
    )


def test_prompt_is_profile_blind_and_requests_atomic_points(tmp_path):
    prompt = build_breakdown_prompt(_trial(tmp_path))

    assert "atomic screening" in prompt
    assert "Entity-\nAttribute-Value row" in prompt
    assert "Do not normalize terminology" in prompt
    assert "Return only conditions that determine" in prompt
    assert "Never emit an exception or exemption" in prompt
    assert '"Men and women" and' in prompt
    assert "serum creatinine > 2 mg/dL" in prompt
    assert '"source_id": 1' in prompt
    assert "target profile" not in prompt.casefold()


def test_validation_attaches_authoritative_criteria_and_context(tmp_path):
    trial = _trial(tmp_path)
    result = {
        "points": [
            {"source_id": 1, "item": "Age 18 to 80 years"},
            {"source_id": 1, "item": "able to give informed consent"},
            {"source_id": 2, "item": "\u2022 renal impairment"},
            {"source_id": 2, "item": "hepatic impairment"},
        ]
    }

    rows = validate_breakdown_result(result, trial.source_rows)

    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[0] == {
        "criteria": "inclusion",
        "item": "Age 18 to 80 years",
        "context": (
            "Age 18 to 80 years and able to give informed consent"
        ),
    }
    assert rows[2]["criteria"] == "exclusion"
    assert rows[2]["item"] == "renal impairment"
    assert rows[2]["context"] == "renal or hepatic impairment"


def test_validation_preserves_finer_attribute_like_breakdowns(tmp_path):
    source = tmp_path / "NCT-ATTRIBUTES.json"
    _write_trial(
        source,
        """Inclusion Criteria:
* Men and women aged from 18 to 85

Exclusion Criteria:
* Permanent renal failure (Creatinin >180 micromol/l)
* Hepatic failure (TGO and TGP >2N)
* Aphasia preventing evaluation of motor and depression scales
""",
        trial_id="NCT-ATTRIBUTES",
    )
    trial = load_eligibility_trial(source)

    rows = validate_breakdown_result(
        {
            "points": [
                {"source_id": 1, "item": "Men and women"},
                {"source_id": 1, "item": "aged from 18 to 85"},
                {"source_id": 2, "item": "Permanent renal failure"},
                {
                    "source_id": 2,
                    "item": "Creatinin >180 micromol/l",
                },
                {"source_id": 3, "item": "Hepatic failure"},
                {"source_id": 3, "item": "TGO >2N"},
                {"source_id": 3, "item": "TGP >2N"},
                {
                    "source_id": 4,
                    "item": (
                        "Aphasia preventing evaluation of motor scales"
                    ),
                },
                {
                    "source_id": 4,
                    "item": (
                        "Aphasia preventing evaluation of depression scales"
                    ),
                },
            ]
        },
        trial.source_rows,
    )

    assert [row["item"] for row in rows] == [
        "Men and women",
        "aged from 18 to 85",
        "Permanent renal failure",
        "Creatinin >180 micromol/l",
        "Hepatic failure",
        "TGO >2N",
        "TGP >2N",
        "Aphasia preventing evaluation of motor scales",
        "Aphasia preventing evaluation of depression scales",
    ]
    assert {row["context"] for row in rows[:2]} == {
        "Men and women aged from 18 to 85"
    }
    assert {row["context"] for row in rows[2:4]} == {
        "Permanent renal failure (Creatinin >180 micromol/l)"
    }
    assert {row["context"] for row in rows[4:7]} == {
        "Hepatic failure (TGO and TGP >2N)"
    }
    assert {row["context"] for row in rows[7:]} == {
        "Aphasia preventing evaluation of motor and depression scales"
    }


def test_validation_rejects_missing_source_coverage(tmp_path):
    trial = _trial(tmp_path)

    with pytest.raises(ValueError, match="omitted source_id values: 2"):
        validate_breakdown_result(
            {
                "points": [
                    {"source_id": 1, "item": "Age 18 to 80 years"}
                ]
            },
            trial.source_rows,
        )


def test_process_trial_writes_exact_csv_and_reuses_output(tmp_path):
    trial = _trial(tmp_path)
    provider = FakeBreakdownProvider(
        {
            "points": [
                {"source_id": 1, "item": "Age 18 to 80 years"},
                {
                    "source_id": 1,
                    "item": "able to give informed consent",
                },
                {"source_id": 2, "item": "renal impairment"},
                {"source_id": 2, "item": "hepatic impairment"},
            ]
        }
    )
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"

    first = process_trial(
        trial=trial,
        provider=provider,
        output_directory=output_root,
        checkpoint_directory=checkpoint_root,
    )
    second = process_trial(
        trial=trial,
        provider=provider,
        output_directory=output_root,
        checkpoint_directory=checkpoint_root,
    )

    assert first.status == "generated"
    assert second.status == "reused_output"
    assert provider.calls == 1
    assert first.output_path == (
        output_root / "NCT-TEST" / "elig_breakdown_points.csv"
    ).resolve()
    with first.output_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        assert len(list(reader)) == 4
    assert len(read_breakdown_csv(first.output_path)) == 4


def test_parallel_processing_keeps_trials_independent(tmp_path):
    first_source = tmp_path / "TRIAL-A.json"
    second_source = tmp_path / "TRIAL-B.json"
    eligibility = """Inclusion Criteria:
* Adults
Exclusion Criteria:
* Pregnancy
"""
    _write_trial(first_source, eligibility, trial_id="TRIAL-A")
    _write_trial(second_source, eligibility, trial_id="TRIAL-B")
    trials = [
        load_eligibility_trial(first_source),
        load_eligibility_trial(second_source),
    ]
    providers = {}

    def provider_factory(trial):
        provider = FakeBreakdownProvider(
            {
                "points": [
                    {"source_id": 1, "item": "Adults"},
                    {"source_id": 2, "item": "Pregnancy"},
                ]
            }
        )
        providers[trial.trial_key] = provider
        return provider

    results, errors = process_trials_parallel(
        trials=trials,
        provider_factory=provider_factory,
        output_directory=tmp_path / "output",
        checkpoint_directory=tmp_path / "checkpoints",
        workers=2,
    )

    assert not errors
    assert [result.trial_key for result in results] == [
        "TRIAL-A",
        "TRIAL-B",
    ]
    assert all(provider.calls == 1 for provider in providers.values())
    assert all(provider.closed for provider in providers.values())
