import json
import threading

import pandas as pd

from backend.schemas.settings_schema import LLMSettings
from backend.services.llm.base import BaseLLMProvider, LLMResponse
from backend.validation_agent import IntegrityValidationAgent


class FakeProvider(BaseLLMProvider):
    provider_name = "fake"

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    @property
    def default_model(self):
        return "fake-model"

    def generate(self, prompt, *, model, temperature=0.0, max_output_tokens=4096):
        self.prompts.append(prompt)
        return LLMResponse(
            text=json.dumps(self.payload),
            provider="fake",
            model=model,
        )


def _write_trial(path):
    path.write_text(
        """<?xml version="1.0"?>
<clinical_study>
  <nct_id>NCT12345678</nct_id>
  <brief_title>Integrity Test</brief_title>
  <eligibility><criteria><textblock>
Inclusion Criteria:
Age 18 years or older.
Platelet count at least 100000/mm3.
Exclusion Criteria:
Active infection.
  </textblock></criteria></eligibility>
</clinical_study>
""",
        encoding="utf-8",
    )


def test_validation_computes_metrics_and_writes_report(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _write_trial(xml_dir / "NCT12345678.xml")
    extraction_file = tmp_path / "extractions.xlsx"
    pd.DataFrame(
        [
            {
                "trial_id": "NCT12345678",
                "criteria_type": "Inclusion",
                "entity": "Demographic",
                "attribute": "Age",
                "value": ">= 18 years",
                "source_sentence": "Age 18 years or older.",
            },
            {
                "trial_id": "NCT12345678",
                "criteria_type": "Exclusion",
                "entity": "Comorbidity",
                "attribute": "Diabetes",
                "value": "Yes",
                "source_sentence": "Patients with diabetes are excluded.",
            },
        ]
    ).to_excel(extraction_file, index=False)

    provider = FakeProvider(
        {
            "source_criteria": [
                {
                    "criteria_type": "Inclusion",
                    "source_clause": "Age 18 years or older.",
                    "coverage_status": "covered",
                    "matched_row_indices": [0],
                    "missing_concepts": [],
                    "explanation": "Age is represented.",
                    "confidence": 0.99,
                },
                {
                    "criteria_type": "Inclusion",
                    "source_clause": "Platelet count at least 100000/mm3.",
                    "coverage_status": "uncovered",
                    "matched_row_indices": [],
                    "missing_concepts": ["platelet count"],
                    "explanation": "No matching row.",
                    "confidence": 0.98,
                },
                {
                    "criteria_type": "Exclusion",
                    "source_clause": "Active infection.",
                    "coverage_status": "uncovered",
                    "matched_row_indices": [],
                    "missing_concepts": ["active infection"],
                    "explanation": "No matching row.",
                    "confidence": 0.97,
                },
            ],
            "extracted_rows": [
                {
                    "row_index": 0,
                    "support_status": "supported",
                    "source_clause": "Age 18 years or older.",
                    "explanation": "Direct support.",
                    "confidence": 0.99,
                },
                {
                    "row_index": 1,
                    "support_status": "unsupported",
                    "source_clause": None,
                    "explanation": "Diabetes is absent from the source.",
                    "confidence": 0.99,
                },
            ],
            "summary": "Two criteria were missed and one row was fabricated.",
            "quality_notes": ["Review the exclusion extraction."],
        }
    )
    agent = IntegrityValidationAgent(
        LLMSettings(api_key="unused", model="fake-model"),
        provider=provider,
    )
    report_file = tmp_path / "integrity.xlsx"

    result = agent.validate(xml_dir, extraction_file, report_file=report_file)

    assert result.trial_count == 1
    metrics = result.trials[0].metrics
    assert metrics.total_source_clauses == 3
    assert metrics.uncovered_clauses == 2
    assert metrics.unsupported_rows == 1
    assert metrics.estimated_precision == 0.5
    assert metrics.estimated_recall == 1 / 3
    assert report_file.exists()
    assert set(pd.ExcelFile(report_file).sheet_names) == {
        "Run Status",
        "Summary",
        "Source Coverage",
        "Extraction Support",
        "Quality Notes",
        "Errors",
    }


def test_missing_requested_trial_is_reported(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    extraction_file = tmp_path / "extractions.xlsx"
    pd.DataFrame(columns=["trial_id"]).to_excel(extraction_file, index=False)
    agent = IntegrityValidationAgent(
        LLMSettings(api_key="unused"),
        provider=FakeProvider({}),
    )

    result = agent.validate(xml_dir, extraction_file, trial_ids=["NCT00000000"])

    assert result.trials == []
    assert "no matching XML" in result.errors[0]


def test_unreturned_extraction_gets_a_conservative_assessment():
    provider = FakeProvider(
        {
            "source_criteria": [],
            "extracted_rows": [],
            "summary": "",
            "quality_notes": [],
        }
    )
    agent = IntegrityValidationAgent(
        LLMSettings(api_key="unused"),
        provider=provider,
    )

    result = agent.validate_trial(
        "NCT12345678",
        "Inclusion Criteria: Adults.",
        pd.DataFrame([{"entity": "Demographic", "attribute": "Age"}]),
    )

    assert result.extractions[0].support_status == "weakly_supported"
    assert "did not return" in result.extractions[0].explanation


def test_validation_can_be_stopped_before_next_trial(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _write_trial(xml_dir / "NCT12345678.xml")
    extraction_file = tmp_path / "extractions.xlsx"
    pd.DataFrame(columns=["trial_id"]).to_excel(extraction_file, index=False)
    stop_event = threading.Event()
    stop_event.set()
    progress = []
    agent = IntegrityValidationAgent(
        LLMSettings(api_key="unused"),
        provider=FakeProvider({}),
    )

    result = agent.validate(
        xml_dir,
        extraction_file,
        stop_event=stop_event,
        progress_callback=lambda completed, total, trial_id: progress.append(
            (completed, total, trial_id)
        ),
    )

    assert result.cancelled is True
    assert result.trials == []
    assert progress == [(0, 1, "")]
