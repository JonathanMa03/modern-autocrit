from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.schemas.settings_schema import LLMSettings
from backend.schemas.validation_schema import (
    ClauseValidation,
    ExtractionValidation,
    ValidationMetrics,
)
from backend.services.clinicaltrials_service import ClinicalTrialsService
from backend.services.cost_monitor import CostMonitor
from backend.services.file_manager import FileManager
from backend.services.llm.base import BaseLLMProvider
from backend.services.llm.factory import create_llm_provider
from backend.validation_agent.prompts import (
    build_chat_prompt,
    build_validation_prompt,
)


@dataclass
class TrialValidationResult:
    trial_id: str
    metrics: ValidationMetrics
    clauses: list[ClauseValidation]
    extractions: list[ExtractionValidation]
    summary: str = ""
    quality_notes: list[str] = field(default_factory=list)

    def context_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "metrics": self.metrics.model_dump(),
            "summary": self.summary,
            "quality_notes": self.quality_notes,
            "missed_or_partial_criteria": [
                clause.model_dump()
                for clause in self.clauses
                if clause.coverage_status != "covered"
            ],
            "weak_or_unsupported_extractions": [
                extraction.model_dump()
                for extraction in self.extractions
                if extraction.support_status != "supported"
            ],
        }


@dataclass
class ValidationRunResult:
    trials: list[TrialValidationResult]
    report_file: Path | None = None
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    def report_context(self) -> str:
        return json.dumps(
            [trial.context_dict() for trial in self.trials],
            ensure_ascii=False,
            indent=2,
        )


class IntegrityValidationAgent:
    """Independently audits extracted criteria against source trial XML."""

    def __init__(
        self,
        llm_settings: LLMSettings,
        cost_monitor: CostMonitor | None = None,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self.llm_settings = llm_settings
        self.cost_monitor = cost_monitor or CostMonitor()
        self.provider = provider or create_llm_provider(
            settings=llm_settings,
            cost_monitor=self.cost_monitor,
        )

    def validate(
        self,
        xml_directory: str | Path,
        extraction_file: str | Path,
        trial_ids: list[str] | None = None,
        report_file: str | Path | None = None,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ValidationRunResult:
        service = ClinicalTrialsService(Path(xml_directory))
        dataframe = FileManager.load_excel(Path(extraction_file))
        if "trial_id" not in dataframe.columns:
            raise ValueError("Extraction workbook must contain a trial_id column.")

        requested = {item.strip().upper() for item in trial_ids or [] if item.strip()}
        results: list[TrialValidationResult] = []
        errors: list[str] = []
        found: set[str] = set()
        candidates: list[tuple[dict[str, Any], Path]] = []

        for xml_path in service.xml_files():
            trial = service.load_trial(xml_path)
            trial_id = str(trial.get("trial_id", "")).upper()
            if requested and trial_id not in requested:
                continue
            found.add(trial_id)
            candidates.append((trial, xml_path))

        total = len(candidates)
        if progress_callback:
            progress_callback(0, total, "")

        cancelled = False
        for completed, (trial, _xml_path) in enumerate(candidates):
            if stop_event and stop_event.is_set():
                cancelled = True
                break
            trial_id = str(trial.get("trial_id", "")).upper()
            if progress_callback:
                progress_callback(completed, total, trial_id)

            rows = dataframe[
                dataframe["trial_id"].astype(str).str.upper() == trial_id
            ].copy()
            rows = rows.reset_index().rename(columns={"index": "workbook_row_index"})

            try:
                results.append(
                    self.validate_trial(
                        trial_id=trial_id,
                        eligibility_text=trial.get("eligibility_text", ""),
                        extracted_rows=rows,
                    )
                )
            except Exception as exc:
                errors.append(f"{trial_id}: {exc}")
            if progress_callback:
                progress_callback(completed + 1, total, trial_id)

        for missing_id in sorted(requested - found):
            errors.append(f"{missing_id}: no matching XML file was found.")

        run = ValidationRunResult(
            trials=results,
            errors=errors,
            cancelled=cancelled,
        )
        if report_file:
            run.report_file = self.save_report(run, Path(report_file))
        return run

    def validate_trial(
        self,
        trial_id: str,
        eligibility_text: str,
        extracted_rows: pd.DataFrame,
    ) -> TrialValidationResult:
        if not eligibility_text.strip():
            raise ValueError("source XML has no eligibility text")

        prompt_rows = self._rows_for_prompt(extracted_rows)
        response = self.provider.generate(
            build_validation_prompt(trial_id, eligibility_text, prompt_rows),
            model=self.llm_settings.model,
            temperature=0.0,
            max_output_tokens=self.llm_settings.max_output_tokens,
        )
        payload = self._parse_json_object(response.text)

        clauses = [
            ClauseValidation(
                trial_id=trial_id,
                criteria_type=str(item.get("criteria_type", "")),
                source_clause=str(item.get("source_clause", "")),
                coverage_status=item.get("coverage_status", "uncovered"),
                matched_row_indices=[
                    int(prompt_rows[index].get("workbook_row_index", index))
                    for index in self._valid_indices(
                        item.get("matched_row_indices", []), len(prompt_rows)
                    )
                ],
                missing_concepts=[str(value) for value in item.get("missing_concepts", [])],
                explanation=str(item.get("explanation", "")),
                confidence=item.get("confidence", 0.0),
            )
            for item in payload.get("source_criteria", [])
            if isinstance(item, dict) and item.get("source_clause")
        ]

        extraction_by_index = {
            int(item["row_index"]): item
            for item in payload.get("extracted_rows", [])
            if isinstance(item, dict)
            and self._is_valid_index(item.get("row_index"), len(prompt_rows))
        }
        extractions = []
        for index in range(len(prompt_rows)):
            item = extraction_by_index.get(index)
            if item is None:
                item = {
                    "support_status": "weakly_supported",
                    "explanation": "The validator did not return an assessment for this row.",
                    "confidence": 0.0,
                }
            extractions.append(
                ExtractionValidation(
                    trial_id=trial_id,
                    row_index=int(
                        prompt_rows[index].get("workbook_row_index", index)
                    ),
                    support_status=item.get("support_status", "weakly_supported"),
                    source_clause=item.get("source_clause"),
                    explanation=str(item.get("explanation", "")),
                    confidence=item.get("confidence", 0.0),
                )
            )

        metrics = self._calculate_metrics(
            trial_id,
            clauses,
            extractions,
            prompt_rows,
        )
        return TrialValidationResult(
            trial_id=trial_id,
            metrics=metrics,
            clauses=clauses,
            extractions=extractions,
            summary=str(payload.get("summary", "")),
            quality_notes=[str(note) for note in payload.get("quality_notes", [])],
        )

    def chat(self, run: ValidationRunResult, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        response = self.provider.generate(
            build_chat_prompt(run.report_context(), question),
            model=self.llm_settings.model,
            temperature=0.0,
            max_output_tokens=min(self.llm_settings.max_output_tokens, 2048),
        )
        return response.text.strip()

    @staticmethod
    def _rows_for_prompt(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        useful_columns = [
            "criteria_type",
            "entity",
            "attribute",
            "value",
            "temporal",
            "modifier",
            "source_sentence",
            "workbook_row_index",
        ]
        rows = []
        for row_index, (_, row) in enumerate(dataframe.iterrows()):
            record: dict[str, Any] = {"row_index": row_index}
            for column in useful_columns:
                if column in dataframe.columns and not pd.isna(row[column]):
                    record[column] = str(row[column])
            rows.append(record)
        return rows

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        fenced = re.findall(
            r"```(?:json)?\s*(.*?)```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            stripped = fenced[0].strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            start, end = stripped.find("{"), stripped.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Validator response did not contain a JSON object.")
            payload = json.loads(stripped[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Validator response must be a JSON object.")
        return payload

    @staticmethod
    def _is_valid_index(value: Any, row_count: int) -> bool:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= index < row_count

    @classmethod
    def _valid_indices(cls, values: Any, row_count: int) -> list[int]:
        if not isinstance(values, list):
            return []
        return sorted(
            {int(value) for value in values if cls._is_valid_index(value, row_count)}
        )

    @staticmethod
    def _calculate_metrics(
        trial_id: str,
        clauses: list[ClauseValidation],
        extractions: list[ExtractionValidation],
        rows: list[dict[str, Any]],
    ) -> ValidationMetrics:
        total_clauses = len(clauses)
        covered = sum(item.coverage_status == "covered" for item in clauses)
        partial = sum(item.coverage_status == "partially_covered" for item in clauses)
        uncovered = sum(item.coverage_status == "uncovered" for item in clauses)
        total_rows = len(extractions)
        supported = sum(item.support_status == "supported" for item in extractions)
        weak = sum(item.support_status == "weakly_supported" for item in extractions)
        unsupported = sum(item.support_status == "unsupported" for item in extractions)
        missing_values = sum(not row.get("value") for row in rows)
        recall = (covered + 0.5 * partial) / total_clauses if total_clauses else 0.0
        precision = (supported + 0.5 * weak) / total_rows if total_rows else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        matched_counts: dict[int, int] = {}
        for clause in clauses:
            for index in clause.matched_row_indices:
                matched_counts[index] = matched_counts.get(index, 0) + 1
        atomicity_issues = sum(count > 1 for count in matched_counts.values())

        return ValidationMetrics(
            trial_id=trial_id,
            total_source_clauses=total_clauses,
            covered_clauses=covered,
            partially_covered_clauses=partial,
            uncovered_clauses=uncovered,
            total_extracted_rows=total_rows,
            supported_rows=supported,
            weakly_supported_rows=weak,
            unsupported_rows=unsupported,
            clause_coverage_rate=recall,
            full_coverage_rate=covered / total_clauses if total_clauses else 0.0,
            unsupported_extraction_rate=unsupported / total_rows if total_rows else 0.0,
            estimated_precision=precision,
            estimated_recall=recall,
            estimated_f1=f1,
            missing_value_rate=missing_values / len(rows) if rows else 0.0,
            atomicity_issue_rate=atomicity_issues / total_rows if total_rows else 0.0,
        )

    @staticmethod
    def save_report(run: ValidationRunResult, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        clause_rows = []
        extraction_rows = []
        note_rows = []
        for trial in run.trials:
            summary = trial.metrics.model_dump()
            summary["summary"] = trial.summary
            summary_rows.append(summary)
            clause_rows.extend(item.model_dump() for item in trial.clauses)
            extraction_rows.extend(item.model_dump() for item in trial.extractions)
            note_rows.extend(
                {"trial_id": trial.trial_id, "quality_note": note}
                for note in trial.quality_notes
            )

        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(
                [
                    {
                        "status": "cancelled" if run.cancelled else "complete",
                        "trials_validated": run.trial_count,
                        "error_count": len(run.errors),
                    }
                ]
            ).to_excel(writer, sheet_name="Run Status", index=False)
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(clause_rows).to_excel(writer, sheet_name="Source Coverage", index=False)
            pd.DataFrame(extraction_rows).to_excel(writer, sheet_name="Extraction Support", index=False)
            pd.DataFrame(note_rows).to_excel(writer, sheet_name="Quality Notes", index=False)
            pd.DataFrame({"error": run.errors}).to_excel(writer, sheet_name="Errors", index=False)
        return path
