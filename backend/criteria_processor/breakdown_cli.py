"""Build per-trial atomic eligibility-point CSV files with the Codex bridge."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.criteria_processor.llm_providers import (
    CodexJsonLLMProvider,
    RuntimeRecordingJsonLLMProvider,
)
from backend.criteria_processor.segmentation import iter_eligibility_source_items
from backend.criteria_processor.atomic_io import atomic_write_json, atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "selected_stroke" / "raw_ctg"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "selected_stroke"
    / "elig_points_breakdown"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT / "tmp" / "eligibility_breakdown" / "checkpoints"
)
DEFAULT_MODEL = "gpt-5.5"
CSV_COLUMNS = ("criteria", "item", "context")
ELIGIBILITY_SOURCE_PATH = (
    "protocolSection.eligibilityModule.eligibilityCriteria"
)


class JsonLLMProvider(Protocol):
    """Minimal provider boundary used by the eligibility breakdown workflow."""

    def generate_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class EligibilityTrial:
    """One source record and its deterministically segmented eligibility rows."""

    trial_key: str
    trial_id: str
    source_path: Path
    source_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TrialBreakdownResult:
    """Completion metadata for one trial output."""

    trial_key: str
    trial_id: str
    output_path: Path
    point_count: int
    status: str


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _nested_text(payload: Mapping[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return _text(current)


def load_eligibility_trial(source_path: str | Path) -> EligibilityTrial:
    """Load one raw study record and preserve each headed source row."""

    path = Path(source_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name}: raw trial record must be an object.")

    trial_id = _nested_text(
        payload,
        "protocolSection",
        "identificationModule",
        "nctId",
    )
    trial_id = trial_id or path.stem
    eligibility_text = _nested_text(
        payload,
        "protocolSection",
        "eligibilityModule",
        "eligibilityCriteria",
    )
    if not eligibility_text:
        raise ValueError(
            f"{path.name}: missing {ELIGIBILITY_SOURCE_PATH}."
        )

    source_items = iter_eligibility_source_items(
        eligibility_text,
        base_path=ELIGIBILITY_SOURCE_PATH,
    )
    if not source_items:
        raise ValueError(
            f"{path.name}: no headed inclusion or exclusion rows were found."
        )

    source_rows = tuple(
        {
            "source_id": position,
            "criteria": item.criterion_type.casefold(),
            "context": item.source_text,
            "source_path": item.source_path,
        }
        for position, item in enumerate(source_items, start=1)
    )
    return EligibilityTrial(
        trial_key=path.stem,
        trial_id=trial_id,
        source_path=path.resolve(),
        source_rows=source_rows,
    )


def load_eligibility_trials(
    input_directory: str | Path,
) -> list[EligibilityTrial]:
    """Load all JSON trial records in stable filename order."""

    directory = Path(input_directory)
    source_paths = sorted(directory.glob("*.json"))
    if not source_paths:
        raise ValueError(f"No JSON trial records found in {directory}.")
    return [load_eligibility_trial(path) for path in source_paths]


def breakdown_output_schema() -> dict[str, Any]:
    """Return the strict JSON schema for one trial-level LLM response."""

    return {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "item": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "required": ["source_id", "item"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["points"],
        "additionalProperties": False,
    }


def build_breakdown_prompt(
    trial: EligibilityTrial,
    *,
    prior_error: str = "",
) -> str:
    """Build a profile-blind prompt for atomic eligibility decomposition."""

    input_rows = [
        {
            "source_id": row["source_id"],
            "criteria": row["criteria"],
            "context": row["context"],
        }
        for row in trial.source_rows
    ]
    error_block = (
        "\nA prior response was rejected for the following reason. "
        "Correct the response completely:\n"
        f"{prior_error}\n"
        if prior_error
        else ""
    )
    return f"""Break raw clinical-trial eligibility rows into atomic screening
points. An atomic point contains exactly one independently assessable
participant screening concept that should become one downstream Entity-
Attribute-Value row.

Rules:
1. Return one or more points for every source_id. Never omit a source row.
2. Split conjunctions, disjunctions, lists, and parenthetical mixed concepts
   when their members could map to different downstream Attributes. Duplicate
   shared qualifiers when needed to keep every resulting point understandable.
3. Split demographic bundles into separate Attribute-like concepts, but keep
   one sex/gender eligibility concept together. For example, "men and women
   aged from 18 to 85" becomes "men and women" and "aged from 18 to 85";
   "all genders aged 18 to 85" becomes "all genders" and
   "aged 18 to 85".
4. Split disease/status concepts from numerical thresholds or lab values even
   when the number appears to define the disease in parentheses. The full
   source row remains available as context for later reasoning. For example,
   "permanent renal failure (Creatinin >180 micromol/l)" becomes "permanent
   renal failure" and "Creatinin >180 micromol/l".
5. When one threshold applies to multiple lab tests, scores, scales, target
   organs, or assessment domains, split each measured concept and duplicate
   the shared threshold or qualifier. For example, "TGO and TGP >2N" becomes
   "TGO >2N" and "TGP >2N".
6. Keep one numerical constraint together when it is itself the Attribute.
   For example, an age interval, a score interval, one biomarker threshold, and
   a treatment timing window each remain one point.
7. Preserve inclusion/exclusion meaning, negation, comparison operators,
   numeric values, units, timing, anatomy, laterality, and exceptions.
8. Return only conditions that determine whether a participant can enroll.
   Do not emit explanatory rationale, investigator instructions, a statement
   that a list will be supplied, reassessment or treatment procedures, test
   scheduling, or other protocol operations as eligibility points.
9. Never emit an exception or exemption as a standalone exclusion point.
   Attach it to the parent eligibility condition when it changes who is
   included or excluded.
10. Copy or minimally edit the source wording. Do not normalize terminology,
   assign entities, infer missing facts, summarize, or add clinical knowledge.
11. Do not include bullet or numbering prefixes. Do not return duplicates.
12. The program will attach criteria and context from source_id, so return only
   source_id and item.

Examples:
- "renal or hepatic impairment" becomes "renal impairment" and
  "hepatic impairment".
- "Men and women aged from 18 to 85" becomes "Men and women" and
  "aged from 18 to 85".
- "FMA-UE score between 20 and 50" remains one point.
- "pregnant or breastfeeding" becomes "pregnant" and "breastfeeding".
- "Kidney impairment with serum creatinine > 2 mg/dL" becomes
  "kidney impairment" and "serum creatinine > 2 mg/dL".
- "Hepatic failure (TGO and TGP >2N)" becomes "Hepatic failure",
  "TGO >2N", and "TGP >2N".
- "Aphasia preventing evaluation of motor and depression scales" becomes
  "Aphasia preventing evaluation of motor scales" and
  "Aphasia preventing evaluation of depression scales".
- "Medication is excluded. A list will be supplied to investigators" returns
  the medication exclusion only.
- "No safe contraception; surgically sterilized participants are exempt"
  remains one contraception condition with its exemption attached.
{error_block}
Trial identifier: {trial.trial_id}

SOURCE_ROWS_JSON:
{json.dumps(input_rows, ensure_ascii=False, indent=2)}
"""


_BULLET_PREFIX = re.compile(
    r"^\s*(?:[-*\u2022]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)"
)


def _clean_point(value: Any) -> str:
    point = _BULLET_PREFIX.sub("", _text(value))
    return re.sub(r"\s+", " ", point).strip()


def validate_breakdown_result(
    result: Mapping[str, Any],
    source_rows: tuple[dict[str, Any], ...],
) -> list[dict[str, str]]:
    """Validate coverage and attach authoritative criteria and context."""

    raw_points = result.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("Breakdown response has no points array.")

    source_by_id = {
        int(row["source_id"]): row for row in source_rows
    }
    seen_source_ids: set[int] = set()
    seen_points: set[tuple[int, str]] = set()
    ordered: list[tuple[int, int, dict[str, str]]] = []

    for position, raw_point in enumerate(raw_points, start=1):
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"Output point {position} is not an object.")
        try:
            source_id = int(raw_point.get("source_id", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Output point {position} has an invalid source_id."
            ) from exc
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(
                f"Output point {position} has unknown source_id {source_id}."
            )
        item = _clean_point(raw_point.get("item"))
        if not item:
            raise ValueError(f"Output point {position} has an empty item.")

        duplicate_key = (source_id, item.casefold())
        if duplicate_key in seen_points:
            raise ValueError(
                f"Output point {position} duplicates an earlier point for "
                f"source_id {source_id}."
            )
        seen_points.add(duplicate_key)
        seen_source_ids.add(source_id)
        ordered.append(
            (
                source_id,
                position,
                {
                    "criteria": _text(source["criteria"]).casefold(),
                    "item": item,
                    "context": _text(source["context"]),
                },
            )
        )

    missing_source_ids = sorted(set(source_by_id) - seen_source_ids)
    if missing_source_ids:
        raise ValueError(
            "Breakdown response omitted source_id values: "
            + ", ".join(str(value) for value in missing_source_ids)
        )

    ordered.sort(key=lambda value: (value[0], value[1]))
    return [row for _source_id, _position, row in ordered]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def write_breakdown_csv(
    path: str | Path,
    rows: list[dict[str, str]],
) -> Path:
    """Write exactly the requested criteria/item/context CSV contract."""

    output_path = Path(path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    writer.writerows(
        {column: _text(row.get(column)) for column in CSV_COLUMNS}
        for row in rows
    )
    atomic_write_text(output_path, buffer.getvalue(), encoding="utf-8-sig")
    return output_path


def read_breakdown_csv(path: str | Path) -> list[dict[str, str]]:
    """Read and validate an existing final CSV before reusing it."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"{input_path}: expected columns {CSV_COLUMNS}, found "
                f"{tuple(reader.fieldnames or ())}."
            )
        rows = [
            {column: _text(row.get(column)) for column in CSV_COLUMNS}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{input_path}: breakdown CSV has no points.")
    return rows


def process_trial(
    *,
    trial: EligibilityTrial,
    provider: JsonLLMProvider,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    max_attempts: int = 3,
    force: bool = False,
    progress_callback: Callable[[str], None] = print,
) -> TrialBreakdownResult:
    """Process one trial with resumable model output and atomic CSV writing."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    output_path = (
        Path(output_directory)
        / trial.trial_key
        / "elig_breakdown_points.csv"
    )
    checkpoint_path = (
        Path(checkpoint_directory) / f"{trial.trial_key}.json"
    )

    if output_path.exists() and not force:
        rows = read_breakdown_csv(output_path)
        progress_callback(
            f"[{trial.trial_key}] reused {output_path.name}"
        )
        return TrialBreakdownResult(
            trial_key=trial.trial_key,
            trial_id=trial.trial_id,
            output_path=output_path.resolve(),
            point_count=len(rows),
            status="reused_output",
        )

    if checkpoint_path.exists() and not force:
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        rows = validate_breakdown_result(
            checkpoint,
            trial.source_rows,
        )
        write_breakdown_csv(output_path, rows)
        progress_callback(
            f"[{trial.trial_key}] reused checkpoint"
        )
        return TrialBreakdownResult(
            trial_key=trial.trial_key,
            trial_id=trial.trial_id,
            output_path=output_path.resolve(),
            point_count=len(rows),
            status="reused_checkpoint",
        )

    prior_error = ""
    for attempt in range(1, max_attempts + 1):
        progress_callback(
            f"[{trial.trial_key}] model attempt "
            f"{attempt}/{max_attempts}"
        )
        result = provider.generate_json(
            build_breakdown_prompt(trial, prior_error=prior_error),
            output_schema=breakdown_output_schema(),
        )
        try:
            rows = validate_breakdown_result(
                result,
                trial.source_rows,
            )
        except (TypeError, ValueError) as exc:
            prior_error = str(exc)
            if attempt == max_attempts:
                raise
            continue

        _atomic_write_json(checkpoint_path, result)
        write_breakdown_csv(output_path, rows)
        return TrialBreakdownResult(
            trial_key=trial.trial_key,
            trial_id=trial.trial_id,
            output_path=output_path.resolve(),
            point_count=len(rows),
            status="generated",
        )

    raise RuntimeError(f"{trial.trial_key}: no breakdown result produced.")


ProviderFactory = Callable[[EligibilityTrial], JsonLLMProvider]


def process_trials_parallel(
    *,
    trials: list[EligibilityTrial],
    provider_factory: ProviderFactory,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    workers: int = 4,
    max_attempts: int = 3,
    force: bool = False,
    progress_callback: Callable[[str], None] = print,
) -> tuple[list[TrialBreakdownResult], dict[str, str]]:
    """Process independent trials concurrently and retain successful outputs."""

    if workers < 1:
        raise ValueError("workers must be at least 1.")

    results: list[TrialBreakdownResult] = []
    errors: dict[str, str] = {}

    def run_one(trial: EligibilityTrial) -> TrialBreakdownResult:
        provider = provider_factory(trial)
        try:
            return process_trial(
                trial=trial,
                provider=provider,
                output_directory=output_directory,
                checkpoint_directory=checkpoint_directory,
                max_attempts=max_attempts,
                force=force,
                progress_callback=progress_callback,
            )
        finally:
            provider.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_by_trial = {
            executor.submit(run_one, trial): trial for trial in trials
        }
        for future in as_completed(future_by_trial):
            trial = future_by_trial[future]
            try:
                result = future.result()
            except Exception as exc:  # retain other independent successes
                errors[trial.trial_key] = (
                    f"{type(exc).__name__}: {exc}"
                )
                progress_callback(
                    f"[{trial.trial_key}] failed: {errors[trial.trial_key]}"
                )
            else:
                results.append(result)
                progress_callback(
                    f"[{trial.trial_key}] {result.status}; "
                    f"{result.point_count} points"
                )

    results.sort(key=lambda result: result.trial_key.casefold())
    return results, dict(sorted(errors.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Break raw inclusion and exclusion criteria into atomic "
            "per-trial eligibility points with the local Codex bridge."
        )
    )
    parser.add_argument(
        "--input-directory",
        default=str(DEFAULT_INPUT_DIRECTORY),
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    parser.add_argument(
        "--checkpoint-directory",
        default=str(DEFAULT_CHECKPOINT_DIRECTORY),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--service-tier", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate checkpoints and overwrite existing CSV outputs.",
    )
    parser.add_argument(
        "--trial",
        action="append",
        default=[],
        help=(
            "Optional trial folder/file stem to process. Repeat to select "
            "multiple trials; by default all input trials are processed."
        ),
    )
    parser.add_argument(
        "--runtime-record-directory",
        default=None,
        help=(
            "Optional directory for complete structured LLM runtime "
            "records; each trial receives an independent run."
        ),
    )
    parser.add_argument(
        "--runtime-run-id",
        default="eligibility_breakdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trials = load_eligibility_trials(args.input_directory)
    if args.trial:
        requested = {value.casefold() for value in args.trial}
        trials = [
            trial
            for trial in trials
            if trial.trial_key.casefold() in requested
            or trial.trial_id.casefold() in requested
        ]
        found = {trial.trial_key.casefold() for trial in trials} | {
            trial.trial_id.casefold() for trial in trials
        }
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(
                "Requested trial(s) not found: " + ", ".join(missing)
            )
        if not trials:
            raise SystemExit("No trials selected.")

    def provider_factory(trial: EligibilityTrial) -> JsonLLMProvider:
        base_provider = CodexJsonLLMProvider(
            model=args.model,
            cwd=PROJECT_ROOT,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
        )
        if not args.runtime_record_directory:
            return base_provider
        return RuntimeRecordingJsonLLMProvider(
            base_provider,
            record_directory=args.runtime_record_directory,
            run_id=f"{args.runtime_run_id}_{trial.trial_key}",
            run_metadata={
                "workflow": "eligibility_point_breakdown",
                "trial_key": trial.trial_key,
                "trial_id": trial.trial_id,
                "source_path": str(trial.source_path),
                "source_rows": len(trial.source_rows),
            },
        )

    results, errors = process_trials_parallel(
        trials=trials,
        provider_factory=provider_factory,
        output_directory=args.output_directory,
        checkpoint_directory=args.checkpoint_directory,
        workers=args.workers,
        max_attempts=args.max_attempts,
        force=args.force,
        progress_callback=lambda message: print(message, flush=True),
    )
    summary = {
        "source_trials": len(trials),
        "completed_trials": len(results),
        "failed_trials": len(errors),
        "total_points": sum(result.point_count for result in results),
        "output_directory": str(Path(args.output_directory).resolve()),
        "checkpoint_directory": str(
            Path(args.checkpoint_directory).resolve()
        ),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "workers": args.workers,
        "results": [
            {
                "trial_key": result.trial_key,
                "trial_id": result.trial_id,
                "point_count": result.point_count,
                "status": result.status,
                "output_path": str(result.output_path),
            }
            for result in results
        ],
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
