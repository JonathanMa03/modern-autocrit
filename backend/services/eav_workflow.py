"""Extraction jobs and append-only structured-criterion snapshots."""

from __future__ import annotations

import csv
import base64
import io
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from pypdf import PdfReader

from backend.api.app_state import AppState
from backend.atomic_io import atomic_write_json
from backend.eav_pipeline import ROW_SCHEMA, deterministic_criterion, extraction_prompt_for_items, interval_notation, normalize_rows, segment_criteria, semantic_relations
from backend.services.ctgov_v2_service import eligibility_text_from_study, fetch_study
from backend.services.llm.factory import create_llm_provider
from backend.services.llm.json_adapter import StructuredJsonAdapter


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    trial_id: str
    input_mode: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start."
    progress: int = 0
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    error: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, str]] = field(default_factory=list)


class EavWorkflowService:
    def __init__(self, state: AppState, root: Path, base_library: Path) -> None:
        self.state = state
        self.root = root
        self.jobs = root / "jobs"
        self.libraries = root / "libraries"
        self.base_library = base_library
        self.config_libraries = base_library.parent
        self._lock = Lock()
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.libraries.mkdir(parents=True, exist_ok=True)

    def submit(self, *, input_mode: str, trial_id: str, protocol_text: str, protocol_pdf: str = "") -> dict[str, Any]:
        if input_mode not in {"manual", "nct", "pdf"}:
            raise ValueError("Input mode must be manual, nct, or pdf.")
        if input_mode == "manual" and not protocol_text.strip():
            raise ValueError("Paste protocol eligibility criteria.")
        if input_mode == "nct" and not trial_id.strip():
            raise ValueError("Enter an NCT ID.")
        if input_mode == "pdf" and not protocol_pdf:
            raise ValueError("Choose a protocol PDF.")
        job = Job(job_id=uuid4().hex, trial_id=trial_id.strip() or "MANUAL", input_mode=input_mode)
        self._save(job)
        Thread(target=self._run, args=(job.job_id, protocol_text, protocol_pdf), daemon=True).start()
        return asdict(job)

    def get(self, job_id: str) -> dict[str, Any]:
        path = self.jobs / f"{job_id}.json"
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, job: Job) -> None:
        job.updated_at = now()
        atomic_write_json(self.jobs / f"{job.job_id}.json", asdict(job))

    def _event(self, job: Job, stage: str, message: str, progress: int) -> None:
        job.stage, job.message, job.progress = stage, message, progress
        job.events.append({"timestamp": now(), "stage": stage, "message": message})
        self._save(job)

    @staticmethod
    def _eligibility_from_pdf(encoded_pdf: str) -> str:
        try:
            raw = base64.b64decode(encoded_pdf, validate=True)
            pages = [(page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages]
        except Exception as exc:
            raise ValueError(f"Could not read protocol PDF: {exc}") from exc
        numbered_inclusion = re.compile(
            r"^\s*\d+(?:\.\d+)+\s+Inclusion\s+criteria\s*$", re.I | re.M
        )
        start_page = next(
            (index for index, page in enumerate(pages) if numbered_inclusion.search(page)),
            None,
        )
        if start_page is None:
            start_page = next(
                (
                    index
                    for index, page in enumerate(pages)
                    if re.search(r"^\s*Key inclusion criteria\s*$", page, re.I | re.M)
                    and "......" not in page
                ),
                None,
            )
        if start_page is None:
            raise ValueError("The PDF did not contain a recognizable Inclusion or Exclusion Criteria section.")
        text = "\n".join(pages[start_page : start_page + 8])
        start = re.search(r"^\s*(?:\d+(?:\.\d+)+\s+)?(?:Key\s+)?Inclusion criteria\s*$", text, re.I | re.M)
        if start:
            text = text[start.start() :]
        # Page furniture inside a long protocol must not become part of the
        # preceding wrapped criterion.
        text = "\n".join(
            line for line in text.splitlines()
            if not (
                "Protocol Date:" in line
                or line.lstrip().startswith("Trial ID:")
                or "VV-TMF-" in line
                or re.search(r"\|\s*of\s+Protocol\b", line, re.I)
                or re.fullmatch(r"\s*Page:?\s+\d+\s+of\s+\d+\s*", line, re.I)
                or re.fullmatch(r"\s*(?:Confidential|(?:Status:\s*)?Final)\s*", line, re.I)
            )
        )
        headings = list(re.finditer(r"^\s*\d+(?:\.\d+)+\s+[^\n]+$", text, re.M))
        eligibility_headings_seen = 0
        for heading in headings:
            title = heading.group(0).casefold()
            if "inclusion criteria" in title or "exclusion criteria" in title:
                eligibility_headings_seen += 1
                continue
            if eligibility_headings_seen >= 2:
                text = text[: heading.start()]
                break
        return text

    def _run(self, job_id: str, protocol_text: str, protocol_pdf: str) -> None:
        job = Job(**self.get(job_id))
        try:
            job.status = "running"
            self._event(job, "ingestion", "Loading protocol eligibility criteria.", 15)
            if job.input_mode == "nct":
                protocol_text = eligibility_text_from_study(fetch_study(job.trial_id))
            elif job.input_mode == "pdf":
                self._event(job, "pdf", "Reading the full protocol PDF and locating eligibility sections.", 10)
                protocol_text = self._eligibility_from_pdf(protocol_pdf)
                if job.trial_id == "MANUAL":
                    job.trial_id = "PDF"
            source_items = segment_criteria(protocol_text)
            if not source_items:
                raise ValueError("No eligibility criteria could be segmented from the input.")
            self._event(job, "located", f"Located {len(source_items)} atomic eligibility criteria.", 30)
            self._event(job, "extraction", f"Extracting 0 of {len(source_items)} criteria.", 35)
            provider = StructuredJsonAdapter(
                create_llm_provider(self.state.settings.llm, cost_monitor=self.state.cost_monitor),
                model=self.state.settings.llm.model,
            )
            extracted_rows = []
            batches = [source_items[index:index + 20] for index in range(0, len(source_items), 20)]
            for batch_number, batch in enumerate(batches, start=1):
                last_error: Exception | None = None
                payload = None
                for attempt in range(1, 3):
                    try:
                        payload = provider.generate_json(
                            extraction_prompt_for_items(batch), output_schema=ROW_SCHEMA
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        self._event(job, "retry", f"Batch {batch_number} attempt {attempt} failed; retrying.", 45)
                if payload is None:
                    raise RuntimeError(f"Structured extraction failed in batch {batch_number}: {last_error}")
                extracted_rows.extend(payload.get("rows") or [])
                completed = min(batch_number * 20, len(source_items))
                self._event(job, "extraction", f"Extracted {completed} of {len(source_items)} criteria.", 40 + round(30 * completed / len(source_items)))
            self._event(job, "normalization", "Separating domains, constraints, constructs, and Boolean semantics.", 75)
            job.rows = normalize_rows({"rows": extracted_rows}, source_items)
            job.status = "completed"
            self._event(job, "completed", f"Extracted {len(job.rows)} criteria.", 100)
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self._event(job, "failed", job.error, 100)

    @staticmethod
    def _upgrade_legacy_entry(row: dict[str, Any]) -> dict[str, Any]:
        """Read former EAV snapshots without rewriting the historical files."""
        if "domain" in row:
            upgraded = dict(row)
            for key in ("lower_bound", "lower_inclusive", "upper_bound", "upper_inclusive"):
                upgraded.setdefault(key, None)
            upgraded.setdefault("interval", interval_notation(
                upgraded["lower_bound"], upgraded["lower_inclusive"],
                upgraded["upper_bound"], upgraded["upper_inclusive"],
            ))
            upgraded.setdefault("relations", semantic_relations(
                comparator=upgraded.get("comparator") or ("interval" if upgraded.get("interval") else None),
                negated=bool(upgraded.get("negated")), temporal=upgraded.get("temporal"),
                repetition=upgraded.get("repetition"), qualifier=upgraded.get("qualifier"),
                logical_operator=str(upgraded.get("logical_operator") or "standalone"),
            ))
            return upgraded
        semantic = deterministic_criterion(
            source=str(row.get("source") or ""),
            criterion_type=str(row.get("criterion_type") or "inclusion"),
            entity=str(row.get("entity") or ""),
            comparator=row.get("attribute"), value=row.get("value"),
        )
        return {
            **{key: value for key, value in row.items() if key != "attribute"},
            **semantic,
            "source_id": row.get("source_id") or row.get("row_id"),
        }

    def library(self) -> dict[str, Any]:
        base = json.loads(self.base_library.read_text(encoding="utf-8"))
        snapshots = sorted(self.config_libraries.glob("criteriaLibrary_*.json"))
        legacy = sorted(self.config_libraries.glob("eavLibrary_*.json"))
        selected = snapshots[-1] if snapshots else (legacy[-1] if legacy else None)
        latest = json.loads(selected.read_text(encoding="utf-8")) if selected else {"entries": []}
        base["entries"] = [self._upgrade_legacy_entry(row) for row in base.get("entries", [])]
        entries = [self._upgrade_legacy_entry(row) for row in latest.get("entries", [])]
        return {"base": base, "entries": entries, "snapshot": selected.name if selected else None}

    def apply_review(self, job_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("status") != "completed":
            raise ValueError("Only completed jobs can be reviewed.")
        expected = {row["row_id"] for row in job.get("rows", [])}
        supplied = {str(row.get("row_id")) for row in rows}
        if expected != supplied:
            raise ValueError("Every extracted row must be accepted or corrected.")
        clean = []
        for row in rows:
            if row.get("review_status") != "accepted":
                raise ValueError("Every row must be accepted or corrected before apply.")
            criterion_type = str(row.get("criterion_type") or "").casefold()
            if criterion_type not in {"inclusion", "exclusion"}:
                raise ValueError("Criterion type must be inclusion or exclusion.")
            domain = str(row.get("domain") or "").casefold().strip()
            if domain not in {"observation", "condition", "person", "device", "drug", "visit", "procedure", "measurement"}:
                raise ValueError("Every reviewed row requires a valid Chia domain.")
            entity = str(row.get("entity") or "").strip()
            value = str(row.get("value") if row.get("value") is not None else "").strip()
            def optional_number(field: str) -> int | float | None:
                raw = row.get(field)
                if raw in {None, ""}: return None
                number = float(raw)
                return int(number) if number.is_integer() else number
            lower_bound = optional_number("lower_bound")
            upper_bound = optional_number("upper_bound")
            lower_inclusive = bool(row.get("lower_inclusive")) if lower_bound is not None else None
            upper_inclusive = bool(row.get("upper_inclusive")) if upper_bound is not None else None
            if not entity or (value == "" and lower_bound is None and upper_bound is None):
                raise ValueError("Entity and a value or interval bound are required for every reviewed row.")
            logical_operator = str(row.get("logical_operator") or "standalone").casefold()
            if logical_operator not in {"standalone", "and", "or"}:
                raise ValueError("Logical operator must be standalone, and, or or.")
            comparator = str(row.get("comparator") or "").strip() or None
            if comparator not in {None, ">", ">=", "<", "<=", "="}:
                raise ValueError("Comparator must be one of >, >=, <, <=, =, or blank.")
            temporal = str(row.get("temporal") or "").strip() or None
            repetition = str(row.get("repetition") or "").strip() or None
            qualifier = str(row.get("qualifier") or "").strip() or None
            negated = bool(row.get("negated"))
            clean.append({
                "row_id": str(row["row_id"]), "trial_id": job["trial_id"],
                "source_id": str(row.get("source_id") or row["row_id"]),
                "criterion_type": criterion_type, "domain": domain, "entity": entity,
                "comparator": comparator,
                "value": value, "unit": str(row.get("unit") or "").strip() or None,
                "value_kind": "interval" if lower_bound is not None and upper_bound is not None else str(row.get("value_kind") or "categorical"),
                "lower_bound": lower_bound, "lower_inclusive": lower_inclusive,
                "upper_bound": upper_bound, "upper_inclusive": upper_inclusive,
                "interval": interval_notation(lower_bound, lower_inclusive, upper_bound, upper_inclusive),
                "negated": negated,
                "temporal": temporal,
                "repetition": repetition,
                "qualifier": qualifier,
                "logical_operator": logical_operator,
                "relations": semantic_relations(comparator=comparator or ("interval" if lower_bound is not None or upper_bound is not None else None), negated=negated, temporal=temporal, repetition=repetition, qualifier=qualifier, logical_operator=logical_operator),
                "source": str(row.get("source") or "").strip(),
                "reviewed_at": now(),
            })
        with self._lock:
            current = self.library()["entries"]
            by_key = {(r["trial_id"], r["source"], r["criterion_type"]): r for r in current}
            for row in clean:
                by_key[(row["trial_id"], row["source"], row["criterion_type"])] = row
            entries = sorted(by_key.values(), key=lambda r: (r["trial_id"], r["criterion_type"], r["entity"]))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            json_path = self.libraries / f"criteriaLibrary_{stamp}.json"
            csv_path = self.libraries / f"criteriaLibrary_{stamp}.csv"
            config_json_path = self.config_libraries / json_path.name
            config_csv_path = self.config_libraries / csv_path.name
            document = {"schema_version": "modern-autocrit.structured-criteria.v3", "created_at": now(), "entries": entries}
            atomic_write_json(json_path, document)
            atomic_write_json(config_json_path, document)
            for target in (csv_path, config_csv_path):
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["row_id", "source_id", "trial_id", "criterion_type", "domain", "entity", "comparator", "value", "unit", "value_kind", "lower_bound", "lower_inclusive", "upper_bound", "upper_inclusive", "interval", "negated", "temporal", "repetition", "qualifier", "logical_operator", "relations", "source", "reviewed_at"])
                    writer.writeheader(); writer.writerows(entries)
        return {"added": len(clean), "total": len(entries), "json": json_path.name, "csv": csv_path.name, "config_json": str(config_json_path), "config_csv": str(config_csv_path), "entries": entries}
