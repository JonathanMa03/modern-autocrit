"""LLM provider adapters for criteria-processor workflows."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

try:  # Optional: only needed when the Codex-backed provider is selected.
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
    from openai_codex.api import ReasoningEffort
except ImportError:  # Keep deterministic processors importable without SDK.
    ApprovalMode = Codex = CodexConfig = Sandbox = ReasoningEffort = None


DEFAULT_CODEX_MODEL = "gpt-5.4-mini"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or f"run_{uuid4().hex[:12]}"


def _record_stage(output_schema: dict[str, Any]) -> str:
    properties = output_schema.get("properties", {})
    if "facts" in properties:
        return "manuscript_eav_extractor"
    if {
        "variable_name",
        "confidence_score",
        "source_evidences",
    }.issubset(properties) and (
        "extracted_terms" in properties or "extracted_term" in properties
    ):
        return "trial_information_extractor"
    if {
        "variable_name",
        "fields",
        "extraction_instructions",
        "rationale",
    }.issubset(properties):
        return "variable_schema_designer"
    if "items" in properties:
        return "eligibility_eav_extractor"
    if "expanded_attribute_terms" in properties:
        return "terminology_search_expander"
    if {"lexical_terms", "semantic_terms"}.issubset(properties):
        return "eligibility_eva_search_expander"
    if {
        "decision",
        "current_mapping_decision",
        "reviewed_mapping_decision",
        "semantic_relationship",
        "additional_search_terms",
    }.issubset(properties):
        return "eligibility_eva_mapping_reviewer"
    if "correction_acknowledged" in properties:
        return "eligibility_eva_mapping_corrector"
    if {
        "decision",
        "reviewed_attribute_id",
        "explanation",
    }.issubset(properties):
        return "eligibility_eva_attribute_id_reviewer"
    if {
        "mapping_decision",
        "value_type",
        "confidence",
    }.issubset(properties):
        return "eligibility_eva_reasoner"
    if "suggestions" in properties:
        return "penalty_hyperparameter_initializer"
    if "decision" in properties:
        return "terminology_mapping_selector"
    return "json_generation"


def _prompt_input_payload(prompt: str, stage: str) -> Any:
    marker_by_stage = {
        "eligibility_eav_extractor": "Input JSON:\n",
        "terminology_search_expander": "EVIDENCE_JSON:\n",
        "terminology_mapping_selector": "SELECTION_INPUT_JSON:\n",
        "trial_information_extractor": "EXTRACTION_INPUT_JSON:\n",
        "variable_schema_designer": "SCHEMA_DESIGN_INPUT_JSON:\n",
        "eligibility_eva_search_expander": "EVIDENCE_JSON:\n",
        "eligibility_eva_reasoner": "SELECTION_INPUT_JSON:\n",
        "eligibility_eva_mapping_reviewer": (
            "MAPPING_REVIEW_INPUT_JSON:\n"
        ),
        "eligibility_eva_mapping_corrector": (
            "MAPPING_CORRECTION_INPUT_JSON:\n"
        ),
        "eligibility_eva_attribute_id_reviewer": (
            "ATTRIBUTE_ID_REVIEW_INPUT_JSON:\n"
        ),
        "manuscript_eav_extractor": (
            "MANUSCRIPT_EXTRACTION_INPUT_JSON:\n"
        ),
        "penalty_hyperparameter_initializer": (
            "PENALTY_INITIALIZATION_INPUT_JSON:\n"
        ),
    }
    marker = marker_by_stage.get(stage)
    if marker is None or marker not in prompt:
        return None
    try:
        return json.loads(prompt.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return None


class RuntimeRecordingJsonLLMProvider:
    """Persist complete structured-LLM calls for replay and fine-tuning."""

    def __init__(
        self,
        provider: Any,
        *,
        record_directory: str | Path,
        run_id: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.model = getattr(provider, "model", "")
        self.reasoning_effort = str(
            getattr(provider, "reasoning_effort", "") or ""
        )
        self.service_tier = getattr(provider, "service_tier", None)
        self.last_completed_at: str | None = None
        self.run_id = _safe_identifier(
            run_id
            or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_"
            f"{uuid4().hex[:8]}"
        )
        self.record_root = Path(record_directory) / self.run_id
        self.calls_directory = self.record_root / "calls"
        self.calls_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.record_root / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "eligcrit.runtime_record.v1",
                        "run_id": self.run_id,
                        "created_at": _utc_now(),
                        "provider": type(provider).__name__,
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "service_tier": self.service_tier,
                        "metadata": run_metadata or {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()

    def generate_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        call_id = uuid4().hex
        stage = _record_stage(output_schema)
        started_at = _utc_now()
        started = self._base_record(
            call_id=call_id,
            stage=stage,
            prompt=prompt,
            output_schema=output_schema,
            started_at=started_at,
        )
        self._write_record(call_id, "started", started)
        started_clock = time.perf_counter()
        try:
            result = self.provider.generate_json(
                prompt,
                output_schema=output_schema,
            )
        except Exception as exc:
            failed = {
                **started,
                "status": "failed",
                "completed_at": _utc_now(),
                "duration_ms": round(
                    (time.perf_counter() - started_clock) * 1000,
                    3,
                ),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            self._write_record(call_id, "failed", failed)
            raise
        completed = self._completed_record(
            started,
            result=result,
            duration_ms=round(
                (time.perf_counter() - started_clock) * 1000,
                3,
            ),
        )
        self.last_completed_at = str(completed["completed_at"])
        self._write_record(call_id, "completed", completed)
        return result

    def record_existing_completion(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        result: dict[str, Any],
        source_metadata: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> Path:
        """Record a previously completed call reconstructed losslessly."""

        call_id = uuid4().hex
        stage = _record_stage(output_schema)
        started = self._base_record(
            call_id=call_id,
            stage=stage,
            prompt=prompt,
            output_schema=output_schema,
            started_at=started_at or _utc_now(),
        )
        completed = self._completed_record(
            started,
            result=result,
            duration_ms=None,
        )
        completed["recording_mode"] = "reconstructed_completed_call"
        completed["source_metadata"] = source_metadata or {}
        return self._write_record(call_id, "completed", completed)

    def _base_record(
        self,
        *,
        call_id: str,
        stage: str,
        prompt: str,
        output_schema: dict[str, Any],
        started_at: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "eligcrit.runtime_record.v1",
            "record_type": "structured_llm_call",
            "run_id": self.run_id,
            "call_id": call_id,
            "stage": stage,
            "status": "started",
            "started_at": started_at,
            "completed_at": None,
            "duration_ms": None,
            "provider": type(self.provider).__name__,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "service_tier": self.service_tier,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "prompt": prompt,
            "input_payload": _prompt_input_payload(prompt, stage),
            "output_schema": output_schema,
            "response_json": None,
            "messages": [{"role": "user", "content": prompt}],
        }

    @staticmethod
    def _completed_record(
        started: dict[str, Any],
        *,
        result: dict[str, Any],
        duration_ms: float | None,
    ) -> dict[str, Any]:
        assistant_content = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return {
            **started,
            "status": "completed",
            "completed_at": _utc_now(),
            "duration_ms": duration_ms,
            "response_sha256": hashlib.sha256(
                assistant_content.encode("utf-8")
            ).hexdigest(),
            "response_json": result,
            "messages": [
                *started["messages"],
                {"role": "assistant", "content": assistant_content},
            ],
        }

    def _write_record(
        self,
        call_id: str,
        status: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self.calls_directory / f"{call_id}.{status}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


class CodexJsonLLMProvider:
    """Structured JSON provider backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        model: str | None = None,
        cwd: str | Path | None = None,
        reasoning_effort: str | ReasoningEffort | None = "low",
        service_tier: str | None = None,
    ) -> None:
        if ReasoningEffort is None:
            raise RuntimeError(
                "Codex support requires the optional 'openai-codex' package. "
                "Install it or select the OpenAI provider."
            )
        self.model = model or DEFAULT_CODEX_MODEL
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else None
        self.reasoning_effort = (
            reasoning_effort
            if isinstance(reasoning_effort, ReasoningEffort)
            else ReasoningEffort(reasoning_effort)
            if reasoning_effort
            else None
        )
        self.service_tier = service_tier

    def close(self) -> None:
        """Release the underlying Codex SDK client."""

        return None

    def generate_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one ephemeral Codex turn constrained by a JSON schema."""

        codex = Codex(CodexConfig(cwd=self.cwd))
        try:
            thread = codex.thread_start(
                approval_mode=ApprovalMode.deny_all,
                cwd=self.cwd,
                ephemeral=True,
                model=self.model,
                sandbox=Sandbox.read_only,
                service_tier=self.service_tier,
            )
            result = thread.run(
                prompt,
                effort=self.reasoning_effort,
                model=self.model,
                output_schema=output_schema,
                cwd=self.cwd,
                sandbox=Sandbox.read_only,
                service_tier=self.service_tier,
            )
            if result.error is not None:
                raise RuntimeError(f"Codex turn failed: {result.error}")
            if not result.final_response:
                raise RuntimeError("Codex turn returned no final response.")
            return json.loads(result.final_response)
        finally:
            codex.close()
