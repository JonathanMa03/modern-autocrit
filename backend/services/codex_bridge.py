"""
Codex bridge service.

This is a placeholder layer for routing extraction or code-assistance
tasks through a future Codex-backed workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CodexTaskResult:
    success: bool
    message: str
    payload: dict[str, Any] | None = None


class CodexBridge:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def is_available(self) -> bool:
        return self.enabled

    def run_task(
        self,
        task_name: str,
        payload: dict[str, Any] | None = None,
    ) -> CodexTaskResult:
        if not self.enabled:
            return CodexTaskResult(
                success=False,
                message="Codex bridge is not enabled.",
                payload=payload or {},
            )

        return CodexTaskResult(
            success=False,
            message=f"Codex task '{task_name}' is not implemented yet.",
            payload=payload or {},
        )