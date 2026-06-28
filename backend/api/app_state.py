from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.schemas.settings_schema import AppSettings
from backend.services.cost_monitor import CostMonitor
from backend.services.file_manager import FileManager
from backend.services.codex_bridge import CodexBridge


@dataclass
class AppState:
    settings: AppSettings
    cost_monitor: CostMonitor
    codex_bridge: CodexBridge
    current_output_file: Path | None = None
    last_summary: object | None = None


def create_app_state() -> AppState:
    settings = FileManager.load_settings()

    return AppState(
        settings=settings,
        cost_monitor=CostMonitor(
            warning_threshold_usd=settings.cost.warning_threshold_usd,
            hard_limit_usd=settings.cost.hard_limit_usd,
        ),
        codex_bridge=CodexBridge(enabled=False),
    )