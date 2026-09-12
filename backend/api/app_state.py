from __future__ import annotations

from dataclasses import dataclass
from backend.schemas.settings_schema import AppSettings
from backend.services.cost_monitor import CostMonitor
from backend.services.file_manager import FileManager


@dataclass
class AppState:
    settings: AppSettings
    cost_monitor: CostMonitor


def create_app_state() -> AppState:
    settings = FileManager.load_settings()

    return AppState(
        settings=settings,
        cost_monitor=CostMonitor(
            warning_threshold_usd=settings.cost.warning_threshold_usd,
            hard_limit_usd=settings.cost.hard_limit_usd,
        ),
    )
