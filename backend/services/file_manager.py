from __future__ import annotations

import json
from pathlib import Path

from backend.schemas.settings_schema import AppSettings
from backend.utils.paths import DEFAULT_SETTINGS


class FileManager:
    """Persist the small browser-application settings document."""

    @staticmethod
    def load_settings(settings_path: Path | None = None) -> AppSettings:
        path = settings_path or DEFAULT_SETTINGS
        if not path.exists():
            settings = AppSettings()
            FileManager.save_settings(settings, path)
            return settings
        return AppSettings.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )

    @staticmethod
    def save_settings(
        settings: AppSettings,
        settings_path: Path | None = None,
    ) -> None:
        path = settings_path or DEFAULT_SETTINGS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(settings.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
