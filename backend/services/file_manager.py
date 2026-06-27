from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from backend.schemas.criteria_schema import EligibilityCriterion
from backend.schemas.settings_schema import AppSettings
from backend.utils.paths import (
    DEFAULT_SETTINGS,
    ENTITY_MAP,
    ATTRIBUTE_MAP,
    DISEASE_MAP,
)

class FileManager:
    @staticmethod
    def load_settings(settings_path: Path | None = None) -> AppSettings:
        path = settings_path or DEFAULT_SETTINGS

        if not path.exists():
            settings = AppSettings()
            FileManager.save_settings(settings, path)
            return settings

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return AppSettings.model_validate(data)

    @staticmethod
    def save_settings(settings: AppSettings, settings_path: Path | None = None) -> None:
        path = settings_path or DEFAULT_SETTINGS
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings.model_dump(mode="json"), f, indent=4)

    @staticmethod
    def load_dictionary(path: Path) -> dict:
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_dictionary(dictionary: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, indent=4, sort_keys=True)

    @staticmethod
    def load_entity_dictionary() -> dict:
        return FileManager.load_dictionary(ENTITY_MAP)

    @staticmethod
    def load_attribute_dictionary() -> dict:
        return FileManager.load_dictionary(ATTRIBUTE_MAP)

    @staticmethod
    def load_disease_dictionary() -> dict:
        return FileManager.load_dictionary(DISEASE_MAP)

    @staticmethod
    def criteria_to_dataframe(criteria: Iterable[EligibilityCriterion]) -> pd.DataFrame:
        return pd.DataFrame([c.model_dump() for c in criteria])

    @staticmethod
    def save_criteria_csv(
        criteria: Iterable[EligibilityCriterion],
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = FileManager.criteria_to_dataframe(criteria)
        df.to_csv(path, index=False)

    @staticmethod
    def save_criteria_excel(
        criteria: Iterable[EligibilityCriterion],
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = FileManager.criteria_to_dataframe(criteria)
        df.to_excel(path, index=False)

    @staticmethod
    def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def save_dataframe_excel(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False)

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def load_excel(path: Path) -> pd.DataFrame:
        return pd.read_excel(path)