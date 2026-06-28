from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.api.app_state import AppState
from backend.core.extractor import ModernAutoCritExtractor
from backend.core.normalizer import CriteriaNormalizer
from backend.services.pipeline import ModernAutoCritPipeline
from backend.services.file_manager import FileManager


def save_settings(state: AppState) -> None:
    FileManager.save_settings(state.settings)


def run_pipeline(
    state: AppState,
    xml_directory: str | Path,
    output_excel: str | Path,
):
    extractor = ModernAutoCritExtractor(
        openai_settings=state.settings.openai,
        extraction_settings=state.settings.extraction,
        cost_monitor=state.cost_monitor,
    )

    normalizer = CriteriaNormalizer()

    pipeline = ModernAutoCritPipeline(
        extractor=extractor,
        normalizer=normalizer,
        cost_monitor=state.cost_monitor,
    )

    summary = pipeline.run(
        xml_directory=Path(xml_directory),
        output_excel=Path(output_excel),
    )

    state.current_output_file = Path(output_excel)
    state.last_summary = summary

    return summary


def preview_output(
    output_excel: str | Path,
    n: int = 20,
) -> pd.DataFrame:
    return pd.read_excel(output_excel).head(n)


def load_dictionary(dictionary_type: str) -> dict:
    from backend.services.dictionary_service import DictionaryService

    return DictionaryService.load(dictionary_type)


def save_dictionary(dictionary_type: str, mapping: dict) -> None:
    from backend.services.dictionary_service import DictionaryService

    DictionaryService.save(dictionary_type, mapping)


def add_dictionary_mapping(
    dictionary_type: str,
    raw_term: str,
    canonical_term: str,
) -> dict:
    from backend.services.dictionary_service import DictionaryService

    return DictionaryService.add_mapping(
        dictionary_type=dictionary_type,
        raw_term=raw_term,
        canonical_term=canonical_term,
    )