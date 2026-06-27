"""
Dictionary management service.

This handles user-editable terminology mappings:
- entity normalization
- attribute normalization
- disease normalization
"""

from __future__ import annotations

from pathlib import Path

from backend.services.file_manager import FileManager
from backend.utils.paths import ENTITY_MAP, ATTRIBUTE_MAP, DISEASE_MAP
from backend.utils.text_utils import clean_attribute_key


class DictionaryService:
    DICTIONARY_PATHS = {
        "entity": ENTITY_MAP,
        "attribute": ATTRIBUTE_MAP,
        "disease": DISEASE_MAP,
    }

    @classmethod
    def load(cls, dictionary_type: str) -> dict:
        path = cls._get_path(dictionary_type)
        return FileManager.load_dictionary(path)

    @classmethod
    def save(cls, dictionary_type: str, mapping: dict) -> None:
        path = cls._get_path(dictionary_type)
        cleaned = cls.clean_mapping(mapping)
        FileManager.save_dictionary(cleaned, path)

    @classmethod
    def add_mapping(
        cls,
        dictionary_type: str,
        raw_term: str,
        canonical_term: str,
    ) -> dict:
        mapping = cls.load(dictionary_type)
        mapping[clean_attribute_key(raw_term)] = canonical_term
        cls.save(dictionary_type, mapping)
        return mapping

    @classmethod
    def remove_mapping(
        cls,
        dictionary_type: str,
        raw_term: str,
    ) -> dict:
        mapping = cls.load(dictionary_type)
        mapping.pop(clean_attribute_key(raw_term), None)
        cls.save(dictionary_type, mapping)
        return mapping

    @staticmethod
    def clean_mapping(mapping: dict) -> dict:
        return {
            clean_attribute_key(raw): canonical
            for raw, canonical in mapping.items()
            if str(raw).strip() and str(canonical).strip()
        }

    @classmethod
    def _get_path(cls, dictionary_type: str) -> Path:
        if dictionary_type not in cls.DICTIONARY_PATHS:
            raise ValueError(
                f"Unknown dictionary type: {dictionary_type}. "
                f"Expected one of {list(cls.DICTIONARY_PATHS)}."
            )

        return cls.DICTIONARY_PATHS[dictionary_type]