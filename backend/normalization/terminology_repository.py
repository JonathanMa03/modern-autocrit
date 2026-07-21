from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


TerminologyType = Literal[
    "attribute",
    "entity",
    "disease",
]


class TerminologyRepository:
    """
    Loads an alias-to-canonical terminology map.

    Expected JSON format:

    {
        "alt": "Alanine aminotransferase (ALT)",
        "alanine aminotransferase":
            "Alanine aminotransferase (ALT)"
    }
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DEFAULT_DICTIONARY_PATHS = {
        "attribute": PROJECT_ROOT
        / "config"
        / "dictionaries"
        / "attribute_map.json",

        "entity": PROJECT_ROOT
        / "config"
        / "dictionaries"
        / "entity_map.json",

        "disease": PROJECT_ROOT
        / "config"
        / "dictionaries"
        / "disease_map.json",
    }

    def __init__(
        self,
        terminology_type: TerminologyType,
        dictionary_path: str | Path | None = None,
    ) -> None:
        if terminology_type not in (
            "attribute",
            "entity",
            "disease",
        ):
            raise ValueError(
                "terminology_type must be one of: "
                "'attribute', 'entity', or 'disease'."
            )

        self.terminology_type = terminology_type

        if dictionary_path is None:
            self.dictionary_path = (
                self.DEFAULT_DICTIONARY_PATHS[
                    terminology_type
                ]
            )
        else:
            self.dictionary_path = Path(
                dictionary_path
            )

        if not self.dictionary_path.exists():
            raise FileNotFoundError(
                "Terminology dictionary not found: "
                f"{self.dictionary_path}"
            )

        self.alias_to_canonical = (
            self._load_dictionary()
        )

        self.canonical_to_aliases = (
            self._group_aliases()
        )

    def _load_dictionary(
        self,
    ) -> dict[str, str]:
        with self.dictionary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, dict):
            raise ValueError(
                "Terminology dictionary must contain "
                "a JSON object."
            )

        cleaned_mapping: dict[str, str] = {}

        for raw_alias, raw_canonical in (
            raw_data.items()
        ):
            alias = self._clean_text(raw_alias)
            canonical = str(
                raw_canonical
            ).strip()

            if not alias or not canonical:
                continue

            cleaned_mapping[alias] = canonical

        return cleaned_mapping

    def _group_aliases(
        self,
    ) -> dict[str, list[str]]:
        """
        Groups aliases by canonical term while treating
        capitalization-only differences as the same concept.
        """

        grouped: dict[str, list[str]] = {}
        normalized_to_preferred: dict[str, str] = {}

        for alias, canonical in (
            self.alias_to_canonical.items()
        ):
            normalized_canonical = (
                self._normalize_canonical_key(
                    canonical
                )
            )

            if (
                normalized_canonical
                not in normalized_to_preferred
            ):
                normalized_to_preferred[
                    normalized_canonical
                ] = canonical

            preferred_canonical = (
                normalized_to_preferred[
                    normalized_canonical
                ]
            )

            grouped.setdefault(
                preferred_canonical,
                [],
            )

            if alias not in grouped[
                preferred_canonical
            ]:
                grouped[
                    preferred_canonical
                ].append(alias)

        return grouped

    def exact_lookup(
        self,
        raw_term: str,
    ) -> str | None:
        normalized_term = self._clean_text(
            raw_term
        )

        if not normalized_term:
            return None

        return self.alias_to_canonical.get(
            normalized_term
        )

    def canonical_terms(
        self,
    ) -> list[str]:
        return sorted(
            self.canonical_to_aliases.keys()
        )

    def aliases_for(
        self,
        canonical_term: str,
    ) -> list[str]:
        return list(
            self.canonical_to_aliases.get(
                canonical_term,
                [],
            )
        )

    def search_entries(
        self,
    ) -> list[dict[str, str]]:
        """
        Returns aliases and canonical terms as
        searchable embedding entries.
        """

        entries: list[dict[str, str]] = []

        for canonical, aliases in (
            self.canonical_to_aliases.items()
        ):
            entries.append(
                {
                    "text": canonical,
                    "canonical_term": canonical,
                    "source": "canonical",
                }
            )

            for alias in aliases:
                entries.append(
                    {
                        "text": alias,
                        "canonical_term": canonical,
                        "source": "alias",
                    }
                )

        return entries

    def add_mapping(
        self,
        alias: str,
        canonical_term: str,
    ) -> None:
        """
        Adds or updates a mapping in memory.

        Call save() to persist it to disk.
        """

        cleaned_alias = self._clean_text(alias)
        cleaned_canonical = str(
            canonical_term
        ).strip()

        if not cleaned_alias:
            raise ValueError(
                "Alias cannot be empty."
            )

        if not cleaned_canonical:
            raise ValueError(
                "Canonical term cannot be empty."
            )

        self.alias_to_canonical[
            cleaned_alias
        ] = cleaned_canonical

        self.canonical_to_aliases = (
            self._group_aliases()
        )

    def save(
        self,
    ) -> None:
        sorted_mapping = dict(
            sorted(
                self.alias_to_canonical.items(),
                key=lambda item: item[0],
            )
        )

        with self.dictionary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                sorted_mapping,
                file,
                indent=4,
                ensure_ascii=False,
            )

            file.write("\n")

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        return " ".join(
            str(text)
            .strip()
            .lower()
            .split()
        )
    
    @staticmethod
    def _normalize_canonical_key(
        text: str,
    ) -> str:
        """
        Normalizes a canonical term for identity comparison
        without changing the displayed capitalization.
        """

        return " ".join(
            str(text)
            .strip()
            .casefold()
            .split()
        )