from __future__ import annotations

from pathlib import Path

from backend.normalization.terminology_repository import (
    TerminologyRepository,
    TerminologyType,
)


class TerminologyService:
    """
    Coordinates terminology lookup, updates, reloads,
    and dictionary persistence.

    This service sits between the GUI/pipeline and the
    underlying TerminologyRepository.
    """

    def __init__(
        self,
        terminology_type: TerminologyType,
        dictionary_path: str | Path | None = None,
    ) -> None:
        self.terminology_type = terminology_type
        self.dictionary_path = (
            Path(dictionary_path)
            if dictionary_path is not None
            else None
        )

        self.repository = self._create_repository()

    def _create_repository(
        self,
    ) -> TerminologyRepository:
        return TerminologyRepository(
            terminology_type=self.terminology_type,
            dictionary_path=self.dictionary_path,
        )

    def reload(
        self,
    ) -> None:
        """
        Reloads the dictionary from disk.
        """

        self.repository = self._create_repository()

    def exact_lookup(
        self,
        raw_term: str,
    ) -> str | None:
        return self.repository.exact_lookup(
            raw_term
        )

    def canonical_terms(
        self,
    ) -> list[str]:
        return self.repository.canonical_terms()

    def aliases_for(
        self,
        canonical_term: str,
    ) -> list[str]:
        return self.repository.aliases_for(
            canonical_term
        )

    def search_entries(
        self,
    ) -> list[dict[str, str]]:
        return self.repository.search_entries()

    def add_mapping(
        self,
        alias: str,
        canonical_term: str,
        save: bool = True,
    ) -> None:
        """
        Adds or updates an alias-to-canonical mapping.

        By default, the change is saved immediately.
        """

        self.repository.add_mapping(
            alias=alias,
            canonical_term=canonical_term,
        )

        if save:
            self.repository.save()

    def add_mappings(
        self,
        mappings: dict[str, str],
        save: bool = True,
    ) -> None:
        """
        Adds multiple alias-to-canonical mappings.
        """

        for alias, canonical_term in (
            mappings.items()
        ):
            self.repository.add_mapping(
                alias=alias,
                canonical_term=canonical_term,
            )

        if save:
            self.repository.save()

    def remove_mapping(
        self,
        alias: str,
        save: bool = True,
    ) -> bool:
        """
        Removes an alias mapping.

        Returns True if the alias existed.
        """

        cleaned_alias = (
            self.repository._clean_text(alias)
        )

        if (
            cleaned_alias
            not in self.repository.alias_to_canonical
        ):
            return False

        del self.repository.alias_to_canonical[
            cleaned_alias
        ]

        self.repository.canonical_to_aliases = (
            self.repository._group_aliases()
        )

        if save:
            self.repository.save()

        return True

    def has_mapping(
        self,
        alias: str,
    ) -> bool:
        return (
            self.repository.exact_lookup(alias)
            is not None
        )

    def get_mapping_count(
        self,
    ) -> int:
        return len(
            self.repository.alias_to_canonical
        )

    def get_canonical_count(
        self,
    ) -> int:
        return len(
            self.repository.canonical_to_aliases
        )