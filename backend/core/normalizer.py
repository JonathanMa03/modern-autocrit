"""
Terminology normalization and semantic deduplication utilities.

This module handles:
    • entity normalization
    • attribute normalization
    • disease normalization
    • exact duplicate removal
    • simple fuzzy duplicate detection

More advanced semantic deduplication can later be added here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz

from backend.schemas.criteria_schema import EligibilityCriterion
from backend.services.file_manager import FileManager
from backend.utils.text_utils import clean_attribute_key


class CriteriaNormalizer:
    def __init__(
        self,
        entity_map: dict | None = None,
        attribute_map: dict | None = None,
        disease_map: dict | None = None,
    ):
        self.entity_map = entity_map or FileManager.load_entity_dictionary()
        self.attribute_map = attribute_map or FileManager.load_attribute_dictionary()
        self.disease_map = disease_map or FileManager.load_disease_dictionary()

        self.entity_map = self._clean_map_keys(self.entity_map)
        self.attribute_map = self._clean_map_keys(self.attribute_map)
        self.disease_map = self._clean_map_keys(self.disease_map)

    @staticmethod
    def _clean_map_keys(mapping: dict) -> dict:
        return {
            clean_attribute_key(k): v
            for k, v in mapping.items()
        }

    def normalize_entity(self, entity: str | None) -> str | None:
        if entity is None:
            return None

        key = clean_attribute_key(entity)

        return self.entity_map.get(key, entity)

    def normalize_attribute(self, attribute: str | None) -> str | None:
        if attribute is None:
            return None

        key = clean_attribute_key(attribute)

        if key in self.attribute_map:
            return self.attribute_map[key]

        if key in self.disease_map:
            return self.disease_map[key]

        return attribute

    def normalize_criterion(
        self,
        criterion: EligibilityCriterion,
    ) -> dict:
        row = criterion.model_dump()

        row["raw_entity"] = row["entity"]
        row["raw_attribute"] = row["attribute"]

        row["canonical_entity"] = self.normalize_entity(row["entity"])
        row["canonical_attribute"] = self.normalize_attribute(row["attribute"])

        row["attribute_key"] = clean_attribute_key(row["attribute"])
        row["canonical_attribute_key"] = clean_attribute_key(
            row["canonical_attribute"]
        )

        return row

    def normalize_many(
        self,
        criteria: Iterable[EligibilityCriterion],
    ) -> pd.DataFrame:
        rows = [
            self.normalize_criterion(c)
            for c in criteria
        ]

        return pd.DataFrame(rows)

    @staticmethod
    def remove_exact_duplicates(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove exact duplicate criteria after normalization.
        """

        dedup_cols = [
            "trial_id",
            "criteria_type",
            "canonical_entity",
            "canonical_attribute_key",
            "value",
            "modifier",
        ]

        existing_cols = [
            col for col in dedup_cols
            if col in df.columns
        ]

        return (
            df
            .drop_duplicates(subset=existing_cols)
            .reset_index(drop=True)
        )
    
    @staticmethod
    def find_unmapped_terms(
        df: pd.DataFrame,
        min_count: int = 1,
    ) -> pd.DataFrame:
        """
        Return extracted attributes that have not been changed by dictionary normalization.
        """

        required_cols = {
            "attribute",
            "canonical_attribute",
            "attribute_key",
            "canonical_attribute_key",
        }

        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        unmapped = df[
            df["attribute_key"].astype(str).str.strip() == df["canonical_attribute_key"].astype(str).str.strip()
        ].copy()

        counts = (
            unmapped
            .groupby(["attribute", "canonical_attribute", "attribute_key"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        return counts[counts["count"] >= min_count].reset_index(drop=True)

    @staticmethod
    def find_fuzzy_duplicates(
        df: pd.DataFrame,
        threshold: int = 90,
    ) -> pd.DataFrame:
        """
        Identify possible fuzzy duplicates within each trial.

        This does not remove rows. It returns candidate duplicate pairs
        for human review or later automated filtering.
        """

        if "canonical_attribute" not in df.columns:
            raise ValueError(
                "DataFrame must contain canonical_attribute column."
            )

        rows = []

        group_cols = ["trial_id"]

        if "criteria_type" in df.columns:
            group_cols.append("criteria_type")

        for group_key, group in df.groupby(group_cols):
            records = group.reset_index(drop=True)

            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    a = str(records.loc[i, "canonical_attribute"])
                    b = str(records.loc[j, "canonical_attribute"])

                    score = fuzz.token_sort_ratio(a, b)

                    if score >= threshold:
                        rows.append(
                            {
                                "group": group_key,
                                "row_i": i,
                                "row_j": j,
                                "attribute_i": a,
                                "attribute_j": b,
                                "similarity": score,
                            }
                        )

        return pd.DataFrame(rows)

    @staticmethod
    def criteria_to_dataframe(
        criteria: Iterable[EligibilityCriterion],
    ) -> pd.DataFrame:
        """
        Convert raw EligibilityCriterion objects to a DataFrame
        without normalization.
        """

        return pd.DataFrame(
            [c.model_dump() for c in criteria]
        )