"""Generic terminology normalization and duplicate-review utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from rapidfuzz import fuzz

from backend.criteria_processor.eav_schema import canonical_entity
from backend.criteria_processor.terminology import TerminologyRepository
from backend.criteria_processor.text import clean_attribute_key
from backend.criteria_processor.value_normalization import normalize_structured_value


class CriterionModel(Protocol):
    """Structural boundary for Pydantic-like extracted criterion models."""

    def model_dump(self) -> dict[str, Any]:
        """Return the criterion as a mutable row mapping."""


CriterionInput = CriterionModel | Mapping[str, Any]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DICTIONARY_DIR = _PROJECT_ROOT / "config" / "dictionaries"


def _load_dictionary(name: str) -> dict[str, str]:
    path = _DEFAULT_DICTIONARY_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _criterion_row(criterion: CriterionInput) -> dict[str, Any]:
    if isinstance(criterion, Mapping):
        return dict(criterion)
    return criterion.model_dump()


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


class CriteriaNormalizer:
    def __init__(
        self,
        entity_map: dict | None = None,
        attribute_map: dict | None = None,
        disease_map: dict | None = None,
        terminology_repository: TerminologyRepository | None = None,
    ):
        self.entity_map = (
            entity_map
            if entity_map is not None
            else _load_dictionary("entity_map.json")
        )
        self.attribute_map = (
            attribute_map
            if attribute_map is not None
            else _load_dictionary("attribute_map.json")
        )
        self.disease_map = (
            disease_map
            if disease_map is not None
            else _load_dictionary("disease_map.json")
        )

        self.entity_map = self._clean_map_keys(self.entity_map)
        self.attribute_map = self._clean_map_keys(self.attribute_map)
        self.disease_map = self._clean_map_keys(self.disease_map)
        self.terminology_repository = terminology_repository

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

        mapped = self.entity_map.get(key, entity)
        return canonical_entity(mapped)

    def normalize_attribute(self, attribute: str | None) -> str | None:
        if attribute is None:
            return None

        key = clean_attribute_key(attribute)

        if key in self.attribute_map:
            return self.attribute_map[key]

        if key in self.disease_map:
            return self.disease_map[key]

        return attribute

    def normalize_value(
        self,
        attribute: str | None,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value_clean = clean_attribute_key(value)
        attribute_clean = clean_attribute_key(attribute)

        if attribute_clean == "gender":
            all_gender_values = {
                "all",
                "all genders",
                "both",
                "both sexes",
                "both genders",
                "male or female",
                "female or male",
                "male and female",
                "female and male",
                "men and women",
                "women and men",
                "males and females",
                "females and males",
            }

            male_values = {
                "male",
                "males",
                "men",
            }

            female_values = {
                "female",
                "females",
                "women",
            }

            if value_clean in all_gender_values:
                return "All"

            if value_clean in male_values:
                return "Male"

            if value_clean in female_values:
                return "Female"

        return value

    def normalize_criterion(
        self,
        criterion: CriterionInput,
    ) -> dict:
        row = _criterion_row(criterion)

        entity = _first_present(row, "entity", "raw_entity")
        attribute = _first_present(row, "attribute", "raw_attribute")
        value = _first_present(row, "value", "raw_value")

        row["entity"] = entity
        row["attribute"] = attribute
        row["value"] = value
        row["raw_entity"] = entity
        row["raw_attribute"] = attribute
        row["raw_value"] = value
        if "modifier" not in row and "condition" in row:
            row["modifier"] = row["condition"]

        terminology_mapping = None
        if self.terminology_repository is not None:
            terminology_mapping = (
                self.terminology_repository.map_criterion(
                    attribute=row["attribute"],
                    value=row["value"],
                    entity=row["entity"],
                )
            )

        row["canonical_entity"] = (
            terminology_mapping.canonical_entity
            if (
                terminology_mapping is not None
                and terminology_mapping.canonical_entity is not None
            )
            else self.normalize_entity(row["entity"])
        )
        row["canonical_attribute"] = (
            terminology_mapping.canonical_attribute
            if (
                terminology_mapping is not None
                and terminology_mapping.canonical_attribute is not None
            )
            else self.normalize_attribute(row["attribute"])
        )
        canonical_value = (
            terminology_mapping.canonical_value
            if terminology_mapping is not None
            else self.normalize_value(
                row["attribute"],
                row["value"],
            )
        )
        row["canonical_value"] = canonical_value

        row["canonical_attribute_id"] = (
            terminology_mapping.attribute_id
            if terminology_mapping is not None
            else None
        )
        row["canonical_value_id"] = (
            terminology_mapping.value_id
            if terminology_mapping is not None
            else None
        )
        row["terminology_version"] = (
            terminology_mapping.terminology_version
            if terminology_mapping is not None
            else None
        )
        row["mapping_method"] = (
            terminology_mapping.mapping_method
            if terminology_mapping is not None
            else "legacy_dictionary"
        )
        row["mapping_status"] = (
            terminology_mapping.mapping_status
            if terminology_mapping is not None
            else "legacy"
        )
        row["mapping_confidence"] = (
            terminology_mapping.confidence
            if terminology_mapping is not None
            else None
        )
        value_schema = (
            terminology_mapping.value_schema
            if (
                terminology_mapping is not None
                and terminology_mapping.value_schema
            )
            else "free_text"
        )
        canonical_unit = (
            terminology_mapping.canonical_unit
            if terminology_mapping is not None
            else None
        )
        structured_value = normalize_structured_value(
            raw_value=row["value"],
            attribute_id=row["canonical_attribute_id"],
            canonical_attribute=row["canonical_attribute"],
            value_schema=value_schema,
            canonical_unit=canonical_unit,
            canonical_value=canonical_value,
            context_text="; ".join(
                str(item).strip()
                for item in [
                    row.get("condition"),
                    row.get("source_text"),
                ]
                if item is not None and str(item).strip()
            ),
            criterion_type=_first_present(
                row,
                "criterion_type",
                "criteria_type",
            ),
        )
        row["value_schema"] = structured_value["schema"]
        row["canonical_unit"] = canonical_unit
        row["normalized_value_json"] = json.dumps(
            structured_value,
            ensure_ascii=False,
            sort_keys=True,
        )
        row["value_category"] = structured_value["category"]
        row["value_lower"] = structured_value["lower"]
        row["value_upper"] = structured_value["upper"]
        row["value_lower_inclusive"] = structured_value["lower_inclusive"]
        row["value_upper_inclusive"] = structured_value["upper_inclusive"]
        row["value_unit"] = structured_value["unit"]
        row["value_lower_unit"] = structured_value["lower_unit"]
        row["value_upper_unit"] = structured_value["upper_unit"]
        row["normalized_value_lower"] = structured_value[
            "normalized_lower"
        ]
        row["normalized_value_upper"] = structured_value[
            "normalized_upper"
        ]
        row["normalized_value_unit"] = structured_value["normalized_unit"]
        row["selection_effect"] = structured_value["selection_effect"]
        row["value_assertion"] = structured_value["assertion"]
        row["eligible_values_json"] = json.dumps(
            structured_value["eligible_values"],
            ensure_ascii=False,
        )
        row["excluded_values_json"] = json.dumps(
            structured_value["excluded_values"],
            ensure_ascii=False,
        )
        row["eligible_ranges_json"] = json.dumps(
            structured_value["eligible_ranges"],
            ensure_ascii=False,
            sort_keys=True,
        )
        row["excluded_ranges_json"] = json.dumps(
            structured_value["excluded_ranges"],
            ensure_ascii=False,
            sort_keys=True,
        )
        row["value_parse_status"] = structured_value["parse_status"]
        if (
            row["mapping_status"] == "mapped"
            and value_schema not in {"free_text", "categorical"}
            and structured_value["parse_status"] != "parsed"
        ):
            row["mapping_status"] = "needs_value_review"
        if structured_value["category"] is not None:
            row["canonical_value"] = structured_value["category"]

        row["attribute_key"] = clean_attribute_key(row["attribute"])
        row["canonical_attribute_key"] = clean_attribute_key(
            row["canonical_attribute"]
        )
        row["canonical_value_key"] = clean_attribute_key(row["canonical_value"])

        return row

    def normalize_many(
        self,
        criteria: Iterable[CriterionInput],
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
            "canonical_value_key",
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

    def find_unmapped_terms(
    self,
    df: pd.DataFrame,
    min_count: int = 1,
    ) -> pd.DataFrame:
        """
        Return extracted attributes that are not covered by the attribute
        or disease dictionaries.
        """

        required_cols = {
            "attribute",
            "canonical_attribute",
            "attribute_key",
        }

        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "mapping_status" in df.columns:
            unmapped = df[
                ~df["mapping_status"].isin({"mapped", "legacy"})
            ].copy()
        else:
            mapped_keys = (
                set(self.attribute_map.keys())
                | set(self.disease_map.keys())
            )
            unmapped = df[
                ~df["attribute_key"].isin(mapped_keys)
            ].copy()

        counts = (
            unmapped
            .groupby(
                [
                    "attribute",
                    "canonical_attribute",
                    "attribute_key",
                ]
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        return counts[counts["count"] >= min_count].reset_index(drop=True)

    def build_terminology_review_queue(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.terminology_repository is None:
            return pd.DataFrame()
        return self.terminology_repository.build_review_queue(df)

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
        criteria: Iterable[CriterionInput],
    ) -> pd.DataFrame:
        """
        Convert criterion models or row mappings to a DataFrame unchanged.
        """

        return pd.DataFrame(
            [_criterion_row(c) for c in criteria]
        )
