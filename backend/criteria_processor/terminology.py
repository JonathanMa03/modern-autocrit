"""Validated, read-only access to the controlled EAV attribute library."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from backend.criteria_processor.eav_schema import ALLOWED_ENTITIES, canonical_entity
from backend.criteria_processor.models import (
    TerminologyAttribute,
    TerminologyMapping,
    TerminologyRelationship,
    TerminologySuggestion,
    TerminologyValue,
)
from backend.criteria_processor.text import clean_attribute_key
from backend.criteria_processor.value_normalization import SUPPORTED_VALUE_SCHEMAS


REQUIRED_COLUMNS = {
    "Attributes": {
        "attribute_id",
        "canonical_name",
        "parent_entity",
        "value_schema",
        "canonical_unit",
        "description",
        "active",
    },
    "Values": {
        "value_id",
        "attribute_id",
        "canonical_value",
        "sort_order",
        "description",
        "active",
    },
    "Aliases": {
        "alias_id",
        "alias_text",
        "source_entity",
        "attribute_id",
        "value_id",
        "alias_type",
        "status",
        "source",
        "active",
        "notes",
    },
    "Relationships": {
        "relationship_id",
        "source_attribute_id",
        "target_attribute_id",
        "relationship_type",
        "status",
        "source",
        "notes",
    },
    "Metadata": {
        "key",
        "value",
        "description",
    },
}

REVIEW_QUEUE_COLUMNS = [
    "review_id",
    "raw_entity",
    "raw_attribute",
    "raw_value",
    "suggested_attribute_id",
    "suggested_value_id",
    "confidence",
    "occurrence_count",
    "example_trial_id",
    "example_sentence",
    "status",
    "reviewer",
    "notes",
]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().casefold() in {
        "1",
        "true",
        "yes",
        "y",
        "active",
    }


def _sort_order(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid attribute-library sort_order value: {value!r}"
        ) from exc


class TerminologyRepository:
    """Load and query an expert-maintained EAV attribute workbook.

    The workbook contains canonical attributes, attribute-specific value
    definitions, aliases, and relationships. It is never modified during
    extraction; only active, approved rows participate in normalization.
    """

    def __init__(
        self,
        workbook_path: str | Path,
        *,
        fuzzy_threshold: int = 90,
        max_prompt_attributes: int = 8,
        enable_prompt_retrieval: bool = True,
    ) -> None:
        self.workbook_path = Path(workbook_path).resolve()
        self.fuzzy_threshold = fuzzy_threshold
        self.max_prompt_attributes = max_prompt_attributes
        self.enable_prompt_retrieval = enable_prompt_retrieval

        self.attributes: dict[str, TerminologyAttribute] = {}
        self.values: dict[str, TerminologyValue] = {}
        self.relationships: list[TerminologyRelationship] = []
        self.terminology_version = ""

        self._attribute_lookup: dict[tuple[str, str], str] = {}
        self._value_lookup: dict[str, dict[str, str]] = {}
        self._retrieval_aliases: dict[
            tuple[str, str],
            tuple[str, str | None],
        ] = {}

        self._load()

    def _load(self) -> None:
        if not self.workbook_path.is_file():
            raise FileNotFoundError(
                f"Attribute library not found: {self.workbook_path}"
            )

        try:
            frames = pd.read_excel(
                self.workbook_path,
                sheet_name=list(REQUIRED_COLUMNS),
                dtype=object,
                engine="openpyxl",
            )
        except ValueError as exc:
            raise ValueError(
                f"Attribute library is missing a required sheet: {exc}"
            ) from exc

        self._validate_columns(frames)
        self._load_metadata(frames["Metadata"])
        self._load_attributes(frames["Attributes"])
        self._load_values(frames["Values"])
        self._load_aliases(frames["Aliases"])
        self._load_relationships(frames["Relationships"])

    @staticmethod
    def _validate_columns(frames: dict[str, pd.DataFrame]) -> None:
        for sheet_name, required in REQUIRED_COLUMNS.items():
            actual = set(frames[sheet_name].columns)
            missing = required - actual
            if missing:
                raise ValueError(
                    f"Attribute-library sheet '{sheet_name}' is missing "
                    f"columns: {sorted(missing)}"
                )

    def _load_metadata(self, frame: pd.DataFrame) -> None:
        metadata = {
            _text(row["key"]): _text(row["value"])
            for _, row in frame.iterrows()
            if _text(row["key"])
        }
        self.terminology_version = metadata.get(
            "terminology_version",
            "",
        )
        if not self.terminology_version:
            raise ValueError(
                "Attribute-library Metadata must define terminology_version."
            )

    def _load_attributes(self, frame: pd.DataFrame) -> None:
        for _, row in frame.iterrows():
            attribute_id = _text(row["attribute_id"])
            if not attribute_id:
                continue
            if attribute_id in self.attributes:
                raise ValueError(
                    f"Duplicate attribute_id: {attribute_id}"
                )

            value_schema = _text(row["value_schema"]).casefold()
            if value_schema not in SUPPORTED_VALUE_SCHEMAS:
                raise ValueError(
                    f"Attribute '{attribute_id}' uses unsupported value_schema "
                    f"'{value_schema}'. Supported schemas: "
                    f"{sorted(SUPPORTED_VALUE_SCHEMAS)}"
                )
            parent_entity = _text(row["parent_entity"])
            if parent_entity not in ALLOWED_ENTITIES:
                raise ValueError(
                    f"Attribute '{attribute_id}' uses uncontrolled parent "
                    f"entity '{parent_entity}'. Allowed entities: "
                    f"{list(ALLOWED_ENTITIES)}"
                )
            attribute = TerminologyAttribute(
                attribute_id=attribute_id,
                canonical_name=_text(row["canonical_name"]),
                parent_entity=parent_entity,
                value_schema=value_schema,
                canonical_unit=_text(row["canonical_unit"]).casefold(),
                description=_text(row["description"]),
                active=_active(row["active"]),
            )
            if not attribute.canonical_name:
                raise ValueError(
                    f"Attribute '{attribute_id}' has no canonical_name."
                )
            self.attributes[attribute_id] = attribute

            if attribute.active:
                self._register_attribute_lookup(
                    clean_attribute_key(attribute.canonical_name),
                    attribute_id,
                    attribute.parent_entity,
                )
                self._register_attribute_lookup(
                    clean_attribute_key(attribute_id),
                    attribute_id,
                    attribute.parent_entity,
                )

    def _load_values(self, frame: pd.DataFrame) -> None:
        for _, row in frame.iterrows():
            value_id = _text(row["value_id"])
            if not value_id:
                continue
            if value_id in self.values:
                raise ValueError(
                    f"Duplicate value_id: {value_id}"
                )

            attribute_id = _text(row["attribute_id"])
            if attribute_id not in self.attributes:
                raise ValueError(
                    f"Value '{value_id}' references unknown attribute "
                    f"'{attribute_id}'."
                )
            if self.attributes[attribute_id].value_schema != "categorical":
                raise ValueError(
                    f"Value '{value_id}' belongs to non-categorical attribute "
                    f"'{attribute_id}'."
                )

            value = TerminologyValue(
                value_id=value_id,
                attribute_id=attribute_id,
                canonical_value=_text(row["canonical_value"]),
                sort_order=_sort_order(row["sort_order"]),
                description=_text(row["description"]),
                active=_active(row["active"]),
            )
            if not value.canonical_value:
                raise ValueError(
                    f"Value '{value_id}' has no canonical_value."
                )
            self.values[value_id] = value

            if value.active and self.attributes[attribute_id].active:
                attribute_values = self._value_lookup.setdefault(
                    attribute_id,
                    {},
                )
                attribute_values[
                    clean_attribute_key(value.canonical_value)
                ] = value_id
                attribute_values[
                    clean_attribute_key(value_id)
                ] = value_id

    def _load_aliases(self, frame: pd.DataFrame) -> None:
        alias_ids: set[str] = set()
        for _, row in frame.iterrows():
            alias_id = _text(row["alias_id"])
            if not alias_id:
                continue
            if alias_id in alias_ids:
                raise ValueError(f"Duplicate alias_id: {alias_id}")
            alias_ids.add(alias_id)

            alias_text = _text(row["alias_text"])
            attribute_id = _text(row["attribute_id"])
            source_entity = _text(row["source_entity"])
            value_id = _text(row["value_id"]) or None
            alias_type = _text(row["alias_type"]).casefold()
            status = _text(row["status"]).casefold()

            if not alias_text:
                raise ValueError(f"Alias '{alias_id}' has no alias_text.")
            if attribute_id not in self.attributes:
                raise ValueError(
                    f"Alias '{alias_id}' references unknown attribute "
                    f"'{attribute_id}'."
                )
            if not source_entity:
                source_entity = self.attributes[attribute_id].parent_entity
            if alias_type not in {"attribute", "value"}:
                raise ValueError(
                    f"Alias '{alias_id}' has invalid alias_type "
                    f"'{alias_type}'."
                )
            if alias_type == "value":
                if value_id not in self.values:
                    raise ValueError(
                        f"Alias '{alias_id}' references unknown value "
                        f"'{value_id}'."
                    )
                if self.values[value_id].attribute_id != attribute_id:
                    raise ValueError(
                        f"Alias '{alias_id}' has mismatched attribute/value "
                        "references."
                    )

            if (
                status != "approved"
                or not _active(row["active"])
                or not self.attributes[attribute_id].active
            ):
                continue

            alias_key = clean_attribute_key(alias_text)
            if not alias_key:
                continue
            if alias_type == "attribute":
                self._register_attribute_lookup(
                    alias_key,
                    attribute_id,
                    source_entity,
                )
            else:
                attribute_values = self._value_lookup.setdefault(
                    attribute_id,
                    {},
                )
                previous_value = attribute_values.get(alias_key)
                if previous_value is not None and previous_value != value_id:
                    raise ValueError(
                        f"Alias '{alias_text}' maps to conflicting values "
                        f"for attribute '{attribute_id}'."
                    )
                attribute_values[alias_key] = value_id

            retrieval_key = (
                clean_attribute_key(
                    source_entity
                ),
                alias_key,
            )
            previous_retrieval = self._retrieval_aliases.get(retrieval_key)
            target = (attribute_id, value_id)
            if (
                previous_retrieval is not None
                and previous_retrieval[0] != attribute_id
            ):
                raise ValueError(
                    f"Alias '{alias_text}' has conflicting approved targets."
                )
            if previous_retrieval is None or value_id is None:
                self._retrieval_aliases[retrieval_key] = target

    def _load_relationships(self, frame: pd.DataFrame) -> None:
        relationship_ids: set[str] = set()
        for _, row in frame.iterrows():
            relationship_id = _text(row["relationship_id"])
            if not relationship_id:
                continue
            if relationship_id in relationship_ids:
                raise ValueError(
                    f"Duplicate relationship_id: {relationship_id}"
                )
            relationship_ids.add(relationship_id)

            source_id = _text(row["source_attribute_id"])
            target_id = _text(row["target_attribute_id"])
            if source_id not in self.attributes:
                raise ValueError(
                    f"Relationship '{relationship_id}' references unknown "
                    f"source attribute '{source_id}'."
                )
            if target_id not in self.attributes:
                raise ValueError(
                    f"Relationship '{relationship_id}' references unknown "
                    f"target attribute '{target_id}'."
                )

            relationship = TerminologyRelationship(
                relationship_id=relationship_id,
                source_attribute_id=source_id,
                target_attribute_id=target_id,
                relationship_type=_text(
                    row["relationship_type"]
                ).casefold(),
                status=_text(row["status"]).casefold(),
            )
            if relationship.status == "approved":
                self.relationships.append(relationship)

    def _register_attribute_lookup(
        self,
        alias_key: str,
        attribute_id: str,
        parent_entity: str,
    ) -> None:
        lookup_key = (clean_attribute_key(parent_entity), alias_key)
        previous = self._attribute_lookup.get(lookup_key)
        if previous is not None and previous != attribute_id:
            raise ValueError(
                f"Attribute alias '{alias_key}' maps to conflicting "
                f"attributes under parent entity '{parent_entity}'."
            )
        self._attribute_lookup[lookup_key] = attribute_id

    def map_criterion(
        self,
        attribute: str | None,
        value: str | None,
        entity: str | None = None,
    ) -> TerminologyMapping:
        """Map one extracted EAV item to an approved attribute and value."""

        raw_entity = _text(entity) or None
        raw_attribute = _text(attribute)
        raw_value = _text(value) or None
        attribute_key = clean_attribute_key(raw_attribute)
        entity_key = clean_attribute_key(raw_entity)
        canonical_entity_key = clean_attribute_key(
            canonical_entity(raw_entity)
        )
        attribute_id = (
            self._attribute_lookup.get((entity_key, attribute_key))
            if entity_key
            else None
        )
        if attribute_id is None and canonical_entity_key:
            attribute_id = self._attribute_lookup.get(
                (canonical_entity_key, attribute_key)
            )
        if attribute_id is None:
            matching_ids = {
                candidate_id
                for (_parent_key, alias_key), candidate_id
                in self._attribute_lookup.items()
                if alias_key == attribute_key
            }
            if len(matching_ids) == 1:
                attribute_id = next(iter(matching_ids))

        if attribute_id is None:
            return TerminologyMapping(
                raw_attribute=raw_attribute,
                raw_value=raw_value,
                canonical_entity=canonical_entity(raw_entity),
                canonical_attribute=raw_attribute or None,
                canonical_value=raw_value,
                terminology_version=self.terminology_version,
            )

        definition = self.attributes[attribute_id]
        mapping_method = (
            "canonical_label"
            if attribute_key == clean_attribute_key(definition.canonical_name)
            else "exact_alias"
        )
        common = {
            "raw_attribute": raw_attribute,
            "raw_value": raw_value,
            "attribute_id": attribute_id,
            "canonical_entity": definition.parent_entity,
            "canonical_attribute": definition.canonical_name,
            "value_schema": definition.value_schema,
            "canonical_unit": definition.canonical_unit or None,
            "mapping_method": mapping_method,
            "confidence": 1.0,
            "terminology_version": self.terminology_version,
        }
        if definition.value_schema != "categorical":
            return TerminologyMapping(
                **common,
                canonical_value=raw_value,
                mapping_status="mapped",
            )

        value_key = clean_attribute_key(raw_value)
        value_id = self._value_lookup.get(attribute_id, {}).get(value_key)
        if value_id is None:
            return TerminologyMapping(
                **common,
                canonical_value=raw_value,
                mapping_status="needs_value_review",
            )

        canonical_value = self.values[value_id]
        return TerminologyMapping(
            **{
                **common,
                "mapping_method": f"{mapping_method}+exact_value",
            },
            value_id=value_id,
            canonical_value=canonical_value.canonical_value,
            mapping_status="mapped",
        )

    def attribute_lookup_items(
        self,
    ) -> list[tuple[str, TerminologyAttribute]]:
        """Return approved cleaned attribute lookup keys and definitions."""

        return [
            (alias_key, self.attributes[attribute_id])
            for (_parent_key, alias_key), attribute_id
            in self._attribute_lookup.items()
            if attribute_id in self.attributes
        ]

    def value_lookup_items(
        self,
        attribute_id: str,
    ) -> list[tuple[str, TerminologyValue]]:
        """Return approved cleaned value lookup keys for one attribute."""

        return [
            (lookup_key, self.values[value_id])
            for lookup_key, value_id in self._value_lookup.get(
                attribute_id,
                {},
            ).items()
            if value_id in self.values
        ]

    def build_prompt_context(self, criteria_text: str) -> str | None:
        if not self.enable_prompt_retrieval:
            return None
        text_key = clean_attribute_key(criteria_text)
        if not text_key:
            return None

        matched: list[tuple[int, str]] = []
        for (_parent_key, alias_key), (attribute_id, value_id) in (
            self._retrieval_aliases.items()
        ):
            if value_id is not None and len(alias_key) < 5:
                continue
            if self._contains_phrase(text_key, alias_key):
                matched.append((len(alias_key), attribute_id))

        if not matched:
            for (_parent_key, alias_key), (attribute_id, _value_id) in (
                self._retrieval_aliases.items()
            ):
                if len(alias_key) < 5:
                    continue
                score = fuzz.partial_ratio(alias_key, text_key)
                if score >= self.fuzzy_threshold:
                    matched.append((int(score), attribute_id))

        attribute_ids: list[str] = []
        for _score, attribute_id in sorted(matched, reverse=True):
            if attribute_id not in attribute_ids:
                attribute_ids.append(attribute_id)
            if len(attribute_ids) >= self.max_prompt_attributes:
                break
        if not attribute_ids:
            return None

        lines = [
            "Approved EAV attributes from the controlled workbook:",
        ]
        for attribute_id in attribute_ids:
            definition = self.attributes[attribute_id]
            values = sorted(
                (
                    item
                    for item in self.values.values()
                    if item.attribute_id == attribute_id and item.active
                ),
                key=lambda item: (item.sort_order, item.canonical_value),
            )
            value_text = (
                ", ".join(
                    f"{item.canonical_value} [{item.value_id}]"
                    for item in values
                )
                if values
                else definition.value_schema
            )
            lines.append(
                f"- {definition.canonical_name} [{attribute_id}] "
                f"(Parent entity: {definition.parent_entity}; "
                f"value schema: {definition.value_schema}; "
                f"unit: {definition.canonical_unit or 'none'}; "
                f"allowed values/representation: {value_text})"
            )
            for relationship in self.relationships:
                if relationship.source_attribute_id == attribute_id:
                    target = self.attributes[
                        relationship.target_attribute_id
                    ]
                    lines.append(
                        f"  Relationship: {relationship.relationship_type} "
                        f"{target.canonical_name} [{target.attribute_id}]"
                    )

        lines.extend(
            [
                "Use a canonical attribute and value exactly as shown only "
                "when the source meaning matches.",
                "Do not put eligibility wording, comparison operators, cutoffs, "
                "ranges, or categories in the attribute name; those belong to "
                "the value representation.",
                "Do not force an unmatched item into these candidates.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _contains_phrase(text_key: str, alias_key: str) -> bool:
        if not alias_key:
            return False
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(alias_key)}(?![a-z0-9])",
                text_key,
            )
            is not None
        )

    def suggest_mapping(
        self,
        attribute: str | None,
        value: str | None,
        entity: str | None = None,
    ) -> TerminologySuggestion:
        attribute_key = clean_attribute_key(attribute)
        if not attribute_key or not self._attribute_lookup:
            return TerminologySuggestion()

        entity_key = clean_attribute_key(entity)
        lookup = {
            alias_key: attribute_id
            for (parent_key, alias_key), attribute_id
            in self._attribute_lookup.items()
            if not entity_key or parent_key == entity_key
        }
        exact_attribute_id = lookup.get(attribute_key)
        if exact_attribute_id is not None:
            attribute_id = exact_attribute_id
            attribute_score = 100.0
        else:
            match = process.extractOne(
                attribute_key,
                list(lookup),
                scorer=fuzz.WRatio,
            )
            if match is None:
                return TerminologySuggestion()
            alias_key, attribute_score, _index = match
            if attribute_score < self.fuzzy_threshold:
                return TerminologySuggestion()
            attribute_id = lookup[alias_key]

        value_id: str | None = None
        value_score = 0.0
        value_key = clean_attribute_key(value)
        value_lookup = self._value_lookup.get(attribute_id, {})
        if value_key and value_lookup:
            exact_value_id = value_lookup.get(value_key)
            if exact_value_id is not None:
                value_id = exact_value_id
                value_score = 100.0
            else:
                value_match = process.extractOne(
                    value_key,
                    list(value_lookup),
                    scorer=fuzz.WRatio,
                )
                if value_match is not None:
                    matched_value_key, value_score, _index = value_match
                    if value_score >= self.fuzzy_threshold:
                        value_id = value_lookup[matched_value_key]

        combined_score = (
            min(attribute_score, value_score)
            if value_id is not None
            else attribute_score
        )
        return TerminologySuggestion(
            attribute_id=attribute_id,
            value_id=value_id,
            confidence=round(combined_score / 100.0, 4),
        )

    def build_review_queue(
        self,
        normalized_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if normalized_df.empty or "mapping_status" not in normalized_df:
            return pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

        pending = normalized_df[
            normalized_df["mapping_status"] != "mapped"
        ].copy()
        if pending.empty:
            return pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

        rows: list[dict[str, Any]] = []
        group_columns = ["raw_entity", "raw_attribute", "raw_value"]
        for group_key, group in pending.groupby(
            group_columns,
            dropna=False,
            sort=True,
        ):
            raw_entity, raw_attribute, raw_value = group_key
            suggestion = self.suggest_mapping(
                _text(raw_attribute),
                _text(raw_value),
                _text(raw_entity),
            )
            stable_key = "|".join(
                [
                    _text(raw_entity),
                    _text(raw_attribute),
                    _text(raw_value),
                ]
            )
            review_id = (
                "review_"
                + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:12]
            )
            first = group.iloc[0]
            rows.append(
                {
                    "review_id": review_id,
                    "raw_entity": _text(raw_entity),
                    "raw_attribute": _text(raw_attribute),
                    "raw_value": _text(raw_value),
                    "suggested_attribute_id": suggestion.attribute_id or "",
                    "suggested_value_id": suggestion.value_id or "",
                    "confidence": suggestion.confidence,
                    "occurrence_count": len(group),
                    "example_trial_id": _text(first.get("trial_id")),
                    "example_sentence": _text(
                        first.get("source_sentence")
                    ),
                    "status": "pending",
                    "reviewer": "",
                    "notes": "",
                }
            )
        return pd.DataFrame(rows, columns=REVIEW_QUEUE_COLUMNS)
