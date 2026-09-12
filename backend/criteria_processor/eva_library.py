"""Revisioned JSON terminology library for eligibility EVA normalization."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.criteria_processor.atomic_io import atomic_write_json, atomic_write_text
from backend.criteria_processor.eva_models import (
    EvaApplyResolution,
    EvaAttributeDefinition,
    EvaAuditDocument,
    EvaEntityDefinition,
    EvaLibraryApplyReceipt,
    EvaLibraryApplyResult,
    EvaReconciliationDecisionInput,
    EvaReconciliationSummary,
)
from backend.criteria_processor.eva_reconciliation import (
    proposal_fingerprint,
    validate_reconciliation_decisions_locked,
)
from backend.criteria_processor.text import clean_attribute_key


LIBRARY_SCHEMA_VERSION = "eligcrit.eligibility_eva_library.v1"
_UPDATE_LOCK = threading.Lock()
_POINT_VALUE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_RANGE_VALUE = re.compile(
    r"^[\[(]\s*(?:[+-]?(?:\d+(?:\.\d+)?)|[-+]inf)\s*,\s*"
    r"(?:[+-]?(?:\d+(?:\.\d+)?)|[-+]inf)\s*[\])]$",
    flags=re.IGNORECASE,
)


class EvaLibraryStaleError(ValueError):
    """The audit snapshot no longer matches the live library."""


class EvaAttributeIdConflictError(ValueError):
    """A reviewed Attribute ID collides with another definition."""


class EvaApplyReceiptConflictError(ValueError):
    """An audit_id was already committed with different input."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_slug(value: str, *, fallback: str) -> str:
    source = str(value or "").strip().casefold().replace("_", " ")
    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        source,
    ).strip("_")
    return slug or fallback


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _unique_text(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = clean_attribute_key(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def library_sha256(payload: Mapping[str, Any]) -> str:
    # A receipt is embedded in the payload it identifies. Blank its own
    # new-SHA slot before hashing to avoid an impossible self-referential
    # digest while keeping every other receipt field integrity-protected.
    normalized = deepcopy(dict(payload))
    for row in normalized.get("audit_history", []):
        if (
            row.get("receipt_schema_version")
            == "eligcrit.eligibility_eva_apply_receipt.v1"
        ):
            row["new_library_sha256"] = ""
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _apply_input_fingerprint(
    audit: EvaAuditDocument,
    decisions: Sequence[EvaReconciliationDecisionInput],
    trusted_semantic_preview: Mapping[str, Any] | None,
) -> str:
    semantic_payload = (
        trusted_semantic_preview.model_dump(mode="json")
        if hasattr(trusted_semantic_preview, "model_dump")
        else trusted_semantic_preview
    )
    minimal_decisions = sorted(
        [
            (
                decision
                if isinstance(decision, EvaReconciliationDecisionInput)
                else EvaReconciliationDecisionInput.model_validate(decision)
            ).model_dump(mode="json")
            for decision in decisions
        ],
        key=lambda row: (row["eva_id"], row["decision"]),
    )
    items = sorted(
        [
            {
                "eva_id": item.eva_id,
                "proposal_fingerprint": proposal_fingerprint(item),
                "review_status": item.review_status,
                "categorical_value": item.categorical_value,
                "numerical_value": item.numerical_value,
                "confidence": item.confidence,
            }
            for item in audit.items
        ],
        key=lambda row: row["eva_id"],
    )
    return library_sha256(
        {
            "audit_id": audit.audit_id,
            "items": items,
            "decisions": minimal_decisions,
            "trusted_semantic_preview": semantic_payload,
        }
    )


class EvaLibraryRepository:
    """Load, query, and explicitly update the eligibility EVA library."""

    def __init__(
        self,
        path: str | Path,
        *,
        snapshot_directory: str | Path | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.snapshot_directory = (
            Path(snapshot_directory).resolve()
            if snapshot_directory is not None
            else None
        )
        self.payload: dict[str, Any] = {}
        self.entities: dict[str, EvaEntityDefinition] = {}
        self.attributes: dict[str, EvaAttributeDefinition] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Eligibility EVA library not found: {self.path}"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Eligibility EVA library must be a JSON object.")
        if payload.get("schema_version") != LIBRARY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported eligibility EVA library schema_version."
            )
        revision = payload.get("revision")
        if not isinstance(revision, int) or revision < 0:
            raise ValueError("Eligibility EVA library revision is invalid.")

        entities: dict[str, EvaEntityDefinition] = {}
        entity_names: dict[str, str] = {}
        for raw in payload.get("entities") or []:
            entity = EvaEntityDefinition.model_validate(raw)
            if entity.entity_id in entities:
                raise ValueError(
                    f"Duplicate EVA entity_id: {entity.entity_id}"
                )
            name_key = clean_attribute_key(entity.canonical_name)
            if name_key in entity_names:
                raise ValueError(
                    "Duplicate EVA Entity canonical_name: "
                    f"{entity.canonical_name}"
                )
            entities[entity.entity_id] = entity
            entity_names[name_key] = entity.entity_id

        attributes: dict[str, EvaAttributeDefinition] = {}
        attribute_names: dict[tuple[str, str], str] = {}
        for raw in payload.get("attributes") or []:
            attribute = EvaAttributeDefinition.model_validate(raw)
            if attribute.attribute_id in attributes:
                raise ValueError(
                    f"Duplicate EVA attribute_id: {attribute.attribute_id}"
                )
            if attribute.entity_id not in entities:
                raise ValueError(
                    f"EVA Attribute '{attribute.attribute_id}' references "
                    f"unknown Entity '{attribute.entity_id}'."
                )
            key = (
                attribute.entity_id,
                clean_attribute_key(attribute.canonical_name),
            )
            if key in attribute_names:
                raise ValueError(
                    "Duplicate EVA Attribute canonical_name within Entity: "
                    f"{attribute.canonical_name}"
                )
            if (
                attribute.value_type in {"Categorical", "SexGender"}
                and attribute.numerical_type is not None
            ):
                raise ValueError(
                    f"{attribute.value_type} Attribute "
                    f"'{attribute.attribute_id}' "
                    "cannot define numerical_type."
                )
            if (
                attribute.value_type == "Numerical"
                and attribute.numerical_type is None
            ):
                raise ValueError(
                    f"Numerical Attribute '{attribute.attribute_id}' "
                    "must define Range or Point."
                )
            attributes[attribute.attribute_id] = attribute
            attribute_names[key] = attribute.attribute_id

        fixed_values = payload.get("value_definitions") or {}
        if fixed_values.get("Categorical") != ["Included", "Excluded"]:
            raise ValueError(
                "Categorical EVA values must be Included and Excluded."
            )
        if fixed_values.get("SexGender") != ["male", "female", "all"]:
            raise ValueError(
                "SexGender EVA values must be male, female, and all."
            )
        if fixed_values.get("Numerical") != ["Range", "Point"]:
            raise ValueError(
                "Numerical EVA types must be Range and Point."
            )

        self.payload = payload
        self.entities = entities
        self.attributes = attributes

    @property
    def revision(self) -> int:
        return int(self.payload["revision"])

    @property
    def sha256(self) -> str:
        return library_sha256(self.payload)

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload["schema_version"],
            "library_name": self.payload.get("library_name", ""),
            "revision": self.revision,
            "updated_at": self.payload.get("updated_at"),
            "value_definitions": deepcopy(
                self.payload.get("value_definitions") or {}
            ),
            "entities": [
                entity.model_dump(mode="json")
                for entity in self.entities.values()
                if entity.active
            ],
            "attributes": [
                attribute.model_dump(mode="json")
                for attribute in self.attributes.values()
                if attribute.active
            ],
            "entity_count": sum(
                1 for entity in self.entities.values() if entity.active
            ),
            "attribute_count": sum(
                1
                for attribute in self.attributes.values()
                if attribute.active
            ),
        }

    def read_raw_bytes(self) -> bytes:
        return self.path.read_bytes()

    def get_attribute(self, attribute_id: str) -> EvaAttributeDefinition:
        return self.attributes[attribute_id]

    def find_entity(
        self,
        *,
        entity_id: str = "",
        canonical_name: str = "",
    ) -> EvaEntityDefinition | None:
        if entity_id and entity_id in self.entities:
            return self.entities[entity_id]
        name_key = clean_attribute_key(canonical_name)
        if not name_key:
            return None
        for entity in self.entities.values():
            names = [entity.canonical_name, *entity.aliases]
            if any(clean_attribute_key(name) == name_key for name in names):
                return entity
        return None

    def find_attribute(
        self,
        *,
        attribute_id: str = "",
        entity_id: str = "",
        canonical_name: str = "",
    ) -> EvaAttributeDefinition | None:
        if attribute_id and attribute_id in self.attributes:
            return self.attributes[attribute_id]
        name_key = clean_attribute_key(canonical_name)
        if not name_key:
            return None
        matches = []
        for attribute in self.attributes.values():
            if entity_id and attribute.entity_id != entity_id:
                continue
            names = [attribute.canonical_name, *attribute.aliases]
            if any(clean_attribute_key(name) == name_key for name in names):
                matches.append(attribute)
        return matches[0] if len(matches) == 1 else None

    def apply_audit(
        self,
        audit: EvaAuditDocument,
        *,
        reconciliation_decisions: Sequence[
            EvaReconciliationDecisionInput
        ] = (),
        trusted_semantic_preview: Mapping[str, Any] | None = None,
    ) -> EvaLibraryApplyResult:
        """Merge reviewer-approved definitions and return normalized rows."""

        input_fingerprint = _apply_input_fingerprint(
            audit,
            reconciliation_decisions,
            trusted_semantic_preview,
        )
        with _UPDATE_LOCK:
            self.reload()
            receipts = [
                row
                for row in self.payload.get("audit_history", [])
                if row.get("receipt_schema_version")
                == "eligcrit.eligibility_eva_apply_receipt.v1"
                and row.get("audit_id") == audit.audit_id
            ]
            if receipts:
                matching = next(
                    (
                        row
                        for row in receipts
                        if row.get("input_fingerprint") == input_fingerprint
                    ),
                    None,
                )
                if matching is None:
                    raise EvaApplyReceiptConflictError(
                        "audit_id already has a different apply receipt."
                    )
                receipt = EvaLibraryApplyReceipt.model_validate(matching)
                approved = self._replay_receipt(audit, receipt)
                normalized = self._normalized_rows(approved)
                return EvaLibraryApplyResult(
                    audit=approved,
                    normalized_rows=normalized,
                    receipt=receipt,
                    replayed=True,
                    reconciliation_summary=receipt.reconciliation_summary,
                )
            if audit.library_revision != self.revision:
                raise EvaLibraryStaleError(
                    "Audit library revision is stale."
                )
            if audit.library_sha256 != self.sha256:
                raise EvaLibraryStaleError("Audit library SHA is stale.")
            resolved_audit, reconciliation_summary = (
                validate_reconciliation_decisions_locked(
                    audit=audit,
                    decisions=reconciliation_decisions,
                    library=self,
                    trusted_semantic_preview=trusted_semantic_preview,
                )
            )
            payload = deepcopy(self.payload)
            entities = {
                row["entity_id"]: row for row in payload["entities"]
            }
            attributes = {
                row["attribute_id"]: row for row in payload["attributes"]
            }
            normalized: list[dict[str, Any]] = []

            for item in resolved_audit.items:
                if item.review_status == "rejected":
                    continue
                self._validate_item_value(item)
                entity_id = self._merge_entity(item, entities)
                attribute_id = self._merge_attribute(
                    item,
                    entity_id=entity_id,
                    attributes=attributes,
                )
                normalized.append(
                    {
                        "eva_id": item.eva_id,
                        "trial_key": audit.trial_key,
                        "trial_id": audit.trial_id,
                        "criteria": item.criteria,
                        "source_item": item.item,
                        "context": item.context,
                        "entity_id": entity_id,
                        "entity": entities[entity_id]["canonical_name"],
                        "attribute_id": attribute_id,
                        "attribute": attributes[attribute_id][
                            "canonical_name"
                        ],
                        "value_type": item.value_type,
                        "categorical_value": item.categorical_value,
                        "numerical_type": item.numerical_type,
                        "numerical_value": item.numerical_value,
                        "unit": item.unit,
                        "confidence": item.confidence,
                        "rationale": item.rationale,
                        "review_status": "approved",
                    }
                )
                item.entity_id = entity_id
                item.attribute_id = attribute_id
                item.entity_name = entities[entity_id]["canonical_name"]
                item.attribute_name = attributes[attribute_id][
                    "canonical_name"
                ]
                item.mapping_decision = "existing_attribute"
                item.new_entity = False
                item.review_status = "approved"

            payload["entities"] = sorted(
                entities.values(),
                key=lambda row: (
                    clean_attribute_key(row["canonical_name"]),
                    row["entity_id"],
                ),
            )
            payload["attributes"] = sorted(
                attributes.values(),
                key=lambda row: (
                    row["entity_id"],
                    clean_attribute_key(row["canonical_name"]),
                    row["attribute_id"],
                ),
            )
            previous_revision = int(payload["revision"])
            previous_sha256 = self.sha256
            payload["revision"] = previous_revision + 1
            payload["updated_at"] = utc_now()
            resolutions = [
                EvaApplyResolution(
                    eva_id=item.eva_id,
                    entity_id=item.entity_id,
                    entity_name=item.entity_name,
                    attribute_id=item.attribute_id,
                    attribute_name=item.attribute_name,
                    mapping_decision=item.mapping_decision,
                    library_reconciliation=item.library_reconciliation,
                )
                for item in resolved_audit.items
                if item.review_status != "rejected"
            ]
            receipt = EvaLibraryApplyReceipt(
                audit_id=resolved_audit.audit_id,
                trial_key=resolved_audit.trial_key,
                trial_id=resolved_audit.trial_id,
                input_fingerprint=input_fingerprint,
                previous_revision=previous_revision,
                new_revision=payload["revision"],
                previous_library_sha256=previous_sha256,
                new_library_sha256="",
                resolutions=resolutions,
                reconciliation_summary=reconciliation_summary,
                approved_item_count=len(normalized),
                applied_at=payload["updated_at"],
            )
            self._write_snapshot(previous_revision, previous_sha256)
            payload.setdefault("audit_history", []).append(
                receipt.model_dump(mode="json")
            )
            receipt.new_library_sha256 = library_sha256(payload)
            payload["audit_history"][-1] = receipt.model_dump(mode="json")
            atomic_write_json(self.path, payload)
            self.reload()

            resolved_audit.review_status = "approved"
            resolved_audit.updated_at = receipt.applied_at
            resolved_audit.approved_at = receipt.applied_at
            resolved_audit.approved_library_revision = self.revision
            resolved_audit.approved_library_sha256 = self.sha256
            return EvaLibraryApplyResult(
                audit=resolved_audit,
                normalized_rows=normalized,
                receipt=receipt,
                replayed=False,
                reconciliation_summary=reconciliation_summary,
            )

    def _write_snapshot(self, revision: int, sha256: str) -> None:
        if self.snapshot_directory is None:
            return
        target = self.snapshot_directory / f"revision-{revision}-{sha256}.json"
        atomic_write_text(target, self.path.read_text(encoding="utf-8"))

    def _replay_receipt(
        self,
        audit: EvaAuditDocument,
        receipt: EvaLibraryApplyReceipt,
    ) -> EvaAuditDocument:
        approved = audit.model_copy(deep=True)
        by_id = {row.eva_id: row for row in receipt.resolutions}
        for item in approved.items:
            if item.review_status == "rejected":
                continue
            resolution = by_id.get(item.eva_id)
            if resolution is None:
                raise EvaApplyReceiptConflictError(
                    f"Receipt is missing resolution for {item.eva_id}."
                )
            item.entity_id = resolution.entity_id
            item.entity_name = resolution.entity_name
            item.attribute_id = resolution.attribute_id
            item.attribute_name = resolution.attribute_name
            item.mapping_decision = resolution.mapping_decision
            item.new_entity = False
            item.library_reconciliation = resolution.library_reconciliation
            item.review_status = "approved"
        approved.review_status = "approved"
        approved.updated_at = receipt.applied_at
        approved.approved_at = receipt.applied_at
        approved.approved_library_revision = receipt.new_revision
        approved.approved_library_sha256 = self.sha256
        return approved

    def _normalized_rows(
        self,
        audit: EvaAuditDocument,
    ) -> list[dict[str, Any]]:
        rows = []
        for item in audit.items:
            if item.review_status == "rejected":
                continue
            entity = self.entities.get(item.entity_id)
            attribute = self.attributes.get(item.attribute_id)
            rows.append(
                {
                    "eva_id": item.eva_id,
                    "trial_key": audit.trial_key,
                    "trial_id": audit.trial_id,
                    "criteria": item.criteria,
                    "source_item": item.item,
                    "context": item.context,
                    "entity_id": item.entity_id,
                    "entity": (
                        entity.canonical_name if entity else item.entity_name
                    ),
                    "attribute_id": item.attribute_id,
                    "attribute": (
                        attribute.canonical_name
                        if attribute
                        else item.attribute_name
                    ),
                    "value_type": item.value_type,
                    "categorical_value": item.categorical_value,
                    "numerical_type": item.numerical_type,
                    "numerical_value": item.numerical_value,
                    "unit": item.unit,
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                    "review_status": "approved",
                }
            )
        return rows

    def _merge_entity(
        self,
        item: Any,
        entities: dict[str, dict[str, Any]],
    ) -> str:
        if item.entity_id in entities:
            row = entities[item.entity_id]
            names = [row["canonical_name"], *row.get("aliases", [])]
            if any(
                clean_attribute_key(name)
                == clean_attribute_key(item.entity_name)
                for name in names
            ):
                return item.entity_id
        name_key = clean_attribute_key(item.entity_name)
        for row in entities.values():
            names = [row["canonical_name"], *row.get("aliases", [])]
            if any(
                clean_attribute_key(name) == name_key for name in names
            ):
                return row["entity_id"]

        canonical_name = item.entity_name.strip()
        if not canonical_name:
            raise ValueError(
                f"{item.eva_id}: Entity name is required."
            )
        desired_id = stable_slug(
            item.entity_id or canonical_name,
            fallback="entity",
        )
        entity_id = desired_id
        if entity_id in entities:
            entity_id = stable_id("entity", canonical_name)
        entities[entity_id] = {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "description": (
                f"Reviewer-approved eligibility Entity from "
                f"{item.source_item_id}."
            ),
            "aliases": [],
            "active": True,
        }
        return entity_id

    def _merge_attribute(
        self,
        item: Any,
        *,
        entity_id: str,
        attributes: dict[str, dict[str, Any]],
    ) -> str:
        reconciliation_keep_new = (
            item.library_reconciliation.status == "resolved"
            and item.library_reconciliation.decision == "KEEP_NEW"
            and item.library_reconciliation.target_attribute_id is None
            and bool(item.library_reconciliation.reason.strip())
        )
        existing_id = ""
        if not reconciliation_keep_new and item.attribute_id in attributes:
            candidate = attributes[item.attribute_id]
            names = [
                candidate["canonical_name"],
                *candidate.get("aliases", []),
            ]
            if (
                candidate["entity_id"] == entity_id
                and any(
                    clean_attribute_key(name)
                    == clean_attribute_key(item.attribute_name)
                    for name in names
                )
            ):
                existing_id = item.attribute_id
        if not reconciliation_keep_new and not existing_id:
            name_key = clean_attribute_key(item.attribute_name)
            matches = [
                row["attribute_id"]
                for row in attributes.values()
                if row["entity_id"] == entity_id
                and any(
                    clean_attribute_key(name) == name_key
                    for name in [
                        row["canonical_name"],
                        *row.get("aliases", []),
                    ]
                )
            ]
            if len(matches) == 1:
                existing_id = matches[0]
        if existing_id:
            row = attributes[existing_id]
            if row["value_type"] != item.value_type:
                raise ValueError(
                    f"{item.eva_id}: reviewed value type conflicts with "
                    f"existing Attribute '{existing_id}'."
                )
            if (
                item.value_type == "Numerical"
                and row.get("numerical_type") != item.numerical_type
            ):
                raise ValueError(
                    f"{item.eva_id}: reviewed numerical type conflicts with "
                    f"existing Attribute '{existing_id}'."
                )
            reconciliation_reuse = (
                item.library_reconciliation.status == "resolved"
                and item.library_reconciliation.decision
                == "REUSE_EXISTING"
            )
            if not reconciliation_reuse:
                row["aliases"] = _unique_text(
                    [
                        *row.get("aliases", []),
                        *item.attribute_aliases,
                        item.attribute_name,
                    ]
                )
            return existing_id

        canonical_name = item.attribute_name.strip()
        description = item.attribute_description.strip()
        if not canonical_name or not description:
            raise ValueError(
                f"{item.eva_id}: new Attribute name and description are "
                "required."
            )
        if item.value_type in {"Categorical", "SexGender"}:
            if item.categorical_value not in {"Included", "Excluded"}:
                if item.value_type == "Categorical":
                    raise ValueError(
                        f"{item.eva_id}: categorical value must be "
                        "Included or Excluded."
                    )
            if item.value_type == "SexGender" and item.categorical_value not in {
                "male",
                "female",
                "all",
            }:
                raise ValueError(
                    f"{item.eva_id}: SexGender value must be male, female, "
                    "or all."
                )
            numerical_type = None
        else:
            if item.numerical_type not in {"Range", "Point"}:
                raise ValueError(
                    f"{item.eva_id}: numerical type must be Range or Point."
                )
            if not str(item.numerical_value or "").strip():
                raise ValueError(
                    f"{item.eva_id}: numerical value is required."
                )
            numerical_type = item.numerical_type

        desired_id = stable_slug(
            item.attribute_id or canonical_name,
            fallback="attribute",
        )
        attribute_id = desired_id
        if attribute_id in attributes:
            raise EvaAttributeIdConflictError(
                f"{item.eva_id}: attribute_id {attribute_id!r} conflicts "
                "with a non-equivalent existing Attribute."
            )
        canonical_key = clean_attribute_key(canonical_name)
        if reconciliation_keep_new and any(
            row["entity_id"] == entity_id
            and clean_attribute_key(row["canonical_name"])
            == canonical_key
            for row in attributes.values()
        ):
            raise ValueError(
                f"{item.eva_id}: canonical name {canonical_name!r} conflicts "
                "with an existing Attribute in this Entity."
            )
        attributes[attribute_id] = {
            "attribute_id": attribute_id,
            "canonical_name": canonical_name,
            "entity_id": entity_id,
            "description": description,
            "aliases": _unique_text(
                [*item.attribute_aliases, canonical_name]
            ),
            "value_type": item.value_type,
            "numerical_type": numerical_type,
            "canonical_unit": (
                item.unit.strip() if item.unit else None
            ),
            "active": True,
        }
        return attribute_id

    @staticmethod
    def _validate_item_value(item: Any) -> None:
        if item.value_type == "Categorical":
            if item.categorical_value not in {"Included", "Excluded"}:
                raise ValueError(
                    f"{item.eva_id}: categorical value must be Included or "
                    "Excluded."
                )
            return
        if item.value_type == "SexGender":
            if item.categorical_value not in {"male", "female", "all"}:
                raise ValueError(
                    f"{item.eva_id}: SexGender value must be male, female, "
                    "or all."
                )
            return
        value = str(item.numerical_value or "").strip()
        if item.numerical_type == "Point":
            if not _POINT_VALUE.fullmatch(value):
                raise ValueError(
                    f"{item.eva_id}: Numerical Point must contain only an "
                    "exact scalar."
                )
            return
        if item.numerical_type == "Range":
            if not _RANGE_VALUE.fullmatch(value):
                raise ValueError(
                    f"{item.eva_id}: Numerical Range must use interval "
                    "notation such as [18, 80] or (30, +inf)."
                )
            return
        raise ValueError(
            f"{item.eva_id}: Numerical value requires Range or Point."
        )
