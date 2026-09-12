"""Hybrid LLM and fuzzy EAV attribute-mapping proposal generation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

from rapidfuzz import fuzz, process

from backend.criteria_processor.eav_schema import ALLOWED_ENTITIES, canonical_entity
from backend.criteria_processor.models import (
    MappingEvidence,
    MappingExpansion,
    MappingProposal,
    MappingSelection,
    TerminologyAuditRecord,
    TerminologyCandidate,
)
from backend.criteria_processor.terminology import TerminologyRepository
from backend.criteria_processor.text import clean_attribute_key
from backend.criteria_processor.value_normalization import (
    SUPPORTED_VALUE_SCHEMAS,
    normalize_structured_value,
)


class JsonLLMProvider(Protocol):
    """Minimal provider boundary for structured LLM calls."""

    def generate_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any] | str:
        """Return JSON-compatible data or a JSON string."""


EXPANSION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expanded_attribute_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "expanded_value_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "term_type_guess": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "expanded_attribute_terms",
        "expanded_value_terms",
        "term_type_guess",
        "rationale",
    ],
    "additionalProperties": False,
}


SELECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "alias_existing_attribute",
                "alias_existing_value",
                "new_value_for_existing_attribute",
                "new_attribute",
                "reject_invalid_or_noise",
                "ambiguous_needs_review",
            ],
        },
        "attribute_id": {"type": ["string", "null"]},
        "value_id": {"type": ["string", "null"]},
        "proposed_attribute_id": {"type": ["string", "null"]},
        "proposed_canonical_name": {"type": ["string", "null"]},
        "proposed_parent_entity": {
            "type": ["string", "null"],
            "enum": [*ALLOWED_ENTITIES, None],
        },
        "proposed_value_schema": {
            "type": ["string", "null"],
            "enum": [
                *sorted(SUPPORTED_VALUE_SCHEMAS),
                None,
            ],
        },
        "proposed_canonical_unit": {"type": ["string", "null"]},
        "proposed_value_id": {"type": ["string", "null"]},
        "proposed_canonical_value": {"type": ["string", "null"]},
        "proposed_aliases": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": [
        "decision",
        "attribute_id",
        "value_id",
        "proposed_attribute_id",
        "proposed_canonical_name",
        "proposed_parent_entity",
        "proposed_value_schema",
        "proposed_canonical_unit",
        "proposed_value_id",
        "proposed_canonical_value",
        "proposed_aliases",
        "confidence",
        "rationale",
    ],
    "additionalProperties": False,
}


def _json_result(value: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", clean_attribute_key(value))
    return slug.strip("_") or "term"


def _unique_aliases(values: list[str]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = clean_attribute_key(text)
        if text and key and key not in seen:
            aliases.append(text)
            seen.add(key)
    return aliases


_GENERIC_ATTRIBUTE_KEYS = {
    "ability",
    "condition",
    "comorbidity",
    "concomitant disease",
    "concurrent condition",
    "concurrent disease",
    "criteria",
    "criterion",
    "current use",
    "disease",
    "diagnosis",
    "disorder",
    "eligibility",
    "exclusion",
    "history",
    "illness",
    "inclusion",
    "medical condition",
    "medical history",
    "medication",
    "medication use",
    "other",
    "presence",
    "procedure",
    "requirement",
    "status",
    "treatment",
    "use",
}

_IDENTITY_IN_VALUE_ATTRIBUTE_KEYS = {
    "condition",
    "comorbidity",
    "concomitant disease",
    "concurrent condition",
    "concurrent disease",
    "disease",
    "diagnosis",
    "disorder",
    "drug",
    "illness",
    "medical condition",
    "medical history",
    "medication",
    "medication use",
    "procedure",
    "treatment",
}

_SELECTION_STATE_ATTRIBUTE_KEYS = {
    "pregnancy",
    "pregnancy status",
}

_CONTROLLED_CATEGORY_VALUES = {
    "sex": {"male", "female", "all gender"},
    "stroke stage": {"acute", "subacute", "chronic"},
    "stroke type": {"ischemic", "hemorrhagic", "any stroke"},
}


def _safe_attribute_aliases(values: list[str]) -> list[str]:
    aliases = []
    for alias in _unique_aliases(values):
        key = clean_attribute_key(alias)
        if key in _GENERIC_ATTRIBUTE_KEYS:
            continue
        if len(alias) > 100:
            continue
        if re.search(r"(?:[<>≤≥]|\d+\s*(?:-|to)\s*\d+)", alias):
            continue
        if re.search(r"\d", alias) and re.search(
            r"\b(?:less|more|greater|fewer|under|over|above|below|"
            r"minimum|maximum|threshold|cutoff|at\s+least|at\s+most|"
            r"or\s+less|or\s+more)\b",
            key,
        ):
            continue
        aliases.append(alias)
    return aliases


class MappingSuggestionEngine:
    """Generate attribute-library update proposals from unmapped EAV evidence."""

    def __init__(
        self,
        terminology: TerminologyRepository,
        llm_provider: JsonLLMProvider,
        *,
        fuzzy_threshold: int = 85,
        top_k: int = 5,
    ) -> None:
        self.terminology = terminology
        self.llm_provider = llm_provider
        self.fuzzy_threshold = fuzzy_threshold
        self.top_k = top_k

    def expand_search_terms(
        self,
        evidence: MappingEvidence,
    ) -> MappingExpansion:
        payload = evidence.model_dump(mode="json")
        prompt = (
            "Generate search terms for one unmapped clinical eligibility EAV "
            "item. The attribute is the variable being constrained (for "
            "example Age, Sex, Smoking Status, or Baseline FMA-UE); the value "
            "contains categories, cutoffs, ranges, operators, and units. "
            "Every final Attribute must have exactly one parent from this "
            f"closed Entity set: {', '.join(ALLOWED_ENTITIES)}. When the raw "
            "attribute is generic but its value identifies the clinical "
            "variable, move that identity into the attribute search terms. "
            "For example, Diagnosis | Parkinson's disease must search for the "
            "specific Attribute Parkinson's Disease, not a generic Diagnosis "
            "attribute with an open-ended disease-name value. Pregnancy is "
            "likewise a specific Attribute. "
            "Expand attribute abbreviations and synonyms separately from value "
            "representations. Do not choose a final mapping.\n\n"
            f"EVIDENCE_JSON:\n{json.dumps(payload, indent=2)}"
        )
        result = self.llm_provider.generate_json(
            prompt,
            output_schema=EXPANSION_OUTPUT_SCHEMA,
        )
        return MappingExpansion.model_validate(_json_result(result))

    def fuzzy_attribute_candidates(
        self,
        search_terms: list[str],
    ) -> list[TerminologyCandidate]:
        best_by_attribute: dict[str, TerminologyCandidate] = {}
        lookup_items = self.terminology.attribute_lookup_items()
        for term in search_terms:
            term_key = clean_attribute_key(term)
            if not term_key:
                continue
            for matched_key, definition in lookup_items:
                score = fuzz.WRatio(term_key, matched_key)
                if score < self.fuzzy_threshold:
                    continue
                candidate = TerminologyCandidate(
                    candidate_type="attribute",
                    attribute_id=definition.attribute_id,
                    canonical_name=definition.canonical_name,
                    parent_entity=definition.parent_entity,
                    value_schema=definition.value_schema,
                    canonical_unit=definition.canonical_unit,
                    matched_key=matched_key,
                    matched_term=term,
                    score=float(score),
                    method="llm_expansion+fuzzy_wratio",
                )
                current = best_by_attribute.get(definition.attribute_id)
                if current is None or candidate.score > current.score:
                    best_by_attribute[definition.attribute_id] = candidate
        return sorted(
            best_by_attribute.values(),
            key=lambda item: item.score,
            reverse=True,
        )[: self.top_k]

    def fuzzy_value_candidates(
        self,
        attribute_id: str,
        search_terms: list[str],
    ) -> list[TerminologyCandidate]:
        definition = self.terminology.attributes[attribute_id]
        best_by_value: dict[str, TerminologyCandidate] = {}
        lookup_items = self.terminology.value_lookup_items(attribute_id)
        choices = [key for key, _value in lookup_items]
        value_by_key = {key: value for key, value in lookup_items}
        for term in search_terms:
            term_key = clean_attribute_key(term)
            if not term_key:
                continue
            matches = process.extract(
                term_key,
                choices,
                scorer=fuzz.WRatio,
                score_cutoff=self.fuzzy_threshold,
                limit=self.top_k,
            )
            for matched_key, score, _index in matches:
                value = value_by_key[matched_key]
                candidate = TerminologyCandidate(
                    candidate_type="value",
                    attribute_id=definition.attribute_id,
                    canonical_name=definition.canonical_name,
                    parent_entity=definition.parent_entity,
                    value_schema=definition.value_schema,
                    canonical_unit=definition.canonical_unit,
                    value_id=value.value_id,
                    canonical_value=value.canonical_value,
                    matched_key=matched_key,
                    matched_term=term,
                    score=float(score),
                    method="llm_expansion+fuzzy_wratio",
                )
                current = best_by_value.get(value.value_id)
                if current is None or candidate.score > current.score:
                    best_by_value[value.value_id] = candidate
        return sorted(
            best_by_value.values(),
            key=lambda item: item.score,
            reverse=True,
        )[: self.top_k]

    def select_mapping(
        self,
        *,
        evidence: MappingEvidence,
        expansion: MappingExpansion,
        attribute_candidates: list[TerminologyCandidate],
        value_candidates: list[TerminologyCandidate],
    ) -> MappingSelection:
        payload = {
            "evidence": evidence.model_dump(mode="json"),
            "expansion": expansion.model_dump(mode="json"),
            "attribute_candidates": [
                item.model_dump(mode="json") for item in attribute_candidates
            ],
            "value_candidates": [
                item.model_dump(mode="json") for item in value_candidates
            ],
        }
        prompt = (
            "Select the best EAV attribute-library action for an unmapped "
            "clinical eligibility item. Existing matches must come only from "
            "the supplied candidates. The parent Entity MUST be exactly one "
            f"of: {', '.join(ALLOWED_ENTITIES)}. The canonical attribute must "
            "be specific and must name only "
            "the variable being constrained, such as Age, Sex, Smoking Status, "
            "Baseline FMA-UE, Parkinson's Disease, or Pregnancy. A disease, "
            "condition, medication, treatment, or procedure identity belongs "
            "in the Attribute name, never in an open-ended Value. Thus "
            "Diagnosis | Parkinson's disease becomes Attribute Parkinson's "
            "Disease under Entity Diagnosis. Never include "
            "eligibility/criterion wording, "
            "comparison operators, cutoffs, ranges, units, or categories in "
            "the attribute name. Those belong to the value schema and value. "
            "Use selection_state for binary presence/absence population "
            "constraints such as a specific disease or Pregnancy. Do not "
            "propose included/excluded as library values: they are derived "
            "from criterion type and statement polarity for every criterion. "
            "Use categorical only for a small, closed vocabulary; "
            "numeric_range for "
            "numeric bounds or scores, duration_range for elapsed-time bounds, "
            "and free_text only when no safer structure applies and human "
            "review is required. For Sex, use canonical categories "
            "male, female, and all gender. `proposed_aliases` must describe "
            "only the object being added or aliased: variable-name synonyms "
            "for attributes and category synonyms for categorical values; do "
            "not place cutoffs, ranges, or full criterion statements in "
            "attribute aliases. If no candidate fits, propose a new "
            "attribute. A proposed new categorical value requires human review "
            "unless it is one of the system-registered closed-vocabulary "
            "values. Do not invent existing IDs."
            "\n\n"
            f"SELECTION_INPUT_JSON:\n{json.dumps(payload, indent=2)}"
        )
        result = self.llm_provider.generate_json(
            prompt,
            output_schema=SELECTION_OUTPUT_SCHEMA,
        )
        return MappingSelection.model_validate(_json_result(result))

    def propose_mapping(
        self,
        evidence: MappingEvidence,
    ) -> MappingProposal:
        expansion = self.expand_search_terms(evidence)
        attribute_terms = [
            evidence.raw_attribute,
            *expansion.expanded_attribute_terms,
        ]
        if (
            clean_attribute_key(evidence.raw_attribute)
            in _IDENTITY_IN_VALUE_ATTRIBUTE_KEYS
            and evidence.raw_value
        ):
            attribute_terms.append(evidence.raw_value)
        value_terms = [
            term
            for term in [evidence.raw_value, *expansion.expanded_value_terms]
            if term
        ]
        attribute_candidates = self.fuzzy_attribute_candidates(attribute_terms)
        value_candidates: list[TerminologyCandidate] = []
        for attribute_candidate in attribute_candidates:
            if attribute_candidate.value_schema == "categorical":
                value_candidates.extend(
                    self.fuzzy_value_candidates(
                        attribute_candidate.attribute_id,
                        value_terms,
                    )
                )
        value_candidates = sorted(
            {
                (item.attribute_id, item.value_id): item
                for item in value_candidates
            }.values(),
            key=lambda item: item.score,
            reverse=True,
        )[: self.top_k]
        selection = self.select_mapping(
            evidence=evidence,
            expansion=expansion,
            attribute_candidates=attribute_candidates,
            value_candidates=value_candidates,
        )
        selection = self._guard_selection(
            evidence=evidence,
            selection=selection,
            attribute_candidates=attribute_candidates,
            value_candidates=value_candidates,
        )
        proposal_id = _stable_id(
            "proposal",
            evidence.raw_entity,
            evidence.raw_attribute,
            evidence.raw_value,
            evidence.source_text,
        )
        return MappingProposal(
            proposal_id=proposal_id,
            evidence=evidence,
            expansion=expansion,
            attribute_candidates=attribute_candidates,
            value_candidates=value_candidates,
            selection=selection,
            terminology_version=self.terminology.terminology_version,
        )

    def _guard_selection(
        self,
        *,
        evidence: MappingEvidence,
        selection: MappingSelection,
        attribute_candidates: list[TerminologyCandidate],
        value_candidates: list[TerminologyCandidate],
    ) -> MappingSelection:
        """Keep selector output applicable to the current library state."""

        attribute_ids = {
            candidate.attribute_id for candidate in attribute_candidates
        }
        value_ids = {
            candidate.value_id
            for candidate in value_candidates
            if candidate.value_id
        }
        if (
            selection.decision
            in {"alias_existing_attribute", "alias_existing_value"}
            and selection.attribute_id not in attribute_ids
        ):
            return self._new_attribute_fallback(
                evidence,
                confidence=min(selection.confidence, 0.75),
                rationale=(
                    "Selector chose an existing attribute that was not among "
                    "the retrieved candidates; converted to a new attribute "
                    "proposal."
                ),
            )
        if (
            selection.decision == "alias_existing_value"
            and selection.value_id not in value_ids
        ):
            return MappingSelection(
                decision="ambiguous_needs_review",
                confidence=min(selection.confidence, 0.5),
                rationale=(
                    "Selector chose an existing value that was not among the "
                    "retrieved candidates. New categorical values are not "
                    "added automatically because that would make the value "
                    "vocabulary unbounded."
                ),
            )
        if (
            selection.decision == "new_value_for_existing_attribute"
            and selection.attribute_id not in self.terminology.attributes
        ):
            return self._new_attribute_fallback(
                evidence,
                confidence=min(selection.confidence, 0.75),
                rationale=(
                    f"{selection.rationale} Guardrail: selector proposed a "
                    "value without a valid existing attribute_id; converted "
                    "to a new attribute proposal."
                ).strip(),
                selection=selection,
            )
        if selection.decision == "new_value_for_existing_attribute":
            definition = self.terminology.attributes[
                selection.attribute_id or ""
            ]
            if definition.value_schema != "categorical":
                return MappingSelection(
                    decision="alias_existing_attribute",
                    attribute_id=definition.attribute_id,
                    proposed_aliases=[evidence.raw_attribute],
                    confidence=min(selection.confidence, 0.75),
                    rationale=(
                        f"{selection.rationale} Guardrail: values may be added "
                        "to the library only for categorical attributes, but "
                        f"'{definition.attribute_id}' uses "
                        f"'{definition.value_schema}'; converted to an "
                        "attribute-alias proposal so the raw value can be "
                        "parsed by that schema."
                    ).strip(),
                )
            category = normalize_structured_value(
                raw_value=evidence.raw_value,
                attribute_id=definition.attribute_id,
                canonical_attribute=definition.canonical_name,
                value_schema="categorical",
                canonical_value=selection.proposed_canonical_value,
            )
            selection.proposed_canonical_value = category["category"]
            allowed_values = _CONTROLLED_CATEGORY_VALUES.get(
                clean_attribute_key(definition.canonical_name)
            )
            if (
                allowed_values is None
                or clean_attribute_key(selection.proposed_canonical_value)
                not in allowed_values
            ):
                return MappingSelection(
                    decision="ambiguous_needs_review",
                    confidence=min(selection.confidence, 0.5),
                    rationale=(
                        f"{selection.rationale} Guardrail: proposed expansion "
                        f"of the controlled categorical values for "
                        f"'{definition.canonical_name}' requires human review."
                    ).strip(),
                )
        if selection.decision == "new_attribute":
            parent_entity = canonical_entity(
                selection.proposed_parent_entity
            )
            if parent_entity is None:
                return MappingSelection(
                    decision="ambiguous_needs_review",
                    confidence=min(selection.confidence, 0.5),
                    rationale=(
                        f"{selection.rationale} Guardrail: every Attribute "
                        "must use one of the ten controlled parent Entities."
                    ).strip(),
                )
            selection.proposed_parent_entity = parent_entity
            schema = (
                selection.proposed_value_schema or "free_text"
            ).casefold()
            if schema == "boolean":
                schema = "selection_state"
                selection.proposed_value_schema = schema
            raw_attribute_key = clean_attribute_key(evidence.raw_attribute)
            canonical_name_key = clean_attribute_key(
                selection.proposed_canonical_name
            )
            if (
                raw_attribute_key in _IDENTITY_IN_VALUE_ATTRIBUTE_KEYS
                and canonical_name_key not in _GENERIC_ATTRIBUTE_KEYS
            ):
                schema = "selection_state"
                selection.proposed_value_schema = schema
            if (
                canonical_name_key in _SELECTION_STATE_ATTRIBUTE_KEYS
                or raw_attribute_key in _SELECTION_STATE_ATTRIBUTE_KEYS
            ):
                schema = "selection_state"
                selection.proposed_value_schema = schema
            if schema not in SUPPORTED_VALUE_SCHEMAS:
                selection.proposed_value_schema = "free_text"
                selection.confidence = min(selection.confidence, 0.75)
            if (
                parent_entity == "Score"
                and schema == "numeric_range"
                and not selection.proposed_canonical_unit
            ):
                selection.proposed_canonical_unit = "point"
            selection.proposed_aliases = _unique_aliases(
                [evidence.raw_attribute, *selection.proposed_aliases]
            )
            if clean_attribute_key(
                selection.proposed_canonical_name
            ) in _GENERIC_ATTRIBUTE_KEYS:
                return MappingSelection(
                    decision="ambiguous_needs_review",
                    confidence=min(selection.confidence, 0.5),
                    rationale=(
                        f"{selection.rationale} Guardrail: proposed canonical "
                        "attribute name is too generic for safe reuse and "
                        "requires extraction or human review."
                    ).strip(),
                )
            if schema == "free_text":
                return MappingSelection(
                    decision="ambiguous_needs_review",
                    confidence=min(selection.confidence, 0.5),
                    rationale=(
                        f"{selection.rationale} Guardrail: free-text Values "
                        "are not added to the filter-oriented working library "
                        "without human review."
                    ).strip(),
                )
            if schema == "selection_state":
                selection.proposed_value_id = None
                selection.proposed_canonical_value = None
            if schema == "categorical" and not (
                selection.proposed_canonical_value
            ):
                parsed = normalize_structured_value(
                    raw_value=evidence.raw_value,
                    attribute_id=selection.proposed_attribute_id,
                    canonical_attribute=selection.proposed_canonical_name,
                    value_schema="categorical",
                )
                selection.proposed_canonical_value = (
                    parsed["category"] or evidence.raw_value
                )
            if (
                schema == "categorical"
                and selection.proposed_canonical_value
                and not selection.proposed_value_id
            ):
                selection.proposed_value_id = (
                    f"{selection.proposed_attribute_id or _slug(evidence.raw_attribute)}_"
                    f"{_slug(selection.proposed_canonical_value)}"
                )
            if schema == "categorical":
                allowed_values = _CONTROLLED_CATEGORY_VALUES.get(
                    clean_attribute_key(
                        selection.proposed_canonical_name
                        or selection.proposed_attribute_id
                    )
                )
                if (
                    allowed_values is None
                    or clean_attribute_key(
                        selection.proposed_canonical_value
                    )
                    not in allowed_values
                ):
                    return MappingSelection(
                        decision="ambiguous_needs_review",
                        confidence=min(selection.confidence, 0.5),
                        rationale=(
                            f"{selection.rationale} Guardrail: categorical "
                            "attributes can update the working library only "
                            "when their closed vocabulary is registered."
                        ).strip(),
                    )
        return selection

    def _new_attribute_fallback(
        self,
        evidence: MappingEvidence,
        *,
        confidence: float,
        rationale: str,
        selection: MappingSelection | None = None,
    ) -> MappingSelection:
        parent_entity = canonical_entity(
            selection.proposed_parent_entity
            if selection
            else evidence.raw_entity
        )
        if parent_entity is None:
            return MappingSelection(
                decision="ambiguous_needs_review",
                confidence=min(confidence, 0.5),
                rationale=(
                    f"{rationale} Guardrail: a new Attribute cannot be "
                    "created until it has one of the ten controlled parent "
                    "Entities."
                ).strip(),
            )
        return MappingSelection(
            decision="new_attribute",
            proposed_attribute_id=(
                selection.proposed_attribute_id
                if selection and selection.proposed_attribute_id
                else _slug(evidence.raw_attribute)
            ),
            proposed_canonical_name=(
                selection.proposed_canonical_name
                if selection and selection.proposed_canonical_name
                else evidence.raw_attribute
            ),
            proposed_parent_entity=parent_entity,
            proposed_value_schema=(
                selection.proposed_value_schema
                if selection and selection.proposed_value_schema
                else "free_text"
            ),
            proposed_canonical_unit=(
                selection.proposed_canonical_unit if selection else None
            ),
            proposed_value_id=(
                selection.proposed_value_id if selection else None
            ),
            proposed_canonical_value=(
                selection.proposed_canonical_value if selection else None
            ),
            proposed_aliases=_unique_aliases(
                [
                    evidence.raw_attribute,
                    *(selection.proposed_aliases if selection else []),
                ]
            ),
            confidence=confidence,
            rationale=rationale,
        )


def audit_records_from_proposal(
    proposal: MappingProposal,
) -> list[TerminologyAuditRecord]:
    """Convert one mapping proposal into compact reviewer-editable changes."""

    selection = proposal.selection
    evidence = proposal.evidence
    examples = evidence.example_source_texts or [evidence.source_text]
    trial_ids = evidence.example_trial_ids or (
        [evidence.trial_id] if evidence.trial_id else []
    )
    common = {
        "proposal_id": proposal.proposal_id,
        "review_status": "proposed",
        "rationale": selection.rationale,
        "confidence": selection.confidence,
        "evidence_count": evidence.occurrence_count,
        "example_trial_ids": trial_ids,
        "example_source_texts": examples,
    }

    if selection.decision == "alias_existing_attribute":
        aliases = _safe_attribute_aliases(
            selection.proposed_aliases or [evidence.raw_attribute]
        )
        if not aliases:
            return [
                TerminologyAuditRecord(
                    **common,
                    change_id=_stable_id(
                        "change",
                        proposal.proposal_id,
                        "contextual_mapping",
                    ),
                    target_object="contextual_mapping",
                    action="no_change",
                    object_id=f"context_{proposal.proposal_id}",
                    attribute_id=selection.attribute_id or "",
                    source_entity=evidence.raw_entity,
                    notes=(
                        "Selector target is used for this evidence only; the "
                        "raw attribute label is too generic to promote as a "
                        "reusable alias."
                    ),
                )
            ]
        return [
            TerminologyAuditRecord(
                **common,
                change_id=_stable_id(
                    "change",
                    proposal.proposal_id,
                    "attribute_alias",
                ),
                target_object="attribute_alias",
                action="add",
                object_id=f"aliases_{selection.attribute_id}",
                attribute_id=selection.attribute_id or "",
                source_entity=evidence.raw_entity,
                aliases=aliases,
                notes="Proposed aliases for an existing attribute.",
            )
        ]

    if selection.decision == "alias_existing_value":
        aliases = _unique_aliases(
            selection.proposed_aliases or [evidence.raw_value or ""]
        )
        return [
            TerminologyAuditRecord(
                **common,
                change_id=_stable_id(
                    "change",
                    proposal.proposal_id,
                    "value_alias",
                ),
                target_object="value_alias",
                action="add",
                object_id=f"aliases_{selection.value_id}",
                attribute_id=selection.attribute_id or "",
                value_id=selection.value_id or "",
                source_entity=evidence.raw_entity,
                aliases=aliases,
                notes="Proposed aliases for an existing categorical value.",
            )
        ]

    if selection.decision == "new_value_for_existing_attribute":
        value_id = selection.proposed_value_id or (
            f"{selection.attribute_id}_"
            f"{_slug(selection.proposed_canonical_value or '')}"
        )
        return [
            TerminologyAuditRecord(
                **common,
                change_id=_stable_id(
                    "change",
                    proposal.proposal_id,
                    "value",
                    value_id,
                ),
                target_object="value",
                action="add",
                object_id=value_id,
                attribute_id=selection.attribute_id or "",
                value_id=value_id,
                source_entity=evidence.raw_entity,
                aliases=_unique_aliases(selection.proposed_aliases),
                canonical_value=selection.proposed_canonical_value or "",
                description=selection.rationale,
                notes="Proposed new categorical value and its aliases.",
            )
        ]

    if selection.decision == "new_attribute":
        attribute_id = selection.proposed_attribute_id or _slug(
            selection.proposed_canonical_name or evidence.raw_attribute
        )
        records = [
            TerminologyAuditRecord(
                **common,
                change_id=_stable_id(
                    "change",
                    proposal.proposal_id,
                    "attribute",
                    attribute_id,
                ),
                target_object="attribute",
                action="add",
                object_id=attribute_id,
                attribute_id=attribute_id,
                source_entity=evidence.raw_entity,
                aliases=_safe_attribute_aliases(
                    [evidence.raw_attribute, *selection.proposed_aliases]
                ),
                canonical_name=(
                    selection.proposed_canonical_name
                    or evidence.raw_attribute
                ),
                parent_entity=(
                    selection.proposed_parent_entity or ""
                ),
                value_schema=(
                    selection.proposed_value_schema or "free_text"
                ),
                canonical_unit=selection.proposed_canonical_unit or "",
                description=selection.rationale,
                notes="Proposed new EAV attribute and its aliases.",
            )
        ]
        if (
            selection.proposed_value_schema == "categorical"
            and selection.proposed_canonical_value
        ):
            value_id = selection.proposed_value_id or (
                f"{attribute_id}_"
                f"{_slug(selection.proposed_canonical_value)}"
            )
            records.append(
                TerminologyAuditRecord(
                    **common,
                    change_id=_stable_id(
                        "change",
                        proposal.proposal_id,
                        "value",
                        value_id,
                    ),
                    target_object="value",
                    action="add",
                    object_id=value_id,
                    attribute_id=attribute_id,
                    value_id=value_id,
                    source_entity=evidence.raw_entity,
                    aliases=_unique_aliases(
                        [evidence.raw_value or ""]
                    ),
                    canonical_value=selection.proposed_canonical_value,
                    description=selection.rationale,
                    notes=(
                        "Proposed first categorical value for the new "
                        "attribute."
                    ),
                )
            )
        return records

    return [
        TerminologyAuditRecord(
            **common,
            change_id=_stable_id(
                "change",
                proposal.proposal_id,
                selection.decision,
            ),
            target_object="suppress_rule",
            action="no_change",
            object_id=_stable_id(
                "suppress",
                evidence.raw_attribute,
                evidence.raw_value,
            ),
            notes=f"Selector decision: {selection.decision}.",
        )
    ]
