"""Data contracts produced and consumed by EAV criteria normalization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.criteria_processor.eav_schema import AllowedEntity


class TerminologyAttribute(BaseModel):
    """One canonical EAV attribute and the schema used for its values."""

    attribute_id: str
    canonical_name: str
    parent_entity: AllowedEntity
    value_schema: str
    canonical_unit: str = ""
    description: str = ""
    active: bool = True


class TerminologyValue(BaseModel):
    value_id: str
    attribute_id: str
    canonical_value: str
    sort_order: int = 0
    description: str = ""
    active: bool = True


class TerminologyRelationship(BaseModel):
    relationship_id: str
    source_attribute_id: str
    target_attribute_id: str
    relationship_type: str
    status: str = "approved"


class TerminologyMapping(BaseModel):
    raw_attribute: str
    raw_value: str | None = None
    attribute_id: str | None = None
    canonical_entity: AllowedEntity | None = None
    canonical_attribute: str | None = None
    value_schema: str | None = None
    canonical_unit: str | None = None
    value_id: str | None = None
    canonical_value: str | None = None
    mapping_method: str = "unmapped"
    mapping_status: str = "unmapped"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    terminology_version: str = ""


class TerminologySuggestion(BaseModel):
    attribute_id: str | None = None
    value_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractedCriterion(BaseModel):
    """One extracted eligibility criterion before terminology mapping.

    This is the generic entity-attribute-value input schema for downstream
    normalization. It can be produced by an LLM, deterministic parser, manual
    curation step, or any other extraction worker as long as source provenance
    is preserved.
    """

    trial_id: str
    criterion_type: Literal["Inclusion", "Exclusion", "Structured", "Outcome"]
    entity: AllowedEntity
    attribute: str
    value: str | None = None
    condition: str | None = None
    source_text: str
    source_path: str | None = None
    source_index: int | None = None
    extraction_method: str = "eav"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MappingEvidence(BaseModel):
    """One unmapped term or corpus cluster needing terminology resolution."""

    raw_entity: str = ""
    raw_attribute: str
    raw_value: str | None = None
    source_text: str = ""
    source_path: str | None = None
    trial_id: str | None = None
    criterion_type: str | None = None
    occurrence_count: int = 1
    example_trial_ids: list[str] = Field(default_factory=list)
    example_source_texts: list[str] = Field(default_factory=list)
    study_context: dict[str, Any] = Field(default_factory=dict)


class MappingExpansion(BaseModel):
    """LLM-generated search terms for terminology candidate retrieval."""

    expanded_attribute_terms: list[str] = Field(default_factory=list)
    expanded_value_terms: list[str] = Field(default_factory=list)
    term_type_guess: str = ""
    rationale: str = ""


class TerminologyCandidate(BaseModel):
    """One fuzzy-retrieved candidate attribute or value from the library."""

    candidate_type: Literal["attribute", "value"]
    attribute_id: str
    canonical_name: str
    parent_entity: AllowedEntity | Literal[""] = ""
    value_schema: str = ""
    canonical_unit: str = ""
    value_id: str | None = None
    canonical_value: str | None = None
    matched_key: str
    matched_term: str
    score: float = Field(ge=0.0, le=100.0)
    method: str


class MappingSelection(BaseModel):
    """LLM-selected terminology action from retrieved candidates."""

    decision: Literal[
        "alias_existing_attribute",
        "alias_existing_value",
        "new_value_for_existing_attribute",
        "new_attribute",
        "reject_invalid_or_noise",
        "ambiguous_needs_review",
    ]
    attribute_id: str | None = None
    value_id: str | None = None
    proposed_attribute_id: str | None = None
    proposed_canonical_name: str | None = None
    proposed_parent_entity: AllowedEntity | None = None
    proposed_value_schema: str | None = None
    proposed_canonical_unit: str | None = None
    proposed_value_id: str | None = None
    proposed_canonical_value: str | None = None
    proposed_aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class MappingProposal(BaseModel):
    """Auditable result of LLM expansion, fuzzy retrieval, and LLM selection."""

    proposal_id: str
    evidence: MappingEvidence
    expansion: MappingExpansion
    attribute_candidates: list[TerminologyCandidate] = Field(
        default_factory=list
    )
    value_candidates: list[TerminologyCandidate] = Field(
        default_factory=list
    )
    selection: MappingSelection
    terminology_version: str
    proposal_method: str = "llm_expansion+fuzzy_retrieval+llm_selection"
    requires_human_review: bool = True


class TerminologyAuditRecord(BaseModel):
    """One proposed/reviewed change for working or verified terminology."""

    change_id: str
    proposal_id: str = ""
    target_object: Literal[
        "attribute",
        "value",
        "attribute_alias",
        "value_alias",
        "contextual_mapping",
        "suppress_rule",
    ]
    action: Literal["add", "update", "delete", "no_change"]
    review_status: Literal[
        "proposed",
        "approved",
        "revised",
        "rejected",
        "conflict",
        "duplicate",
        "needs_discussion",
    ] = "proposed"
    object_id: str = ""
    attribute_id: str = ""
    value_id: str = ""
    source_entity: str = ""
    aliases: list[str] = Field(default_factory=list)
    canonical_name: str = ""
    canonical_value: str = ""
    parent_entity: AllowedEntity | Literal[""] = ""
    value_schema: str = ""
    canonical_unit: str = ""
    description: str = ""
    active: bool = True
    source: str = "mapping_proposal"
    notes: str = ""
    rationale: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = 1
    example_trial_ids: list[str] = Field(default_factory=list)
    example_source_texts: list[str] = Field(default_factory=list)


class EligibilitySourceItem(BaseModel):
    """One source eligibility criterion with registry-style provenance."""

    criterion_type: Literal["Inclusion", "Exclusion"]
    source_text: str
    source_path: str
    source_index: int


class NormalizedCriterion(BaseModel):
    """One provenance-aware, normalized eligibility or evidence row."""

    criterion_id: str
    nct_id: str
    criterion_type: Literal["Inclusion", "Exclusion", "Structured", "Outcome"]
    source_text: str
    source_path: str
    source_index: int | None = None
    raw_attribute: str
    raw_value: str = ""
    attribute_id: str | None = None
    canonical_attribute: str | None = None
    value_schema: str | None = None
    canonical_value_id: str | None = None
    canonical_value: str | None = None
    value_category: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    lower_inclusive: bool | None = None
    upper_inclusive: bool | None = None
    unit: str | None = None
    lower_unit: str | None = None
    upper_unit: str | None = None
    normalized_lower_bound: float | None = None
    normalized_upper_bound: float | None = None
    normalized_unit: str | None = None
    selection_effect: Literal["included", "excluded"] | None = None
    assertion: Literal["present", "absent", "unknown"] | None = None
    eligible_values: list[Any] = Field(default_factory=list)
    excluded_values: list[Any] = Field(default_factory=list)
    eligible_ranges: list[dict[str, Any]] = Field(default_factory=list)
    excluded_ranges: list[dict[str, Any]] = Field(default_factory=list)
    value_parse_status: str = "preserved"
    mapping_status: str = "unmapped"
    mapping_method: str = "unmapped"
    terminology_version: str = ""
    extraction_method: Literal[
        "ctgov_structured",
        "rule",
        "llm",
        "outcome_rule",
        "unmapped",
    ] = "rule"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_review: bool = False
    review_reason: str = ""
