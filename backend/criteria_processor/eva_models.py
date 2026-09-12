"""Data contracts for human-calibrated eligibility EVA normalization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvaValueType = Literal["Categorical", "SexGender", "Numerical"]
EvaNumericalType = Literal["Range", "Point"]
EvaCategoricalValue = Literal["Included", "Excluded", "male", "female", "all"]
EvaMappingDecision = Literal["existing_attribute", "new_attribute"]
EvaReconciliationDecisionName = Literal[
    "REUSE_EXISTING",
    "KEEP_NEW",
    "BYPASS_KEEP_NEW",
]
EvaReconciliationStatus = Literal[
    "not_required",
    "needs_review",
    "blocked",
    "resolved",
]
EvaReconciliationSemanticRelationship = Literal[
    "exact_equivalent",
    "related_not_equivalent",
    "broader",
    "narrower",
    "incompatible",
    "uncertain",
]
EvaSemanticReviewStatus = Literal[
    "not_run",
    "completed",
    "no_candidates",
    "failed",
    "bypassed",
]
EvaMappingReviewDecision = Literal[
    "CONFIRM_EXISTING",
    "CONFIRM_NEW",
    "SWITCH_EXISTING",
    "REQUIRE_NEW",
    "RETRIEVE_AGAIN",
    "SEGMENTATION_ISSUE",
    "NEEDS_HUMAN_REVIEW",
]
EvaSemanticRelationship = Literal[
    "exact_equivalent",
    "broader_than_source",
    "narrower_than_source",
    "related_not_equivalent",
    "no_supported_target",
    "segmentation_issue",
]
EvaPersistedSemanticRelationship = Literal[
    "exact_equivalent",
    "broader_than_source",
    "narrower_than_source",
    "related_not_equivalent",
    "no_supported_target",
    "segmentation_issue",
    "legacy_unverified",
]
EvaAttributeIdPolicyStatus = Literal[
    "compliant",
    "contextual_pre_existing_qualifier",
    "opaque_generated_id",
    "unsupported_specimen_qualifier",
    "trial_specific_qualifier",
    "measurement_identity_mismatch",
    "not_applicable",
]
EvaPersistedAttributeIdPolicyStatus = Literal[
    "compliant",
    "contextual_pre_existing_qualifier",
    "opaque_generated_id",
    "unsupported_specimen_qualifier",
    "trial_specific_qualifier",
    "measurement_identity_mismatch",
    "not_applicable",
    "legacy_unverified",
]
EvaExplanationLanguage = Literal["en", "legacy_unverified"]
EvaMappingReviewStatus = Literal[
    "not_run",
    "completed",
    "failed",
    "blocked",
    "legacy_unreviewed",
]
EvaAttributeIdReviewDecision = Literal[
    "KEEP",
    "MODIFY",
    "NEEDS_HUMAN_REVIEW",
]
EvaAttributeIdReviewStatus = Literal[
    "not_required",
    "not_run",
    "completed",
    "failed",
    "blocked",
    "legacy_unreviewed",
]


class EvaEntityDefinition(BaseModel):
    """One reusable Entity in the eligibility EVA library."""

    entity_id: str
    canonical_name: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    active: bool = True


class EvaAttributeDefinition(BaseModel):
    """One reusable Attribute and its controlled value schema."""

    attribute_id: str
    canonical_name: str
    entity_id: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    value_type: EvaValueType
    numerical_type: EvaNumericalType | None = None
    canonical_unit: str | None = None
    active: bool = True


class EvaSearchExpansion(BaseModel):
    """LLM-generated lexical and semantic terminology search terms."""

    lexical_terms: list[str] = Field(default_factory=list)
    semantic_terms: list[str] = Field(default_factory=list)
    entity_hints: list[str] = Field(default_factory=list)
    concept_summary: str = ""


class EvaAttributeCandidate(BaseModel):
    """One library Attribute retrieved for a source eligibility point."""

    attribute_id: str
    canonical_name: str
    entity_id: str
    entity_name: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    value_type: EvaValueType
    numerical_type: EvaNumericalType | None = None
    canonical_unit: str | None = None
    score: float = Field(ge=0.0, le=100.0)
    matched_term: str
    matched_text: str
    retrieval_methods: list[str] = Field(default_factory=list)


class EvaReasoningSelection(BaseModel):
    """Structured output from the final EVA mapping reasoner."""

    mapping_decision: EvaMappingDecision
    attribute_id: str | None = None
    entity_id: str | None = None
    entity_name: str
    new_entity: bool = False
    attribute_name: str
    attribute_description: str
    attribute_aliases: list[str] = Field(default_factory=list)
    value_type: EvaValueType
    categorical_value: EvaCategoricalValue | None = None
    numerical_type: EvaNumericalType | None = None
    numerical_value: str | None = None
    unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    candidate_ids_considered: list[str] = Field(default_factory=list)


class EvaMappingReview(BaseModel):
    """Strict structured decision returned by Mapping Review."""

    model_config = ConfigDict(extra="forbid")

    decision: EvaMappingReviewDecision
    current_mapping_decision: EvaMappingDecision
    reviewed_mapping_decision: EvaMappingDecision
    target_attribute_id: str | None = None
    semantic_relationship: EvaSemanticRelationship
    attribute_id_policy_status: EvaAttributeIdPolicyStatus
    additional_search_terms: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvaMappingReviewAttempt(BaseModel):
    """One ordered Mapping Review attempt with deterministic lineage."""

    model_config = ConfigDict(extra="forbid")

    decision: EvaMappingReviewDecision
    current_mapping_decision: EvaMappingDecision
    reviewed_mapping_decision: EvaMappingDecision
    target_attribute_id: str | None = None
    semantic_relationship: EvaPersistedSemanticRelationship
    legacy_semantic_relationship: str = ""
    attribute_id_policy_status: EvaPersistedAttributeIdPolicyStatus
    additional_search_terms: list[str] = Field(default_factory=list)
    explanation: str
    explanation_language: EvaExplanationLanguage = "en"
    confidence: float = Field(ge=0.0, le=1.0)
    review_cycle: int = Field(ge=1, le=2)
    candidate_set_fingerprint: str
    selection_fingerprint: str
    model: str = ""
    reasoning_effort: str = ""
    timestamp: str


class EvaMappingReviewTrace(BaseModel):
    """Complete Mapping Review history for one audit item."""

    model_config = ConfigDict(extra="forbid")

    status: EvaMappingReviewStatus = "not_run"
    initial_mapping_decision: EvaMappingDecision
    initial_attribute_id: str
    final_mapping_decision: EvaMappingDecision
    final_attribute_id: str
    final_decision: EvaMappingReviewDecision | None = None
    attempts: list[EvaMappingReviewAttempt] = Field(default_factory=list)
    error: dict[str, Any] | None = None


class EvaAttributeIdReview(BaseModel):
    """Strict structured Attribute ID review response."""

    model_config = ConfigDict(extra="forbid")

    decision: EvaAttributeIdReviewDecision
    reviewed_attribute_id: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvaAttributeIdReviewTrace(BaseModel):
    """Attribute ID review evidence and deterministic lineage."""

    model_config = ConfigDict(extra="forbid")

    status: EvaAttributeIdReviewStatus = "not_run"
    current_attribute_id: str
    reviewed_attribute_id: str
    decision: EvaAttributeIdReviewDecision | None = None
    explanation: str = ""
    explanation_language: EvaExplanationLanguage = "en"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_fingerprint: str = ""
    model: str = ""
    reasoning_effort: str = ""
    timestamp: str | None = None
    error: dict[str, Any] | None = None


def _default_mapping_review() -> EvaMappingReviewTrace:
    return EvaMappingReviewTrace(
        status="not_run",
        initial_mapping_decision="new_attribute",
        initial_attribute_id="",
        final_mapping_decision="new_attribute",
        final_attribute_id="",
    )


def _default_attribute_id_review() -> EvaAttributeIdReviewTrace:
    return EvaAttributeIdReviewTrace(
        status="not_run",
        current_attribute_id="",
        reviewed_attribute_id="",
    )


class EvaReconciliationDecisionInput(BaseModel):
    """Minimal, untrusted reconciliation choice accepted from a client."""

    model_config = ConfigDict(extra="forbid")

    eva_id: str
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: EvaReconciliationDecisionName
    target_attribute_id: str | None = None
    reason: str = ""


class EvaReconciliationCandidateEvidence(BaseModel):
    """Server-built exact-name candidate evidence."""

    model_config = ConfigDict(extra="forbid")

    attribute_id: str
    canonical_name: str
    entity_id: str
    entity_name: str
    value_type: EvaValueType
    numerical_type: EvaNumericalType | None = None
    canonical_unit: str | None = None
    compatible: bool = True
    matched_names: list[str] = Field(default_factory=list)
    conflict_reasons: list[str] = Field(default_factory=list)


class EvaSemanticCandidateEvidence(BaseModel):
    """Canonical Library metadata plus deterministic retrieval evidence."""

    model_config = ConfigDict(extra="forbid")

    attribute_id: str
    canonical_name: str
    entity_id: str
    entity_name: str
    value_type: EvaValueType
    numerical_type: EvaNumericalType | None = None
    canonical_unit: str | None = None
    retrieval_score: float = Field(ge=0.0, le=100.0)
    matched_term: str
    matched_text: str
    retrieval_methods: list[str] = Field(default_factory=list)
    active: bool = True
    schema_compatible: bool = True


class EvaSemanticAssessment(BaseModel):
    """LLM-authored relationship bound to one trusted candidate set."""

    model_config = ConfigDict(extra="forbid")

    attribute_id: str
    candidate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    relationship: EvaReconciliationSemanticRelationship
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class EvaSemanticProposalReview(BaseModel):
    """Persistable result for one proposal's semantic review."""

    model_config = ConfigDict(extra="forbid")

    retrieval_version: str = "legacy-unversioned"
    status: EvaSemanticReviewStatus = "not_run"
    proposal_fingerprint: str = ""
    candidate_fingerprint: str = ""
    candidates: list[EvaSemanticCandidateEvidence] = Field(default_factory=list)
    diagnostic_candidates: list[EvaSemanticCandidateEvidence] = Field(
        default_factory=list
    )
    assessments: list[EvaSemanticAssessment] = Field(default_factory=list)
    prompt_version: str = ""
    provider: str = ""
    model: str = ""
    reasoning_effort: str = ""
    reviewed_at: str | None = None
    retryable: bool = False
    error_type: str = ""
    error_message: str = ""

    @model_validator(mode="after")
    def _validate_candidate_linkage(self):
        candidate_ids = [candidate.attribute_id for candidate in self.candidates]
        assessment_ids = [assessment.attribute_id for assessment in self.assessments]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Semantic candidates must have unique attribute_id values.")
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("Semantic assessments must not contain duplicates.")
        if self.status == "completed" and set(assessment_ids) != set(candidate_ids):
            raise ValueError("Semantic assessments must cover exactly the trusted candidates.")
        if any(
            assessment.candidate_fingerprint != self.candidate_fingerprint
            for assessment in self.assessments
        ):
            raise ValueError("Semantic assessment candidate fingerprint mismatch.")
        return self


def _default_semantic_review() -> EvaSemanticProposalReview:
    return EvaSemanticProposalReview()


class EvaLibraryReconciliationEvidence(BaseModel):
    """Server-authoritative reconciliation evidence persisted on an item."""

    model_config = ConfigDict(extra="forbid")

    status: EvaReconciliationStatus = "not_required"
    library_revision: int | None = Field(default=None, ge=0)
    library_sha256: str = ""
    proposal_fingerprint: str = ""
    candidates: list[EvaReconciliationCandidateEvidence] = Field(
        default_factory=list
    )
    decision: EvaReconciliationDecisionName | None = None
    target_attribute_id: str | None = None
    reason: str = ""
    reviewed_at: str | None = None
    semantic_review: EvaSemanticProposalReview = Field(
        default_factory=_default_semantic_review
    )


def _default_library_reconciliation() -> EvaLibraryReconciliationEvidence:
    return EvaLibraryReconciliationEvidence()


class EvaAuditItem(BaseModel):
    """Reviewer-editable EVA result for one atomic eligibility point."""

    eva_id: str
    source_item_id: str
    criteria: Literal["inclusion", "exclusion"]
    item: str
    context: str
    search_expansion: EvaSearchExpansion
    candidates: list[EvaAttributeCandidate] = Field(default_factory=list)
    mapping_decision: EvaMappingDecision
    attribute_id: str
    entity_id: str
    entity_name: str
    new_entity: bool = False
    attribute_name: str
    attribute_description: str
    attribute_aliases: list[str] = Field(default_factory=list)
    value_type: EvaValueType
    categorical_value: EvaCategoricalValue | None = None
    numerical_type: EvaNumericalType | None = None
    numerical_value: str | None = None
    unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    mapping_review: EvaMappingReviewTrace = Field(
        default_factory=_default_mapping_review
    )
    attribute_id_review: EvaAttributeIdReviewTrace = Field(
        default_factory=_default_attribute_id_review
    )
    library_reconciliation: EvaLibraryReconciliationEvidence = Field(
        default_factory=_default_library_reconciliation
    )
    review_status: Literal[
        "pending",
        "approved",
        "revised",
        "rejected",
    ] = "pending"
    review_notes: str = ""


class EvaAuditDocument(BaseModel):
    """Per-trial audit file produced before any library mutation."""

    schema_version: Literal["eligcrit.eligibility_eva_audit.v3"] = (
        "eligcrit.eligibility_eva_audit.v3"
    )
    audit_id: str
    trial_key: str
    trial_id: str
    source_path: str
    library_revision: int = Field(ge=0)
    library_sha256: str
    created_at: str
    updated_at: str
    review_status: Literal["pending", "approved", "revised"] = "pending"
    search_model: str
    search_reasoning_effort: str
    reasoner_model: str
    reasoner_reasoning_effort: str
    review_mode: Literal["off", "full"] = "full"
    review_prompt_versions: dict[str, str] = Field(default_factory=dict)
    items: list[EvaAuditItem]
    approved_at: str | None = None
    approved_library_revision: int | None = None
    approved_library_sha256: str | None = None


class EvaApplyEnvelope(BaseModel):
    """New apply request shape; legacy callers may still send a raw audit."""

    model_config = ConfigDict(extra="forbid")

    audit: EvaAuditDocument
    reconciliation_decisions: list[EvaReconciliationDecisionInput] = Field(
        default_factory=list
    )
    reconciliation_job_id: str | None = None


class EvaReconciliationSummary(BaseModel):
    """Counts derived from the final server-resolved audit."""

    reused_count: int = Field(default=0, ge=0)
    new_count: int = Field(default=0, ge=0)
    existing_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)


class EvaApplyResolution(BaseModel):
    """Final Entity/Attribute identity recorded for one accepted EVA row."""

    eva_id: str
    entity_id: str
    entity_name: str
    attribute_id: str
    attribute_name: str
    mapping_decision: EvaMappingDecision
    library_reconciliation: EvaLibraryReconciliationEvidence = Field(
        default_factory=_default_library_reconciliation
    )


class EvaLibraryApplyReceipt(BaseModel):
    """Library-embedded receipt used to replay downstream persistence."""

    receipt_schema_version: Literal[
        "eligcrit.eligibility_eva_apply_receipt.v1"
    ] = "eligcrit.eligibility_eva_apply_receipt.v1"
    audit_id: str
    trial_key: str
    trial_id: str
    input_fingerprint: str
    previous_revision: int = Field(ge=0)
    new_revision: int = Field(ge=0)
    previous_library_sha256: str
    new_library_sha256: str
    resolutions: list[EvaApplyResolution] = Field(default_factory=list)
    reconciliation_summary: EvaReconciliationSummary
    approved_item_count: int = Field(ge=0)
    applied_at: str


class EvaLibraryApplyResult(BaseModel):
    """Authoritative repository result for fresh applies and receipt replay."""

    audit: EvaAuditDocument
    normalized_rows: list[dict[str, Any]] = Field(default_factory=list)
    receipt: EvaLibraryApplyReceipt
    replayed: bool = False
    reconciliation_summary: EvaReconciliationSummary

    def __iter__(self):
        """Preserve the historical ``approved, rows = apply_audit(...)`` API."""

        yield self.audit
        yield self.normalized_rows
