"""Deterministic contracts and prompts for Eligibility EVA GPT Review."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from backend.criteria_processor.eva_library import EvaLibraryRepository, stable_slug
from backend.criteria_processor.eva_models import (
    EvaAttributeCandidate,
    EvaAttributeIdReview,
    EvaAttributeIdReviewTrace,
    EvaMappingReview,
    EvaMappingReviewAttempt,
    EvaMappingReviewTrace,
    EvaReasoningSelection,
    EvaSearchExpansion,
)


MAPPING_REVIEW_PROMPT_VERSION = "eligibility-eva-mapping-review-v3"
MAPPING_CORRECTION_PROMPT_VERSION = "eligibility-eva-mapping-correction-v3"
ATTRIBUTE_ID_REVIEW_PROMPT_VERSION = "eligibility-eva-attribute-id-review-v3"

CLINICAL_ATTRIBUTE_NAMING_POLICY = """Clinical Attribute naming policy:
- An Attribute ID names one stable reusable variable, not a trial-specific
  eligibility sentence. Do not encode comparison operators, thresholds, range
  endpoints, units, polarity, contextual list words, or trial-specific severity
  and support requirements.
- Preserve qualifiers that define variable identity: persistent versus
  transient, active versus passive, disease/event subtype, impairment versus
  failure, disorder versus addiction, a validated instrument plus score, and
  the measured physical quantity.
- In general, pre-existing is enrollment context and does not enter a reusable
  Attribute ID. The word other in a coordinated residual category is also
  contextual. Do not generalize away a meaning-bearing follow-up timepoint when
  the schema has no separate field in which to preserve it.
- Do not strengthen impairment into failure, problematic use or abuse into
  addiction, generic use into a disorder, temporary into permanent, or a broad
  concept into a subtype.
- Do not infer an unsupported specimen qualifier such as serum or plasma. For
  laboratory analytes, prefer the standard analyte name when the atomic title
  does not specify a necessary specimen distinction. Use paired names:
  aspartate_aminotransferase for AST/TGO/SGOT and alanine_aminotransferase for
  ALT/TGP/SGPT.
- A Numerical degree measurement names a range_of_motion and preserves active
  versus passive. Do not use an ID that sounds like a categorical ability.
- Aliases contain terminology synonyms and abbreviations only. Do not put
  comparison operators, thresholds, units, or a complete eligibility constraint
  in aliases.
- Aliases, descriptions, matched text, and retrieval scores are retrieval
  evidence only; they do not prove semantic equivalence.
- Boolean grouping is part of criterion meaning. If material AND/OR grouping in
  context is absent from the atomic representation, treat it as a segmentation
  issue rather than approving independent filters."""

ID_REVIEW_NOT_REQUIRED_EXPLANATION = (
    "Attribute ID Review is not required because the final mapping reuses "
    "an existing immutable Attribute ID."
)


MAPPING_REVIEW_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "CONFIRM_EXISTING",
                "CONFIRM_NEW",
                "SWITCH_EXISTING",
                "REQUIRE_NEW",
                "RETRIEVE_AGAIN",
                "SEGMENTATION_ISSUE",
                "NEEDS_HUMAN_REVIEW",
            ],
        },
        "current_mapping_decision": {
            "type": "string",
            "enum": ["existing_attribute", "new_attribute"],
        },
        "reviewed_mapping_decision": {
            "type": "string",
            "enum": ["existing_attribute", "new_attribute"],
        },
        "target_attribute_id": {"type": ["string", "null"]},
        "semantic_relationship": {
            "type": "string",
            "enum": [
                "exact_equivalent",
                "broader_than_source",
                "narrower_than_source",
                "related_not_equivalent",
                "no_supported_target",
                "segmentation_issue",
            ],
        },
        "attribute_id_policy_status": {
            "type": "string",
            "enum": [
                "compliant",
                "contextual_pre_existing_qualifier",
                "opaque_generated_id",
                "unsupported_specimen_qualifier",
                "trial_specific_qualifier",
                "measurement_identity_mismatch",
                "not_applicable",
            ],
        },
        "additional_search_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "explanation": {"type": "string", "minLength": 1},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "decision",
        "current_mapping_decision",
        "reviewed_mapping_decision",
        "target_attribute_id",
        "semantic_relationship",
        "attribute_id_policy_status",
        "additional_search_terms",
        "explanation",
        "confidence",
    ],
    "additionalProperties": False,
}


_REASONING_PROPERTIES: dict[str, Any] = {
    "mapping_decision": {
        "type": "string",
        "enum": ["existing_attribute", "new_attribute"],
    },
    "attribute_id": {"type": ["string", "null"]},
    "entity_id": {"type": ["string", "null"]},
    "entity_name": {"type": "string"},
    "new_entity": {"type": "boolean"},
    "attribute_name": {"type": "string"},
    "attribute_description": {"type": "string"},
    "attribute_aliases": {"type": "array", "items": {"type": "string"}},
    "value_type": {
        "type": "string",
        "enum": ["Categorical", "SexGender", "Numerical"],
    },
    "categorical_value": {
        "type": ["string", "null"],
        "enum": ["Included", "Excluded", "male", "female", "all", None],
    },
    "numerical_type": {
        "type": ["string", "null"],
        "enum": ["Range", "Point", None],
    },
    "numerical_value": {"type": ["string", "null"]},
    "unit": {"type": ["string", "null"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "rationale": {"type": "string"},
    "candidate_ids_considered": {
        "type": "array",
        "items": {"type": "string"},
    },
    "correction_acknowledged": {"type": "boolean"},
}
MAPPING_CORRECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _REASONING_PROPERTIES,
    "required": list(_REASONING_PROPERTIES),
    "additionalProperties": False,
}


ATTRIBUTE_ID_REVIEW_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["KEEP", "MODIFY", "NEEDS_HUMAN_REVIEW"],
        },
        "reviewed_attribute_id": {"type": "string", "minLength": 1},
        "explanation": {"type": "string", "minLength": 1},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "decision",
        "reviewed_attribute_id",
        "explanation",
        "confidence",
    ],
    "additionalProperties": False,
}


def _canonical_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def candidate_set_sha256(candidates: Sequence[Any]) -> str:
    """Hash a candidate set using canonical JSON in supplied order."""

    return _canonical_sha256(
        [candidate.model_dump(mode="json") for candidate in candidates]
    )


def selection_sha256(selection: EvaReasoningSelection) -> str:
    """Hash a validated reasoner selection using canonical JSON."""

    return _canonical_sha256(selection.model_dump(mode="json"))


_CJK_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def validate_english_explanation(text: str, *, field_name: str) -> str:
    """Reject blank or CJK-containing text for newly generated evidence."""

    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"{field_name} explanation must not be empty.")
    if _CJK_TEXT.search(cleaned):
        raise ValueError(
            f"{field_name} explanation must be written in English."
        )
    return cleaned


def build_mapping_review_prompt(
    *,
    source_item_id: str,
    criteria: str,
    item: str,
    context: str,
    selection: EvaReasoningSelection,
    candidates: list[EvaAttributeCandidate],
    review_cycle: int,
    candidate_fingerprint: str,
    selection_fingerprint: str,
    prior_error: str = "",
) -> str:
    """Build the bounded Mapping Review structured prompt."""

    payload = {
        "source_item_id": source_item_id,
        "review_cycle": review_cycle,
        "candidate_set_fingerprint": candidate_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "criteria": criteria,
        "atomic_criterion_title": item,
        "context_for_disambiguation_only": context,
        "current_selection": selection.model_dump(mode="json"),
        "active_candidates": [
            candidate.model_dump(mode="json") for candidate in candidates
        ],
        "prior_validation_error": prior_error,
    }
    return (
        "You are the Eligibility EVA Mapping Reviewer. Independently audit the "
        "validated mapping. Use criterion context only to disambiguate the "
        "atomic title; never broaden or segment it. Existing targets must be "
        "selected only from active_candidates.\n\n"
        "Judge clinical equivalence separately from Attribute-ID naming policy. "
        "A target may be clinically exact yet still require a new Attribute "
        "because its ID stores context that should not be part of reusable "
        "variable identity.\n\n"
        f"{CLINICAL_ATTRIBUTE_NAMING_POLICY}\n\n"
        "Use opaque_generated_id for an ID such as attribute_<hex>. Use "
        "unsupported_specimen_qualifier when an exact analyte ID adds serum or "
        "plasma without source support. Use trial_specific_qualifier when an ID "
        "stores local severity or support wording. Use "
        "measurement_identity_mismatch when a Numerical measurement ID omits "
        "its measured quantity, including range_of_motion for degrees. These "
        "four policies require NEEDS_HUMAN_REVIEW for an exact existing concept; "
        "name the preferred reusable ID in the English explanation so a governed "
        "migration can be planned.\n\n"
        "Aliases, descriptions, matched text, and retrieval scores explain "
        "retrieval; they do not prove equivalence. Compare disease state, "
        "severity, permanence, temporal anchor, subtype, substance class, "
        "measurement, and value schema. Never strengthen impairment to failure, "
        "problematic use or abuse to addiction, or generic use to a disorder.\n\n"
        "In general, pre-existing is enrollment context and does not enter a new "
        "Attribute ID. If an existing pre_existing_* candidate is clinically "
        "equivalent but pre-existing is merely contextual, return "
        "exact_equivalent together with contextual_pre_existing_qualifier and "
        "REQUIRE_NEW. Do not falsely label the clinical concepts as "
        "non-equivalent. The correction must propose the bare reusable concept. "
        "The word other in a coordinated residual category is also contextual "
        "and does not enter the new ID.\n\n"
        "Return exactly one decision. CONFIRM_EXISTING, CONFIRM_NEW, and "
        "SWITCH_EXISTING require exact_equivalent plus compliant. REQUIRE_NEW "
        "requires either a non-equivalent semantic relationship or "
        "exact_equivalent plus contextual_pre_existing_qualifier. RETRIEVE_AGAIN "
        "requires no_supported_target, not_applicable, and non-empty search terms. "
        "SEGMENTATION_ISSUE requires segmentation_issue and not_applicable; use "
        "it when material AND/OR grouping in context is lost by the atomic row. "
        "If deleting a follow-up timepoint would change meaning and there is no "
        "structured timepoint field, use NEEDS_HUMAN_REVIEW. "
        "NEEDS_HUMAN_REVIEW blocks uncertain cases. Write a concise 2-4 sentence "
        "explanation in English.\n"
        "MAPPING_REVIEW_INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def validate_mapping_review(
    review: EvaMappingReview | Mapping[str, Any],
    *,
    selection: EvaReasoningSelection,
    candidates: list[EvaAttributeCandidate],
    review_cycle: int,
) -> EvaMappingReview:
    """Validate one Mapping Review response against its current lineage."""

    validated = (
        review
        if isinstance(review, EvaMappingReview)
        else EvaMappingReview.model_validate(review)
    )
    if review_cycle not in {1, 2}:
        raise ValueError("Mapping review_cycle must be 1 or 2.")
    if validated.current_mapping_decision != selection.mapping_decision:
        raise ValueError("Mapping Review current decision does not match selection.")

    _validate_review_dimensions(validated)
    candidate_ids = {candidate.attribute_id for candidate in candidates}
    target_id = (validated.target_attribute_id or "").strip() or None
    terms = [term.strip() for term in validated.additional_search_terms if term.strip()]
    decision = validated.decision
    expected_reviewed = {
        "CONFIRM_EXISTING": "existing_attribute",
        "CONFIRM_NEW": "new_attribute",
        "SWITCH_EXISTING": "existing_attribute",
        "REQUIRE_NEW": "new_attribute",
    }.get(decision)
    if expected_reviewed and validated.reviewed_mapping_decision != expected_reviewed:
        raise ValueError("Mapping Review decision is inconsistent with reviewed mapping.")
    if decision == "CONFIRM_EXISTING":
        expected_id = selection.attribute_id
        if selection.mapping_decision != "existing_attribute":
            raise ValueError("CONFIRM_EXISTING requires an existing selection.")
        if target_id not in {None, expected_id}:
            raise ValueError("CONFIRM_EXISTING cannot change the Attribute ID.")
    elif decision == "CONFIRM_NEW":
        if selection.mapping_decision != "new_attribute":
            raise ValueError("CONFIRM_NEW requires a new selection.")
        if target_id not in {None, selection.attribute_id}:
            raise ValueError("CONFIRM_NEW cannot change the Attribute ID.")
    elif decision == "SWITCH_EXISTING":
        if target_id not in candidate_ids:
            raise ValueError("SWITCH_EXISTING target must be an active candidate.")
    elif decision == "RETRIEVE_AGAIN" and not terms:
        raise ValueError("RETRIEVE_AGAIN requires additional search terms.")

    return validated.model_copy(
        update={
            "target_attribute_id": target_id,
            "additional_search_terms": terms,
            "explanation": validate_english_explanation(
                validated.explanation,
                field_name="Mapping Review",
            ),
        }
    )


_NON_EQUIVALENT_RELATIONSHIPS = {
    "broader_than_source",
    "narrower_than_source",
    "related_not_equivalent",
    "no_supported_target",
}
_AUTOMATIC_NEW_ID_POLICY_VIOLATIONS = {
    "contextual_pre_existing_qualifier",
}
_MIGRATION_REQUIRED_ID_POLICY_VIOLATIONS = {
    "opaque_generated_id",
    "unsupported_specimen_qualifier",
    "trial_specific_qualifier",
    "measurement_identity_mismatch",
}


def _validate_review_dimensions(review: EvaMappingReview) -> None:
    """Enforce the mapping decision's independent semantic/policy matrix."""

    decision = review.decision
    relationship = review.semantic_relationship
    policy = review.attribute_id_policy_status
    if (
        relationship == "exact_equivalent"
        and policy in _MIGRATION_REQUIRED_ID_POLICY_VIOLATIONS
        and decision != "NEEDS_HUMAN_REVIEW"
    ):
        raise ValueError(
            "Exact existing concepts with migration-required ID-policy issues "
            "must use NEEDS_HUMAN_REVIEW."
        )
    if decision in {"CONFIRM_EXISTING", "CONFIRM_NEW", "SWITCH_EXISTING"}:
        if relationship != "exact_equivalent" or policy != "compliant":
            raise ValueError(
                f"{decision} requires an exact relationship and compliant ID policy."
            )
    elif decision == "REQUIRE_NEW":
        semantic_mismatch = relationship in _NON_EQUIVALENT_RELATIONSHIPS
        contextual_id_mismatch = (
            relationship == "exact_equivalent"
            and policy in _AUTOMATIC_NEW_ID_POLICY_VIOLATIONS
        )
        if not (semantic_mismatch or contextual_id_mismatch):
            raise ValueError(
                "REQUIRE_NEW requires a semantic mismatch or contextual "
                "pre_existing ID-policy violation."
            )
    elif decision == "RETRIEVE_AGAIN":
        if relationship != "no_supported_target" or policy != "not_applicable":
            raise ValueError(
                "RETRIEVE_AGAIN requires no_supported_target and not_applicable."
            )
    elif decision == "SEGMENTATION_ISSUE":
        if relationship != "segmentation_issue" or policy != "not_applicable":
            raise ValueError(
                "SEGMENTATION_ISSUE requires segmentation_issue and not_applicable."
            )


def build_mapping_correction_prompt(
    *,
    source_item_id: str,
    row: Mapping[str, Any],
    locked_decision: str,
    locked_attribute_id: str | None,
    review: EvaMappingReview,
    candidates: Sequence[EvaAttributeCandidate],
    entities: Sequence[Any],
) -> str:
    """Build a correction prompt with an immutable mapping target."""

    def dump(value: Any) -> Any:
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else value

    payload = {
        "source_item_id": source_item_id,
        "locked_mapping_decision": locked_decision,
        "locked_attribute_id": locked_attribute_id,
        "source_row": dict(row),
        "mapping_review": review.model_dump(mode="json"),
        "candidates": [dump(candidate) for candidate in candidates],
        "entities": [dump(entity) for entity in entities],
    }
    return (
        "Correct the Eligibility EVA proposal while preserving the locked "
        "mapping decision and Attribute ID exactly. Recompute all value fields "
        "from the source row and follow the Mapping Review. Set "
        "correction_acknowledged=true and write the rationale in English.\n\n"
        f"{CLINICAL_ATTRIBUTE_NAMING_POLICY}\n\n"
        "The locked mapping decision and locked Attribute ID are immutable. "
        "For a new Attribute, produce terminology-only aliases and keep every "
        "operator, threshold, endpoint, and unit in the value fields.\n"
        "MAPPING_CORRECTION_INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


_ID_REVIEW_RULES = f"""You are the Attribute ID naming reviewer for clinical-trial eligibility criteria. Use only the atomic criterion title and current Attribute ID as semantic evidence.

{CLINICAL_ATTRIBUTE_NAMING_POLICY}

An Attribute ID names one stable reusable clinical variable, not the full eligibility sentence. Preserve disease or event subtype, validated scale plus score, a necessary time_since anchor, self_reported, persistent, impairment, failure, disorder, dependence, and addiction when those words define the clinical variable. Remove comparison operators, cutoffs, units, polarity, severity thresholds, support amounts, and contextual residual words.

In general, remove pre-existing because it describes enrollment context rather than Attribute identity. Preserve it only when pre-existence itself is clearly the reusable variable; use NEEDS_HUMAN_REVIEW if the title cannot establish that distinction. Remove other from coordinated residual categories.

Do not strengthen meaning: impairment is not failure; abuse or a drug use disorder is not automatically addiction; generic drug use is not automatically a disorder. Prefer drug_use_disorder for a new non-alcohol umbrella concept that explicitly describes clinically relevant problematic drug use without proving addiction.

The final ID must be concise lowercase snake_case and reusable across trials. Prefer ability_to_<verb>_<object> for capability concepts.

Calibration examples:
1. ischaemic stroke at least 3 months ago + time_since_stroke_at_enrollment => MODIFY time_since_ischemic_stroke_at_enrollment
2. TIA at least 3 months ago + time_since_transient_ischemic_attack => MODIFY time_since_transient_ischemic_attack_at_enrollment
3. persistent self-reported fatigue + persistent_self_reported_fatigue => KEEP
4. MFI-20 score of 12 or more + mfi_20_score => KEEP
5. modified Rankin Score of 3 or less + modified_rankin_scale => MODIFY modified_rankin_scale_score
6. can speak reasonable English + english_language_proficiency => MODIFY ability_to_speak_english
7. understand instructions + ability_to_understand_instructions => KEEP
8. complete tests independently/minimal support + ability_to_complete_study_assessments => MODIFY ability_to_complete_tests
9. complete questionnaires independently/minimal support + questionnaire_completion_ability => MODIFY ability_to_complete_questionnaires
10. pre-existing dementia + dementia => KEEP
11. pre-existing dementia + pre_existing_dementia => MODIFY dementia
12. pre-existing other neuropsychiatric disease + pre_existing_other_neuropsychiatric_disease => MODIFY neuropsychiatric_disease
13. drug abuse + drug_addiction => MODIFY drug_use_disorder
14. drug addiction + drug_addiction => KEEP
15. hepatic impairment + hepatic_failure => MODIFY hepatic_impairment
16. hepatic failure + hepatic_failure => KEEP
17. persistent focal neurological deficit severe enough to warrant treatment + clinically_significant_persistent_focal_neurological_deficit => MODIFY persistent_focal_neurological_deficit
18. persistent focal neurological deficit + attribute_0863e915208d6860 => MODIFY persistent_focal_neurological_deficit
19. >= 10 degrees of active wrist extension + active_wrist_extension => MODIFY active_wrist_extension_range_of_motion
20. >= 10 degrees of extension of all thumb joints + thumb_joint_extension_range_of_motion => MODIFY active_thumb_joint_extension_range_of_motion
21. TGP >2N + serum_alt_level => MODIFY alanine_aminotransferase
22. TGO >2N + aspartate_aminotransferase => KEEP
23. unlikely to be available for follow up at 12 months + follow_up_availability => NEEDS_HUMAN_REVIEW and preserve the current ID because no separate follow-up timepoint field exists

KEEP must preserve the current ID exactly. MODIFY must return one final ID. NEEDS_HUMAN_REVIEW must preserve the current ID. This reviewer is invoked only for a new Attribute proposal; never infer that the current ID is an approved existing mapping. Therefore an opaque generated ID in a new proposal must be modified to the stable concept name as in example 18. Approved existing mappings bypass this reviewer and are handled by Mapping Review as governed migration. Write a concise 2-4 sentence explanation in English."""


def build_attribute_id_review_prompt(
    *,
    source_item_id: str,
    atomic_criterion_title: str,
    current_attribute_id: str,
    review_cycle: int,
    selection_fingerprint: str,
    prior_error: str = "",
) -> str:
    """Build the calibrated ID prompt with only two semantic input fields."""

    payload = {
        "source_item_id": source_item_id,
        "review_cycle": review_cycle,
        "selection_fingerprint": selection_fingerprint,
        "atomic_criterion_title": atomic_criterion_title,
        "current_attribute_id": current_attribute_id,
        "prior_validation_error": prior_error,
    }
    return (
        _ID_REVIEW_RULES
        + "\nATTRIBUTE_ID_REVIEW_INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


_SNAKE_CASE_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CONSTRAINT_TOKEN = re.compile(
    r"(?:^|_)(?:at_least|at_most|more_than|less_than|or_more|or_less|"
    r"minimum|maximum)(?:_|$)|(?:^|_)\d+(?:_\d+)?_"
    r"(?:day|days|week|weeks|month|months|year|years|percent|percentage|"
    r"mg|g|kg|ml|mmhg)(?:_|$)"
)


def validate_attribute_id_review(
    review: EvaAttributeIdReview | Mapping[str, Any],
    *,
    current_attribute_id: str,
) -> EvaAttributeIdReview:
    """Validate KEEP/MODIFY identity and stable-variable naming constraints."""

    validated = (
        review
        if isinstance(review, EvaAttributeIdReview)
        else EvaAttributeIdReview.model_validate(review)
    )
    current = current_attribute_id.strip()
    reviewed = validated.reviewed_attribute_id.strip()
    if not reviewed:
        raise ValueError("Attribute ID Review must provide one non-empty ID.")
    if validated.decision in {"KEEP", "NEEDS_HUMAN_REVIEW"}:
        if reviewed != current:
            raise ValueError(f"{validated.decision} must preserve the current ID.")
    elif reviewed == current:
        raise ValueError("MODIFY must provide a different Attribute ID.")
    if not _SNAKE_CASE_ID.fullmatch(reviewed):
        raise ValueError("Reviewed Attribute ID must use lowercase snake_case.")
    if stable_slug(reviewed, fallback="attribute") != reviewed:
        raise ValueError("Reviewed Attribute ID is not a stable snake-case ID.")
    if _CONSTRAINT_TOKEN.search(reviewed):
        raise ValueError("Reviewed Attribute ID contains a constraint token.")
    return validated.model_copy(
        update={
            "reviewed_attribute_id": reviewed,
            "explanation": validate_english_explanation(
                validated.explanation,
                field_name="Attribute ID Review",
            ),
        }
    )


def library_attribute_is_equivalent(
    library: EvaLibraryRepository,
    selection: EvaReasoningSelection,
) -> bool:
    """Return whether an active library definition matches a new proposal."""

    attribute = library.attributes.get(selection.attribute_id or "")
    if attribute is None or not attribute.active:
        return False
    return (
        attribute.entity_id == selection.entity_id
        and stable_slug(attribute.canonical_name, fallback="attribute")
        == stable_slug(selection.attribute_name, fallback="attribute")
        and attribute.value_type == selection.value_type
        and attribute.numerical_type == selection.numerical_type
        and attribute.canonical_unit == selection.unit
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_metadata(provider: Any) -> tuple[str, str]:
    return (
        str(getattr(provider, "model", "") or ""),
        str(getattr(provider, "reasoning_effort", "") or ""),
    )


def provider_metadata(provider: Any) -> tuple[str, str]:
    """Return stable public provider model and reasoning-effort lineage."""

    return _provider_metadata(provider)


def _mapping_attempt(
    review: EvaMappingReview,
    *,
    review_cycle: int,
    candidate_fingerprint: str,
    selection_fingerprint: str,
    provider: Any,
) -> EvaMappingReviewAttempt:
    model, effort = _provider_metadata(provider)
    return EvaMappingReviewAttempt(
        **review.model_dump(mode="json"),
        legacy_semantic_relationship="",
        explanation_language="en",
        review_cycle=review_cycle,
        candidate_set_fingerprint=candidate_fingerprint,
        selection_fingerprint=selection_fingerprint,
        model=model,
        reasoning_effort=effort,
        timestamp=str(
            getattr(provider, "last_completed_at", None) or _utc_now()
        ),
    )


def _call_mapping_reviewer(
    *,
    provider: Any,
    source_item_id: str,
    row: Mapping[str, Any],
    selection: EvaReasoningSelection,
    candidates: list[EvaAttributeCandidate],
    review_cycle: int,
    max_attempts: int,
) -> tuple[EvaMappingReview | None, dict[str, Any] | None]:
    candidate_fingerprint = candidate_set_sha256(candidates)
    selection_fingerprint = selection_sha256(selection)
    prior_error = ""
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        try:
            result = provider.generate_json(
                build_mapping_review_prompt(
                    source_item_id=source_item_id,
                    criteria=str(row["criteria"]),
                    item=str(row["item"]),
                    context=str(row["context"]),
                    selection=selection,
                    candidates=candidates,
                    review_cycle=review_cycle,
                    candidate_fingerprint=candidate_fingerprint,
                    selection_fingerprint=selection_fingerprint,
                    prior_error=prior_error,
                ),
                output_schema=MAPPING_REVIEW_OUTPUT_SCHEMA,
            )
            return (
                validate_mapping_review(
                    result,
                    selection=selection,
                    candidates=candidates,
                    review_cycle=review_cycle,
                ),
                None,
            )
        except Exception as exc:  # reviewer failures are item-level issues
            last_error = exc
            prior_error = str(exc)
    assert last_error is not None
    return None, {
        "type": type(last_error).__name__,
        "message": str(last_error),
        "review_cycle": review_cycle,
    }


def _correct_mapping_selection(
    *,
    library: EvaLibraryRepository,
    provider: Any,
    source_item_id: str,
    row: Mapping[str, Any],
    locked_decision: str,
    locked_attribute_id: str | None,
    review: EvaMappingReview,
    candidates: list[EvaAttributeCandidate],
    max_attempts: int,
) -> EvaReasoningSelection:
    from backend.criteria_processor.eva_pipeline import validate_reasoning_selection

    prior_error = ""
    for attempt in range(max_attempts):
        prompt = build_mapping_correction_prompt(
            source_item_id=source_item_id,
            row={**dict(row), "prior_validation_error": prior_error},
            locked_decision=locked_decision,
            locked_attribute_id=locked_attribute_id,
            review=review,
            candidates=candidates,
            entities=[entity for entity in library.entities.values() if entity.active],
        )
        result = provider.generate_json(
            prompt,
            output_schema=MAPPING_CORRECTION_OUTPUT_SCHEMA,
        )
        try:
            if result.get("correction_acknowledged") is not True:
                raise ValueError("Mapping correction was not acknowledged.")
            selection_payload = dict(result)
            selection_payload.pop("correction_acknowledged", None)
            corrected = EvaReasoningSelection.model_validate(selection_payload)
            if corrected.mapping_decision != locked_decision:
                raise ValueError("Correction changed the locked mapping decision.")
            if locked_attribute_id and corrected.attribute_id != locked_attribute_id:
                raise ValueError("Correction changed the locked Attribute ID.")
            return validate_reasoning_selection(
                selection=corrected,
                candidates=candidates,
                library=library,
                source_item=str(row["item"]),
            )
        except (TypeError, ValueError) as exc:
            prior_error = str(exc)
            if attempt + 1 == max_attempts:
                raise
    raise RuntimeError("No valid Mapping correction was produced.")


def review_eva_mapping(
    *,
    library: EvaLibraryRepository,
    reasoner_provider: Any,
    trial_id: str,
    source_item_id: str,
    row: Mapping[str, Any],
    expansion: EvaSearchExpansion,
    candidates: list[EvaAttributeCandidate],
    selection: EvaReasoningSelection,
    top_k: int,
    max_attempts: int = 2,
) -> tuple[
    EvaReasoningSelection,
    EvaSearchExpansion,
    list[EvaAttributeCandidate],
    EvaMappingReviewTrace,
]:
    """Run the bounded two-cycle Mapping Review state machine."""

    from backend.criteria_processor.eva_pipeline import (
        REASONING_OUTPUT_SCHEMA,
        build_reasoning_prompt,
        normalize_search_expansion,
        retrieve_attribute_candidates,
        validate_reasoning_selection,
    )

    initial = selection.model_copy(deep=True)
    current = selection
    current_expansion = expansion
    current_candidates = candidates
    attempts: list[EvaMappingReviewAttempt] = []
    final_decision = None
    status = "completed"
    error = None

    for cycle in (1, 2):
        review, review_error = _call_mapping_reviewer(
            provider=reasoner_provider,
            source_item_id=source_item_id,
            row=row,
            selection=current,
            candidates=current_candidates,
            review_cycle=cycle,
            max_attempts=max_attempts,
        )
        if review is None:
            status = "failed"
            error = review_error
            break
        attempts.append(
            _mapping_attempt(
                review,
                review_cycle=cycle,
                candidate_fingerprint=candidate_set_sha256(current_candidates),
                selection_fingerprint=selection_sha256(current),
                provider=reasoner_provider,
            )
        )
        final_decision = review.decision
        if review.decision in {"CONFIRM_EXISTING", "CONFIRM_NEW"}:
            break
        if review.decision == "SWITCH_EXISTING":
            try:
                current = _correct_mapping_selection(
                    library=library,
                    provider=reasoner_provider,
                    source_item_id=source_item_id,
                    row=row,
                    locked_decision="existing_attribute",
                    locked_attribute_id=review.target_attribute_id,
                    review=review,
                    candidates=current_candidates,
                    max_attempts=max_attempts,
                )
            except Exception as exc:
                status = "failed"
                error = {"type": type(exc).__name__, "message": str(exc), "review_cycle": cycle}
            break
        if cycle == 2:
            status = "blocked"
            error = {
                "type": "UnresolvedMappingReview",
                "message": f"Cycle 2 ended with {review.decision}.",
                "review_cycle": 2,
            }
            break
        if review.decision == "REQUIRE_NEW":
            try:
                current = _correct_mapping_selection(
                    library=library,
                    provider=reasoner_provider,
                    source_item_id=source_item_id,
                    row=row,
                    locked_decision="new_attribute",
                    locked_attribute_id=None,
                    review=review,
                    candidates=current_candidates,
                    max_attempts=max_attempts,
                )
            except Exception as exc:
                status = "failed"
                error = {"type": type(exc).__name__, "message": str(exc), "review_cycle": cycle}
                break
            break
        if review.decision == "RETRIEVE_AGAIN":
            try:
                current_expansion = normalize_search_expansion(
                    current_expansion.model_copy(
                        deep=True,
                        update={
                            "semantic_terms": [
                                *current_expansion.semantic_terms,
                                *review.additional_search_terms,
                            ]
                        },
                    )
                )
                current_candidates = retrieve_attribute_candidates(
                    library=library,
                    expansion=current_expansion,
                    source_item=str(row["item"]),
                    top_k=top_k,
                )
                prior_error = ""
                for reason_attempt in range(max_attempts):
                    result = reasoner_provider.generate_json(
                        build_reasoning_prompt(
                        trial_id=trial_id,
                        source_item_id=source_item_id,
                        criteria=str(row["criteria"]),
                        item=str(row["item"]),
                        context=str(row["context"]),
                        expansion=current_expansion,
                        candidates=current_candidates,
                        entities=[
                            entity.model_dump(mode="json")
                            for entity in library.entities.values()
                            if entity.active
                        ],
                        prior_error=prior_error,
                        ),
                        output_schema=REASONING_OUTPUT_SCHEMA,
                    )
                    try:
                        current = validate_reasoning_selection(
                            selection=EvaReasoningSelection.model_validate(result),
                            candidates=current_candidates,
                            library=library,
                            source_item=str(row["item"]),
                        )
                        break
                    except (TypeError, ValueError) as exc:
                        prior_error = str(exc)
                        if reason_attempt + 1 == max_attempts:
                            raise
            except Exception as exc:
                status = "failed"
                error = {"type": type(exc).__name__, "message": str(exc), "review_cycle": cycle}
                break
            continue
        status = "blocked"
        error = {
            "type": review.decision,
            "message": review.explanation,
            "review_cycle": cycle,
        }
        break

    trace = EvaMappingReviewTrace(
        status=status,
        initial_mapping_decision=initial.mapping_decision,
        initial_attribute_id=initial.attribute_id or "",
        final_mapping_decision=current.mapping_decision,
        final_attribute_id=current.attribute_id or "",
        final_decision=final_decision,
        attempts=attempts,
        error=error,
    )
    return current, current_expansion, current_candidates, trace


def review_new_attribute_id(
    *,
    reasoner_provider: Any,
    source_item_id: str,
    atomic_criterion_title: str,
    selection: EvaReasoningSelection,
    library: EvaLibraryRepository,
    max_attempts: int = 2,
) -> tuple[EvaReasoningSelection, EvaAttributeIdReviewTrace]:
    """Review a final new Attribute ID without exposing other semantics."""

    current_id = selection.attribute_id or ""
    fingerprint = selection_sha256(selection)
    if selection.mapping_decision == "existing_attribute":
        return selection, EvaAttributeIdReviewTrace(
            status="not_required",
            current_attribute_id=current_id,
            reviewed_attribute_id=current_id,
            explanation=ID_REVIEW_NOT_REQUIRED_EXPLANATION,
            explanation_language="en",
            selection_fingerprint=fingerprint,
        )
    prior_error = ""
    last_error: Exception | None = None
    for cycle in range(1, max_attempts + 1):
        try:
            result = reasoner_provider.generate_json(
                build_attribute_id_review_prompt(
                    source_item_id=source_item_id,
                    atomic_criterion_title=atomic_criterion_title,
                    current_attribute_id=current_id,
                    review_cycle=cycle,
                    selection_fingerprint=fingerprint,
                    prior_error=prior_error,
                ),
                output_schema=ATTRIBUTE_ID_REVIEW_OUTPUT_SCHEMA,
            )
            review = validate_attribute_id_review(
                result,
                current_attribute_id=current_id,
            )
            status = (
                "blocked"
                if review.decision == "NEEDS_HUMAN_REVIEW"
                else "completed"
            )
            reviewed_selection = selection.model_copy(deep=True)
            if review.decision == "MODIFY":
                reviewed_selection.attribute_id = review.reviewed_attribute_id
            model, effort = _provider_metadata(reasoner_provider)
            return reviewed_selection, EvaAttributeIdReviewTrace(
                status=status,
                current_attribute_id=current_id,
                reviewed_attribute_id=review.reviewed_attribute_id,
                decision=review.decision,
                explanation=review.explanation,
                explanation_language="en",
                confidence=review.confidence,
                selection_fingerprint=fingerprint,
                model=model,
                reasoning_effort=effort,
                timestamp=str(
                    getattr(reasoner_provider, "last_completed_at", None)
                    or _utc_now()
                ),
                error=(
                    {
                        "type": "NEEDS_HUMAN_REVIEW",
                        "message": review.explanation,
                    }
                    if status == "blocked"
                    else None
                ),
            )
        except Exception as exc:  # reviewer failures are item-level issues
            last_error = exc
            prior_error = str(exc)
    assert last_error is not None
    model, effort = _provider_metadata(reasoner_provider)
    return selection, EvaAttributeIdReviewTrace(
        status="failed",
        current_attribute_id=current_id,
        reviewed_attribute_id=current_id,
        explanation=(
            "Attribute ID Review failed before valid English evidence was produced."
        ),
        explanation_language="en",
        selection_fingerprint=fingerprint,
        model=model,
        reasoning_effort=effort,
        timestamp=str(
            getattr(reasoner_provider, "last_completed_at", None)
            or _utc_now()
        ),
        error={"type": type(last_error).__name__, "message": str(last_error)},
    )
