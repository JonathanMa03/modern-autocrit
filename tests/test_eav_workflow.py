import csv
import json
import re
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.criteria_processor.atomic_io import atomic_write_json
from backend.criteria_processor.eva_audit_migration import (
    load_eva_audit,
    migrate_eva_audit_v1,
)
from backend.criteria_processor.eva_jobs import EligibilityEvaJobManager
from backend.criteria_processor.eva_library import (
    EvaAttributeIdConflictError,
    EvaLibraryRepository,
    EvaLibraryStaleError,
)
from backend.criteria_processor.eva_models import (
    EvaAttributeDefinition,
    EvaAttributeIdReview,
    EvaMappingReview,
    EvaReasoningSelection,
    EvaSearchExpansion,
)
from backend.criteria_processor.eva_pipeline import (
    REASONER_PROMPT_VERSION,
    _is_reusable_attribute_alias,
    build_search_prompt,
    build_reasoning_prompt,
    extract_trial_eva,
    finalize_trial_attribute_id_conflicts,
    retrieve_attribute_candidates,
    validate_reasoning_selection,
)
from backend.criteria_processor.eva_retrieval import (
    EvaReconciliationRetrievalQuery,
    EvaRetrievalTerm,
    score_reconciliation_candidate,
)
from backend.criteria_processor.eva_recovery import (
    _assert_single_review_contract,
    recover_trial_eva_from_runtime_records,
)
from backend.criteria_processor.eva_review import (
    ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
    MAPPING_CORRECTION_PROMPT_VERSION,
    MAPPING_REVIEW_OUTPUT_SCHEMA,
    MAPPING_REVIEW_PROMPT_VERSION,
    _validate_review_dimensions,
    ID_REVIEW_NOT_REQUIRED_EXPLANATION,
    build_attribute_id_review_prompt,
    build_mapping_correction_prompt,
    build_mapping_review_prompt,
    candidate_set_sha256,
    review_eva_mapping,
    review_new_attribute_id,
    selection_sha256,
    validate_attribute_id_review,
    validate_english_explanation,
)
from backend.criteria_processor.llm_providers import (
    RuntimeRecordingJsonLLMProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _legacy_audit_payload() -> dict:
    return {
        "schema_version": "eligcrit.eligibility_eva_audit.v1",
        "audit_id": "eva_audit_test",
        "trial_key": "TEST",
        "trial_id": "TEST",
        "source_path": "points/TEST/elig_breakdown_points.csv",
        "library_revision": 0,
        "library_sha256": "abc123",
        "created_at": "2026-08-10T00:00:00+00:00",
        "updated_at": "2026-08-10T00:00:00+00:00",
        "review_status": "pending",
        "search_model": "search-model",
        "search_reasoning_effort": "low",
        "reasoner_model": "reasoner-model",
        "reasoner_reasoning_effort": "medium",
        "items": [
            {
                "eva_id": "eva_test_0001",
                "source_item_id": "TEST:0001",
                "criteria": "inclusion",
                "item": "non-pregnant",
                "context": "Participants must be non-pregnant.",
                "search_expansion": {},
                "candidates": [],
                "mapping_decision": "new_attribute",
                "attribute_id": "pregnancy",
                "entity_id": "contraceptive",
                "entity_name": "Contraceptive",
                "attribute_name": "pregnancy",
                "attribute_description": "Pregnancy status.",
                "value_type": "Categorical",
                "categorical_value": "Excluded",
                "confidence": 0.9,
                "rationale": "Test rationale.",
            }
        ],
    }


def test_v1_audit_loads_as_v3():
    audit = load_eva_audit(_legacy_audit_payload())

    assert audit.schema_version == "eligcrit.eligibility_eva_audit.v3"
    assert audit.review_mode == "off"
    assert audit.items[0].mapping_review.status == "legacy_unreviewed"
    assert audit.items[0].attribute_id_review.status == "legacy_unreviewed"


def test_existing_v3_audit_without_reasoner_prompt_version_still_loads():
    payload = load_eva_audit(_legacy_audit_payload()).model_dump(mode="json")
    payload["review_prompt_versions"] = {
        "mapping_review": "eligibility-eva-mapping-review-v2"
    }

    audit = load_eva_audit(payload)

    assert "reasoner" not in audit.review_prompt_versions
    assert audit.review_prompt_versions["mapping_review"].endswith("-v2")


def test_v2_unknown_relationship_is_preserved_without_inference():
    payload = migrate_eva_audit_v1(_legacy_audit_payload())
    payload["items"][0]["mapping_review"]["attempts"] = [
        {
            "decision": "CONFIRM_NEW",
            "current_mapping_decision": "new_attribute",
            "reviewed_mapping_decision": "new_attribute",
            "target_attribute_id": "pregnancy",
            "semantic_relationship": "当前概念完全对应。",
            "additional_search_terms": [],
            "rationale_zh": "旧版中文解释。",
            "confidence": 0.9,
            "review_cycle": 1,
            "candidate_set_fingerprint": "candidate-hash",
            "selection_fingerprint": "selection-hash",
            "model": "legacy-model",
            "reasoning_effort": "medium",
            "timestamp": "2026-08-10T00:00:00+00:00",
        }
    ]
    payload["items"][0]["attribute_id_review"].update(
        {
            "status": "completed",
            "decision": "KEEP",
            "notes_zh": "旧版 ID 说明。",
            "confidence": 0.9,
        }
    )

    audit = load_eva_audit(payload)
    attempt = audit.items[0].mapping_review.attempts[0]
    id_trace = audit.items[0].attribute_id_review

    assert attempt.semantic_relationship == "legacy_unverified"
    assert attempt.legacy_semantic_relationship == "当前概念完全对应。"
    assert attempt.attribute_id_policy_status == "legacy_unverified"
    assert attempt.explanation == "旧版中文解释。"
    assert attempt.explanation_language == "legacy_unverified"
    assert id_trace.explanation == "旧版 ID 说明。"
    assert id_trace.explanation_language == "legacy_unverified"


def test_v2_review_decisions_are_strict_literals():
    payload = load_eva_audit(_legacy_audit_payload()).model_dump(mode="json")
    payload["items"][0]["mapping_review"]["decision"] = "false"

    with pytest.raises(ValidationError):
        load_eva_audit(payload)


def _minimal_selection() -> EvaReasoningSelection:
    return EvaReasoningSelection(
        mapping_decision="new_attribute",
        attribute_id="hepatic_impairment",
        entity_id="comorbidity",
        entity_name="Comorbidity",
        new_entity=False,
        attribute_name="hepatic impairment",
        attribute_description="Impaired hepatic function.",
        attribute_aliases=[],
        value_type="Categorical",
        categorical_value="Excluded",
        confidence=0.9,
        rationale="The criterion excludes participants with hepatic impairment.",
    )


def _minimal_reasoning_prompt() -> str:
    return build_reasoning_prompt(
        trial_id="TEST",
        source_item_id="TEST:0001",
        criteria="exclusion",
        item="hepatic impairment",
        context="Participants with hepatic impairment are excluded.",
        expansion=EvaSearchExpansion(),
        candidates=[],
        entities=[],
    )


def _minimal_mapping_review_prompt() -> str:
    selection = _minimal_selection()
    return build_mapping_review_prompt(
        source_item_id="TEST:0001",
        criteria="exclusion",
        item="hepatic impairment",
        context="Participants with hepatic impairment are excluded.",
        selection=selection,
        candidates=[],
        review_cycle=1,
        candidate_fingerprint=candidate_set_sha256([]),
        selection_fingerprint=selection_sha256(selection),
    )


def _minimal_mapping_correction_prompt() -> str:
    review = EvaMappingReview(
        decision="REQUIRE_NEW",
        current_mapping_decision="existing_attribute",
        reviewed_mapping_decision="new_attribute",
        target_attribute_id=None,
        semantic_relationship="narrower_than_source",
        attribute_id_policy_status="compliant",
        additional_search_terms=[],
        explanation=(
            "Hepatic failure is narrower and more severe than hepatic impairment."
        ),
        confidence=0.99,
    )
    return build_mapping_correction_prompt(
        source_item_id="TEST:0001",
        row={
            "criteria": "exclusion",
            "item": "hepatic impairment",
            "context": "Participants with hepatic impairment are excluded.",
        },
        locked_decision="new_attribute",
        locked_attribute_id=None,
        review=review,
        candidates=[],
        entities=[],
    )


@pytest.mark.parametrize(
    ("name", "prompt", "marker"),
    [
        (
            "search",
            build_search_prompt(
                trial_id="TEST",
                source_item_id="TEST:0001",
                criteria="exclusion",
                item="hepatic impairment",
                context="",
            ),
            "EVIDENCE_JSON:\n",
        ),
        ("reasoner", _minimal_reasoning_prompt(), "SELECTION_INPUT_JSON:\n"),
        ("mapping", _minimal_mapping_review_prompt(), "MAPPING_REVIEW_INPUT_JSON:\n"),
        (
            "correction",
            _minimal_mapping_correction_prompt(),
            "MAPPING_CORRECTION_INPUT_JSON:\n",
        ),
        (
            "attribute_id",
            build_attribute_id_review_prompt(
                source_item_id="TEST:0001",
                atomic_criterion_title="hepatic impairment",
                current_attribute_id="hepatic_failure",
                review_cycle=1,
                selection_fingerprint="selection-hash",
            ),
            "ATTRIBUTE_ID_REVIEW_INPUT_JSON:\n",
        ),
    ],
)
def test_all_eva_prompt_instructions_are_english(name, prompt, marker):
    instructions = prompt.split(marker, 1)[0]
    assert not re.search(r"[\u3400-\u9fff]", instructions), name


@pytest.mark.parametrize("text", ["", "   ", "当前映射准确。"])
def test_new_review_explanations_require_english(text):
    with pytest.raises(ValueError, match="English|empty"):
        validate_english_explanation(text, field_name="Mapping Review")


def test_english_clinical_explanation_is_accepted():
    assert validate_english_explanation(
        "Hepatic failure is narrower and more severe than hepatic impairment.",
        field_name="Mapping Review",
    ).startswith("Hepatic failure")


@pytest.mark.parametrize(
    ("decision", "relationship", "policy"),
    [
        ("CONFIRM_EXISTING", "narrower_than_source", "compliant"),
        (
            "CONFIRM_EXISTING",
            "exact_equivalent",
            "contextual_pre_existing_qualifier",
        ),
        ("REQUIRE_NEW", "exact_equivalent", "compliant"),
        ("RETRIEVE_AGAIN", "no_supported_target", "compliant"),
        ("SEGMENTATION_ISSUE", "segmentation_issue", "compliant"),
    ],
)
def test_mapping_review_rejects_inconsistent_dimensions(
    decision, relationship, policy
):
    review = EvaMappingReview(
        decision=decision,
        current_mapping_decision="existing_attribute",
        reviewed_mapping_decision=(
            "new_attribute" if decision == "REQUIRE_NEW" else "existing_attribute"
        ),
        target_attribute_id="pre_existing_dementia",
        semantic_relationship=relationship,
        attribute_id_policy_status=policy,
        additional_search_terms=(
            ["dementia"] if decision == "RETRIEVE_AGAIN" else []
        ),
        explanation="The relationship and decision are intentionally inconsistent.",
        confidence=0.9,
    )

    with pytest.raises(ValueError, match="relationship|policy|not_applicable"):
        _validate_review_dimensions(review)


@pytest.mark.parametrize(
    ("decision", "relationship", "policy"),
    [
        ("CONFIRM_EXISTING", "exact_equivalent", "compliant"),
        (
            "REQUIRE_NEW",
            "exact_equivalent",
            "contextual_pre_existing_qualifier",
        ),
        ("REQUIRE_NEW", "narrower_than_source", "compliant"),
        ("RETRIEVE_AGAIN", "no_supported_target", "not_applicable"),
        ("SEGMENTATION_ISSUE", "segmentation_issue", "not_applicable"),
    ],
)
def test_mapping_review_accepts_valid_review_dimensions(
    decision, relationship, policy
):
    review = EvaMappingReview(
        decision=decision,
        current_mapping_decision="existing_attribute",
        reviewed_mapping_decision=(
            "new_attribute" if decision == "REQUIRE_NEW" else "existing_attribute"
        ),
        target_attribute_id="pre_existing_dementia",
        semantic_relationship=relationship,
        attribute_id_policy_status=policy,
        additional_search_terms=(
            ["dementia"] if decision == "RETRIEVE_AGAIN" else []
        ),
        explanation="The relationship and policy support this decision.",
        confidence=0.9,
    )

    _validate_review_dimensions(review)


@pytest.mark.parametrize(
    "policy",
    [
        "opaque_generated_id",
        "unsupported_specimen_qualifier",
        "trial_specific_qualifier",
        "measurement_identity_mismatch",
    ],
)
def test_mapping_review_accepts_current_id_policy_literals(policy):
    review = EvaMappingReview(
        decision="NEEDS_HUMAN_REVIEW",
        current_mapping_decision="existing_attribute",
        reviewed_mapping_decision="existing_attribute",
        target_attribute_id="legacy_attribute",
        semantic_relationship="exact_equivalent",
        attribute_id_policy_status=policy,
        additional_search_terms=[],
        explanation="The exact concept requires governed ID migration.",
        confidence=0.95,
    )

    _validate_review_dimensions(review)
    assert review.attribute_id_policy_status == policy


def test_mapping_review_schema_exposes_all_current_id_policy_values():
    assert set(
        MAPPING_REVIEW_OUTPUT_SCHEMA["properties"]
        ["attribute_id_policy_status"]["enum"]
    ) == {
        "compliant",
        "contextual_pre_existing_qualifier",
        "opaque_generated_id",
        "unsupported_specimen_qualifier",
        "trial_specific_qualifier",
        "measurement_identity_mismatch",
        "not_applicable",
    }


@pytest.mark.parametrize(
    "policy",
    [
        "opaque_generated_id",
        "unsupported_specimen_qualifier",
        "trial_specific_qualifier",
        "measurement_identity_mismatch",
    ],
)
@pytest.mark.parametrize("decision", ["CONFIRM_EXISTING", "REQUIRE_NEW"])
def test_exact_migration_policy_requires_human_review(policy, decision):
    review = EvaMappingReview(
        decision=decision,
        current_mapping_decision="existing_attribute",
        reviewed_mapping_decision=(
            "new_attribute" if decision == "REQUIRE_NEW" else "existing_attribute"
        ),
        target_attribute_id=None,
        semantic_relationship="exact_equivalent",
        attribute_id_policy_status=policy,
        additional_search_terms=[],
        explanation="The existing ID requires governed migration.",
        confidence=0.95,
    )

    with pytest.raises(ValueError, match="NEEDS_HUMAN_REVIEW"):
        _validate_review_dimensions(review)


def test_mapping_prompt_separates_semantics_from_id_policy():
    prompt = _minimal_mapping_review_prompt()

    assert "clinically exact yet still require a new Attribute" in prompt
    assert "contextual_pre_existing_qualifier" in prompt
    assert "Do not falsely label the clinical concepts as non-equivalent" in prompt
    assert "aliases" in prompt.casefold()
    assert "retrieval" in prompt.casefold()


def test_mapping_review_v3_covers_general_naming_and_logic_rules():
    prompt = _minimal_mapping_review_prompt()

    for text in (
        "opaque_generated_id",
        "unsupported_specimen_qualifier",
        "trial_specific_qualifier",
        "measurement_identity_mismatch",
        "range_of_motion",
        "alanine_aminotransferase",
        "AND/OR grouping",
        "follow-up timepoint",
    ):
        assert text in prompt


def test_mapping_correction_v3_reuses_clinical_naming_policy():
    prompt = _minimal_mapping_correction_prompt()

    assert "stable reusable variable" in prompt
    assert "range_of_motion" in prompt
    assert "alanine_aminotransferase" in prompt
    assert "Do not infer" in prompt and "specimen" in prompt
    assert "locked mapping decision" in prompt
    assert "locked Attribute ID" in prompt


def test_reasoner_prompt_contains_shared_clinical_naming_policy():
    prompt = _minimal_reasoning_prompt()

    for text in (
        "stable reusable variable",
        "trial-specific severity",
        "unsupported specimen",
        "range_of_motion",
        "active versus passive",
        "alanine_aminotransferase",
        "Aliases contain terminology synonyms",
        "Boolean grouping",
        "follow-up timepoint",
    ):
        assert text in prompt


@pytest.mark.parametrize(
    "alias",
    ["TGP >2N", "TGO >= 2 x ULN", ">= 10 degrees", "MFI-20 score >= 12"],
)
def test_constraint_text_is_not_a_reusable_alias(alias):
    assert _is_reusable_attribute_alias(alias) is False


@pytest.mark.parametrize(
    "alias",
    ["ALT", "TGP", "SGPT", "alanine aminotransferase", "MFI-20 score"],
)
def test_terminology_is_a_reusable_alias(alias):
    assert _is_reusable_attribute_alias(alias) is True


def test_reasoning_validation_removes_threshold_source_alias(tmp_path):
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    selection = EvaReasoningSelection(
        mapping_decision="new_attribute",
        attribute_id="alanine_aminotransferase",
        entity_id="lab_test",
        entity_name="Lab test",
        new_entity=False,
        attribute_name="alanine aminotransferase",
        attribute_description="Alanine aminotransferase laboratory value.",
        attribute_aliases=["ALT", "TGP", "SGPT", "TGP >2N"],
        value_type="Numerical",
        numerical_type="Range",
        numerical_value="(-inf, 2]",
        unit="x ULN",
        confidence=0.97,
        rationale="The excluded range above two times ULN is complemented.",
    )

    guarded = validate_reasoning_selection(
        selection=selection,
        candidates=[],
        library=library,
        source_item="TGP >2N",
    )

    assert guarded.attribute_aliases == ["ALT", "TGP", "SGPT"]
    assert guarded.numerical_value == "(-inf, 2]"
    assert guarded.unit == "x ULN"


def test_attribute_id_review_prompt_contains_only_whitelisted_semantics():
    prompt = build_attribute_id_review_prompt(
        source_item_id="TEST:0001",
        atomic_criterion_title="MFI-20 score of 12 or more",
        current_attribute_id="mfi_20_score_12_or_more",
        review_cycle=1,
        selection_fingerprint="abc123",
    )

    assert "MFI-20 score of 12 or more" in prompt
    assert "mfi_20_score_12_or_more" in prompt
    payload = json.loads(
        prompt.split("ATTRIBUTE_ID_REVIEW_INPUT_JSON:\n", 1)[1]
    )
    assert set(payload) == {
        "source_item_id",
        "review_cycle",
        "selection_fingerprint",
        "atomic_criterion_title",
        "current_attribute_id",
        "prior_validation_error",
    }


def test_id_prompt_calibrates_contextual_pre_existing_and_scope():
    prompt = build_attribute_id_review_prompt(
        source_item_id="TEST:0001",
        atomic_criterion_title="pre-existing other neuropsychiatric disease",
        current_attribute_id="pre_existing_other_neuropsychiatric_disease",
        review_cycle=1,
        selection_fingerprint="selection-hash",
    )

    assert "pre-existing dementia + dementia => KEEP" in prompt
    assert "pre-existing dementia + pre_existing_dementia => MODIFY dementia" in prompt
    assert "MODIFY neuropsychiatric_disease" in prompt
    assert "drug abuse + drug_addiction => MODIFY drug_use_disorder" in prompt
    assert "hepatic impairment + hepatic_failure => MODIFY hepatic_impairment" in prompt
    assert "Write a concise 2-4 sentence explanation in English" in prompt


def test_id_prompt_v3_covers_measurement_analyte_and_historical_rules():
    prompt = build_attribute_id_review_prompt(
        source_item_id="TEST:0001",
        atomic_criterion_title=">= 10 degrees of active wrist extension",
        current_attribute_id="active_wrist_extension",
        review_cycle=1,
        selection_fingerprint="selection-hash",
    )

    for text in (
        "persistent_focal_neurological_deficit",
        "active_wrist_extension_range_of_motion",
        "active_thumb_joint_extension_range_of_motion",
        "alanine_aminotransferase",
        "aspartate_aminotransferase",
        "unsupported specimen",
        "opaque generated ID",
        "follow-up timepoint",
    ):
        assert text in prompt


def test_id_review_modify_rejects_constraint_suffix():
    review = EvaAttributeIdReview(
        decision="MODIFY",
        reviewed_attribute_id="mfi_20_score_12_or_more",
        explanation="The current ID contains a threshold instead of only the scale score concept.",
        confidence=0.99,
    )
    with pytest.raises(ValueError, match="constraint"):
        validate_attribute_id_review(
            review,
            current_attribute_id="mfi_20_score",
        )


def test_id_review_allows_measurement_identity_digits():
    review = EvaAttributeIdReview(
        decision="KEEP",
        reviewed_attribute_id="mfi_20_score",
        explanation="The current ID accurately represents the MFI-20 score without encoding a threshold.",
        confidence=0.99,
    )
    validated = validate_attribute_id_review(
        review,
        current_attribute_id="mfi_20_score",
    )

    assert validated.reviewed_attribute_id == "mfi_20_score"


def _copy_fresh_library(tmp_path: Path) -> Path:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "config"
            / "eligibility_eva_library.json"
        ).read_text(encoding="utf-8")
    )
    payload["revision"] = 0
    payload["audit_history"] = []
    payload["attributes"] = [
        attribute
        for attribute in payload.get("attributes", [])
        if attribute.get("attribute_id") == "sex_gender"
    ]
    target = tmp_path / "eligibility_eva_library.json"
    target.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return target


def _copy_full_library(tmp_path: Path) -> Path:
    payload = json.loads(
        (
            PROJECT_ROOT / "config" / "eligibility_eva_library.json"
        ).read_text(encoding="utf-8")
    )
    payload["revision"] = 0
    payload["audit_history"] = []
    target = tmp_path / "full_eligibility_eva_library.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _write_points(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["criteria", "item", "context"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "criteria": "inclusion",
                    "item": "non-pregnant",
                    "context": "Participants must be non-pregnant.",
                },
                {
                    "criteria": "inclusion",
                    "item": "one previous stroke",
                    "context": "Exactly one previous stroke is required.",
                },
            ]
        )


class FakeSearchProvider:
    def generate_json(self, prompt, *, output_schema):
        if "non-pregnant" in prompt:
            return {
                "lexical_terms": ["non-pregnant", "pregnancy"],
                "semantic_terms": [
                    "pregnancy status",
                    "reproductive status",
                ],
                "entity_hints": ["Contraceptive"],
                "concept_summary": "Pregnancy status",
            }
        return {
            "lexical_terms": ["previous stroke", "prior stroke count"],
            "semantic_terms": ["stroke history", "recurrent stroke count"],
            "entity_hints": ["Diagnosis"],
            "concept_summary": "Number of previous strokes",
        }

    def close(self):
        return None


class FakeReasonerProvider:
    def generate_json(self, prompt, *, output_schema):
        properties = output_schema.get("properties", {})
        if "current_mapping_decision" in properties:
            payload = json.loads(
                prompt.split("MAPPING_REVIEW_INPUT_JSON:\n", 1)[1]
            )
            current = payload["current_selection"]
            is_existing = current["mapping_decision"] == "existing_attribute"
            return {
                "decision": (
                    "CONFIRM_EXISTING" if is_existing else "CONFIRM_NEW"
                ),
                "current_mapping_decision": current["mapping_decision"],
                "reviewed_mapping_decision": current["mapping_decision"],
                "target_attribute_id": current["attribute_id"],
                "semantic_relationship": "exact_equivalent",
                "attribute_id_policy_status": "compliant",
                "additional_search_terms": [],
                "explanation": "The mapping exactly matches the atomic criterion.",
                "confidence": 0.98,
            }
        if "reviewed_attribute_id" in properties:
            payload = json.loads(
                prompt.split("ATTRIBUTE_ID_REVIEW_INPUT_JSON:\n", 1)[1]
            )
            return {
                "decision": "KEEP",
                "reviewed_attribute_id": payload["current_attribute_id"],
                "explanation": "The current ID names the stable clinical variable without encoding a threshold.",
                "confidence": 0.98,
            }
        if '"item": "non-pregnant"' in prompt:
            return {
                "mapping_decision": "new_attribute",
                "attribute_id": "pregnancy",
                "entity_id": "contraceptive",
                "entity_name": "Contraceptive",
                "new_entity": False,
                "attribute_name": "pregnancy",
                "attribute_description": (
                    "Whether the participant is pregnant."
                ),
                "attribute_aliases": ["pregnancy status"],
                "value_type": "Categorical",
                "categorical_value": "Excluded",
                "numerical_type": None,
                "numerical_value": None,
                "unit": None,
                "confidence": 0.98,
                "rationale": (
                    "Inclusion requires the canonical pregnancy concept to "
                    "be absent."
                ),
                "candidate_ids_considered": [],
            }
        return {
            "mapping_decision": "new_attribute",
            "attribute_id": "previous_stroke_times",
            "entity_id": "diagnosis",
            "entity_name": "Diagnosis",
            "new_entity": False,
            "attribute_name": "previous_stroke_times",
            "attribute_description": (
                "Exact number of strokes experienced before enrollment."
            ),
            "attribute_aliases": ["prior stroke count"],
            "value_type": "Numerical",
            "categorical_value": None,
            "numerical_type": "Point",
            "numerical_value": "1",
            "unit": "stroke",
            "confidence": 0.96,
            "rationale": "The criterion requires exactly one prior stroke.",
            "candidate_ids_considered": [],
        }

    def close(self):
        return None


def test_fresh_library_has_fixed_values_and_base_sex_gender_attribute():
    repository = EvaLibraryRepository(
        PROJECT_ROOT / "config" / "eligibility_eva_library.json"
    )

    assert repository.revision >= 0
    assert len(repository.entities) >= 10
    assert "sex_gender" in repository.attributes
    assert repository.attributes["sex_gender"].value_type == "SexGender"
    assert repository.payload["value_definitions"]["Categorical"] == [
        "Included",
        "Excluded",
    ]
    assert repository.payload["value_definitions"]["SexGender"] == [
        "male",
        "female",
        "all",
    ]
    assert repository.payload["value_definitions"]["Numerical"] == [
        "Range",
        "Point",
    ]


def test_reasoning_prompt_does_not_reduce_sex_qualified_disease_to_sex_gender():
    prompt = build_reasoning_prompt(
        trial_id="NCT00006174",
        source_item_id="test:0001",
        criteria="exclusion",
        item="Boys with MAS will be excluded.",
        context="Boys with MAS will be excluded.",
        expansion=EvaSearchExpansion(),
        candidates=[],
        entities=[],
    )
    assert "Do not choose sex_gender merely because" in prompt
    assert "map the condition itself" in prompt


def test_exact_attribute_name_repairs_entity_id_returned_as_attribute_id():
    repository = EvaLibraryRepository(
        PROJECT_ROOT / "config" / "eligibility_eva_library.json"
    )
    selection = EvaReasoningSelection(
        mapping_decision="existing_attribute",
        attribute_id="demographic",
        entity_id="demographic",
        entity_name="Demographic",
        attribute_name="sex_gender",
        attribute_description="Participant sex or gender eligibility.",
        value_type="SexGender",
        categorical_value="male",
        confidence=0.9,
        rationale="The criterion explicitly limits eligibility to male participants.",
    )

    repaired = validate_reasoning_selection(
        selection=selection,
        candidates=[],
        library=repository,
        source_item="male",
        allow_exact_name_repair=True,
    )

    assert repaired.attribute_id == "sex_gender"


def test_exact_attribute_name_repair_is_disabled_by_default():
    repository = EvaLibraryRepository(
        PROJECT_ROOT / "config" / "eligibility_eva_library.json"
    )
    selection = EvaReasoningSelection(
        mapping_decision="existing_attribute",
        attribute_id="demographic",
        entity_id="demographic",
        entity_name="Demographic",
        attribute_name="sex_gender",
        attribute_description="Participant sex or gender eligibility.",
        value_type="SexGender",
        categorical_value="male",
        confidence=0.9,
        rationale="The criterion explicitly limits eligibility to male participants.",
    )

    with pytest.raises(ValueError, match="top candidate"):
        validate_reasoning_selection(
            selection=selection,
            candidates=[],
            library=repository,
            source_item="male",
        )


def test_retrieval_combines_semantic_lexical_and_fuzzy_terms(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    payload["attributes"].append(
        {
            "attribute_id": "pregnancy",
            "canonical_name": "pregnancy",
            "entity_id": "contraceptive",
            "description": "Whether the participant is pregnant.",
            "aliases": ["pregnancy status", "gestation"],
            "value_type": "Categorical",
            "numerical_type": None,
            "canonical_unit": None,
            "active": True,
        }
    )
    library_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    repository = EvaLibraryRepository(library_path)

    candidates = retrieve_attribute_candidates(
        library=repository,
        expansion=EvaSearchExpansion(
            lexical_terms=["pregnant"],
            semantic_terms=["gestational status"],
        ),
        source_item="non-pregnant",
        top_k=10,
    )

    assert candidates[0].attribute_id == "pregnancy"
    assert candidates[0].value_type == "Categorical"
    assert any(
        "llm_semantic" in method
        for method in candidates[0].retrieval_methods
    )
    assert any(
        "fuzzy" in method for method in candidates[0].retrieval_methods
    )


def test_default_retrieval_profile_remains_legacy(tmp_path):
    repository = EvaLibraryRepository(_copy_full_library(tmp_path))
    candidates = retrieve_attribute_candidates(
        library=repository,
        expansion=EvaSearchExpansion(
            lexical_terms=["participant age"],
            semantic_terms=["age in years"],
        ),
        source_item="18-80 years old",
        top_k=10,
    )
    assert candidates[0].attribute_id == "participant_age"
    assert any(
        method.startswith("source_lexical:") or method.startswith("llm_")
        for method in candidates[0].retrieval_methods
    )
    assert all(
        "semantic_reconciliation_v2" not in method
        for method in candidates[0].retrieval_methods
    )


def _entity(entity_id: str, canonical_name: str):
    from backend.criteria_processor.eva_models import EvaEntityDefinition

    return EvaEntityDefinition(
        entity_id=entity_id,
        canonical_name=canonical_name,
        description="",
        aliases=[],
        active=True,
    )


def _attribute(
    attribute_id: str,
    canonical_name: str,
    entity_id: str,
    aliases: list[str],
    *,
    value_type: str = "Categorical",
    numerical_type: str | None = None,
    canonical_unit: str | None = None,
):
    return EvaAttributeDefinition(
        attribute_id=attribute_id,
        canonical_name=canonical_name,
        entity_id=entity_id,
        description=f"Reusable {canonical_name} eligibility Attribute.",
        aliases=aliases,
        value_type=value_type,
        numerical_type=numerical_type,
        canonical_unit=canonical_unit,
        active=True,
    )


def _query(
    terms: tuple[EvaRetrievalTerm, ...],
    entity_id: str,
    value_type: str,
    *,
    numerical_type: str | None = None,
    canonical_unit: str | None = None,
) -> EvaReconciliationRetrievalQuery:
    return EvaReconciliationRetrievalQuery(
        terms=terms,
        canonical_entity_id=entity_id,
        value_type=value_type,
        numerical_type=numerical_type,
        canonical_unit=canonical_unit,
    )


def test_v2_exact_primary_match_with_compatible_metadata_scores_100():
    query = _query(
        (EvaRetrievalTerm("participant age", "edited_alias", 1.0),),
        "demographic",
        "Numerical",
        numerical_type="Range",
        canonical_unit="years",
    )
    match = score_reconciliation_candidate(
        query,
        attribute=_attribute(
            "participant_age",
            "participant_age",
            "demographic",
            ["participant age"],
            value_type="Numerical",
            numerical_type="Range",
            canonical_unit="years",
        ),
        entity=_entity("demographic", "Demographic"),
    )
    assert match.score == 100
    assert match.matched_term == "participant age"
    assert match.methods == (
        "edited_alias:exact_alias",
        "entity:compatible",
        "schema:compatible",
    )


def test_v2_supplemental_exact_is_below_primary_exact():
    supplemental = EvaRetrievalTerm(
        "hepatic failure", "supplemental_lexical", 0.70
    )
    primary = EvaRetrievalTerm("hepatic failure", "edited_alias", 1.0)
    candidate = _attribute(
        "hepatic_failure", "hepatic_failure", "comorbidity", ["hepatic failure"]
    )
    entity = _entity("comorbidity", "Comorbidity")
    supplemental_match = score_reconciliation_candidate(
        _query((supplemental,), "comorbidity", "Categorical"),
        attribute=candidate,
        entity=entity,
    )
    primary_match = score_reconciliation_candidate(
        _query((primary,), "comorbidity", "Categorical"),
        attribute=candidate,
        entity=entity,
    )
    assert supplemental_match.score < primary_match.score
    assert primary_match.score == 100


def test_v2_token_matching_does_not_match_inside_a_token():
    query = _query(
        (EvaRetrievalTerm("hepatic impairment", "supplemental_semantic", 0.60),),
        "comorbidity",
        "Categorical",
    )
    match = score_reconciliation_candidate(
        query,
        attribute=_attribute(
            "sex_gender",
            "sex_gender",
            "demographic",
            ["men"],
            value_type="SexGender",
        ),
        entity=_entity("demographic", "Demographic"),
    )
    assert match is None or match.score < 30


def test_existing_attribute_selection_keeps_library_schema(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    payload["attributes"].append(
        {
            "attribute_id": "pregnancy",
            "canonical_name": "pregnancy",
            "entity_id": "contraceptive",
            "description": "Whether the participant is pregnant.",
            "aliases": [],
            "value_type": "Categorical",
            "numerical_type": None,
            "canonical_unit": None,
            "active": True,
        }
    )
    library_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    repository = EvaLibraryRepository(library_path)
    candidates = retrieve_attribute_candidates(
        library=repository,
        expansion=EvaSearchExpansion(
            lexical_terms=["pregnancy"],
            semantic_terms=["pregnancy status"],
        ),
        source_item="non-pregnant",
    )
    selection = EvaReasoningSelection(
        mapping_decision="existing_attribute",
        attribute_id="pregnancy",
        entity_name="wrong",
        attribute_name="wrong",
        attribute_description="wrong",
        value_type="Numerical",
        numerical_type="Point",
        numerical_value="1",
        categorical_value="Excluded",
        confidence=0.9,
        rationale="Test",
    )

    guarded = validate_reasoning_selection(
        selection=selection,
        candidates=candidates,
        library=repository,
        source_item="non-pregnant",
    )

    assert guarded.entity_id == "contraceptive"
    assert guarded.attribute_name == "pregnancy"
    assert guarded.value_type == "Categorical"
    assert guarded.categorical_value == "Excluded"
    assert guarded.numerical_type is None
    assert guarded.numerical_value is None


def test_reasoning_prompt_contains_polarity_truth_table():
    prompt = build_reasoning_prompt(
        trial_id="TEST",
        source_item_id="TEST:0001",
        criteria="inclusion",
        item="non-pregnant",
        context="Participants must be non-pregnant.",
        expansion=EvaSearchExpansion(),
        candidates=[],
        entities=[],
    )

    assert "inclusion + concept absent => Excluded" in prompt
    assert "exclusion + concept absent => Included" in prompt
    assert "previous_stroke_times" in prompt
    assert "top_attribute_candidates" in prompt
    assert "sex_gender = all" in prompt
    assert "Numerical values must always describe the final eligible population" in prompt
    assert "exclusion score > 10 means eligible score <= 10" in prompt
    assert "approved NCT00657163 audit" in prompt
    assert "Fugl Meyer Motor Scale > 55" in prompt
    assert "(-inf, 55]" in prompt


def test_sex_gender_item_uses_single_controlled_attribute(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    repository = EvaLibraryRepository(library_path)
    selection = EvaReasoningSelection(
        mapping_decision="existing_attribute",
        attribute_id="sex_gender",
        entity_id="demographic",
        entity_name="Demographic",
        new_entity=False,
        attribute_name="wrong",
        attribute_description="wrong",
        attribute_aliases=[],
        value_type="SexGender",
        categorical_value="all",
        confidence=0.9,
        rationale="Men and women are eligible.",
    )
    candidates = retrieve_attribute_candidates(
        library=repository,
        expansion=EvaSearchExpansion(
            lexical_terms=["men and women"],
            semantic_terms=["sex gender eligibility"],
        ),
        source_item="Men and women",
    )

    guarded = validate_reasoning_selection(
        selection=selection,
        candidates=candidates,
        library=repository,
        source_item="Men and women",
    )

    assert guarded.attribute_id == "sex_gender"
    assert guarded.attribute_name == "sex_gender"
    assert guarded.value_type == "SexGender"
    assert guarded.categorical_value == "all"


def test_trial_extraction_creates_pending_audit_without_library_update(
    tmp_path,
):
    library_path = _copy_fresh_library(tmp_path)
    points_path = (
        tmp_path
        / "points"
        / "TEST"
        / "elig_breakdown_points.csv"
    )
    _write_points(points_path)
    repository = EvaLibraryRepository(library_path)

    audit = extract_trial_eva(
        trial_key="TEST",
        trial_id="TEST",
        breakdown_path=points_path,
        library=repository,
        search_provider=FakeSearchProvider(),
        reasoner_provider=FakeReasonerProvider(),
        search_model="gpt-5.4-mini",
        search_reasoning_effort="low",
        reasoner_model="gpt-5.5",
        reasoner_reasoning_effort="medium",
        workers=2,
    )

    assert audit.review_status == "pending"
    assert len(audit.items) == 2
    assert audit.items[0].attribute_name == "pregnancy"
    assert audit.items[0].categorical_value == "Excluded"
    assert audit.items[1].attribute_name == "previous_stroke_times"
    assert audit.items[1].numerical_type == "Point"
    assert audit.items[1].numerical_value == "1"
    assert audit.review_prompt_versions == {
        "reasoner": REASONER_PROMPT_VERSION,
        "mapping_review": MAPPING_REVIEW_PROMPT_VERSION,
        "mapping_correction": MAPPING_CORRECTION_PROMPT_VERSION,
        "attribute_id_review": ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
    }
    assert set(EvaLibraryRepository(library_path).attributes) == {
        "sex_gender"
    }


def test_runtime_records_recover_complete_audit_without_new_calls(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    points_path = (
        tmp_path
        / "points"
        / "TEST"
        / "elig_breakdown_points.csv"
    )
    _write_points(points_path)
    runtime_root = tmp_path / "runtime"
    search = RuntimeRecordingJsonLLMProvider(
        FakeSearchProvider(),
        record_directory=runtime_root,
        run_id="job_search",
    )
    reasoner = RuntimeRecordingJsonLLMProvider(
        FakeReasonerProvider(),
        record_directory=runtime_root,
        run_id="job_reasoner",
    )
    try:
        original = extract_trial_eva(
            trial_key="TEST",
            trial_id="TEST",
            breakdown_path=points_path,
            library=EvaLibraryRepository(library_path),
            search_provider=search,
            reasoner_provider=reasoner,
            search_model="gpt-5.4-mini",
            search_reasoning_effort="low",
            reasoner_model="gpt-5.5",
            reasoner_reasoning_effort="medium",
            workers=2,
            review_mode="off",
        )
    finally:
        search.close()
        reasoner.close()

    recovered = recover_trial_eva_from_runtime_records(
        trial_key="TEST",
        trial_id="TEST",
        breakdown_path=points_path,
        library=EvaLibraryRepository(library_path),
        search_run_directory=runtime_root / "job_search",
        reasoner_run_directory=runtime_root / "job_reasoner",
        search_model="gpt-5.4-mini",
        search_reasoning_effort="low",
        reasoner_model="gpt-5.5",
        reasoner_reasoning_effort="medium",
    )

    assert recovered.audit_id == original.audit_id
    for recovered_item, original_item in zip(
        recovered.items, original.items, strict=True
    ):
        assert recovered_item.attribute_id == original_item.attribute_id
        assert recovered_item.mapping_review.status == "legacy_unreviewed"
    assert recovered.review_status == "pending"
    assert recovered.review_prompt_versions == {
        "reasoner": REASONER_PROMPT_VERSION,
        "mapping_review": MAPPING_REVIEW_PROMPT_VERSION,
        "mapping_correction": MAPPING_CORRECTION_PROMPT_VERSION,
        "attribute_id_review": ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
    }


def test_recovery_reconstructs_confirmed_new_and_id_keep(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    points_path = tmp_path / "points" / "TEST" / "elig_breakdown_points.csv"
    _write_points(points_path)
    runtime_root = tmp_path / "runtime_reviewed"
    search = RuntimeRecordingJsonLLMProvider(
        FakeSearchProvider(), record_directory=runtime_root, run_id="search"
    )
    reasoner = RuntimeRecordingJsonLLMProvider(
        FakeReasonerProvider(), record_directory=runtime_root, run_id="reasoner"
    )
    try:
        original = extract_trial_eva(
            trial_key="TEST",
            trial_id="TEST",
            breakdown_path=points_path,
            library=EvaLibraryRepository(library_path),
            search_provider=search,
            reasoner_provider=reasoner,
            search_model="search-model",
            search_reasoning_effort="low",
            reasoner_model="reasoner-model",
            reasoner_reasoning_effort="medium",
            workers=1,
            review_mode="full",
        )
    finally:
        search.close()
        reasoner.close()

    recovered = recover_trial_eva_from_runtime_records(
        trial_key="TEST",
        trial_id="TEST",
        breakdown_path=points_path,
        library=EvaLibraryRepository(library_path),
        search_run_directory=runtime_root / "search",
        reasoner_run_directory=runtime_root / "reasoner",
        search_model="search-model",
        search_reasoning_effort="low",
        reasoner_model="reasoner-model",
        reasoner_reasoning_effort="medium",
        created_at=original.created_at,
    )

    assert [item.model_dump(mode="json") for item in recovered.items] == [
        item.model_dump(mode="json") for item in original.items
    ]
    assert recovered.schema_version == "eligcrit.eligibility_eva_audit.v3"


def _rewrite_review_records_as_v2(run_directory: Path) -> None:
    for path in (run_directory / "calls").glob("*.completed.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        stage = record.get("stage")
        properties = record.get("output_schema", {}).get("properties", {})
        required = record.get("output_schema", {}).get("required", [])
        response = record.get("response_json") or {}
        if stage == "eligibility_eva_mapping_reviewer":
            response["rationale_zh"] = response.pop("explanation")
            response.pop("attribute_id_policy_status", None)
            properties["rationale_zh"] = properties.pop("explanation")
            properties.pop("attribute_id_policy_status", None)
            record["output_schema"]["required"] = [
                "rationale_zh" if value == "explanation" else value
                for value in required
                if value != "attribute_id_policy_status"
            ]
        elif stage == "eligibility_eva_attribute_id_reviewer":
            response["notes_zh"] = response.pop("explanation")
            properties["notes_zh"] = properties.pop("explanation")
            record["output_schema"]["required"] = [
                "notes_zh" if value == "explanation" else value
                for value in required
            ]
        else:
            continue
        path.write_text(json.dumps(record), encoding="utf-8")


def test_recovery_preserves_v2_review_provenance(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    points_path = tmp_path / "points" / "TEST" / "elig_breakdown_points.csv"
    _write_points(points_path)
    runtime_root = tmp_path / "runtime_legacy_reviewed"
    search = RuntimeRecordingJsonLLMProvider(
        FakeSearchProvider(), record_directory=runtime_root, run_id="search"
    )
    reasoner = RuntimeRecordingJsonLLMProvider(
        FakeReasonerProvider(), record_directory=runtime_root, run_id="reasoner"
    )
    try:
        extract_trial_eva(
            trial_key="TEST",
            trial_id="TEST",
            breakdown_path=points_path,
            library=EvaLibraryRepository(library_path),
            search_provider=search,
            reasoner_provider=reasoner,
            search_model="search-model",
            search_reasoning_effort="low",
            reasoner_model="reasoner-model",
            reasoner_reasoning_effort="medium",
            workers=1,
            review_mode="full",
        )
    finally:
        search.close()
        reasoner.close()

    mapping_path = next(
        path
        for path in (runtime_root / "reasoner" / "calls").glob(
            "*.completed.json"
        )
        if json.loads(path.read_text(encoding="utf-8")).get("stage")
        == "eligibility_eva_mapping_reviewer"
    )
    original_mapping_record = json.loads(mapping_path.read_text(encoding="utf-8"))
    original_relationship = str(
        original_mapping_record["response_json"]["semantic_relationship"]
    )
    original_explanation = str(
        original_mapping_record["response_json"]["explanation"]
    )
    _rewrite_review_records_as_v2(runtime_root / "reasoner")

    recovered = recover_trial_eva_from_runtime_records(
        trial_key="TEST",
        trial_id="TEST",
        breakdown_path=points_path,
        library=EvaLibraryRepository(library_path),
        search_run_directory=runtime_root / "search",
        reasoner_run_directory=runtime_root / "reasoner",
        search_model="search-model",
        search_reasoning_effort="low",
        reasoner_model="reasoner-model",
        reasoner_reasoning_effort="medium",
    )

    attempt = recovered.items[0].mapping_review.attempts[0]
    assert attempt.semantic_relationship == "legacy_unverified"
    assert attempt.legacy_semantic_relationship == original_relationship
    assert attempt.attribute_id_policy_status == "legacy_unverified"
    assert attempt.explanation == original_explanation
    assert attempt.explanation_language == "legacy_unverified"


def test_mixed_review_contracts_cannot_be_replayed():
    v2_record = {
        "stage": "eligibility_eva_mapping_reviewer",
        "output_schema": {"properties": {"rationale_zh": {"type": "string"}}},
    }
    v3_record = {
        "stage": "eligibility_eva_mapping_reviewer",
        "output_schema": {"properties": {"explanation": {"type": "string"}}},
    }

    with pytest.raises(
        ValueError,
        match="Mixed Eligibility EVA review prompt contracts cannot be replayed",
    ):
        _assert_single_review_contract([v2_record, v3_record])


class ScriptedJsonProvider:
    model = "scripted-model"
    reasoning_effort = "medium"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, prompt, *, output_schema):
        self.calls.append({"prompt": prompt, "output_schema": output_schema})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        return None


def _categorical_new_attribute_correction(
    *, attribute_id: str, attribute_name: str, description: str
) -> dict:
    return {
        "mapping_decision": "new_attribute",
        "attribute_id": attribute_id,
        "entity_id": "comorbidity",
        "entity_name": "Comorbidity",
        "new_entity": False,
        "attribute_name": attribute_name,
        "attribute_description": description,
        "attribute_aliases": [],
        "value_type": "Categorical",
        "categorical_value": "Excluded",
        "numerical_type": None,
        "numerical_value": None,
        "unit": None,
        "confidence": 0.99,
        "rationale": (
            "The new Attribute preserves the source concept without "
            "strengthening its meaning."
        ),
        "candidate_ids_considered": [],
        "correction_acknowledged": True,
    }


def _run_existing_candidate_to_new(
    *,
    tmp_path: Path,
    item: str,
    candidate_id: str,
    relationship: str,
    policy: str,
    final_id: str,
):
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    expansion = EvaSearchExpansion(lexical_terms=[item])
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=expansion,
        source_item=item,
        top_k=1000,
    )
    candidate = next(
        value for value in candidates if value.attribute_id == candidate_id
    )
    selection = validate_reasoning_selection(
        selection=EvaReasoningSelection(
            mapping_decision="existing_attribute",
            attribute_id=candidate.attribute_id,
            entity_id=candidate.entity_id,
            entity_name=candidate.entity_name,
            new_entity=False,
            attribute_name=candidate.canonical_name,
            attribute_description=candidate.description,
            attribute_aliases=candidate.aliases,
            value_type=candidate.value_type,
            categorical_value="Excluded",
            numerical_type=candidate.numerical_type,
            unit=candidate.canonical_unit,
            confidence=0.95,
            rationale=(
                "The retrieved candidate was selected for independent review."
            ),
            candidate_ids_considered=[candidate.attribute_id],
        ),
        candidates=candidates,
        library=library,
        source_item=item,
    )
    provider = ScriptedJsonProvider(
        [
            {
                "decision": "REQUIRE_NEW",
                "current_mapping_decision": "existing_attribute",
                "reviewed_mapping_decision": "new_attribute",
                "target_attribute_id": None,
                "semantic_relationship": relationship,
                "attribute_id_policy_status": policy,
                "additional_search_terms": [],
                "explanation": (
                    "The current candidate cannot be the final reusable target "
                    "under the reviewed semantic and naming rules."
                ),
                "confidence": 0.99,
            },
            _categorical_new_attribute_correction(
                attribute_id=final_id,
                attribute_name=final_id.replace("_", " "),
                description=f"Eligibility concept for {final_id.replace('_', ' ')}.",
            ),
        ]
    )
    reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={"criteria": "exclusion", "item": item, "context": item},
        expansion=expansion,
        candidates=candidates,
        selection=selection,
        top_k=10,
    )
    return library, reviewed, trace


@pytest.mark.parametrize(
    ("item", "candidate_id", "relationship", "policy", "final_id"),
    [
        (
            "pre-existing dementia",
            "pre_existing_dementia",
            "exact_equivalent",
            "contextual_pre_existing_qualifier",
            "dementia",
        ),
        (
            "pre-existing other neuropsychiatric disease",
            "pre_existing_other_neuropsychiatric_disease",
            "exact_equivalent",
            "contextual_pre_existing_qualifier",
            "neuropsychiatric_disease",
        ),
        (
            "drug abuse",
            "drug_addiction",
            "narrower_than_source",
            "compliant",
            "drug_use_disorder",
        ),
        (
            "hepatic impairment",
            "hepatic_failure",
            "narrower_than_source",
            "compliant",
            "hepatic_impairment",
        ),
    ],
)
def test_scope_mismatch_requires_precise_new_id(
    tmp_path, item, candidate_id, relationship, policy, final_id
):
    library, reviewed, trace = _run_existing_candidate_to_new(
        tmp_path=tmp_path,
        item=item,
        candidate_id=candidate_id,
        relationship=relationship,
        policy=policy,
        final_id=final_id,
    )

    assert reviewed.mapping_decision == "new_attribute"
    assert reviewed.attribute_id == final_id
    assert trace.final_decision == "REQUIRE_NEW"
    assert trace.attempts[0].semantic_relationship == relationship
    assert trace.attempts[0].attribute_id_policy_status == policy
    assert library.attributes[candidate_id].active is True


def _run_existing_confirmation(
    *, tmp_path: Path, item: str, candidate_id: str
):
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    expansion = EvaSearchExpansion(lexical_terms=[item])
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=expansion,
        source_item=item,
        top_k=1000,
    )
    candidate = next(
        value for value in candidates if value.attribute_id == candidate_id
    )
    selection = validate_reasoning_selection(
        selection=EvaReasoningSelection(
            mapping_decision="existing_attribute",
            attribute_id=candidate.attribute_id,
            entity_id=candidate.entity_id,
            entity_name=candidate.entity_name,
            new_entity=False,
            attribute_name=candidate.canonical_name,
            attribute_description=candidate.description,
            attribute_aliases=candidate.aliases,
            value_type=candidate.value_type,
            categorical_value="Excluded",
            numerical_type=candidate.numerical_type,
            unit=candidate.canonical_unit,
            confidence=0.99,
            rationale="The source explicitly names the existing concept.",
            candidate_ids_considered=[candidate.attribute_id],
        ),
        candidates=candidates,
        library=library,
        source_item=item,
    )
    provider = ScriptedJsonProvider(
        [
            {
                "decision": "CONFIRM_EXISTING",
                "current_mapping_decision": "existing_attribute",
                "reviewed_mapping_decision": "existing_attribute",
                "target_attribute_id": candidate_id,
                "semantic_relationship": "exact_equivalent",
                "attribute_id_policy_status": "compliant",
                "additional_search_terms": [],
                "explanation": (
                    "The source explicitly names the same reusable clinical concept."
                ),
                "confidence": 0.99,
            }
        ]
    )
    reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={"criteria": "exclusion", "item": item, "context": item},
        expansion=expansion,
        candidates=candidates,
        selection=selection,
        top_k=10,
    )
    return reviewed, trace, provider


@pytest.mark.parametrize(
    ("item", "candidate_id"),
    [
        ("drug addiction", "drug_addiction"),
        ("hepatic failure", "hepatic_failure"),
    ],
)
def test_explicit_narrow_concept_confirms_existing(
    tmp_path, item, candidate_id
):
    reviewed, trace, provider = _run_existing_confirmation(
        tmp_path=tmp_path,
        item=item,
        candidate_id=candidate_id,
    )

    assert reviewed.attribute_id == candidate_id
    assert trace.final_decision == "CONFIRM_EXISTING"
    assert trace.attempts[0].semantic_relationship == "exact_equivalent"
    assert trace.attempts[0].attribute_id_policy_status == "compliant"
    assert len(provider.calls) == 1


def _existing_selection_for_candidate(candidate, *, numerical_value=None):
    return EvaReasoningSelection(
        mapping_decision="existing_attribute",
        attribute_id=candidate.attribute_id,
        entity_id=candidate.entity_id,
        entity_name=candidate.entity_name,
        new_entity=False,
        attribute_name=candidate.canonical_name,
        attribute_description=candidate.description,
        attribute_aliases=candidate.aliases,
        value_type=candidate.value_type,
        categorical_value=(
            "Excluded" if candidate.value_type == "Categorical" else None
        ),
        numerical_type=candidate.numerical_type,
        numerical_value=numerical_value,
        unit=candidate.canonical_unit,
        confidence=0.97,
        rationale="The existing concept was selected for governed review.",
        candidate_ids_considered=[candidate.attribute_id],
    )


@pytest.mark.parametrize(
    (
        "criteria",
        "item",
        "candidate_id",
        "policy",
        "recommended_id",
        "numerical_value",
    ),
    [
        (
            "inclusion",
            "Persisting focal neurological deficit severe enough to warrant treatment",
            "attribute_0863e915208d6860",
            "opaque_generated_id",
            "persistent_focal_neurological_deficit",
            None,
        ),
        (
            "exclusion",
            "TGP >2N",
            "serum_alt_level",
            "unsupported_specimen_qualifier",
            "alanine_aminotransferase",
            "(-inf, 2]",
        ),
        (
            "inclusion",
            ">= 10 degrees of active wrist extension",
            "active_wrist_extension",
            "measurement_identity_mismatch",
            "active_wrist_extension_range_of_motion",
            "[10, +inf)",
        ),
    ],
)
def test_exact_existing_naming_problem_blocks_for_migration(
    tmp_path,
    criteria,
    item,
    candidate_id,
    policy,
    recommended_id,
    numerical_value,
):
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=EvaSearchExpansion(lexical_terms=[item]),
        source_item=item,
        top_k=1000,
    )
    candidate = next(value for value in candidates if value.attribute_id == candidate_id)
    selection = validate_reasoning_selection(
        selection=_existing_selection_for_candidate(
            candidate,
            numerical_value=numerical_value,
        ),
        candidates=candidates,
        library=library,
        source_item=item,
    )
    provider = ScriptedJsonProvider([{
        "decision": "NEEDS_HUMAN_REVIEW",
        "current_mapping_decision": "existing_attribute",
        "reviewed_mapping_decision": "existing_attribute",
        "target_attribute_id": candidate_id,
        "semantic_relationship": "exact_equivalent",
        "attribute_id_policy_status": policy,
        "additional_search_terms": [],
        "explanation": (
            "The concept is exact, but governed migration should use "
            f"{recommended_id}."
        ),
        "confidence": 0.98,
    }])

    reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={"criteria": criteria, "item": item, "context": item},
        expansion=EvaSearchExpansion(lexical_terms=[item]),
        candidates=candidates,
        selection=selection,
        top_k=10,
    )

    assert reviewed.attribute_id == candidate_id
    assert trace.status == "blocked"
    assert trace.final_decision == "NEEDS_HUMAN_REVIEW"
    assert trace.attempts[0].attribute_id_policy_status == policy
    assert recommended_id in trace.attempts[0].explanation
    assert library.attributes[candidate_id].active is True


def test_composite_lab_fragment_is_blocked_as_segmentation_issue(tmp_path):
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=EvaSearchExpansion(lexical_terms=["TGP"]),
        source_item="TGP >2N",
        top_k=1000,
    )
    candidate = next(
        value for value in candidates if value.attribute_id == "serum_alt_level"
    )
    selection = validate_reasoning_selection(
        selection=_existing_selection_for_candidate(
            candidate,
            numerical_value="(-inf, 2]",
        ),
        candidates=candidates,
        library=library,
        source_item="TGP >2N",
    )
    provider = ScriptedJsonProvider([{
        "decision": "SEGMENTATION_ISSUE",
        "current_mapping_decision": "existing_attribute",
        "reviewed_mapping_decision": "existing_attribute",
        "target_attribute_id": None,
        "semantic_relationship": "segmentation_issue",
        "attribute_id_policy_status": "not_applicable",
        "additional_search_terms": [],
        "explanation": (
            "The atomic fragment omits the AND relationship in TGO and TGP >2N."
        ),
        "confidence": 0.99,
    }])

    _reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={
            "criteria": "exclusion",
            "item": "TGP >2N",
            "context": "Hepatic failure (TGO and TGP >2N)",
        },
        expansion=EvaSearchExpansion(lexical_terms=["TGP"]),
        candidates=candidates,
        selection=selection,
        top_k=10,
    )

    assert trace.status == "blocked"
    assert trace.final_decision == "SEGMENTATION_ISSUE"


def test_follow_up_timepoint_is_not_silently_generalized(tmp_path):
    item = "Unlikely to be available for follow up at 12 months"
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=EvaSearchExpansion(lexical_terms=[item]),
        source_item=item,
        top_k=1000,
    )
    candidate = next(
        value
        for value in candidates
        if value.attribute_id == "follow_up_availability_at_12_months"
    )
    selection = validate_reasoning_selection(
        selection=_existing_selection_for_candidate(candidate),
        candidates=candidates,
        library=library,
        source_item=item,
    )
    provider = ScriptedJsonProvider([{
        "decision": "NEEDS_HUMAN_REVIEW",
        "current_mapping_decision": "existing_attribute",
        "reviewed_mapping_decision": "existing_attribute",
        "target_attribute_id": candidate.attribute_id,
        "semantic_relationship": "exact_equivalent",
        "attribute_id_policy_status": "not_applicable",
        "additional_search_terms": [],
        "explanation": (
            "The 12-month timepoint is meaning-bearing and has no structured field."
        ),
        "confidence": 0.97,
    }])

    reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={"criteria": "exclusion", "item": item, "context": item},
        expansion=EvaSearchExpansion(lexical_terms=[item]),
        candidates=candidates,
        selection=selection,
        top_k=10,
    )

    assert reviewed.attribute_id == "follow_up_availability_at_12_months"
    assert trace.status == "blocked"
    assert trace.final_decision == "NEEDS_HUMAN_REVIEW"


def test_existing_attribute_id_review_has_english_not_required_explanation(
    tmp_path,
):
    library = EvaLibraryRepository(_copy_full_library(tmp_path))
    definition = library.attributes["hepatic_failure"]
    selection = EvaReasoningSelection(
        mapping_decision="existing_attribute",
        attribute_id=definition.attribute_id,
        entity_id=definition.entity_id,
        entity_name=library.entities[definition.entity_id].canonical_name,
        new_entity=False,
        attribute_name=definition.canonical_name,
        attribute_description=definition.description,
        attribute_aliases=definition.aliases,
        value_type=definition.value_type,
        categorical_value="Excluded",
        confidence=0.99,
        rationale="The source explicitly states hepatic failure.",
    )

    reviewed, trace = review_new_attribute_id(
        reasoner_provider=ScriptedJsonProvider([]),
        source_item_id="TEST:0001",
        atomic_criterion_title="hepatic failure",
        selection=selection,
        library=library,
    )

    assert reviewed.attribute_id == "hepatic_failure"
    assert trace.status == "not_required"
    assert trace.explanation == ID_REVIEW_NOT_REQUIRED_EXPLANATION
    assert trace.explanation_language == "en"


def _library_with_pregnancy(tmp_path: Path) -> EvaLibraryRepository:
    library_path = _copy_fresh_library(tmp_path)
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    payload["attributes"].append(
        {
            "attribute_id": "pregnancy",
            "canonical_name": "pregnancy",
            "entity_id": "contraceptive",
            "description": "Whether the participant is pregnant.",
            "aliases": ["pregnancy status"],
            "value_type": "Categorical",
            "numerical_type": None,
            "canonical_unit": None,
            "active": True,
        }
    )
    library_path.write_text(json.dumps(payload), encoding="utf-8")
    return EvaLibraryRepository(library_path)


def _mapping_review_response(decision: str, **overrides):
    reviewed = (
        "existing_attribute"
        if decision in {"CONFIRM_EXISTING", "SWITCH_EXISTING"}
        else "new_attribute"
    )
    payload = {
        "decision": decision,
        "current_mapping_decision": reviewed,
        "reviewed_mapping_decision": reviewed,
        "target_attribute_id": "pregnancy",
        "semantic_relationship": "exact_equivalent",
        "attribute_id_policy_status": "compliant",
        "additional_search_terms": [],
        "explanation": "The mapping exactly matches the atomic criterion.",
        "confidence": 0.99,
    }
    payload.update(overrides)
    return payload


def test_mapping_review_confirms_existing_without_correction(tmp_path):
    library = _library_with_pregnancy(tmp_path)
    expansion = EvaSearchExpansion(
        lexical_terms=["pregnancy"], semantic_terms=["pregnancy status"]
    )
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=expansion,
        source_item="non-pregnant",
    )
    selection = validate_reasoning_selection(
        selection=EvaReasoningSelection(
            mapping_decision="existing_attribute",
            attribute_id="pregnancy",
            entity_name="Contraceptive",
            attribute_name="pregnancy",
            attribute_description="Pregnancy status.",
            value_type="Categorical",
            categorical_value="Excluded",
            confidence=0.9,
            rationale="Test.",
        ),
        candidates=candidates,
        library=library,
        source_item="non-pregnant",
    )
    provider = ScriptedJsonProvider([
        _mapping_review_response("CONFIRM_EXISTING")
    ])

    reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={
            "criteria": "inclusion",
            "item": "non-pregnant",
            "context": "Participants must be non-pregnant.",
        },
        expansion=expansion,
        candidates=candidates,
        selection=selection,
        top_k=10,
    )

    assert reviewed.attribute_id == "pregnancy"
    assert trace.status == "completed"
    assert trace.final_decision == "CONFIRM_EXISTING"
    assert len(provider.calls) == 1


def test_new_mapping_applies_valid_id_review(tmp_path):
    library = EvaLibraryRepository(_copy_fresh_library(tmp_path))
    selection = validate_reasoning_selection(
        selection=EvaReasoningSelection(
            mapping_decision="new_attribute",
            attribute_id="time_since_stroke",
            entity_id="diagnosis",
            entity_name="Diagnosis",
            attribute_name="time since ischemic stroke",
            attribute_description="Time since ischemic stroke.",
            value_type="Numerical",
            numerical_type="Range",
            numerical_value="[3, +inf)",
            unit="months",
            confidence=0.9,
            rationale="Test.",
        ),
        candidates=[],
        library=library,
        source_item="ischemic stroke at least 3 months ago",
    )
    provider = ScriptedJsonProvider([
        {
            "decision": "MODIFY",
            "reviewed_attribute_id": "time_since_ischemic_stroke_at_enrollment",
            "explanation": "The current ID is too broad; the revision preserves the event subtype and temporal anchor.",
            "confidence": 0.99,
        }
    ])

    reviewed, trace = review_new_attribute_id(
        reasoner_provider=provider,
        source_item_id="TEST:0001",
        atomic_criterion_title="ischemic stroke at least 3 months ago",
        selection=selection,
        library=library,
    )

    assert reviewed.attribute_id == "time_since_ischemic_stroke_at_enrollment"
    assert trace.status == "completed"
    assert trace.decision == "MODIFY"


def test_mapping_review_failure_retains_initial_selection(tmp_path):
    library = EvaLibraryRepository(_copy_fresh_library(tmp_path))
    selection = validate_reasoning_selection(
        selection=EvaReasoningSelection(
            mapping_decision="new_attribute",
            attribute_id="fatigue",
            entity_id="symptom",
            entity_name="Symptom",
            attribute_name="fatigue",
            attribute_description="Fatigue status.",
            value_type="Categorical",
            categorical_value="Included",
            confidence=0.8,
            rationale="Test.",
        ),
        candidates=[],
        library=library,
        source_item="persistent fatigue",
    )
    provider = ScriptedJsonProvider(
        [RuntimeError("first failure"), RuntimeError("retry failure")]
    )

    reviewed, _expansion, _candidates, trace = review_eva_mapping(
        library=library,
        reasoner_provider=provider,
        trial_id="TEST",
        source_item_id="TEST:0001",
        row={
            "criteria": "inclusion",
            "item": "persistent fatigue",
            "context": "Persistent fatigue is required.",
        },
        expansion=EvaSearchExpansion(),
        candidates=[],
        selection=selection,
        top_k=10,
    )

    assert reviewed.attribute_id == "fatigue"
    assert trace.status == "failed"
    assert "retry failure" in trace.error["message"]


def test_conflicting_trial_definitions_with_same_id_are_blocked(tmp_path):
    library = EvaLibraryRepository(_copy_fresh_library(tmp_path))
    points_path = tmp_path / "points" / "points.csv"
    _write_points(points_path)
    audit = extract_trial_eva(
        trial_key="TEST",
        trial_id="TEST",
        breakdown_path=points_path,
        library=library,
        search_provider=FakeSearchProvider(),
        reasoner_provider=FakeReasonerProvider(),
        search_model="search",
        search_reasoning_effort="low",
        reasoner_model="reasoner",
        reasoner_reasoning_effort="medium",
        workers=1,
        review_mode="off",
    )
    audit.items[0].attribute_id = "shared_id"
    audit.items[0].attribute_name = "first_name"
    audit.items[1].attribute_id = "shared_id"
    audit.items[1].attribute_name = "second_name"

    finalized = finalize_trial_attribute_id_conflicts(audit.items, library)

    assert [item.attribute_id_review.status for item in finalized] == [
        "blocked",
        "blocked",
    ]


def test_transient_progress_write_error_does_not_abort_job(
    tmp_path,
    monkeypatch,
):
    library_path = _copy_fresh_library(tmp_path)
    points_root = tmp_path / "points"
    _write_points(
        points_root / "TEST" / "elig_breakdown_points.csv"
    )
    manager = EligibilityEvaJobManager(
        points_directory=points_root,
        library_path=library_path,
        work_directory=tmp_path / "work",
        runtime_record_directory=None,
        provider_builder=lambda *_args: (
            FakeSearchProvider(),
            FakeReasonerProvider(),
        ),
        item_workers=1,
    )
    original_write_job = manager._write_job
    failed_once = False

    def flaky_write_job(payload):
        nonlocal failed_once
        if (
            not failed_once
            and payload.get("status") == "running"
            and payload.get("completed_items") == 1
        ):
            failed_once = True
            raise PermissionError(5, "simulated transient lock")
        original_write_job(payload)

    monkeypatch.setattr(manager, "_write_job", flaky_write_job)
    try:
        started = manager.start("TEST")
        deadline = time.monotonic() + 5
        job = started
        while (
            job.get("status") not in {"completed", "failed"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            job = manager.get(started["job_id"]) or job
    finally:
        manager.close()

    assert failed_once is True
    assert job["status"] == "completed"
    assert job["completed_items"] == 2
    assert job["persistence_warning"]["type"] == "PermissionError"
    assert manager.get_audit("TEST") is not None


def test_atomic_json_write_retries_transient_replace_lock(
    tmp_path,
    monkeypatch,
):
    import backend.criteria_processor.atomic_io as atomic_io

    destination = tmp_path / "job.json"
    original_replace = atomic_io.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "simulated transient lock")
        original_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    atomic_write_json(destination, {"status": "completed"})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "completed"
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_reviewer_apply_merges_terms_and_writes_normalized_records(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    points_root = tmp_path / "points"
    points_path = points_root / "TEST" / "elig_breakdown_points.csv"
    _write_points(points_path)
    repository = EvaLibraryRepository(library_path)
    audit = extract_trial_eva(
        trial_key="TEST",
        trial_id="TEST",
        breakdown_path=points_path,
        library=repository,
        search_provider=FakeSearchProvider(),
        reasoner_provider=FakeReasonerProvider(),
        search_model="gpt-5.4-mini",
        search_reasoning_effort="low",
        reasoner_model="gpt-5.5",
        reasoner_reasoning_effort="medium",
        workers=1,
    )
    manager = EligibilityEvaJobManager(
        points_directory=points_root,
        library_path=library_path,
        work_directory=tmp_path / "work",
        verified_output_directory=tmp_path / "normalized_elig",
        runtime_record_directory=None,
    )
    try:
        audit_path = manager.audit_directory / "TEST.eva_audit.json"
        audit_path.write_text(
            json.dumps(audit.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        saved_revision = manager.save_audit(
            "TEST",
            audit.model_dump(mode="json"),
        )
        result = manager.apply_audit("TEST", saved_revision)
        verified_path = (
            tmp_path
            / "normalized_elig"
            / "TEST"
            / "eligibility_eva.jsonl"
        )
        assert verified_path.is_file()
        verified_path.unlink()
        repeated = manager.apply_audit("TEST", result["audit"])
    finally:
        manager.close()

    assert saved_revision["review_status"] == "revised"
    assert result["audit"]["review_status"] == "approved"
    updated = EvaLibraryRepository(library_path)
    assert updated.revision == 1
    assert set(updated.attributes) == {
        "pregnancy",
        "previous_stroke_times",
        "sex_gender",
    }
    assert updated.attributes["pregnancy"].value_type == "Categorical"
    assert (
        updated.attributes["previous_stroke_times"].numerical_type
        == "Point"
    )
    assert result["normalized_count"] == 2
    assert result["verified_normalized_path"] == str(
        tmp_path
        / "normalized_elig"
        / "TEST"
        / "eligibility_eva.jsonl"
    )
    assert repeated["library"]["revision"] == 1
    assert repeated["normalized_count"] == 2
    normalized_path = (
        tmp_path
        / "work"
        / "normalized"
        / "TEST"
        / "eligibility_eva.jsonl"
    )
    rows = [
        json.loads(line)
        for line in normalized_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["categorical_value"] == "Excluded"
    assert rows[1]["numerical_value"] == "1"
    verified_rows = [
        json.loads(line)
        for line in verified_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert verified_rows == rows


def test_frontend_contains_per_trial_audit_and_library_update_controls():
    static = PROJECT_ROOT / "webapp" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="criteria"' in html
    assert 'id="run"' in html
    assert 'id="library-list"' in html
    assert "/api/jobs/extract" in script
    assert "/api/jobs/${id}/result" in script
    assert "/api/library" in script


def test_apply_rejects_stale_library_revision_inside_update_lock(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    points_path = tmp_path / "points" / "points.csv"
    _write_points(points_path)
    audit = extract_trial_eva(
        trial_key="TEST", trial_id="TEST", breakdown_path=points_path,
        library=EvaLibraryRepository(library_path),
        search_provider=FakeSearchProvider(), reasoner_provider=FakeReasonerProvider(),
        search_model="search", search_reasoning_effort="low",
        reasoner_model="reasoner", reasoner_reasoning_effort="medium",
        workers=1, review_mode="off",
    )
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    payload["revision"] += 1
    library_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaLibraryStaleError, match="revision"):
        EvaLibraryRepository(library_path).apply_audit(audit)


def test_apply_rejects_non_equivalent_attribute_id_collision(tmp_path):
    library_path = _copy_fresh_library(tmp_path)
    points_path = tmp_path / "points" / "points.csv"
    _write_points(points_path)
    repository = EvaLibraryRepository(library_path)
    audit = extract_trial_eva(
        trial_key="TEST", trial_id="TEST", breakdown_path=points_path,
        library=repository, search_provider=FakeSearchProvider(),
        reasoner_provider=FakeReasonerProvider(), search_model="search",
        search_reasoning_effort="low", reasoner_model="reasoner",
        reasoner_reasoning_effort="medium", workers=1, review_mode="off",
    )
    audit.items[0].attribute_id = "sex_gender"
    with pytest.raises(EvaAttributeIdConflictError, match="attribute_id"):
        repository.apply_audit(audit)


def test_eva_frontend_renders_mapping_and_id_review_panels():
    server = (PROJECT_ROOT / "webapp" / "server.py").read_text(encoding="utf-8")
    assert "/api/jobs/" in server
    assert 'parts[3] == "result"' in server
    assert "audit_path" in server


def test_eva_frontend_renders_v3_review_evidence():
    service = (
        PROJECT_ROOT / "backend" / "services" / "eligcrit_extraction.py"
    ).read_text(encoding="utf-8")
    assert 'review_mode="full"' in service
    assert "RuntimeRecordingJsonLLMProvider" in service
    assert "resume_incomplete" in service
