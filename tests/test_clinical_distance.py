import math

import pytest

from backend.statistics.clinical_distance import (
    PenaltyValidationError,
    build_penalty_initializer_pool_context,
    calculate_condition_penalty_analysis,
    calculate_numerical_penalty_analysis,
    categorical_label_penalty,
    categorical_unresolved_penalty,
    clinical_distance_penalty,
    default_numerical_penalty,
    initialize_numerical_penalties,
    validate_categorical_penalty_policy,
    validate_numerical_penalty,
    validate_numerical_penalty_policy,
)
from backend.statistics.catalog import load_condition_catalog


def _policy():
    catalog = load_condition_catalog(
        "config/meta_analysis_condition_catalog.json"
    )
    return catalog["condition_builder"]["numerical_penalty"]


def _categorical_policy():
    catalog = load_condition_catalog(
        "config/meta_analysis_condition_catalog.json"
    )
    return catalog["condition_builder"]["categorical_penalty"]


def _attribute():
    return {
        "attribute_id": "baseline_uefma",
        "canonical_name": "Baseline UEFMA",
        "canonical_unit": "point",
        "integer_only": False,
        "description": "Baseline upper-extremity Fugl-Meyer score.",
        "trial_values": [
            {
                "trial_id": "NCT1",
                "trial_label": "Trial 1",
                "reported": True,
                "observations": [
                    {
                        "value_kind": "point",
                        "value": 24,
                        "unit": "point",
                    }
                ],
            },
            {
                "trial_id": "NCT2",
                "trial_label": "Trial 2",
                "reported": True,
                "observations": [
                    {
                        "value_kind": "range",
                        "value": None,
                        "intervals": [{"lower": 20, "upper": 50}],
                        "unit": "point",
                    }
                ],
            },
        ],
    }


def _point_target():
    return {
        "value_kind": "point",
        "value": 30.0,
        "unit": "point",
    }


def _pooling_catalog():
    return {
        "default_method_id": "fixed_effect_inverse_variance",
        "default_attribute_id": "uefma_score_mean_difference",
        "methods": [
            {
                "method_id": "fixed_effect_inverse_variance",
                "label": "Fixed-effect inverse-variance",
                "model_label": "Fixed-effect model",
                "weight_definition": "1 / standard_error^2",
                "available": True,
            }
        ],
        "targets": [
            {
                "attribute_id": "uefma_score_mean_difference",
                "canonical_name": "UEFMA Score Difference",
                "description": "Change from baseline.",
                "canonical_unit": "point",
                "analysis_kind": "single_arm_summary",
                "trials": [
                    {
                        "trial_id": "NCT1",
                        "trial_label": "Trial 1",
                        "audit_status": "approved",
                        "selection_eligible": True,
                        "endpoints": [
                            {
                                "observations": [
                                    {
                                        "observation_id": "OBS1",
                                        "endpoint_label": "Primary endpoint",
                                        "endpoint_role": "primary",
                                        "arm_id": "intervention_1",
                                        "arm_label": "Intervention",
                                        "estimate": 8.0,
                                        "unit": "point",
                                        "statistic_type": "mean change",
                                        "precision_options": [
                                            {
                                                "precision_option_id": "CI1",
                                                "source_label": (
                                                    "Reported 95% CI"
                                                ),
                                            }
                                        ],
                                        "sample_size_options": [],
                                    }
                                ]
                            }
                        ],
                    },
                    {
                        "trial_id": "NCT2",
                        "trial_label": "Trial 2",
                        "audit_status": "approved",
                        "selection_eligible": True,
                        "endpoints": [],
                    },
                ],
            }
        ],
    }


def _model():
    return {
        "model_id": "absolute_clinical_distance",
        "parameters": {
            "full_relevance_tolerance": 2.0,
            "clinical_distance_scale": 10.0,
        },
        "display_domain": {"minimum": 0.0, "maximum": 60.0},
        "initialization": {
            "source": "user",
            "status": "edited",
            "model": None,
            "reasoning_effort": None,
            "generated_at": None,
            "rationale": "User tuned.",
        },
    }


def test_clinical_distance_model_has_tolerance_and_weighted_decay():
    target = _point_target()

    at_target = clinical_distance_penalty(
        30,
        target=target,
        numerical_penalty=_model(),
    )
    at_tolerance = clinical_distance_penalty(
        32,
        target=target,
        numerical_penalty=_model(),
    )
    ten_beyond_tolerance = clinical_distance_penalty(
        42,
        target=target,
        numerical_penalty=_model(),
        condition_weight=2,
    )

    assert at_target["relevance"] == 1.0
    assert at_tolerance["relevance"] == 1.0
    assert ten_beyond_tolerance["raw_penalty"] == pytest.approx(1.0)
    assert ten_beyond_tolerance["weighted_penalty"] == pytest.approx(2.0)
    assert ten_beyond_tolerance["relevance"] == pytest.approx(math.exp(-2))


def test_categorical_label_penalty_is_direct_and_weighted():
    matched = categorical_label_penalty(
        "stroke_stage__subacute",
        target_label_ids=[
            "stroke_stage__subacute",
            "stroke_stage__chronic",
        ],
        condition_weight=2.5,
    )
    mismatched = categorical_label_penalty(
        "stroke_stage__acute",
        target_label_ids=["stroke_stage__subacute"],
        condition_weight=2.5,
    )

    assert matched == {
        "model_id": "direct_label_match",
        "match": True,
        "penalty_kind": "match",
        "raw_penalty": 0.0,
        "weighted_penalty": 0.0,
        "relevance": 1.0,
    }
    assert mismatched["match"] is False
    assert mismatched["penalty_kind"] == "mismatch"
    assert mismatched["raw_penalty"] == 1.0
    assert mismatched["weighted_penalty"] == 2.5
    assert mismatched["relevance"] == pytest.approx(math.exp(-2.5))


def test_unresolved_categorical_penalty_is_small_and_weighted():
    unresolved = categorical_unresolved_penalty(
        condition_weight=2.0,
        raw_penalty=0.1,
    )

    assert unresolved["match"] is None
    assert unresolved["penalty_kind"] == "unresolved"
    assert unresolved["raw_penalty"] == 0.1
    assert unresolved["weighted_penalty"] == 0.2
    assert unresolved["relevance"] == pytest.approx(math.exp(-0.2))


def test_legacy_taper_is_folded_into_delta_and_removed():
    legacy = _model()
    legacy["parameters"]["taper"] = 4.0

    normalized = validate_numerical_penalty(
        legacy,
        condition_id="reported_results__baseline_uefma",
        target=_point_target(),
        attribute=_attribute(),
        policy=_policy(),
    )

    assert normalized["parameters"] == {
        "full_relevance_tolerance": 2.0,
        "clinical_distance_scale": 5.0,
    }


def test_clinical_distance_model_uses_nearest_target_interval_boundary():
    target = {
        "value_kind": "range",
        "intervals": [{"lower": 20.0, "upper": 40.0}],
        "unit": "point",
    }

    inside = clinical_distance_penalty(
        35,
        target=target,
        numerical_penalty=_model(),
    )
    outside = clinical_distance_penalty(
        50,
        target=target,
        numerical_penalty=_model(),
    )

    assert inside["distance"] == 0.0
    assert inside["relevance"] == 1.0
    assert outside["distance"] == 10.0
    assert outside["raw_penalty"] == pytest.approx(0.64)


def test_default_model_covers_target_and_observed_values():
    model = default_numerical_penalty(
        _point_target(),
        _attribute(),
        _policy(),
    )

    assert model["model_id"] == "absolute_clinical_distance"
    assert model["initialization"]["status"] == "pending"
    assert model["display_domain"]["minimum"] <= 20
    assert model["display_domain"]["maximum"] >= 50
    assert model["parameters"]["clinical_distance_scale"] > 0


def test_penalty_initializer_pool_context_resolves_controlled_setup():
    context = build_penalty_initializer_pool_context(
        _pooling_catalog(),
        {
            "method": "fixed_effect_inverse_variance",
            "attribute_id": "uefma_score_mean_difference",
            "selections": [
                {
                    "trial_id": "NCT1",
                    "observation_id": "OBS1",
                    "precision_option_id": "CI1",
                }
            ],
        },
    )

    assert context["endpoint"] == {
        "attribute_id": "uefma_score_mean_difference",
        "attribute": "UEFMA Score Difference",
        "description": "Change from baseline.",
        "unit": "point",
        "analysis_kind": "single_arm_summary",
    }
    assert context["selected_trial_count"] == 1
    assert context["selected_trials"][0]["trial_id"] == "NCT1"
    assert context["selected_trials"][0]["arm_label"] == "Intervention"
    assert (
        context["selected_trials"][0]["precision_source"]
        == "Reported 95% CI"
    )
    assert len(context["available_trials"]) == 2


def test_penalty_initializer_pool_context_rejects_unknown_observation():
    with pytest.raises(
        PenaltyValidationError,
        match="unavailable endpoint observation",
    ):
        build_penalty_initializer_pool_context(
            _pooling_catalog(),
            {
                "attribute_id": "uefma_score_mean_difference",
                "selections": [
                    {
                        "trial_id": "NCT1",
                        "observation_id": "UNKNOWN",
                    }
                ],
            },
        )


class _FakeProvider:
    model = "fake-penalty-model"
    reasoning_effort = "medium"

    def __init__(self):
        self.prompt = ""
        self.output_schema = None

    def generate_json(self, prompt, *, output_schema):
        self.prompt = prompt
        self.output_schema = output_schema
        return {
            "suggestions": [
                {
                    "condition_id": (
                        "reported_results__baseline_uefma"
                    ),
                    "full_relevance_tolerance": 2.0,
                    "clinical_distance_scale": 8.0,
                    "display_minimum": 1.0,
                    "display_maximum": 60.0,
                    "rationale": (
                        "Two points are treated as practically similar; "
                        "larger distances decay on an eight-point scale."
                    ),
                }
            ]
        }


def test_llm_initialization_is_structured_validated_and_auditable():
    attribute = _attribute()
    default = default_numerical_penalty(
        _point_target(),
        attribute,
        _policy(),
    )
    condition = {
        "condition_id": "reported_results__baseline_uefma",
        "condition_source": "reported_results",
        "attribute_id": "baseline_uefma",
        "attribute": "Baseline UEFMA",
        "entity": "Study Population",
        "value_type": "numerical",
        "target": _point_target(),
        "penalty_weight": 1.0,
        "numerical_penalty": default,
    }
    provider = _FakeProvider()

    result = initialize_numerical_penalties(
        [condition],
        attributes={
            ("reported_results", "baseline_uefma"): attribute
        },
        policy=_policy(),
        task_context={
            "profile_name": "90-day motor recovery",
            "endpoint": "UEFMA change at 90 days",
            "selected_pool_context": {
                "selected_trials": [{"trial_id": "NCT1"}]
            },
        },
        provider=provider,
    )

    suggestion = result["suggestions"][0]["numerical_penalty"]
    assert suggestion["parameters"] == {
        "full_relevance_tolerance": 2.0,
        "clinical_distance_scale": 8.0,
    }
    assert suggestion["display_domain"] == {
        "minimum": 1.0,
        "maximum": 60.0,
    }
    assert suggestion["initialization"]["source"] == "llm"
    assert suggestion["initialization"]["status"] == "suggested"
    assert suggestion["initialization"]["model"] == "gpt-5.4-mini"
    assert suggestion["initialization"]["reasoning_effort"] == "medium"
    assert suggestion["initialization"]["generated_at"]
    assert "PENALTY_INITIALIZATION_INPUT_JSON" in provider.prompt
    assert '"value": 30.0' in provider.prompt
    assert '"selected_in_pool": true' in provider.prompt
    assert '"selected_in_pool": false' in provider.prompt
    assert provider.output_schema["properties"]["suggestions"]


def test_llm_initialization_rejects_domain_that_excludes_target():
    provider = _FakeProvider()
    original = provider.generate_json

    def invalid_response(prompt, *, output_schema):
        payload = original(prompt, output_schema=output_schema)
        payload["suggestions"][0]["display_maximum"] = 25.0
        return payload

    provider.generate_json = invalid_response
    attribute = _attribute()
    condition = {
        "condition_id": "reported_results__baseline_uefma",
        "condition_source": "reported_results",
        "attribute_id": "baseline_uefma",
        "attribute": "Baseline UEFMA",
        "entity": "Study Population",
        "value_type": "numerical",
        "target": _point_target(),
        "numerical_penalty": default_numerical_penalty(
            _point_target(),
            attribute,
            _policy(),
        ),
    }

    with pytest.raises(
        PenaltyValidationError,
        match="display domain must contain",
    ):
        initialize_numerical_penalties(
            [condition],
            attributes={
                ("reported_results", "baseline_uefma"): attribute
            },
            policy=_policy(),
            task_context={},
            provider=provider,
        )


def test_penalty_policy_is_normalized_from_config():
    policy = validate_numerical_penalty_policy(_policy())
    categorical_policy = validate_categorical_penalty_policy(
        _categorical_policy()
    )

    assert policy["model_id"] == "absolute_clinical_distance"
    assert policy["unresolved_raw_penalty"] == 0.1
    assert policy["initializer"] == {
        "model": "gpt-5.4-mini",
        "reasoning_effort": "medium",
    }
    assert categorical_policy == {
        "model_id": "direct_label_match",
        "label": "Direct controlled-label relevance",
        "unresolved_raw_penalty": 0.1,
    }


def test_numerical_penalties_multiply_precision_weights_transparently():
    condition = {
        "condition_id": "reported_results__baseline_uefma",
        "condition_source": "reported_results",
        "attribute_id": "baseline_uefma",
        "attribute": "Baseline UEFMA",
        "value_type": "numerical",
        "target": _point_target(),
        "penalty_weight": 1.0,
        "numerical_penalty": {
            **_model(),
            "parameters": {
                "full_relevance_tolerance": 0.0,
                "clinical_distance_scale": 10.0,
            },
        },
    }
    attribute = _attribute()
    attribute["trial_values"] = [
        {
            "trial_id": "NCT1",
            "audit_status": "approved",
            "item_review_status": "approved",
            "observations": [
                {
                    "status": "found",
                    "audit_status": "approved",
                    "value_kind": "point",
                    "value": 30.0,
                    "unit": "point",
                    "arm_id": "overall",
                }
            ],
        },
        {
            "trial_id": "NCT2",
            "audit_status": "approved",
            "item_review_status": "approved",
            "observations": [
                {
                    "status": "found",
                    "audit_status": "approved",
                    "value_kind": "point",
                    "value": 40.0,
                    "unit": "point",
                    "arm_id": "overall",
                }
            ],
        },
    ]
    precision_result = {
        "status": "completed",
        "method": {"minimum_studies": 2},
        "heterogeneity": {"tau_squared": 0.0},
        "pooled": {"unit": "point"},
        "studies": [
            {
                "trial_id": "NCT1",
                "trial_label": "Trial 1",
                "arm_id": "control_1",
                "estimate": 10.0,
                "variance": 1.0,
                "raw_weight": 1.0,
            },
            {
                "trial_id": "NCT2",
                "trial_label": "Trial 2",
                "arm_id": "control_1",
                "estimate": 20.0,
                "variance": 1.0,
                "raw_weight": 1.0,
            },
        ],
    }

    analysis = calculate_numerical_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("reported_results", "baseline_uefma"): attribute
        },
    )

    assert analysis["status"] == "completed"
    assert analysis["studies"][0]["combined_relevance"] == 1.0
    assert analysis["studies"][1]["combined_relevance"] == pytest.approx(
        math.exp(-1)
    )
    expected = (10 + 20 * math.exp(-1)) / (1 + math.exp(-1))
    assert analysis["pooled"]["estimate"] == pytest.approx(expected)
    assert sum(
        study["adjusted_weight_percent"]
        for study in analysis["studies"]
    ) == pytest.approx(100.0)
    assert "hyperparameter-selection uncertainty" in (
        analysis["warnings"][-1]
    )


def test_random_effects_penalty_weights_hold_unpenalized_tau_squared_fixed():
    condition = {
        "condition_id": "reported_results__baseline_uefma",
        "condition_source": "reported_results",
        "attribute_id": "baseline_uefma",
        "attribute": "Baseline UEFMA",
        "value_type": "numerical",
        "target": _point_target(),
        "penalty_weight": 1.0,
        "numerical_penalty": {
            **_model(),
            "parameters": {
                "full_relevance_tolerance": 0.0,
                "clinical_distance_scale": 10.0,
            },
        },
    }
    attribute = _attribute()
    attribute["trial_values"] = [
        {
            "trial_id": trial_id,
            "audit_status": "approved",
            "item_review_status": "approved",
            "observations": [
                {
                    "status": "found",
                    "audit_status": "approved",
                    "value_kind": "point",
                    "value": value,
                    "unit": "point",
                    "arm_id": "overall",
                }
            ],
        }
        for trial_id, value in (
            ("NCT1", 30.0),
            ("NCT2", 40.0),
            ("NCT3", 30.0),
        )
    ]
    tau_squared = 4.0
    studies = [
        {
            "trial_id": "NCT1",
            "trial_label": "Trial 1",
            "arm_id": "control_1",
            "estimate": 1.0,
            "standard_error": 1.0,
            "variance": 1.0,
            "raw_weight": 1.0 / 5.0,
        },
        {
            "trial_id": "NCT2",
            "trial_label": "Trial 2",
            "arm_id": "control_1",
            "estimate": 9.0,
            "standard_error": 2.0,
            "variance": 4.0,
            "raw_weight": 1.0 / 8.0,
        },
        {
            "trial_id": "NCT3",
            "trial_label": "Trial 3",
            "arm_id": "control_1",
            "estimate": 7.0,
            "standard_error": 1.0,
            "variance": 1.0,
            "raw_weight": 1.0 / 5.0,
        },
    ]
    base_total_weight = sum(study["raw_weight"] for study in studies)
    conventional_estimate = sum(
        study["raw_weight"] * study["estimate"] for study in studies
    ) / base_total_weight
    precision_result = {
        "status": "completed",
        "method": {
            "method_id": "random_effects_inverse_variance_dl",
            "label": "Random-effects inverse-variance (DL)",
            "minimum_studies": 2,
            "weight_definition": "1 / (standard_error^2 + tau^2)",
        },
        "heterogeneity": {
            "tau_squared": tau_squared,
            "tau_squared_estimator": "DerSimonian-Laird",
        },
        "pooled": {
            "estimate": conventional_estimate,
            "unit": "point",
        },
        "studies": studies,
    }

    analysis = calculate_condition_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("reported_results", "baseline_uefma"): attribute
        },
    )

    relevances = [1.0, math.exp(-1.0), 1.0]
    adjusted_weights = [
        study["raw_weight"] * relevance
        for study, relevance in zip(studies, relevances, strict=True)
    ]
    adjusted_total = sum(adjusted_weights)
    expected_estimate = sum(
        weight * study["estimate"]
        for weight, study in zip(adjusted_weights, studies, strict=True)
    ) / adjusted_total
    expected_variance = sum(
        weight**2 * (study["variance"] + tau_squared)
        for weight, study in zip(adjusted_weights, studies, strict=True)
    ) / adjusted_total**2

    assert analysis["status"] == "completed"
    assert analysis["method"]["method_id"] == (
        "random_effects_inverse_variance_dl"
    )
    assert analysis["heterogeneity"]["tau_squared"] == tau_squared
    assert analysis["weighting"]["tau_squared_source"] == (
        "conventional_unpenalized_pool"
    )
    assert analysis["pooled"]["estimate"] == pytest.approx(
        expected_estimate
    )
    assert analysis["pooled"]["variance"] == pytest.approx(
        expected_variance
    )
    assert analysis["pooled"]["difference_from_conventional"] == (
        pytest.approx(expected_estimate - conventional_estimate)
    )
    assert sum(
        study["base_weight_percent"] for study in analysis["studies"]
    ) == pytest.approx(100.0)
    assert sum(
        study["adjusted_weight_percent"] for study in analysis["studies"]
    ) == pytest.approx(100.0)
    assert analysis["effective_study_count"] == pytest.approx(
        adjusted_total**2
        / sum(weight**2 for weight in adjusted_weights)
    )
    first_condition = analysis["studies"][0]["conditions"][0]
    second_condition = analysis["studies"][1]["conditions"][0]
    assert first_condition["condition_source"] == "reported_results"
    assert first_condition["attribute_id"] == "baseline_uefma"
    assert first_condition["target"] == _point_target()
    assert first_condition["penalty_model"] == {
        "model_id": "absolute_clinical_distance",
        "parameters": {
            "full_relevance_tolerance": 0.0,
            "clinical_distance_scale": 10.0,
        },
    }
    assert first_condition["distance"] == 0.0
    assert first_condition["excess_distance"] == 0.0
    assert second_condition["distance"] == 10.0
    assert second_condition["excess_distance"] == 10.0


def test_categorical_labels_directly_adjust_precision_weights():
    condition = {
        "condition_id": "reported_results__stroke_stage",
        "condition_source": "reported_results",
        "attribute_id": "stroke_stage",
        "attribute": "Stroke Stage",
        "value_type": "categorical",
        "target": {
            "value_kind": "category_set",
            "operator": "one_of",
            "label_ids": ["stroke_stage__subacute"],
            "labels": ["subacute"],
        },
        "penalty_weight": 2.0,
    }
    attribute = {
        "categorical_labels": [
            {
                "label_id": "stroke_stage__subacute",
                "canonical_label": "subacute",
            },
            {
                "label_id": "stroke_stage__chronic",
                "canonical_label": "chronic",
            },
        ],
        "trial_values": [
            {
                "trial_id": "NCT1",
                "audit_status": "approved",
                "item_review_status": "approved",
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "category",
                        "category": "subacute",
                        "category_label_id": "stroke_stage__subacute",
                        "arm_id": "overall",
                    }
                ],
            },
            {
                "trial_id": "NCT2",
                "audit_status": "approved",
                "item_review_status": "approved",
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "category",
                        "category": "chronic",
                        "category_label_id": "stroke_stage__chronic",
                        "arm_id": "overall",
                    }
                ],
            },
        ],
    }
    precision_result = {
        "status": "completed",
        "method": {"minimum_studies": 2},
        "heterogeneity": {"tau_squared": 0.0},
        "pooled": {"unit": "point"},
        "studies": [
            {
                "trial_id": "NCT1",
                "trial_label": "Trial 1",
                "arm_id": "control_1",
                "estimate": 10.0,
                "variance": 1.0,
                "raw_weight": 1.0,
            },
            {
                "trial_id": "NCT2",
                "trial_label": "Trial 2",
                "arm_id": "control_1",
                "estimate": 20.0,
                "variance": 1.0,
                "raw_weight": 1.0,
            },
        ],
    }

    analysis = calculate_condition_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("reported_results", "stroke_stage"): attribute
        },
    )

    assert analysis["status"] == "completed"
    assert analysis["model_id"] == "condition_relevance_product"
    assert analysis["numerical_condition_count"] == 0
    assert analysis["categorical_condition_count"] == 1
    matched = analysis["studies"][0]["conditions"][0]
    mismatched = analysis["studies"][1]["conditions"][0]
    assert matched["status"] == "matched"
    assert matched["observed_labels"] == ["subacute"]
    assert matched["target_labels"] == ["subacute"]
    assert matched["relevance"] == 1.0
    assert mismatched["status"] == "mismatched"
    assert mismatched["observed_labels"] == ["chronic"]
    assert mismatched["raw_penalty"] == 1.0
    assert mismatched["weighted_penalty"] == 2.0
    assert mismatched["relevance"] == pytest.approx(math.exp(-2))
    assert mismatched["condition_source"] == "reported_results"
    assert mismatched["attribute_id"] == "stroke_stage"
    assert mismatched["target"] == condition["target"]
    assert mismatched["penalty_model"] == {
        "model_id": "direct_label_match",
        "parameters": {},
    }
    expected = (10 + 20 * math.exp(-2)) / (1 + math.exp(-2))
    assert analysis["pooled"]["estimate"] == pytest.approx(expected)


def test_conflicting_categorical_states_receive_unresolved_penalty():
    condition = {
        "condition_id": "reported_results__stroke_stage",
        "condition_source": "reported_results",
        "attribute_id": "stroke_stage",
        "attribute": "Stroke Stage",
        "value_type": "categorical",
        "target": {
            "value_kind": "category_set",
            "operator": "one_of",
            "label_ids": ["stroke_stage__subacute"],
            "labels": ["subacute"],
        },
        "penalty_weight": 1.0,
    }
    attribute = {
        "categorical_labels": [
            {
                "label_id": "stroke_stage__subacute",
                "canonical_label": "subacute",
            },
            {
                "label_id": "stroke_stage__chronic",
                "canonical_label": "chronic",
            },
        ],
        "trial_values": [
            {
                "trial_id": "NCT1",
                "audit_status": "approved",
                "item_review_status": "approved",
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "category",
                        "category_label_id": "stroke_stage__subacute",
                        "arm_id": "overall",
                    },
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "category",
                        "category_label_id": "stroke_stage__chronic",
                        "arm_id": "overall",
                    },
                ],
            }
        ],
    }
    precision_result = {
        "status": "completed",
        "method": {"minimum_studies": 2},
        "heterogeneity": {"tau_squared": 0.0},
        "pooled": {"unit": "point"},
        "studies": [
            {
                "trial_id": "NCT1",
                "trial_label": "Trial 1",
                "arm_id": "control_1",
                "estimate": 10.0,
                "variance": 1.0,
                "raw_weight": 1.0,
            }
        ],
    }

    analysis = calculate_condition_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("reported_results", "stroke_stage"): attribute
        },
    )

    categorical = analysis["studies"][0]["conditions"][0]
    assert analysis["status"] == "insufficient_complete_data"
    assert analysis["complete_study_count"] == 1
    assert analysis["studies"][0]["status"] == "scored"
    assert analysis["studies"][0]["combined_relevance"] == pytest.approx(
        math.exp(-0.1)
    )
    assert categorical["status"] == "ambiguous"
    assert categorical["penalty_kind"] == "unresolved"
    assert categorical["raw_penalty"] == 0.1
    assert categorical["relevance"] == pytest.approx(math.exp(-0.1))
    assert categorical["observed_labels"] == ["chronic", "subacute"]


def test_missing_not_reported_and_invalid_labels_get_small_penalty():
    condition = {
        "condition_id": "reported_results__stroke_stage",
        "condition_source": "reported_results",
        "attribute_id": "stroke_stage",
        "attribute": "Stroke Stage",
        "value_type": "categorical",
        "target": {
            "value_kind": "category_set",
            "operator": "one_of",
            "label_ids": ["stroke_stage__subacute"],
            "labels": ["subacute"],
        },
        "penalty_weight": 2.0,
    }
    attribute = {
        "categorical_labels": [
            {
                "label_id": "stroke_stage__subacute",
                "canonical_label": "subacute",
            }
        ],
        "trial_values": [
            {
                "trial_id": "NCT2",
                "audit_status": "missing",
                "item_review_status": "missing",
                "reported": False,
                "observations": [],
            },
            {
                "trial_id": "NCT3",
                "audit_status": "approved",
                "item_review_status": "approved",
                "reported": True,
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "category",
                        "category": "invented stage",
                        "category_label_id": "stroke_stage__invented",
                        "arm_id": "overall",
                    }
                ],
            },
        ],
    }
    precision_result = {
        "status": "completed",
        "method": {"minimum_studies": 2},
        "heterogeneity": {"tau_squared": 0.0},
        "pooled": {"unit": "point"},
        "studies": [
            {
                "trial_id": f"NCT{index}",
                "trial_label": f"Trial {index}",
                "arm_id": "control_1",
                "estimate": float(index * 10),
                "variance": 1.0,
                "raw_weight": 1.0,
            }
            for index in range(1, 4)
        ],
    }

    analysis = calculate_condition_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("reported_results", "stroke_stage"): attribute
        },
        categorical_penalty_policy={
            "model_id": "direct_label_match",
            "unresolved_raw_penalty": 0.1,
        },
    )

    states = [
        study["conditions"][0] for study in analysis["studies"]
    ]
    assert analysis["status"] == "completed"
    assert analysis["complete_study_count"] == 3
    assert analysis["unresolved_categorical_state_count"] == 3
    assert [state["status"] for state in states] == [
        "missing",
        "not_reported",
        "invalid_label",
    ]
    assert all(
        state["penalty_kind"] == "unresolved" for state in states
    )
    assert all(state["raw_penalty"] == 0.1 for state in states)
    assert all(
        state["weighted_penalty"] == 0.2 for state in states
    )
    assert all(
        state["relevance"] == pytest.approx(math.exp(-0.2))
        for state in states
    )
    assert analysis["pooled"]["estimate"] == pytest.approx(20.0)
    assert "small raw penalty 0.1" in analysis["warnings"][0]


def test_finite_reported_range_uses_audited_midpoint_and_is_labeled():
    condition = {
        "condition_id": "reported_results__assessment",
        "condition_source": "reported_results",
        "attribute_id": "assessment",
        "attribute": "Assessment Window",
        "value_type": "numerical",
        "target": {
            "value_kind": "point",
            "value": 30.0,
            "unit": "day",
        },
        "penalty_weight": 1.0,
        "numerical_penalty": _model(),
    }
    attribute = {
        "canonical_unit": "day",
        "trial_values": [
            {
                "trial_id": "NCT1",
                "audit_status": "approved",
                "item_review_status": "approved",
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "range",
                        "value": None,
                        "intervals": [{"lower": 13.0, "upper": 17.0}],
                        "unit": "day",
                        "arm_id": "overall",
                    }
                ],
            }
        ],
    }
    precision_result = {
        "status": "completed",
        "method": {"minimum_studies": 2},
        "heterogeneity": {"tau_squared": 0.0},
        "pooled": {"unit": "point"},
        "studies": [
            {
                "trial_id": "NCT1",
                "trial_label": "Trial 1",
                "arm_id": "control_1",
                "estimate": 10.0,
                "variance": 1.0,
                "raw_weight": 1.0,
            }
        ],
    }

    analysis = calculate_numerical_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("reported_results", "assessment"): attribute
        },
    )

    result = analysis["studies"][0]["conditions"][0]
    assert result["value"] == 15.0
    assert result["value_selection"] == "finite_range_midpoint"
    assert analysis["status"] == "insufficient_complete_data"


def test_unresolved_numerical_states_receive_small_penalty_and_stay_poolable():
    condition = {
        "condition_id": "eligibility_criteria__life_expectancy",
        "condition_source": "eligibility_criteria",
        "attribute_id": "life_expectancy",
        "attribute": "Life expectancy",
        "value_type": "numerical",
        "target": {
            "value_kind": "point",
            "value": 1.0,
            "unit": "year",
        },
        "penalty_weight": 1.0,
        "numerical_penalty": _model(),
    }
    approved = {
        "audit_status": "approved",
        "item_review_status": "approved",
        "reported": True,
    }
    attribute = {
        "canonical_unit": "year",
        "trial_values": [
            {
                **approved,
                "trial_id": "NCT1",
                "observations": [],
            },
            {
                "trial_id": "NCT2",
                "audit_status": "missing",
                "item_review_status": "missing",
                "reported": False,
                "observations": [],
            },
            {
                "trial_id": "NCT3",
                "audit_status": "pending",
                "item_review_status": "pending",
                "reported": True,
                "observations": [],
            },
            {
                **approved,
                "trial_id": "NCT4",
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "point",
                        "value": "not numeric",
                        "unit": "year",
                        "arm_id": "overall",
                    }
                ],
            },
            {
                **approved,
                "trial_id": "NCT5",
                "observations": [
                    {
                        "status": "found",
                        "audit_status": "approved",
                        "value_kind": "point",
                        "value": value,
                        "unit": "year",
                        "arm_id": "overall",
                    }
                    for value in (1.0, 2.0)
                ],
            },
        ],
    }
    precision_result = {
        "status": "completed",
        "method": {"minimum_studies": 2},
        "heterogeneity": {"tau_squared": 0.0},
        "pooled": {"estimate": 30.0, "unit": "point"},
        "studies": [
            {
                "trial_id": f"NCT{index}",
                "trial_label": f"Trial {index}",
                "arm_id": "control_1",
                "estimate": float(index * 10),
                "variance": 1.0,
                "raw_weight": 1.0,
            }
            for index in range(1, 6)
        ],
    }

    analysis = calculate_condition_penalty_analysis(
        precision_result,
        conditions=[condition],
        attributes={
            ("eligibility_criteria", "life_expectancy"): attribute
        },
        numerical_penalty_policy=_policy(),
    )

    states = [
        study["conditions"][0] for study in analysis["studies"]
    ]
    assert analysis["status"] == "completed"
    assert analysis["complete_study_count"] == 5
    assert analysis["unresolved_numerical_state_count"] == 5
    assert [state["status"] for state in states] == [
        "missing",
        "not_reported",
        "unverified",
        "invalid_value",
        "ambiguous",
    ]
    assert all(
        state["penalty_kind"] == "unresolved" for state in states
    )
    assert all(state["raw_penalty"] == 0.1 for state in states)
    assert all(
        state["relevance"] == pytest.approx(math.exp(-0.1))
        for state in states
    )
    assert all(
        study["adjusted_weight_percent"] == pytest.approx(20.0)
        for study in analysis["studies"]
    )
    assert analysis["pooled"]["estimate"] == pytest.approx(30.0)
    assert "unresolved numerical" in analysis["warnings"][0]
