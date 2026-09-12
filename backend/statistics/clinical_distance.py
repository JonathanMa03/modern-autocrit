"""Deterministic numerical penalty models and auditable LLM initialization."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


NUMERICAL_PENALTY_MODEL_ID = "absolute_clinical_distance"
CATEGORICAL_PENALTY_MODEL_ID = "direct_label_match"
CONDITION_PENALTY_MODEL_ID = "condition_relevance_product"
DEFAULT_NUMERICAL_UNRESOLVED_RAW_PENALTY = 0.1
DEFAULT_CATEGORICAL_UNRESOLVED_RAW_PENALTY = 0.1
PENALTY_INITIALIZATION_SOURCES = {"default", "llm", "user"}
PENALTY_INITIALIZATION_STATUSES = {"pending", "suggested", "edited"}
PARAMETER_NAMES = (
    "full_relevance_tolerance",
    "clinical_distance_scale",
)


class PenaltyValidationError(ValueError):
    """Raised when a numerical penalty model is invalid."""


def validate_numerical_penalty_policy(payload: Any) -> dict[str, Any]:
    """Validate the configurable numerical-penalty UI and model policy."""

    if not isinstance(payload, dict):
        raise PenaltyValidationError(
            "Condition configuration must define numerical_penalty."
        )
    model_id = str(payload.get("model_id") or "")
    if model_id != NUMERICAL_PENALTY_MODEL_ID:
        raise PenaltyValidationError(
            "Numerical penalty configuration must use "
            f"'{NUMERICAL_PENALTY_MODEL_ID}'."
        )
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise PenaltyValidationError(
            "Numerical penalty configuration must define parameters."
        )
    normalized_parameters: dict[str, dict[str, float]] = {}
    for name in PARAMETER_NAMES:
        definition = parameters.get(name)
        if not isinstance(definition, dict):
            raise PenaltyValidationError(
                f"Numerical penalty parameter '{name}' is required."
            )
        normalized_parameters[name] = _validate_parameter_definition(
            name,
            definition,
        )
    if normalized_parameters["clinical_distance_scale"]["minimum"] <= 0:
        raise PenaltyValidationError(
            "clinical_distance_scale must have a positive minimum."
        )
    unresolved = _finite_number(
        payload.get(
            "unresolved_raw_penalty",
            DEFAULT_NUMERICAL_UNRESOLVED_RAW_PENALTY,
        ),
        "numerical unresolved_raw_penalty",
    )
    if unresolved < 0 or unresolved >= 1:
        raise PenaltyValidationError(
            "Numerical unresolved_raw_penalty must be at least 0 and "
            "smaller than 1."
        )
    display = payload.get("display_domain")
    if not isinstance(display, dict):
        raise PenaltyValidationError(
            "Numerical penalty configuration must define display_domain."
        )
    display_minimum = _finite_number(
        display.get("minimum"),
        "display_domain minimum",
    )
    display_maximum = _finite_number(
        display.get("maximum"),
        "display_domain maximum",
    )
    minimum_span = _finite_number(
        display.get("minimum_span"),
        "display_domain minimum_span",
    )
    display_step = _finite_number(
        display.get("step"),
        "display_domain step",
    )
    if (
        display_maximum <= display_minimum
        or minimum_span <= 0
        or display_step <= 0
    ):
        raise PenaltyValidationError(
            "Numerical penalty display_domain must have an ordered range, "
            "a positive minimum_span, and a positive step."
        )

    point_count = payload.get("curve_point_count", 121)
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 21
        or point_count > 501
    ):
        raise PenaltyValidationError(
            "Numerical penalty curve_point_count must be an integer from "
            "21 to 501."
        )
    initializer = payload.get("initializer")
    if not isinstance(initializer, dict):
        raise PenaltyValidationError(
            "Numerical penalty configuration must define initializer."
        )
    model = str(initializer.get("model") or "").strip()
    effort = str(initializer.get("reasoning_effort") or "").strip()
    if not model or effort not in {"low", "medium", "high"}:
        raise PenaltyValidationError(
            "Numerical penalty initializer requires a model and low, medium, "
            "or high reasoning_effort."
        )
    return {
        "model_id": model_id,
        "label": str(
            payload.get("label") or "Absolute clinical-distance relevance"
        ),
        "unresolved_raw_penalty": unresolved,
        "curve_point_count": point_count,
        "parameters": normalized_parameters,
        "display_domain": {
            "minimum": display_minimum,
            "maximum": display_maximum,
            "minimum_span": minimum_span,
            "step": display_step,
        },
        "initializer": {
            "model": model,
            "reasoning_effort": effort,
        },
    }


def validate_categorical_penalty_policy(payload: Any) -> dict[str, Any]:
    """Validate the direct-match and unresolved-state categorical policy."""

    if not isinstance(payload, dict):
        raise PenaltyValidationError(
            "Condition configuration must define categorical_penalty."
        )
    model_id = str(payload.get("model_id") or "")
    if model_id != CATEGORICAL_PENALTY_MODEL_ID:
        raise PenaltyValidationError(
            "Categorical penalty configuration must use "
            f"'{CATEGORICAL_PENALTY_MODEL_ID}'."
        )
    unresolved = _finite_number(
        payload.get("unresolved_raw_penalty"),
        "categorical unresolved_raw_penalty",
    )
    if unresolved < 0 or unresolved >= 1:
        raise PenaltyValidationError(
            "Categorical unresolved_raw_penalty must be at least 0 and "
            "smaller than the definite-mismatch penalty of 1."
        )
    return {
        "model_id": model_id,
        "label": str(
            payload.get("label") or "Direct controlled-label relevance"
        ),
        "unresolved_raw_penalty": unresolved,
    }


def default_numerical_penalty(
    target: dict[str, Any],
    attribute: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, reviewable model before LLM initialization."""

    validated_policy = validate_numerical_penalty_policy(policy)
    target_lower, target_upper = numerical_target_bounds(target)
    finite_target = [
        value
        for value in (target_lower, target_upper)
        if value is not None
    ]
    observed = observed_numeric_values(attribute)
    values = [*finite_target, *observed]
    center = (
        sum(finite_target) / len(finite_target)
        if finite_target
        else (sum(observed) / len(observed) if observed else 0.0)
    )
    value_minimum = min(values) if values else center
    value_maximum = max(values) if values else center
    observed_span = value_maximum - value_minimum
    target_span = (
        target_upper - target_lower
        if target_lower is not None and target_upper is not None
        else 0.0
    )
    scale_definition = validated_policy["parameters"][
        "clinical_distance_scale"
    ]
    scale_candidate = (
        max(observed_span / 4.0, 1.0)
        if observed_span > 0
        else max(abs(center) * 0.2, 1.0)
    )
    scale = max(
        scale_definition["minimum"],
        min(scale_definition["maximum"], scale_candidate),
    )

    margin = max(scale * 3.0, target_span / 2.0, 1.0)
    display_minimum = min(value_minimum, center) - margin
    display_maximum = max(value_maximum, center) + margin
    nonnegative_values = values and all(value >= 0 for value in values)
    if nonnegative_values:
        display_minimum = max(0.0, display_minimum)
    display_policy = validated_policy["display_domain"]
    display_minimum = max(display_policy["minimum"], display_minimum)
    display_maximum = min(display_policy["maximum"], display_maximum)
    if (
        display_maximum - display_minimum
        < display_policy["minimum_span"]
    ):
        display_maximum = min(
            display_policy["maximum"],
            display_minimum + display_policy["minimum_span"],
        )

    return {
        "model_id": NUMERICAL_PENALTY_MODEL_ID,
        "parameters": {
            "full_relevance_tolerance": validated_policy["parameters"][
                "full_relevance_tolerance"
            ]["default"],
            "clinical_distance_scale": _rounded(scale),
        },
        "display_domain": {
            "minimum": _rounded(display_minimum),
            "maximum": _rounded(display_maximum),
        },
        "initialization": {
            "source": "default",
            "status": "pending",
            "model": None,
            "reasoning_effort": None,
            "generated_at": None,
            "rationale": (
                "Deterministic placeholder settings. Choose Initialize "
                "Weights to request an LLM suggestion, or tune them manually."
            ),
        },
    }


def validate_numerical_penalty(
    payload: Any,
    *,
    condition_id: str,
    target: dict[str, Any],
    attribute: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one numerical clinical-distance penalty contract."""

    validated_policy = validate_numerical_penalty_policy(policy)
    default = default_numerical_penalty(target, attribute, validated_policy)
    if payload is None:
        return default
    if not isinstance(payload, dict):
        raise PenaltyValidationError(
            f"Condition {condition_id} numerical_penalty must be an object."
        )
    model_id = str(payload.get("model_id") or default["model_id"])
    if model_id != NUMERICAL_PENALTY_MODEL_ID:
        raise PenaltyValidationError(
            f"Condition {condition_id} must use numerical penalty model "
            f"'{NUMERICAL_PENALTY_MODEL_ID}'."
        )
    raw_parameters = payload.get("parameters")
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict):
        raise PenaltyValidationError(
            f"Condition {condition_id} numerical penalty parameters must "
            "be an object."
        )
    legacy_taper = raw_parameters.get("taper")
    if legacy_taper is not None:
        legacy_taper = _finite_number(
            legacy_taper,
            f"Condition {condition_id} legacy taper",
        )
        if legacy_taper <= 0:
            raise PenaltyValidationError(
                f"Condition {condition_id} legacy taper must be positive."
            )
    parameters: dict[str, float] = {}
    for name in PARAMETER_NAMES:
        definition = validated_policy["parameters"][name]
        value = raw_parameters.get(name, default["parameters"][name])
        if (
            name == "clinical_distance_scale"
            and legacy_taper is not None
        ):
            value = _finite_number(
                value,
                f"Condition {condition_id} {name}",
            ) / math.sqrt(legacy_taper)
        normalized = _finite_number(
            value,
            f"Condition {condition_id} {name}",
        )
        if normalized < definition["minimum"] or normalized > definition[
            "maximum"
        ]:
            raise PenaltyValidationError(
                f"Condition {condition_id} {name} must be between "
                f"{definition['minimum']:g} and "
                f"{definition['maximum']:g}."
            )
        parameters[name] = normalized

    raw_domain = payload.get("display_domain")
    if raw_domain is None:
        raw_domain = {}
    if not isinstance(raw_domain, dict):
        raise PenaltyValidationError(
            f"Condition {condition_id} display_domain must be an object."
        )
    domain_policy = validated_policy["display_domain"]
    domain_minimum = _finite_number(
        raw_domain.get(
            "minimum",
            default["display_domain"]["minimum"],
        ),
        f"Condition {condition_id} display minimum",
    )
    domain_maximum = _finite_number(
        raw_domain.get(
            "maximum",
            default["display_domain"]["maximum"],
        ),
        f"Condition {condition_id} display maximum",
    )
    if (
        domain_minimum < domain_policy["minimum"]
        or domain_maximum > domain_policy["maximum"]
        or domain_maximum - domain_minimum
        < domain_policy["minimum_span"]
    ):
        raise PenaltyValidationError(
            f"Condition {condition_id} display domain is outside the "
            "configured bounds or too narrow."
        )
    target_lower, target_upper = numerical_target_bounds(target)
    for bound in (target_lower, target_upper):
        if bound is not None and not domain_minimum <= bound <= domain_maximum:
            raise PenaltyValidationError(
                f"Condition {condition_id} display domain must contain its "
                "finite target boundaries."
            )

    raw_initialization = payload.get("initialization")
    if raw_initialization is None:
        raw_initialization = default["initialization"]
    if not isinstance(raw_initialization, dict):
        raise PenaltyValidationError(
            f"Condition {condition_id} numerical penalty initialization "
            "must be an object."
        )
    source = str(raw_initialization.get("source") or "default")
    status = str(raw_initialization.get("status") or "pending")
    if source not in PENALTY_INITIALIZATION_SOURCES:
        raise PenaltyValidationError(
            f"Condition {condition_id} has an invalid penalty initialization "
            "source."
        )
    if status not in PENALTY_INITIALIZATION_STATUSES:
        raise PenaltyValidationError(
            f"Condition {condition_id} has an invalid penalty initialization "
            "status."
        )
    rationale = str(raw_initialization.get("rationale") or "").strip()
    if len(rationale) > 2000:
        raise PenaltyValidationError(
            f"Condition {condition_id} penalty rationale is too long."
        )
    return {
        "model_id": model_id,
        "parameters": parameters,
        "display_domain": {
            "minimum": domain_minimum,
            "maximum": domain_maximum,
        },
        "initialization": {
            "source": source,
            "status": status,
            "model": _optional_text(raw_initialization.get("model"), 120),
            "reasoning_effort": _optional_text(
                raw_initialization.get("reasoning_effort"),
                20,
            ),
            "generated_at": _optional_text(
                raw_initialization.get("generated_at"),
                80,
            ),
            "rationale": rationale,
        },
    }


def clinical_distance_penalty(
    value: float,
    *,
    target: dict[str, Any],
    numerical_penalty: dict[str, Any],
    condition_weight: float = 1.0,
) -> dict[str, float]:
    """Return raw penalty and retained relevance for one numerical value."""

    numeric_value = _finite_number(value, "Numerical study value")
    weight = _finite_number(condition_weight, "Condition penalty weight")
    if weight < 0:
        raise PenaltyValidationError(
            "Condition penalty weight must be non-negative."
        )
    target_lower, target_upper = numerical_target_bounds(target)
    if target_lower is None and target_upper is None:
        raise PenaltyValidationError(
            "A numerical penalty target needs at least one finite boundary."
        )
    if target_lower is not None and numeric_value < target_lower:
        distance = target_lower - numeric_value
    elif target_upper is not None and numeric_value > target_upper:
        distance = numeric_value - target_upper
    else:
        distance = 0.0
    parameters = numerical_penalty.get("parameters") or {}
    tolerance = _finite_number(
        parameters.get("full_relevance_tolerance"),
        "full_relevance_tolerance",
    )
    scale = _finite_number(
        parameters.get("clinical_distance_scale"),
        "clinical_distance_scale",
    )
    if tolerance < 0 or scale <= 0:
        raise PenaltyValidationError(
            "Numerical penalty parameters require h >= 0 and delta > 0."
        )
    excess_distance = max(0.0, distance - tolerance)
    raw_penalty = (excess_distance / scale) ** 2
    weighted_penalty = weight * raw_penalty
    relevance = math.exp(-weighted_penalty)
    return {
        "distance": distance,
        "excess_distance": excess_distance,
        "raw_penalty": raw_penalty,
        "weighted_penalty": weighted_penalty,
        "relevance": relevance,
    }


def categorical_label_penalty(
    observed_label_id: str,
    *,
    target_label_ids: list[str],
    condition_weight: float = 1.0,
) -> dict[str, Any]:
    """Return an exact controlled-label match penalty and relevance."""

    observed = str(observed_label_id or "")
    targets = {
        str(label_id)
        for label_id in target_label_ids
        if str(label_id)
    }
    if not observed:
        raise PenaltyValidationError(
            "Categorical penalty requires a parsed controlled label ID."
        )
    if not targets:
        raise PenaltyValidationError(
            "Categorical penalty requires at least one target label ID."
        )
    weight = _finite_number(
        condition_weight,
        "Categorical condition weight",
    )
    if weight < 0:
        raise PenaltyValidationError(
            "Categorical condition weight must be non-negative."
        )
    matched = observed in targets
    raw_penalty = 0.0 if matched else 1.0
    weighted_penalty = weight * raw_penalty
    return {
        "model_id": CATEGORICAL_PENALTY_MODEL_ID,
        "match": matched,
        "penalty_kind": "match" if matched else "mismatch",
        "raw_penalty": raw_penalty,
        "weighted_penalty": weighted_penalty,
        "relevance": math.exp(-weighted_penalty),
    }


def categorical_unresolved_penalty(
    *,
    condition_weight: float = 1.0,
    raw_penalty: float = DEFAULT_CATEGORICAL_UNRESOLVED_RAW_PENALTY,
) -> dict[str, Any]:
    """Return the low-severity penalty for an unresolved categorical state."""

    weight = _finite_number(
        condition_weight,
        "Categorical condition weight",
    )
    unresolved = _finite_number(
        raw_penalty,
        "Categorical unresolved raw penalty",
    )
    if weight < 0:
        raise PenaltyValidationError(
            "Categorical condition weight must be non-negative."
        )
    if unresolved < 0 or unresolved >= 1:
        raise PenaltyValidationError(
            "Categorical unresolved raw penalty must be at least 0 and "
            "smaller than 1."
        )
    weighted_penalty = weight * unresolved
    return {
        "model_id": CATEGORICAL_PENALTY_MODEL_ID,
        "match": None,
        "penalty_kind": "unresolved",
        "raw_penalty": unresolved,
        "weighted_penalty": weighted_penalty,
        "relevance": math.exp(-weighted_penalty),
    }


def build_penalty_initializer_pool_context(
    pooling_catalog: dict[str, Any],
    payload: Any,
) -> dict[str, Any]:
    """Build a validated, compact trial/setup context for the initializer."""

    if payload is not None and not isinstance(payload, dict):
        raise PenaltyValidationError(
            "Penalty initialization pooling context must be an object."
        )
    raw = payload if isinstance(payload, dict) else {}
    targets = {
        str(item.get("attribute_id") or ""): item
        for item in pooling_catalog.get("targets") or []
        if isinstance(item, dict)
    }
    attribute_id = str(
        raw.get("attribute_id")
        or pooling_catalog.get("default_attribute_id")
        or ""
    )
    target = targets.get(attribute_id)
    if target is None:
        if raw:
            raise PenaltyValidationError(
                "Penalty initialization selected an unavailable endpoint "
                "Attribute."
            )
        return {
            "endpoint": None,
            "method": None,
            "selected_trial_count": 0,
            "selected_trials": [],
            "available_trials": [],
        }

    methods = {
        str(item.get("method_id") or ""): item
        for item in pooling_catalog.get("methods") or []
        if isinstance(item, dict) and item.get("available") is not False
    }
    method_id = str(
        raw.get("method")
        or pooling_catalog.get("default_method_id")
        or ""
    )
    method = methods.get(method_id)
    if raw.get("method") and method is None:
        raise PenaltyValidationError(
            "Penalty initialization selected an unavailable pooling method."
        )
    if method is None and methods:
        method = next(iter(methods.values()))
    trials = {
        str(item.get("trial_id") or ""): item
        for item in target.get("trials") or []
        if isinstance(item, dict)
    }
    raw_selections = raw.get("selections") or []
    if not isinstance(raw_selections, list):
        raise PenaltyValidationError(
            "Penalty initialization pool selections must be an array."
        )

    selected_trials: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    for raw_selection in raw_selections:
        if not isinstance(raw_selection, dict):
            raise PenaltyValidationError(
                "Every penalty initialization pool selection must be an "
                "object."
            )
        trial_id = str(raw_selection.get("trial_id") or "")
        if trial_id in seen_trials:
            raise PenaltyValidationError(
                f"Trial '{trial_id}' can appear only once in penalty "
                "initialization context."
            )
        trial = trials.get(trial_id)
        if trial is None:
            raise PenaltyValidationError(
                f"Trial '{trial_id}' is not available for the selected "
                "endpoint."
            )
        if trial.get("selection_eligible") is not True:
            raise PenaltyValidationError(
                f"Trial '{trial_id}' is not an approved, poolable trial for "
                "the selected endpoint."
            )
        observations = {
            str(observation.get("observation_id") or ""): observation
            for endpoint in trial.get("endpoints") or []
            if isinstance(endpoint, dict)
            for observation in endpoint.get("observations") or []
            if isinstance(observation, dict)
        }
        observation_id = str(
            raw_selection.get("observation_id") or ""
        )
        observation = observations.get(observation_id)
        if observation is None:
            raise PenaltyValidationError(
                f"Trial '{trial_id}' selected an unavailable endpoint "
                "observation."
            )
        precision_options = {
            str(option.get("precision_option_id") or ""): option
            for option in observation.get("precision_options") or []
            if isinstance(option, dict)
        }
        precision_id = str(
            raw_selection.get("precision_option_id") or ""
        )
        precision = precision_options.get(precision_id)
        if precision_id and precision is None:
            raise PenaltyValidationError(
                f"Trial '{trial_id}' selected an unavailable precision "
                "source."
            )
        sample_size_options = {
            str(option.get("sample_size_option_id") or ""): option
            for option in observation.get("sample_size_options") or []
            if isinstance(option, dict)
        }
        sample_size_id = str(
            raw_selection.get("sample_size_option_id") or ""
        )
        sample_size = sample_size_options.get(sample_size_id)
        if sample_size_id and sample_size is None:
            raise PenaltyValidationError(
                f"Trial '{trial_id}' selected an unavailable population-size "
                "source."
            )
        selected_trials.append(
            {
                "trial_id": trial_id,
                "trial_label": trial.get("trial_label") or trial_id,
                "audit_status": trial.get("audit_status"),
                "endpoint_label": observation.get("endpoint_label"),
                "endpoint_role": observation.get("endpoint_role"),
                "arm_id": observation.get("arm_id"),
                "arm_label": observation.get("arm_label"),
                "estimate": observation.get("estimate"),
                "unit": observation.get("unit"),
                "statistic_type": observation.get("statistic_type"),
                "precision_source": (
                    precision.get("source_label")
                    if precision is not None
                    else None
                ),
                "population_size_source": (
                    sample_size.get("source_label")
                    if sample_size is not None
                    else None
                ),
            }
        )
        seen_trials.add(trial_id)

    available_trials = [
        {
            "trial_id": trial_id,
            "trial_label": trial.get("trial_label") or trial_id,
            "audit_status": trial.get("audit_status"),
            "selection_eligible": trial.get("selection_eligible") is True,
            "endpoint_instance_count": len(trial.get("endpoints") or []),
            "observation_count": sum(
                len(endpoint.get("observations") or [])
                for endpoint in trial.get("endpoints") or []
                if isinstance(endpoint, dict)
            ),
        }
        for trial_id, trial in trials.items()
    ]
    return {
        "endpoint": {
            "attribute_id": attribute_id,
            "attribute": target.get("canonical_name") or attribute_id,
            "description": target.get("description"),
            "unit": target.get("canonical_unit"),
            "analysis_kind": target.get("analysis_kind"),
        },
        "method": (
            {
                "method_id": method.get("method_id"),
                "label": method.get("label"),
                "model_label": method.get("model_label"),
                "weight_definition": method.get("weight_definition"),
            }
            if method is not None
            else None
        ),
        "selected_trial_count": len(selected_trials),
        "selected_trials": selected_trials,
        "available_trials": available_trials,
    }


def initialize_numerical_penalties(
    conditions: list[dict[str, Any]],
    *,
    attributes: dict[tuple[str, str], dict[str, Any]],
    policy: dict[str, Any],
    task_context: dict[str, Any],
    provider: Any,
) -> dict[str, Any]:
    """Ask one structured LLM call to initialize pending numerical models."""

    validated_policy = validate_numerical_penalty_policy(policy)
    pending = [
        condition
        for condition in conditions
        if condition.get("value_type") == "numerical"
        and (
            (condition.get("numerical_penalty") or {})
            .get("initialization", {})
            .get("status")
            == "pending"
        )
    ]
    initializer = validated_policy["initializer"]
    if not pending:
        return {
            "suggestions": [],
            "model": initializer["model"],
            "reasoning_effort": initializer["reasoning_effort"],
        }

    selected_trial_ids = {
        str(item.get("trial_id") or "")
        for item in (
            (task_context.get("selected_pool_context") or {}).get(
                "selected_trials"
            )
            or []
        )
        if isinstance(item, dict) and item.get("trial_id")
    }
    request_conditions: list[dict[str, Any]] = []
    for condition in pending:
        key = (
            str(condition.get("condition_source") or ""),
            str(condition.get("attribute_id") or ""),
        )
        attribute = attributes.get(key)
        if attribute is None:
            raise PenaltyValidationError(
                f"Condition {condition.get('condition_id')} Attribute is "
                "not available for penalty initialization."
            )
        request_conditions.append(
            {
                "condition_id": condition["condition_id"],
                "attribute_id": condition["attribute_id"],
                "attribute": condition.get("attribute"),
                "entity": condition.get("entity"),
                "description": attribute.get("description"),
                "unit": attribute.get("canonical_unit"),
                "integer_only": bool(attribute.get("integer_only", False)),
                "target": deepcopy(condition.get("target")),
                "current_model": deepcopy(
                    condition.get("numerical_penalty")
                ),
                "observed_trial_values": _observed_value_context(
                    attribute,
                    selected_trial_ids=selected_trial_ids,
                ),
            }
        )

    schema = _initializer_output_schema(validated_policy)
    prompt_input = {
        "task_context": deepcopy(task_context),
        "model": {
            "model_id": NUMERICAL_PENALTY_MODEL_ID,
            "equation": (
                "c(x) = exp[-"
                "(max(0, d(x,target)-h) / delta)^2]"
            ),
            "definitions": {
                "h": "full_relevance_tolerance",
                "delta": "clinical_distance_scale",
                "d": (
                    "absolute distance to a point target, or distance to the "
                    "nearest boundary of a target interval"
                ),
            },
        },
        "configuration_bounds": {
            "parameters": deepcopy(validated_policy["parameters"]),
            "display_domain": deepcopy(
                validated_policy["display_domain"]
            ),
        },
        "conditions": request_conditions,
    }
    prompt = (
        "You initialize clinically interpretable hyperparameters for a "
        "numerical relevance penalty in an auditable meta-analysis setup.\n"
        "Use only the supplied JSON. Do not inspect files, browse, or invent "
        "reported trial values. Never change a condition target.\n\n"
        "For every condition, choose:\n"
        "- h: a full-relevance tolerance justified by the measurement;\n"
        "- delta: a clinically meaningful distance scale, strictly positive;\n"
        "- a display minimum and maximum that contain the target and make the "
        "curve clinically readable.\n\n"
        "Use the task and endpoint context, the controlled Attribute meaning, "
        "unit, target, selected trial/arm setup, and observed trial-value "
        "ranges. Prioritize observations marked selected_in_pool when the "
        "selected pool is nonempty; use the other verified trials only as "
        "broader distribution context. For follow-up timing "
        "conditions, materially earlier measurements may deserve very low "
        "relevance to a long-term target; choose parameters accordingly when "
        "the supplied context supports that interpretation. For bounded "
        "clinical scales, respect meaningful score distances. Be conservative "
        "and state uncertainty. Do not call a choice validated or evidence-"
        "based unless the supplied input establishes that.\n\n"
        "Return exactly one suggestion for every supplied condition_id and no "
        "others.\n\n"
        "PENALTY_INITIALIZATION_INPUT_JSON:\n"
        + json.dumps(prompt_input, ensure_ascii=False, indent=2)
    )
    raw = provider.generate_json(prompt, output_schema=schema)
    raw_suggestions = raw.get("suggestions") if isinstance(raw, dict) else None
    if not isinstance(raw_suggestions, list):
        raise PenaltyValidationError(
            "Penalty initializer did not return a suggestions array."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for suggestion in raw_suggestions:
        if not isinstance(suggestion, dict):
            raise PenaltyValidationError(
                "Every penalty suggestion must be an object."
            )
        condition_id = str(suggestion.get("condition_id") or "")
        if condition_id in by_id:
            raise PenaltyValidationError(
                f"Penalty initializer repeated condition {condition_id}."
            )
        by_id[condition_id] = suggestion
    expected_ids = {condition["condition_id"] for condition in pending}
    if set(by_id) != expected_ids:
        raise PenaltyValidationError(
            "Penalty initializer must return exactly the requested condition "
            "IDs."
        )

    now = datetime.now(timezone.utc).isoformat()
    suggestions: list[dict[str, Any]] = []
    for condition in pending:
        condition_id = condition["condition_id"]
        raw_suggestion = by_id[condition_id]
        key = (
            str(condition.get("condition_source") or ""),
            str(condition.get("attribute_id") or ""),
        )
        attribute = attributes[key]
        candidate = {
            "model_id": NUMERICAL_PENALTY_MODEL_ID,
            "parameters": {
                name: raw_suggestion.get(name)
                for name in PARAMETER_NAMES
            },
            "display_domain": {
                "minimum": raw_suggestion.get("display_minimum"),
                "maximum": raw_suggestion.get("display_maximum"),
            },
            "initialization": {
                "source": "llm",
                "status": "suggested",
                "model": initializer["model"],
                "reasoning_effort": initializer["reasoning_effort"],
                "generated_at": now,
                "rationale": str(
                    raw_suggestion.get("rationale") or ""
                ).strip(),
            },
        }
        normalized = validate_numerical_penalty(
            candidate,
            condition_id=condition_id,
            target=condition["target"],
            attribute=attribute,
            policy=validated_policy,
        )
        suggestions.append(
            {
                "condition_id": condition_id,
                "numerical_penalty": normalized,
            }
        )
    return {
        "suggestions": suggestions,
        "model": initializer["model"],
        "reasoning_effort": initializer["reasoning_effort"],
    }


def calculate_condition_penalty_analysis(
    precision_result: dict[str, Any],
    *,
    conditions: list[dict[str, Any]],
    attributes: dict[tuple[str, str], dict[str, Any]],
    numerical_penalty_policy: dict[str, Any] | None = None,
    categorical_penalty_policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Apply validated numerical and categorical relevance to study weights."""

    active_conditions = [
        condition
        for condition in conditions
        if condition.get("value_type") in {"numerical", "categorical"}
    ]
    if not active_conditions:
        return None
    numerical_condition_count = sum(
        condition.get("value_type") == "numerical"
        for condition in active_conditions
    )
    categorical_condition_count = sum(
        condition.get("value_type") == "categorical"
        for condition in active_conditions
    )
    validated_numerical_policy = (
        validate_numerical_penalty_policy(numerical_penalty_policy)
        if numerical_penalty_policy is not None
        else {
            "unresolved_raw_penalty": (
                DEFAULT_NUMERICAL_UNRESOLVED_RAW_PENALTY
            )
        }
    )
    validated_categorical_policy = (
        validate_categorical_penalty_policy(
            categorical_penalty_policy
            or {
                "model_id": CATEGORICAL_PENALTY_MODEL_ID,
                "unresolved_raw_penalty": (
                    DEFAULT_CATEGORICAL_UNRESOLVED_RAW_PENALTY
                ),
            }
        )
        if categorical_condition_count
        else None
    )
    method = deepcopy(precision_result.get("method") or {})
    heterogeneity = deepcopy(precision_result.get("heterogeneity"))
    reference_pooled = deepcopy(precision_result.get("pooled"))
    method_id = str(method.get("method_id") or "")
    tau_squared = float(
        (heterogeneity or {}).get("tau_squared") or 0.0
    )
    weighting = {
        "base_weight_definition": method.get("weight_definition"),
        "trial_relevance_definition": (
            "exp(-sum(condition_weight * raw_condition_penalty))"
        ),
        "adjusted_weight_definition": (
            "base_raw_weight * combined_trial_relevance"
        ),
        "tau_squared_source": (
            "conventional_unpenalized_pool"
            if method_id == "random_effects_inverse_variance_dl"
            else None
        ),
    }
    base_total_weight = sum(
        float(study["raw_weight"])
        for study in precision_result.get("studies") or []
        if _is_finite_number(study.get("raw_weight"))
    )
    study_results: list[dict[str, Any]] = []
    complete_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for study in precision_result.get("studies") or []:
        condition_results: list[dict[str, Any]] = []
        combined_relevance = 1.0
        complete = True
        for condition in active_conditions:
            if condition.get("value_type") == "categorical":
                condition_result = (
                    _evaluate_study_categorical_condition(
                        study,
                        condition,
                        attributes=attributes,
                        unresolved_raw_penalty=float(
                            validated_categorical_policy[
                                "unresolved_raw_penalty"
                            ]
                        ),
                    )
                )
            else:
                condition_result = _evaluate_study_numerical_condition(
                    study,
                    condition,
                    attributes=attributes,
                    unresolved_raw_penalty=float(
                        validated_numerical_policy[
                            "unresolved_raw_penalty"
                        ]
                    ),
                )
            condition_results.append(condition_result)
            if condition_result["status"] == "disabled":
                continue
            if _is_finite_number(condition_result.get("relevance")):
                combined_relevance *= float(condition_result["relevance"])
            else:
                complete = False
        base_weight = study.get("raw_weight")
        adjusted_weight = (
            float(base_weight) * combined_relevance
            if complete and _is_finite_number(base_weight)
            else None
        )
        row = {
            "trial_id": study.get("trial_id"),
            "trial_label": study.get("trial_label"),
            "status": "scored" if complete else "incomplete",
            "combined_relevance": (
                combined_relevance if complete else None
            ),
            "estimate": study.get("estimate"),
            "standard_error": study.get("standard_error"),
            "variance": study.get("variance"),
            "base_raw_weight": base_weight,
            "base_weight_percent": (
                100.0 * float(base_weight) / base_total_weight
                if _is_finite_number(base_weight)
                and base_total_weight > 0
                else None
            ),
            "adjusted_raw_weight": adjusted_weight,
            "adjusted_weight_percent": None,
            "unresolved_condition_count": sum(
                result.get("penalty_kind") == "unresolved"
                for result in condition_results
            ),
            "conditions": condition_results,
        }
        study_results.append(row)
        if complete and adjusted_weight is not None:
            complete_rows.append((study, row))

    warnings = []
    unresolved_numerical_state_count = sum(
        result.get("penalty_kind") == "unresolved"
        and result.get("value_type") == "numerical"
        for row in study_results
        for result in row["conditions"]
    )
    unresolved_categorical_state_count = sum(
        result.get("penalty_kind") == "unresolved"
        and result.get("value_type") == "categorical"
        for row in study_results
        for result in row["conditions"]
    )
    if unresolved_numerical_state_count:
        raw_penalty = float(
            validated_numerical_policy["unresolved_raw_penalty"]
        )
        warnings.append(
            f"{unresolved_numerical_state_count} unresolved numerical "
            "state(s)—missing, not reported, unavailable, unverified, "
            f"invalid, or ambiguous—received the small raw penalty "
            f"{raw_penalty:g} instead of being excluded."
        )
    if unresolved_categorical_state_count:
        raw_penalty = float(
            validated_categorical_policy["unresolved_raw_penalty"]
        )
        warnings.append(
            f"{unresolved_categorical_state_count} unresolved categorical "
            "state(s)—missing, not reported, unavailable, unverified, "
            f"invalid, or ambiguous—received the small raw penalty "
            f"{raw_penalty:g} "
            "instead of being treated as a definite mismatch or excluded."
        )
    incomplete_labels = [
        str(row.get("trial_label") or row.get("trial_id"))
        for row in study_results
        if row["status"] != "scored"
    ]
    if incomplete_labels:
        warnings.append(
            "The condition-adjusted estimate excludes trials without a "
            "calculable relevance for every active condition: "
            + ", ".join(incomplete_labels)
            + "."
        )
    minimum = int(
        (precision_result.get("method") or {}).get("minimum_studies") or 2
    )
    total_adjusted_weight = sum(
        row["adjusted_raw_weight"] or 0.0
        for _, row in complete_rows
    )
    if (
        precision_result.get("status") != "completed"
        or len(complete_rows) < minimum
        or total_adjusted_weight <= 0
    ):
        return {
            "status": "insufficient_complete_data",
            "model_id": CONDITION_PENALTY_MODEL_ID,
            "method": method,
            "heterogeneity": heterogeneity,
            "reference_pooled": reference_pooled,
            "weighting": weighting,
            "condition_count": len(active_conditions),
            "numerical_condition_count": numerical_condition_count,
            "categorical_condition_count": categorical_condition_count,
            "numerical_unresolved_raw_penalty": (
                validated_numerical_policy["unresolved_raw_penalty"]
            ),
            "unresolved_numerical_state_count": (
                unresolved_numerical_state_count
            ),
            "categorical_penalty": validated_categorical_policy,
            "unresolved_categorical_state_count": (
                unresolved_categorical_state_count
            ),
            "complete_study_count": len(complete_rows),
            "studies": study_results,
            "pooled": None,
            "warnings": warnings,
            "message": (
                f"At least {minimum} conventionally poolable trials with "
                "complete condition states and nonzero adjusted "
                "weight are required."
            ),
        }

    adjusted_estimate = sum(
        float(row["adjusted_raw_weight"]) * float(study["estimate"])
        for study, row in complete_rows
    ) / total_adjusted_weight
    conditional_variance = sum(
        float(row["adjusted_raw_weight"]) ** 2
        * (float(study["variance"]) + tau_squared)
        for study, row in complete_rows
    ) / total_adjusted_weight**2
    adjusted_standard_error = math.sqrt(conditional_variance)
    for _, row in complete_rows:
        row["adjusted_weight_percent"] = (
            100.0
            * float(row["adjusted_raw_weight"])
            / total_adjusted_weight
        )
    adjusted_weight_square_sum = sum(
        float(row["adjusted_raw_weight"]) ** 2
        for _, row in complete_rows
    )
    effective_study_count = (
        total_adjusted_weight**2 / adjusted_weight_square_sum
        if adjusted_weight_square_sum > 0
        else 0.0
    )
    conventional_estimate = (
        float(reference_pooled["estimate"])
        if isinstance(reference_pooled, dict)
        and _is_finite_number(reference_pooled.get("estimate"))
        else None
    )
    warnings.append(
        "The adjusted confidence interval treats the selected condition "
        "rules, penalty weights, and numerical hyperparameters as fixed "
        "external weights; it does not include condition-specification or "
        "hyperparameter-selection uncertainty."
    )
    return {
        "status": "completed",
        "model_id": CONDITION_PENALTY_MODEL_ID,
        "method": method,
        "heterogeneity": heterogeneity,
        "reference_pooled": reference_pooled,
        "weighting": weighting,
        "condition_count": len(active_conditions),
        "numerical_condition_count": numerical_condition_count,
        "categorical_condition_count": categorical_condition_count,
        "numerical_unresolved_raw_penalty": (
            validated_numerical_policy["unresolved_raw_penalty"]
        ),
        "unresolved_numerical_state_count": (
            unresolved_numerical_state_count
        ),
        "categorical_penalty": validated_categorical_policy,
        "unresolved_categorical_state_count": (
            unresolved_categorical_state_count
        ),
        "complete_study_count": len(complete_rows),
        "effective_study_count": effective_study_count,
        "studies": study_results,
        "pooled": {
            "estimate": adjusted_estimate,
            "standard_error": adjusted_standard_error,
            "variance": conditional_variance,
            "ci_95_lower": (
                adjusted_estimate - 1.959963984540054
                * adjusted_standard_error
            ),
            "ci_95_upper": (
                adjusted_estimate + 1.959963984540054
                * adjusted_standard_error
            ),
            "unit": (
                (precision_result.get("pooled") or {}).get("unit")
            ),
            "difference_from_conventional": (
                adjusted_estimate - conventional_estimate
                if conventional_estimate is not None
                else None
            ),
        },
        "warnings": warnings,
        "message": None,
    }


def calculate_numerical_penalty_analysis(
    precision_result: dict[str, Any],
    *,
    conditions: list[dict[str, Any]],
    attributes: dict[tuple[str, str], dict[str, Any]],
    numerical_penalty_policy: dict[str, Any] | None = None,
    categorical_penalty_policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Backward-compatible alias for the all-condition penalty analysis."""

    return calculate_condition_penalty_analysis(
        precision_result,
        conditions=conditions,
        attributes=attributes,
        numerical_penalty_policy=numerical_penalty_policy,
        categorical_penalty_policy=categorical_penalty_policy,
    )


def _numerical_unresolved_condition(
    base: dict[str, Any],
    *,
    status: str,
    message: str,
    condition_weight: float,
    unresolved_raw_penalty: float,
) -> dict[str, Any]:
    weighted_penalty = condition_weight * unresolved_raw_penalty
    return {
        **base,
        "status": status,
        "penalty_kind": "unresolved",
        "raw_penalty": unresolved_raw_penalty,
        "weighted_penalty": weighted_penalty,
        "relevance": math.exp(-weighted_penalty),
        "message": message,
    }


def _evaluate_study_numerical_condition(
    study: dict[str, Any],
    condition: dict[str, Any],
    *,
    attributes: dict[tuple[str, str], dict[str, Any]],
    unresolved_raw_penalty: float,
) -> dict[str, Any]:
    condition_id = str(condition.get("condition_id") or "")
    supplied_weight = condition.get("penalty_weight")
    condition_weight = float(
        1.0 if supplied_weight is None else supplied_weight
    )
    numerical_penalty = condition.get("numerical_penalty") or {}
    base = {
        "condition_id": condition_id,
        "condition_source": condition.get("condition_source"),
        "attribute_id": condition.get("attribute_id"),
        "attribute": condition.get("attribute"),
        "value_type": "numerical",
        "target": deepcopy(condition.get("target") or {}),
        "penalty_model": {
            "model_id": numerical_penalty.get("model_id"),
            "parameters": deepcopy(
                numerical_penalty.get("parameters") or {}
            ),
        },
        "condition_weight": condition_weight,
        "status": "missing",
        "value": None,
        "value_kind": None,
        "value_selection": None,
        "penalty_kind": None,
        "raw_penalty": None,
        "weighted_penalty": None,
        "relevance": None,
        "message": None,
    }
    if condition_weight == 0:
        return {
            **base,
            "status": "disabled",
            "relevance": 1.0,
            "message": "Condition weight is 0; this condition is disabled.",
        }
    key = (
        str(condition.get("condition_source") or ""),
        str(condition.get("attribute_id") or ""),
    )
    attribute = attributes.get(key)
    if attribute is None:
        return _numerical_unresolved_condition(
            base,
            status="unavailable",
            message="Controlled Attribute is unavailable.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    trial = next(
        (
            row
            for row in attribute.get("trial_values") or []
            if isinstance(row, dict)
            and str(row.get("trial_id") or "")
            == str(study.get("trial_id") or "")
        ),
        None,
    )
    if not isinstance(trial, dict):
        return _numerical_unresolved_condition(
            base,
            status="missing",
            message="No trial-level condition record is available.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    if trial.get("reported") is False:
        return _numerical_unresolved_condition(
            base,
            status="not_reported",
            message="The numerical condition value was not reported.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    if (
        str(trial.get("audit_status") or "")
        not in {"approved", "reviewer_approved"}
        or str(trial.get("item_review_status") or "")
        not in {"approved", "reviewer_approved"}
    ):
        return _numerical_unresolved_condition(
            base,
            status="unverified",
            message="The trial condition value is not reviewer approved.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    canonical_unit = str(attribute.get("canonical_unit") or "").casefold()
    verified_observations = [
        observation
        for observation in trial.get("observations") or []
        if isinstance(observation, dict)
        and observation.get("status") == "found"
        and str(observation.get("audit_status") or "")
        in {"approved", "reviewer_approved"}
    ]
    candidates = [
        observation
        for observation in verified_observations
        if (
            not canonical_unit
            or not observation.get("unit")
            or str(observation.get("unit") or "").casefold()
            == canonical_unit
        )
    ]
    arm_id = str(study.get("arm_id") or "")
    exact_arm = [
        observation
        for observation in candidates
        if str(observation.get("arm_id") or "") == arm_id
    ]
    if exact_arm:
        candidates = exact_arm
    else:
        overall = [
            observation
            for observation in candidates
            if str(observation.get("arm_id") or "") == "overall"
        ]
        if overall:
            candidates = overall
    endpoint_role = str(study.get("endpoint_role") or "")
    if endpoint_role:
        same_role = [
            observation
            for observation in candidates
            if str(observation.get("endpoint_role") or "")
            == endpoint_role
        ]
        if same_role:
            candidates = same_role

    represented = [
        (observation, _representative_numeric_value(observation))
        for observation in candidates
    ]
    represented = [
        (observation, value)
        for observation, value in represented
        if value is not None
    ]
    unique_values = {
        round(float(value[0]), 12)
        for _, value in represented
    }
    if not represented:
        status = "invalid_value" if verified_observations else "missing"
        message = (
            "The verified numerical state is not a finite point or "
            "single finite range in the canonical unit."
            if verified_observations
            else (
                "No verified point or finite single-range condition value "
                "is available in the canonical unit."
            )
        )
        return _numerical_unresolved_condition(
            base,
            status=status,
            message=message,
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    if len(unique_values) != 1:
        return _numerical_unresolved_condition(
            base,
            status="ambiguous",
            message=(
                "Multiple verified condition values remain after arm and "
                "endpoint-role scoping."
            ),
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    observation, (value, selection) = represented[0]
    evaluated = clinical_distance_penalty(
        value,
        target=condition["target"],
        numerical_penalty=condition["numerical_penalty"],
        condition_weight=condition_weight,
    )
    return {
        **base,
        "status": "scored",
        "value": value,
        "unit": attribute.get("canonical_unit"),
        "value_kind": observation.get("value_kind"),
        "value_selection": selection,
        "penalty_kind": "distance",
        "distance": evaluated["distance"],
        "excess_distance": evaluated["excess_distance"],
        "raw_penalty": evaluated["raw_penalty"],
        "weighted_penalty": evaluated["weighted_penalty"],
        "relevance": evaluated["relevance"],
        "message": None,
    }


def _categorical_unresolved_condition(
    base: dict[str, Any],
    *,
    status: str,
    message: str,
    condition_weight: float,
    unresolved_raw_penalty: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluated = categorical_unresolved_penalty(
        condition_weight=condition_weight,
        raw_penalty=unresolved_raw_penalty,
    )
    return {
        **base,
        **(details or {}),
        "status": status,
        "penalty_kind": evaluated["penalty_kind"],
        "match": None,
        "raw_penalty": evaluated["raw_penalty"],
        "weighted_penalty": evaluated["weighted_penalty"],
        "relevance": evaluated["relevance"],
        "message": message,
    }


def _evaluate_study_categorical_condition(
    study: dict[str, Any],
    condition: dict[str, Any],
    *,
    attributes: dict[tuple[str, str], dict[str, Any]],
    unresolved_raw_penalty: float,
) -> dict[str, Any]:
    condition_id = str(condition.get("condition_id") or "")
    supplied_weight = condition.get("penalty_weight")
    condition_weight = float(
        1.0 if supplied_weight is None else supplied_weight
    )
    target_label_ids = [
        str(label_id)
        for label_id in (condition.get("target") or {}).get(
            "label_ids"
        )
        or []
        if str(label_id)
    ]
    base = {
        "condition_id": condition_id,
        "condition_source": condition.get("condition_source"),
        "attribute_id": condition.get("attribute_id"),
        "attribute": condition.get("attribute"),
        "value_type": "categorical",
        "target": deepcopy(condition.get("target") or {}),
        "penalty_model": {
            "model_id": CATEGORICAL_PENALTY_MODEL_ID,
            "parameters": {},
        },
        "condition_weight": condition_weight,
        "status": "missing",
        "value": None,
        "label_id": None,
        "observed_label_ids": [],
        "observed_labels": [],
        "target_label_ids": target_label_ids,
        "target_labels": list(
            (condition.get("target") or {}).get("labels") or []
        ),
        "value_selection": None,
        "match": None,
        "penalty_kind": None,
        "raw_penalty": None,
        "weighted_penalty": None,
        "relevance": None,
        "message": None,
    }
    if condition_weight == 0:
        return {
            **base,
            "status": "disabled",
            "relevance": 1.0,
            "message": "Condition weight is 0; this condition is disabled.",
        }
    key = (
        str(condition.get("condition_source") or ""),
        str(condition.get("attribute_id") or ""),
    )
    attribute = attributes.get(key)
    if attribute is None:
        return _categorical_unresolved_condition(
            base,
            status="unavailable",
            message="Controlled Attribute is unavailable.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    labels_by_id = {
        str(item.get("label_id") or ""): str(
            item.get("canonical_label") or ""
        )
        for item in attribute.get("categorical_labels") or []
        if isinstance(item, dict) and item.get("label_id")
    }
    labels_by_name = {
        label.casefold(): label_id
        for label_id, label in labels_by_id.items()
        if label
    }
    target_labels = [
        labels_by_id[label_id]
        for label_id in target_label_ids
        if label_id in labels_by_id
    ]
    base["target_labels"] = target_labels
    trial = next(
        (
            row
            for row in attribute.get("trial_values") or []
            if isinstance(row, dict)
            and str(row.get("trial_id") or "")
            == str(study.get("trial_id") or "")
        ),
        None,
    )
    if not isinstance(trial, dict):
        return _categorical_unresolved_condition(
            base,
            status="missing",
            message="No trial-level condition record is available.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    if trial.get("reported") is False:
        return _categorical_unresolved_condition(
            base,
            status="not_reported",
            message="The categorical state was not reported for this trial.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    if (
        str(trial.get("audit_status") or "")
        not in {"approved", "reviewer_approved"}
        or str(trial.get("item_review_status") or "")
        not in {"approved", "reviewer_approved"}
    ):
        return _categorical_unresolved_condition(
            base,
            status="unverified",
            message="The trial condition state is not reviewer approved.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    candidates = [
        observation
        for observation in trial.get("observations") or []
        if isinstance(observation, dict)
        and observation.get("status") == "found"
        and observation.get("value_kind") == "category"
        and str(observation.get("audit_status") or "")
        in {"approved", "reviewer_approved"}
    ]
    arm_id = str(study.get("arm_id") or "")
    exact_arm = [
        observation
        for observation in candidates
        if str(observation.get("arm_id") or "") == arm_id
    ]
    if exact_arm:
        candidates = exact_arm
    else:
        overall = [
            observation
            for observation in candidates
            if str(observation.get("arm_id") or "") == "overall"
        ]
        if overall:
            candidates = overall
    endpoint_role = str(study.get("endpoint_role") or "")
    if endpoint_role:
        same_role = [
            observation
            for observation in candidates
            if str(observation.get("endpoint_role") or "")
            == endpoint_role
        ]
        if same_role:
            candidates = same_role
    if not candidates:
        return _categorical_unresolved_condition(
            base,
            status="missing",
            message="No verified parsed categorical state is available.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )

    represented: list[tuple[dict[str, Any], str, str]] = []
    invalid_labels: list[str] = []
    for observation in candidates:
        label_id = str(
            observation.get("category_label_id") or ""
        )
        selection = "reported_label_id"
        if not label_id:
            category = str(observation.get("category") or "").strip()
            label_id = labels_by_name.get(category.casefold(), "")
            selection = "exact_canonical_label"
        if label_id not in labels_by_id:
            invalid_labels.append(
                str(
                    observation.get("category_label_id")
                    or observation.get("category")
                    or "(missing label)"
                )
            )
            continue
        represented.append((observation, label_id, selection))
    if invalid_labels:
        return _categorical_unresolved_condition(
            base,
            status="invalid_label",
            message=(
                "Parsed categorical state is not an approved controlled "
                "label: " + ", ".join(sorted(set(invalid_labels))) + "."
            ),
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    unique_label_ids = {
        label_id for _, label_id, _ in represented
    }
    if not unique_label_ids:
        return _categorical_unresolved_condition(
            base,
            status="missing",
            message="No controlled categorical label was parsed.",
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
        )
    observed_label_ids = sorted(unique_label_ids)
    observed_labels = [
        labels_by_id[label_id] for label_id in observed_label_ids
    ]
    if len(unique_label_ids) != 1:
        return _categorical_unresolved_condition(
            base,
            status="ambiguous",
            message=(
                "Multiple different verified categorical labels remain after "
                "arm and endpoint-role scoping."
            ),
            condition_weight=condition_weight,
            unresolved_raw_penalty=unresolved_raw_penalty,
            details={
                "observed_label_ids": observed_label_ids,
                "observed_labels": observed_labels,
            },
        )
    label_id = observed_label_ids[0]
    observation, _, selection = next(
        row for row in represented if row[1] == label_id
    )
    evaluated = categorical_label_penalty(
        label_id,
        target_label_ids=target_label_ids,
        condition_weight=condition_weight,
    )
    return {
        **base,
        "status": "matched" if evaluated["match"] else "mismatched",
        "value": labels_by_id[label_id],
        "label_id": label_id,
        "observed_label_ids": observed_label_ids,
        "observed_labels": observed_labels,
        "value_selection": selection,
        "match": evaluated["match"],
        "penalty_kind": evaluated["penalty_kind"],
        "raw_penalty": evaluated["raw_penalty"],
        "weighted_penalty": evaluated["weighted_penalty"],
        "relevance": evaluated["relevance"],
        "criterion_type": observation.get("criterion_type"),
        "message": (
            None
            if evaluated["match"]
            else "Parsed label does not match the selected target labels."
        ),
    }


def _representative_numeric_value(
    observation: dict[str, Any],
) -> tuple[float, str] | None:
    if (
        observation.get("value_kind") == "point"
        and _is_finite_number(observation.get("value"))
    ):
        return float(observation["value"]), "reported_point"
    intervals = observation.get("intervals")
    if (
        observation.get("value_kind") == "range"
        and isinstance(intervals, list)
        and len(intervals) == 1
        and isinstance(intervals[0], dict)
        and _is_finite_number(intervals[0].get("lower"))
        and _is_finite_number(intervals[0].get("upper"))
    ):
        lower = float(intervals[0]["lower"])
        upper = float(intervals[0]["upper"])
        return (lower + upper) / 2.0, "finite_range_midpoint"
    return None


def numerical_target_bounds(
    target: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Return finite target boundaries for a point or one interval."""

    if not isinstance(target, dict):
        return None, None
    if target.get("value_kind") == "point":
        value = target.get("value")
        if _is_finite_number(value):
            number = float(value)
            return number, number
        return None, None
    intervals = target.get("intervals")
    interval = (
        intervals[0]
        if isinstance(intervals, list)
        and intervals
        and isinstance(intervals[0], dict)
        else {}
    )
    lower = (
        float(interval["lower"])
        if _is_finite_number(interval.get("lower"))
        else None
    )
    upper = (
        float(interval["upper"])
        if _is_finite_number(interval.get("upper"))
        else None
    )
    return lower, upper


def observed_numeric_values(attribute: dict[str, Any]) -> list[float]:
    """Collect finite, canonical-unit observations for display defaults."""

    canonical_unit = str(attribute.get("canonical_unit") or "").casefold()
    values: list[float] = []
    for trial in attribute.get("trial_values") or []:
        if not isinstance(trial, dict):
            continue
        for observation in trial.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            observation_unit = str(
                observation.get("unit") or ""
            ).casefold()
            if (
                canonical_unit
                and observation_unit
                and canonical_unit != observation_unit
            ):
                continue
            if _is_finite_number(observation.get("value")):
                values.append(float(observation["value"]))
            for interval in observation.get("intervals") or []:
                if not isinstance(interval, dict):
                    continue
                for field in ("lower", "upper"):
                    if _is_finite_number(interval.get(field)):
                        values.append(float(interval[field]))
    return values


def _observed_value_context(
    attribute: dict[str, Any],
    *,
    selected_trial_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in attribute.get("trial_values") or []:
        if not isinstance(trial, dict):
            continue
        observations = []
        for observation in trial.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            observations.append(
                {
                    "value_kind": observation.get("value_kind"),
                    "value": observation.get("value"),
                    "intervals": deepcopy(
                        observation.get("intervals") or []
                    ),
                    "unit": observation.get("unit"),
                    "statistic_type": observation.get("statistic_type"),
                    "criterion_type": observation.get("criterion_type"),
                }
            )
        rows.append(
            {
                "trial_id": trial.get("trial_id"),
                "trial_label": trial.get("trial_label"),
                "selected_in_pool": (
                    str(trial.get("trial_id") or "")
                    in selected_trial_ids
                    if selected_trial_ids
                    else None
                ),
                "reported": trial.get("reported"),
                "observations": observations[:12],
            }
        )
    return rows[:20]


def _initializer_output_schema(
    policy: dict[str, Any],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "condition_id": {"type": "string", "minLength": 1},
        "rationale": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2000,
        },
    }
    for name in PARAMETER_NAMES:
        definition = policy["parameters"][name]
        properties[name] = {
            "type": "number",
            "minimum": definition["minimum"],
            "maximum": definition["maximum"],
        }
    domain = policy["display_domain"]
    properties["display_minimum"] = {
        "type": "number",
        "minimum": domain["minimum"],
        "maximum": domain["maximum"],
    }
    properties["display_maximum"] = deepcopy(
        properties["display_minimum"]
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["suggestions"],
        "properties": {
            "suggestions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(properties),
                    "properties": properties,
                },
            }
        },
    }


def _validate_parameter_definition(
    name: str,
    payload: dict[str, Any],
) -> dict[str, float]:
    minimum = _finite_number(payload.get("minimum"), f"{name} minimum")
    maximum = _finite_number(payload.get("maximum"), f"{name} maximum")
    default = _finite_number(payload.get("default"), f"{name} default")
    step = _finite_number(payload.get("step"), f"{name} step")
    if (
        minimum < 0
        or maximum <= minimum
        or default < minimum
        or default > maximum
        or step <= 0
    ):
        raise PenaltyValidationError(
            f"Numerical penalty parameter '{name}' has invalid bounds."
        )
    return {
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
    }


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PenaltyValidationError(f"{label} must be numeric and finite.")
    return float(value)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _optional_text(value: Any, maximum: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > maximum:
        raise PenaltyValidationError(
            f"Penalty initialization metadata exceeds {maximum} characters."
        )
    return text


def _rounded(value: float) -> float:
    return float(f"{value:.8g}")
