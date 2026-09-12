"""Two-stage LLM extraction and normalization for eligibility EVA points."""

from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

from rapidfuzz import fuzz

from backend.criteria_processor.eva_library import (
    EvaLibraryRepository,
    stable_id,
    stable_slug,
)
from backend.criteria_processor.eva_models import (
    EvaAttributeCandidate,
    EvaAuditDocument,
    EvaAuditItem,
    EvaAttributeIdReviewTrace,
    EvaMappingReviewTrace,
    EvaReasoningSelection,
    EvaSearchExpansion,
)
from backend.criteria_processor.eva_review import (
    ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
    CLINICAL_ATTRIBUTE_NAMING_POLICY,
    ID_REVIEW_NOT_REQUIRED_EXPLANATION,
    MAPPING_CORRECTION_PROMPT_VERSION,
    MAPPING_REVIEW_PROMPT_VERSION,
    review_eva_mapping,
    review_new_attribute_id,
    validate_english_explanation,
)
from backend.criteria_processor.eva_retrieval import (
    EvaReconciliationRetrievalQuery,
    score_reconciliation_candidate,
)
from backend.criteria_processor.text import clean_attribute_key


REASONER_PROMPT_VERSION = "eligibility-eva-reasoner-v3"


SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lexical_terms": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
        },
        "semantic_terms": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
        },
        "entity_hints": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "concept_summary": {"type": "string"},
    },
    "required": [
        "lexical_terms",
        "semantic_terms",
        "entity_hints",
        "concept_summary",
    ],
    "additionalProperties": False,
}


REASONING_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
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
        "attribute_aliases": {
            "type": "array",
            "items": {"type": "string"},
        },
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
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "rationale": {"type": "string"},
        "candidate_ids_considered": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "mapping_decision",
        "attribute_id",
        "entity_id",
        "entity_name",
        "new_entity",
        "attribute_name",
        "attribute_description",
        "attribute_aliases",
        "value_type",
        "categorical_value",
        "numerical_type",
        "numerical_value",
        "unit",
        "confidence",
        "rationale",
        "candidate_ids_considered",
    ],
    "additionalProperties": False,
}


class JsonLLMProvider(Protocol):
    def generate_json(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _unique_terms(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        key = clean_attribute_key(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
    return output


_ALIAS_COMPARISON = re.compile(r"(?:>=|<=|>|<|≥|≤)")
_ALIAS_THRESHOLD_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\s*)?(?:uln|n|degrees?|points?|%|percent|"
    r"mg|g|kg|ml|mmhg|days?|weeks?|months?|years?)\b",
    flags=re.IGNORECASE,
)


def _is_reusable_attribute_alias(value: str) -> bool:
    """Return whether text is terminology rather than a trial constraint."""

    text = str(value or "").strip()
    return bool(text) and not (
        _ALIAS_COMPARISON.search(text)
        or _ALIAS_THRESHOLD_UNIT.search(text)
    )


def normalize_search_expansion(
    expansion: EvaSearchExpansion,
) -> EvaSearchExpansion:
    """Apply the exact deterministic term cleanup used before reasoning."""

    expansion.lexical_terms = _unique_terms(expansion.lexical_terms)
    expansion.semantic_terms = _unique_terms(expansion.semantic_terms)
    return expansion


def load_breakdown_points(path: str | Path) -> list[dict[str, str]]:
    """Load the exact criteria/item/context breakdown contract."""

    input_path = Path(path)
    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        expected = ("criteria", "item", "context")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"{input_path}: expected columns {expected}, found "
                f"{tuple(reader.fieldnames or ())}."
            )
        rows = []
        for row_number, row in enumerate(reader, start=1):
            criteria = _text(row.get("criteria")).casefold()
            item = _text(row.get("item"))
            context = _text(row.get("context"))
            if criteria not in {"inclusion", "exclusion"}:
                raise ValueError(
                    f"{input_path}: row {row_number} has invalid criteria."
                )
            if not item or not context:
                raise ValueError(
                    f"{input_path}: row {row_number} is missing item/context."
                )
            rows.append(
                {
                    "criteria": criteria,
                    "item": item,
                    "context": context,
                }
            )
    if not rows:
        raise ValueError(f"{input_path}: no eligibility points found.")
    return rows


def build_search_prompt(
    *,
    trial_id: str,
    source_item_id: str,
    criteria: str,
    item: str,
    context: str,
) -> str:
    """Ask the inexpensive model only for reusable retrieval terms."""

    evidence = {
        "trial_id": trial_id,
        "source_item_id": source_item_id,
        "criteria": criteria,
        "item": item,
        "context": context,
    }
    return (
        "Generate a small terminology-search expansion for one atomic "
        "clinical-trial eligibility point. Return 2-6 lexical terms that "
        "closely preserve the wording, including corrected spellings and "
        "abbreviation expansions, and 2-6 semantic terms that name closely "
        "related reusable clinical concepts. For example, an "
        "immunoinsufficiency statement may produce immunodeficiency, "
        "immunocompromised status, HIV infection, and immune suppression. "
        "The terms will be used only to retrieve candidate Attributes from a "
        "controlled library. Do not select an Attribute, infer an eligibility "
        "value, invent a cutoff, or normalize the criterion. Keep entity "
        "hints broad and concise.\n\n"
        f"EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(clean_attribute_key(left).split())
    right_tokens = set(clean_attribute_key(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return 100.0 * len(left_tokens & right_tokens) / len(
        left_tokens | right_tokens
    )


def retrieve_attribute_candidates(
    *,
    library: EvaLibraryRepository,
    expansion: EvaSearchExpansion,
    source_item: str,
    top_k: int = 10,
    reconciliation_query: EvaReconciliationRetrievalQuery | None = None,
) -> list[EvaAttributeCandidate]:
    """Combine lexical, LLM-semantic, and fuzzy retrieval signals."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if reconciliation_query is not None:
        candidates: list[EvaAttributeCandidate] = []
        for attribute in library.attributes.values():
            if not attribute.active:
                continue
            entity = library.entities[attribute.entity_id]
            match = score_reconciliation_candidate(
                reconciliation_query,
                attribute=attribute,
                entity=entity,
            )
            if match is None:
                continue
            candidates.append(
                EvaAttributeCandidate(
                    attribute_id=attribute.attribute_id,
                    canonical_name=attribute.canonical_name,
                    entity_id=attribute.entity_id,
                    entity_name=entity.canonical_name,
                    description=attribute.description,
                    aliases=attribute.aliases,
                    value_type=attribute.value_type,
                    numerical_type=attribute.numerical_type,
                    canonical_unit=attribute.canonical_unit,
                    score=match.score,
                    matched_term=match.matched_term,
                    matched_text=match.matched_text,
                    retrieval_methods=list(match.methods),
                )
            )
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.canonical_name.casefold(),
                candidate.attribute_id,
            ),
        )[:top_k]
    terms = [
        ("source_lexical", source_item),
        *[
            ("llm_lexical", term)
            for term in expansion.lexical_terms
        ],
        *[
            ("llm_semantic", term)
            for term in expansion.semantic_terms
        ],
    ]
    best: dict[str, EvaAttributeCandidate] = {}
    method_sets: dict[str, set[str]] = {}

    for attribute in library.attributes.values():
        if not attribute.active:
            continue
        entity = library.entities[attribute.entity_id]
        searchable = _unique_terms(
            [
                attribute.canonical_name,
                *attribute.aliases,
                attribute.description,
                entity.canonical_name,
                *entity.aliases,
            ]
        )
        for term_kind, term in terms:
            term_key = clean_attribute_key(term)
            if not term_key:
                continue
            for candidate_text in searchable:
                candidate_key = clean_attribute_key(candidate_text)
                methods: list[str] = []
                scores = [
                    fuzz.WRatio(term_key, candidate_key),
                    fuzz.token_set_ratio(term_key, candidate_key),
                    fuzz.partial_ratio(term_key, candidate_key),
                    _token_overlap(term_key, candidate_key),
                ]
                if term_key == candidate_key:
                    scores.append(100.0)
                    methods.append(f"{term_kind}:exact")
                elif (
                    term_key in candidate_key
                    or candidate_key in term_key
                ):
                    scores.append(96.0)
                    methods.append(f"{term_kind}:contains")
                score = float(max(scores))
                methods.extend(
                    [
                        f"{term_kind}:fuzzy_wratio",
                        f"{term_kind}:token_set",
                        f"{term_kind}:lexical_overlap",
                    ]
                )
                current = best.get(attribute.attribute_id)
                if current is None or score > current.score:
                    best[attribute.attribute_id] = EvaAttributeCandidate(
                        attribute_id=attribute.attribute_id,
                        canonical_name=attribute.canonical_name,
                        entity_id=attribute.entity_id,
                        entity_name=entity.canonical_name,
                        description=attribute.description,
                        aliases=attribute.aliases,
                        value_type=attribute.value_type,
                        numerical_type=attribute.numerical_type,
                        canonical_unit=attribute.canonical_unit,
                        score=score,
                        matched_term=term,
                        matched_text=candidate_text,
                        retrieval_methods=[],
                    )
                method_sets.setdefault(
                    attribute.attribute_id,
                    set(),
                ).update(methods)

    candidates = sorted(
        best.values(),
        key=lambda candidate: (
            -candidate.score,
            candidate.canonical_name.casefold(),
            candidate.attribute_id,
        ),
    )[:top_k]
    for candidate in candidates:
        candidate.retrieval_methods = sorted(
            method_sets.get(candidate.attribute_id, set())
        )
    return candidates


def build_reasoning_prompt(
    *,
    trial_id: str,
    source_item_id: str,
    criteria: str,
    item: str,
    context: str,
    expansion: EvaSearchExpansion,
    candidates: list[EvaAttributeCandidate],
    entities: list[dict[str, Any]],
    prior_error: str = "",
) -> str:
    """Build the detailed EVA mapping and value-normalization prompt."""

    payload = {
        "trial_id": trial_id,
        "source_item_id": source_item_id,
        "criteria": criteria,
        "item": item,
        "context": context,
        "search_expansion": expansion.model_dump(mode="json"),
        "existing_entities": entities,
        "top_attribute_candidates": [
            candidate.model_dump(mode="json") for candidate in candidates
        ],
    }
    correction = (
        "\nA previous response failed deterministic validation. Correct it "
        f"fully:\n{prior_error}\n"
        if prior_error
        else ""
    )
    return (
        "Normalize one atomic eligibility point into exactly one reusable "
        "Entity-Value-Attribute (EVA) record. Choose an existing Attribute "
        "only from top_attribute_candidates, otherwise propose a new "
        "Attribute. A new Entity is allowed only when none of the supplied "
        "existing Entities is conceptually suitable.\n\n"
        "Attribute rules:\n"
        "- Attribute names must identify the precise reusable clinical "
        "concept being constrained. Avoid generic names such as status, "
        "condition, disease, history, or eligibility.\n"
        "- Use the reusable Attribute sex_gender for participant sex or "
        "gender eligibility. Its value_type is SexGender, and its values are "
        "male, female, or all. Do not create separate male_sex and female_sex "
        "Attributes.\n"
        "- Do not choose sex_gender merely because a disease or condition "
        "criterion mentions boys, girls, men, or women. When sex identifies "
        "the subgroup affected by another clinical condition, map the "
        "condition itself and preserve the sex qualifier in context.\n"
        "- Prefer concise snake_case names. A short phrase is acceptable when "
        "needed to avoid ambiguity.\n"
        "- Do not put operators, cutoff values, range endpoints, Included/"
        "Excluded, male, female, or all in an Attribute name.\n"
        "- `previous_stroke_times` is a good Numerical/Point Attribute for "
        "an exact count of prior strokes.\n"
        "- Existing Attributes retain exactly their supplied Entity, value "
        "type, numerical type, and canonical unit.\n\n"
        f"{CLINICAL_ATTRIBUTE_NAMING_POLICY}\n\n"
        "Value rules:\n"
        "- Only three value types exist: Categorical, SexGender, and "
        "Numerical.\n"
        "- Categorical has exactly two final eligible-population values: "
        "Included and Excluded. Infer the status of the canonical Attribute "
        "concept after combining criterion type, assertion polarity, "
        "negation, exceptions, and full context.\n"
        "- Use this truth table as a check: inclusion + concept present => "
        "Included; inclusion + concept absent => Excluded; exclusion + "
        "concept present => Excluded; exclusion + concept absent => "
        "Included.\n"
        "- Thus an inclusion requirement for non-pregnant participants maps "
        "to Pregnancy = Excluded. If non-pregnant participants themselves "
        "are excluded, Pregnancy = Included. An exclusion of pregnant "
        "participants maps to Pregnancy = Excluded.\n"
        "- SexGender applies only to participant sex/gender eligibility. "
        "Use sex_gender = male for men/male-only eligibility, sex_gender = "
        "female for women/female-only eligibility, and sex_gender = all when "
        "the criterion states all genders, both sexes, men and women, or no "
        "sex/gender restriction.\n"
        "- Numerical has exactly Range or Point. Range must use mathematical "
        "interval notation such as [18, 80], (30, +inf), or [0, 1). Preserve "
        "open/closed boundaries. Point contains only the exact scalar, such "
        "as 1 for exactly one previous stroke.\n"
        "- Numerical values must always describe the final eligible "
        "population after applying the criterion, not merely the range named "
        "in the source text. For inclusion criteria, usually keep the stated "
        "range. For exclusion criteria, take the complement of the excluded "
        "range and flip strict/inclusive boundaries correctly.\n"
        "- Boundary complement checks: exclusion score > 10 means eligible "
        "score <= 10, so output Range (-inf, 10]. Exclusion score >= 10 "
        "means eligible score < 10, so output Range (-inf, 10). Exclusion "
        "score < 10 means eligible score >= 10, so output Range [10, +inf). "
        "Exclusion score <= 10 means eligible score > 10, so output Range "
        "(10, +inf).\n"
        "- Do not convert a categorical presence concept into a number merely "
        "because the source contains incidental numbers.\n\n"
        "Few-shot numerical examples from approved NCT00657163 audit:\n"
        "- Input criteria=inclusion, item='aged from 18 to 85', context='Men "
        "and women aged from 18 to 85' => Attribute participant_age, "
        "Numerical/Range, numerical_value='[18, 85]', unit='years'. "
        "Reason: the stated inclusion interval is the eligible interval.\n"
        "- Input criteria=exclusion, item='Fugl Meyer Motor Scale > 55', "
        "context='Fugl Meyer Motor Scale > 55' => Attribute "
        "fugl_meyer_motor_scale_score, Numerical/Range, "
        "numerical_value='(-inf, 55]', unit='points'. Reason: scores greater "
        "than 55 are excluded, so the eligible complement is scores less "
        "than or equal to 55.\n\n"
        "Set confidence to the confidence in the complete mapping and value "
        "interpretation. Explain the eligibility logic in rationale. New "
        "terminology is only a proposal for human audit and will not be "
        "merged automatically."
        f"{correction}\n\n"
        f"SELECTION_INPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


_POINT_VALUE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_RANGE_VALUE = re.compile(
    r"^[\[(]\s*(?:[+-]?(?:\d+(?:\.\d+)?)|[-+]inf)\s*,\s*"
    r"(?:[+-]?(?:\d+(?:\.\d+)?)|[-+]inf)\s*[\])]$",
    flags=re.IGNORECASE,
)

_SEX_GENDER_VALUES_BY_TERM = {
    "men": "male",
    "man": "male",
    "male": "male",
    "males": "male",
    "women": "female",
    "woman": "female",
    "female": "female",
    "females": "female",
    "men and women": "all",
    "women and men": "all",
    "male and female": "all",
    "female and male": "all",
    "all genders": "all",
    "all gender": "all",
    "all sexes": "all",
    "both genders": "all",
    "both sexes": "all",
}


def _sex_gender_value(source_item: str) -> str:
    cleaned = re.sub(r"[^a-z]+", " ", source_item.casefold()).strip()
    return _SEX_GENDER_VALUES_BY_TERM.get(cleaned, "")


def validate_reasoning_selection(
    *,
    selection: EvaReasoningSelection,
    candidates: list[EvaAttributeCandidate],
    library: EvaLibraryRepository,
    source_item: str,
    allow_exact_name_repair: bool = False,
) -> EvaReasoningSelection:
    """Apply deterministic library and value-schema guardrails."""

    selection.rationale = validate_english_explanation(
        selection.rationale,
        field_name="Initial reasoner",
    )
    candidate_by_id = {
        candidate.attribute_id: candidate for candidate in candidates
    }
    if selection.mapping_decision == "existing_attribute":
        if selection.attribute_id not in candidate_by_id:
            exact_name_matches = [
                definition
                for definition in library.attributes.values()
                if definition.active
                and clean_attribute_key(definition.canonical_name)
                == clean_attribute_key(selection.attribute_name)
            ]
            if allow_exact_name_repair and len(exact_name_matches) == 1:
                selection.attribute_id = exact_name_matches[0].attribute_id
            else:
                raise ValueError(
                    "Existing Attribute selection must use one of the supplied "
                    "top candidate attribute_id values."
                )
        definition = library.attributes[selection.attribute_id or ""]
        entity = library.entities[definition.entity_id]
        selection.entity_id = entity.entity_id
        selection.entity_name = entity.canonical_name
        selection.new_entity = False
        selection.attribute_name = definition.canonical_name
        selection.attribute_description = definition.description
        selection.value_type = definition.value_type
        selection.numerical_type = definition.numerical_type
        selection.unit = definition.canonical_unit
    else:
        if not selection.attribute_name.strip():
            raise ValueError("A new Attribute requires a specific name.")
        selection.attribute_id = stable_slug(
            selection.attribute_id or selection.attribute_name,
            fallback=stable_id("attribute", source_item),
        )
        existing_entity = library.find_entity(
            entity_id=selection.entity_id or "",
            canonical_name=selection.entity_name,
        )
        if existing_entity is not None:
            selection.entity_id = existing_entity.entity_id
            selection.entity_name = existing_entity.canonical_name
            selection.new_entity = False
        else:
            if not selection.entity_name.strip():
                raise ValueError(
                    "A new Attribute requires an Entity name."
                )
            selection.entity_id = stable_slug(
                selection.entity_id or selection.entity_name,
                fallback="entity",
            )
            selection.new_entity = True

    selection.attribute_aliases = _unique_terms([
        alias
        for alias in [source_item, *selection.attribute_aliases]
        if _is_reusable_attribute_alias(alias)
    ])
    if selection.value_type == "Categorical":
        if selection.categorical_value not in {
            "Included",
            "Excluded",
        }:
            raise ValueError(
                "Categorical EVA requires Included or Excluded."
            )
        selection.numerical_type = None
        selection.numerical_value = None
        selection.unit = None
    elif selection.value_type == "SexGender":
        expected_value = _sex_gender_value(source_item)
        if selection.attribute_id != "sex_gender":
            raise ValueError(
                "Sex/gender eligibility must use the reusable sex_gender "
                "Attribute."
            )
        if selection.categorical_value not in {"male", "female", "all"}:
            raise ValueError(
                "SexGender EVA requires male, female, or all."
            )
        if expected_value and selection.categorical_value != expected_value:
            raise ValueError(
                f"Sex/gender item '{source_item}' requires "
                f"sex_gender={expected_value}."
            )
        selection.attribute_id = "sex_gender"
        selection.attribute_name = "sex_gender"
        selection.entity_id = "demographic"
        selection.entity_name = "Demographic"
        selection.new_entity = False
        selection.numerical_type = None
        selection.numerical_value = None
        selection.unit = None
    else:
        selection.categorical_value = None
        numerical_value = _text(selection.numerical_value)
        if selection.numerical_type == "Point":
            if not _POINT_VALUE.fullmatch(numerical_value):
                raise ValueError(
                    "Numerical Point must contain only an exact scalar."
                )
        elif selection.numerical_type == "Range":
            if not _RANGE_VALUE.fullmatch(numerical_value):
                raise ValueError(
                    "Numerical Range must use one interval such as [18, 80] "
                    "or (30, +inf)."
                )
        else:
            raise ValueError(
                "Numerical EVA requires Range or Point."
            )
        selection.numerical_value = numerical_value
    return selection


def build_eva_audit_item(
    *,
    trial_id: str,
    source_item_id: str,
    row: Mapping[str, str],
    expansion: EvaSearchExpansion,
    candidates: list[EvaAttributeCandidate],
    selection: EvaReasoningSelection,
    mapping_review: EvaMappingReviewTrace | None = None,
    attribute_id_review: EvaAttributeIdReviewTrace | None = None,
) -> EvaAuditItem:
    """Build the stable audit row shared by live and recovered calls."""

    return EvaAuditItem(
        eva_id=stable_id(
            "eva",
            trial_id,
            source_item_id,
            row["item"],
        ),
        source_item_id=source_item_id,
        criteria=row["criteria"],
        item=row["item"],
        context=row["context"],
        search_expansion=expansion,
        candidates=candidates,
        mapping_decision=selection.mapping_decision,
        attribute_id=selection.attribute_id or "",
        entity_id=selection.entity_id or "",
        entity_name=selection.entity_name,
        new_entity=selection.new_entity,
        attribute_name=selection.attribute_name,
        attribute_description=selection.attribute_description,
        attribute_aliases=selection.attribute_aliases,
        value_type=selection.value_type,
        categorical_value=selection.categorical_value,
        numerical_type=selection.numerical_type,
        numerical_value=selection.numerical_value,
        unit=selection.unit,
        confidence=selection.confidence,
        rationale=selection.rationale,
        **({"mapping_review": mapping_review} if mapping_review else {}),
        **(
            {"attribute_id_review": attribute_id_review}
            if attribute_id_review
            else {}
        ),
    )


def extract_one_eva(
    *,
    library: EvaLibraryRepository,
    search_provider: JsonLLMProvider,
    reasoner_provider: JsonLLMProvider,
    trial_key: str,
    trial_id: str,
    row_index: int,
    row: Mapping[str, str],
    top_k: int = 10,
    max_attempts: int = 3,
    review_mode: Literal["off", "full"] = "full",
) -> EvaAuditItem:
    """Run search expansion, candidate retrieval, and final reasoning."""

    source_item_id = f"{trial_key}:{row_index:04d}"
    expansion_result = search_provider.generate_json(
        build_search_prompt(
            trial_id=trial_id,
            source_item_id=source_item_id,
            criteria=row["criteria"],
            item=row["item"],
            context=row["context"],
        ),
        output_schema=SEARCH_OUTPUT_SCHEMA,
    )
    expansion = normalize_search_expansion(
        EvaSearchExpansion.model_validate(expansion_result)
    )
    candidates = retrieve_attribute_candidates(
        library=library,
        expansion=expansion,
        source_item=row["item"],
        top_k=top_k,
    )
    entities = [
        entity.model_dump(mode="json")
        for entity in library.entities.values()
        if entity.active
    ]

    prior_error = ""
    selection: EvaReasoningSelection | None = None
    for attempt in range(1, max_attempts + 1):
        result = reasoner_provider.generate_json(
            build_reasoning_prompt(
                trial_id=trial_id,
                source_item_id=source_item_id,
                criteria=row["criteria"],
                item=row["item"],
                context=row["context"],
                expansion=expansion,
                candidates=candidates,
                entities=entities,
                prior_error=prior_error,
            ),
            output_schema=REASONING_OUTPUT_SCHEMA,
        )
        try:
            selection = validate_reasoning_selection(
                selection=EvaReasoningSelection.model_validate(result),
                candidates=candidates,
                library=library,
                source_item=row["item"],
                allow_exact_name_repair=True,
            )
        except (TypeError, ValueError) as exc:
            prior_error = str(exc)
            if attempt == max_attempts:
                raise
            continue
        break
    if selection is None:
        raise RuntimeError(f"{source_item_id}: no valid EVA result.")

    if review_mode == "full":
        selection, expansion, candidates, mapping_review = review_eva_mapping(
            library=library,
            reasoner_provider=reasoner_provider,
            trial_id=trial_id,
            source_item_id=source_item_id,
            row=row,
            expansion=expansion,
            candidates=candidates,
            selection=selection,
            top_k=top_k,
        )
        if (
            mapping_review.status == "completed"
            and selection.mapping_decision == "new_attribute"
        ):
            selection, attribute_id_review = review_new_attribute_id(
                reasoner_provider=reasoner_provider,
                source_item_id=source_item_id,
                atomic_criterion_title=row["item"],
                selection=selection,
                library=library,
            )
        elif selection.mapping_decision == "existing_attribute":
            _selection, attribute_id_review = review_new_attribute_id(
                reasoner_provider=reasoner_provider,
                source_item_id=source_item_id,
                atomic_criterion_title=row["item"],
                selection=selection,
                library=library,
            )
        else:
            attribute_id_review = EvaAttributeIdReviewTrace(
                status="not_run",
                current_attribute_id=selection.attribute_id or "",
                reviewed_attribute_id=selection.attribute_id or "",
                explanation=(
                    "Attribute ID Review was not run because Mapping Review "
                    "did not produce a completed new-Attribute decision."
                ),
                explanation_language="en",
            )
    else:
        mapping_review = EvaMappingReviewTrace(
            status="not_run",
            initial_mapping_decision=selection.mapping_decision,
            initial_attribute_id=selection.attribute_id or "",
            final_mapping_decision=selection.mapping_decision,
            final_attribute_id=selection.attribute_id or "",
        )
        attribute_id_review = EvaAttributeIdReviewTrace(
            status=(
                "not_required"
                if selection.mapping_decision == "existing_attribute"
                else "not_run"
            ),
            current_attribute_id=selection.attribute_id or "",
            reviewed_attribute_id=selection.attribute_id or "",
            explanation=(
                ID_REVIEW_NOT_REQUIRED_EXPLANATION
                if selection.mapping_decision == "existing_attribute"
                else "Attribute ID Review was not run because review mode is off."
            ),
            explanation_language="en",
        )

    return build_eva_audit_item(
        trial_id=trial_id,
        source_item_id=source_item_id,
        row=row,
        expansion=expansion,
        candidates=candidates,
        selection=selection,
        mapping_review=mapping_review,
        attribute_id_review=attribute_id_review,
    )


def _audit_definitions_equivalent(left: EvaAuditItem, right: EvaAuditItem) -> bool:
    return (
        left.entity_id == right.entity_id
        and clean_attribute_key(left.attribute_name)
        == clean_attribute_key(right.attribute_name)
        and left.value_type == right.value_type
        and left.numerical_type == right.numerical_type
        and left.unit == right.unit
    )


def finalize_trial_attribute_id_conflicts(
    items: list[EvaAuditItem],
    library: EvaLibraryRepository,
) -> list[EvaAuditItem]:
    """Block non-equivalent new definitions sharing an Attribute ID."""

    by_id: dict[str, list[EvaAuditItem]] = {}
    for item in items:
        if item.mapping_decision == "new_attribute":
            by_id.setdefault(item.attribute_id, []).append(item)
    for attribute_id, grouped in by_id.items():
        conflict_message = ""
        definition = library.attributes.get(attribute_id)
        if definition is not None and definition.active:
            exemplar = grouped[0]
            equivalent = (
                definition.entity_id == exemplar.entity_id
                and clean_attribute_key(definition.canonical_name)
                == clean_attribute_key(exemplar.attribute_name)
                and definition.value_type == exemplar.value_type
                and definition.numerical_type == exemplar.numerical_type
                and definition.canonical_unit == exemplar.unit
            )
            if not equivalent:
                conflict_message = (
                    f"attribute_id {attribute_id!r} conflicts with an active "
                    "non-equivalent library definition."
                )
        if not conflict_message and any(
            not _audit_definitions_equivalent(grouped[0], other)
            for other in grouped[1:]
        ):
            conflict_message = (
                f"attribute_id {attribute_id!r} has non-equivalent trial definitions."
            )
        if conflict_message:
            for item in grouped:
                item.attribute_id_review.status = "blocked"
                item.attribute_id_review.error = {
                    "type": "AttributeIdConflict",
                    "message": conflict_message,
                }
    return items


def extract_trial_eva(
    *,
    trial_key: str,
    trial_id: str,
    breakdown_path: str | Path,
    library: EvaLibraryRepository,
    search_provider: JsonLLMProvider,
    reasoner_provider: JsonLLMProvider,
    search_model: str,
    search_reasoning_effort: str,
    reasoner_model: str,
    reasoner_reasoning_effort: str,
    top_k: int = 10,
    workers: int = 4,
    max_attempts: int = 3,
    progress_callback: Callable[[int, int], None] | None = None,
    review_mode: Literal["off", "full"] = "full",
) -> EvaAuditDocument:
    """Extract all points for one manually selected trial."""

    rows = load_breakdown_points(breakdown_path)
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    completed = 0
    items: list[tuple[int, EvaAuditItem]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_by_index = {
            executor.submit(
                extract_one_eva,
                library=library,
                search_provider=search_provider,
                reasoner_provider=reasoner_provider,
                trial_key=trial_key,
                trial_id=trial_id,
                row_index=index,
                row=row,
                top_k=top_k,
                max_attempts=max_attempts,
                review_mode=review_mode,
            ): index
            for index, row in enumerate(rows, start=1)
        }
        for future in as_completed(future_by_index):
            index = future_by_index[future]
            items.append((index, future.result()))
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(rows))

    items.sort(key=lambda pair: pair[0])
    finalized_items = finalize_trial_attribute_id_conflicts(
        [item for _index, item in items],
        library,
    )
    now = _utc_now()
    source_path = Path(breakdown_path).resolve()
    return EvaAuditDocument(
        audit_id=stable_id(
            "eva_audit",
            trial_key,
            library.revision,
            library.sha256,
        ),
        trial_key=trial_key,
        trial_id=trial_id,
        source_path=str(source_path),
        library_revision=library.revision,
        library_sha256=library.sha256,
        created_at=now,
        updated_at=now,
        search_model=search_model,
        search_reasoning_effort=search_reasoning_effort,
        reasoner_model=reasoner_model,
        reasoner_reasoning_effort=reasoner_reasoning_effort,
        review_mode=review_mode,
        review_prompt_versions={
            "reasoner": REASONER_PROMPT_VERSION,
            "mapping_review": MAPPING_REVIEW_PROMPT_VERSION,
            "mapping_correction": MAPPING_CORRECTION_PROMPT_VERSION,
            "attribute_id_review": ATTRIBUTE_ID_REVIEW_PROMPT_VERSION,
        },
        items=finalized_items,
    )
