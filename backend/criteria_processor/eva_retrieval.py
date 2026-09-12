"""Pure scoring primitives for semantic EVA reconciliation retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from rapidfuzz import fuzz

from backend.criteria_processor.eva_models import (
    EvaAttributeDefinition,
    EvaEntityDefinition,
    EvaNumericalType,
    EvaValueType,
)
from backend.criteria_processor.text import clean_attribute_key


SEMANTIC_RECONCILIATION_RETRIEVAL_VERSION = (
    "eligibility-eva-semantic-retrieval-v2"
)

EvaRetrievalProvenance = Literal[
    "edited_name",
    "edited_alias",
    "edited_id",
    "edited_description",
    "supplemental_lexical",
    "supplemental_semantic",
]


@dataclass(frozen=True)
class EvaRetrievalTerm:
    text: str
    provenance: EvaRetrievalProvenance
    weight: float

    def __post_init__(self) -> None:
        text = self.text.strip()
        if not text:
            raise ValueError("Retrieval term text must be non-empty.")
        if not 0 < self.weight <= 1:
            raise ValueError("Retrieval term weight must be in (0, 1].")
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class EvaReconciliationRetrievalQuery:
    terms: tuple[EvaRetrievalTerm, ...]
    canonical_entity_id: str | None
    value_type: EvaValueType
    numerical_type: EvaNumericalType | None
    canonical_unit: str | None


@dataclass(frozen=True)
class EvaRetrievalMatch:
    score: float
    matched_term: str
    matched_text: str
    methods: tuple[str, ...]


class EvaSchemaCandidate(Protocol):
    value_type: EvaValueType
    numerical_type: EvaNumericalType | None
    canonical_unit: str | None


def eva_schema_conflicts(
    *,
    value_type: EvaValueType,
    numerical_type: EvaNumericalType | None,
    canonical_unit: str | None,
    candidate: EvaSchemaCandidate,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if candidate.value_type != value_type:
        conflicts.append("value_type")
    if (
        candidate.value_type == "Numerical"
        and candidate.numerical_type != numerical_type
    ):
        conflicts.append("numerical_type")
    if (
        candidate.value_type == "Numerical"
        and clean_attribute_key(candidate.canonical_unit or "")
        != clean_attribute_key(canonical_unit or "")
    ):
        conflicts.append("canonical_unit")
    return tuple(conflicts)


def _text_match(term: str, candidate: str) -> tuple[float, str] | None:
    term_key = clean_attribute_key(term)
    candidate_key = clean_attribute_key(candidate)
    if not term_key or not candidate_key:
        return None
    if term_key == candidate_key:
        return 100.0, "exact"

    term_tokens = tuple(term_key.split())
    candidate_tokens = tuple(candidate_key.split())
    term_set = set(term_tokens)
    candidate_set = set(candidate_tokens)
    if term_set and candidate_set and (
        term_set <= candidate_set or candidate_set <= term_set
    ):
        return 94.0, "token_contains"

    shared_tokens = term_set & candidate_set
    if not shared_tokens:
        if (
            len(term_tokens) == 1
            and len(candidate_tokens) == 1
            and min(len(term_key), len(candidate_key)) >= 5
        ):
            typo_score = min(float(fuzz.ratio(term_key, candidate_key)), 88.0)
            if typo_score >= 85.0:
                return typo_score, "single_token_typo"
        return None

    overlap = 100.0 * len(shared_tokens) / len(term_set | candidate_set)
    token_set = min(float(fuzz.token_set_ratio(term_key, candidate_key)), 92.0)
    ratio = min(float(fuzz.ratio(term_key, candidate_key)), 88.0)
    score = max(overlap, token_set, ratio)
    if score < 35.0:
        return None
    method = "token_overlap" if score == overlap else (
        "token_set" if score == token_set else "fuzzy_ratio"
    )
    return score, method


def score_reconciliation_candidate(
    query: EvaReconciliationRetrievalQuery,
    *,
    attribute: EvaAttributeDefinition,
    entity: EvaEntityDefinition,
) -> EvaRetrievalMatch | None:
    """Score one active Attribute using bounded, provenance-aware evidence."""

    candidate_fields = [
        ("canonical_name", attribute.canonical_name, 1.00),
        *[("alias", alias, 1.00) for alias in attribute.aliases],
        ("description", attribute.description, 0.75),
    ]
    provenance_priority = {
        "edited_name": 0,
        "edited_alias": 1,
        "edited_id": 2,
        "edited_description": 3,
        "supplemental_lexical": 4,
        "supplemental_semantic": 5,
    }
    field_priority = {"canonical_name": 0, "alias": 1, "description": 2}
    matches: list[tuple[float, tuple[object, ...], EvaRetrievalTerm, str, str, str]] = []
    for term in query.terms:
        for field_kind, candidate_text, field_weight in candidate_fields:
            match = _text_match(term.text, candidate_text)
            if match is None:
                continue
            text_score, match_method = match
            contribution = text_score * term.weight * field_weight
            tie_break = (
                provenance_priority[term.provenance],
                field_priority[field_kind],
                clean_attribute_key(term.text),
                clean_attribute_key(candidate_text),
            )
            matches.append(
                (contribution, tie_break, term, field_kind, candidate_text, match_method)
            )
    if not matches:
        return None
    contribution, _, term, field_kind, matched_text, match_method = min(
        matches, key=lambda row: (-row[0], row[1])
    )
    text_component = 0.80 * contribution
    entity_bonus = 12.0 if query.canonical_entity_id == attribute.entity_id else 0.0
    conflicts = eva_schema_conflicts(
        value_type=query.value_type,
        numerical_type=query.numerical_type,
        canonical_unit=query.canonical_unit,
        candidate=attribute,
    )
    schema_bonus = 8.0 if not conflicts else 0.0
    methods = [f"{term.provenance}:{match_method}_{field_kind}"]
    if entity_bonus:
        methods.append("entity:compatible")
    if schema_bonus:
        methods.append("schema:compatible")
    return EvaRetrievalMatch(
        score=min(100.0, text_component + entity_bonus + schema_bonus),
        matched_term=term.text,
        matched_text=matched_text,
        methods=tuple(methods),
    )
