"""Reusable helpers for deterministic eligibility rule extractors."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from backend.criteria_processor.inputs import EligibilityCandidateInput
from backend.criteria_processor.models import EligibilitySourceItem, NormalizedCriterion
from backend.criteria_processor.terminology import TerminologyRepository
from backend.criteria_processor.value_normalization import normalize_structured_value


def stable_criterion_id(*parts: object) -> str:
    digest = hashlib.sha1(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:16]
    return f"criterion_{digest}"


class EligibilityRuleExtractor:
    """Base class for provenance-aware deterministic rule extractors."""

    def __init__(
        self,
        terminology_repository: TerminologyRepository,
    ) -> None:
        self.terminology = terminology_repository

    def _criterion(
        self,
        *,
        candidate: EligibilityCandidateInput,
        criterion_type: str,
        source_text: str,
        source_path: str,
        source_index: int | None,
        raw_attribute: str,
        raw_value: str,
        extraction_method: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        lower_inclusive: bool | None = None,
        upper_inclusive: bool | None = None,
        unit: str | None = None,
        normalized_lower_bound: float | None = None,
        normalized_upper_bound: float | None = None,
        normalized_unit: str | None = None,
        confidence: float = 1.0,
        needs_review: bool = False,
        review_reason: str = "",
    ) -> NormalizedCriterion:
        mapping = self.terminology.map_criterion(raw_attribute, raw_value)
        structured = normalize_structured_value(
            raw_value=raw_value,
            attribute_id=mapping.attribute_id,
            canonical_attribute=mapping.canonical_attribute,
            value_schema=mapping.value_schema,
            canonical_unit=mapping.canonical_unit,
            canonical_value=mapping.canonical_value,
            context_text=source_text,
            criterion_type=criterion_type,
        )
        return NormalizedCriterion(
            criterion_id=stable_criterion_id(
                candidate.nct_id,
                criterion_type,
                source_path,
                source_index,
                raw_attribute,
                raw_value,
                extraction_method,
            ),
            nct_id=candidate.nct_id,
            criterion_type=criterion_type,
            source_text=source_text,
            source_path=source_path,
            source_index=source_index,
            raw_attribute=raw_attribute,
            raw_value=raw_value,
            attribute_id=mapping.attribute_id,
            canonical_attribute=mapping.canonical_attribute,
            value_schema=mapping.value_schema,
            canonical_value_id=mapping.value_id,
            canonical_value=structured["category"]
            or mapping.canonical_value,
            value_category=structured["category"],
            lower_bound=(
                lower_bound
                if lower_bound is not None
                else structured["lower"]
            ),
            upper_bound=(
                upper_bound
                if upper_bound is not None
                else structured["upper"]
            ),
            lower_inclusive=(
                lower_inclusive
                if lower_inclusive is not None
                else structured["lower_inclusive"]
            ),
            upper_inclusive=(
                upper_inclusive
                if upper_inclusive is not None
                else structured["upper_inclusive"]
            ),
            unit=unit or structured["unit"],
            lower_unit=structured["lower_unit"],
            upper_unit=structured["upper_unit"],
            normalized_lower_bound=(
                normalized_lower_bound
                if normalized_lower_bound is not None
                else structured["normalized_lower"]
            ),
            normalized_upper_bound=(
                normalized_upper_bound
                if normalized_upper_bound is not None
                else structured["normalized_upper"]
            ),
            normalized_unit=normalized_unit or structured["normalized_unit"],
            selection_effect=structured["selection_effect"],
            assertion=structured["assertion"],
            eligible_values=structured["eligible_values"],
            excluded_values=structured["excluded_values"],
            eligible_ranges=structured["eligible_ranges"],
            excluded_ranges=structured["excluded_ranges"],
            value_parse_status=structured["parse_status"],
            mapping_status=mapping.mapping_status,
            mapping_method=mapping.mapping_method,
            terminology_version=mapping.terminology_version,
            extraction_method=extraction_method,
            confidence=confidence,
            needs_review=needs_review,
            review_reason=review_reason,
        )

    def _unmapped_source_item(
        self,
        *,
        candidate: EligibilityCandidateInput,
        source_item: EligibilitySourceItem,
        review_reason: str,
    ) -> NormalizedCriterion:
        return NormalizedCriterion(
            criterion_id=stable_criterion_id(
                candidate.nct_id,
                source_item.criterion_type,
                source_item.source_path,
                source_item.source_text,
                "unmapped",
            ),
            nct_id=candidate.nct_id,
            criterion_type=source_item.criterion_type,
            source_text=source_item.source_text,
            source_path=source_item.source_path,
            source_index=source_item.source_index,
            raw_attribute="Unmapped Eligibility Criterion",
            raw_value=source_item.source_text,
            extraction_method="unmapped",
            mapping_status="unmapped",
            mapping_method="unmapped",
            terminology_version=self.terminology.terminology_version,
            confidence=0.0,
            needs_review=True,
            review_reason=review_reason,
        )

    @staticmethod
    def deduplicate(
        rows: Iterable[NormalizedCriterion],
    ) -> list[NormalizedCriterion]:
        deduped: list[NormalizedCriterion] = []
        seen: set[
            tuple[str, str, str | None, str | None, float | None, float | None]
        ] = set()
        for row in rows:
            key = (
                row.criterion_type,
                row.source_text.casefold(),
                row.attribute_id,
                row.canonical_value,
                row.normalized_lower_bound,
                row.normalized_upper_bound,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped
