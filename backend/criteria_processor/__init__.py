"""Lazy public API for eligibility-criteria normalization.

Submodules remain independently importable: using the text or segmentation
helpers does not eagerly import pandas, rapidfuzz, or the workbook repository.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CriteriaNormalizer",
    "CodexJsonLLMProvider",
    "EligibilityRuleExtractor",
    "EligibilitySourceItem",
    "ExtractedCriterion",
    "MappingEvidence",
    "MappingExpansion",
    "MappingProposal",
    "MappingSelection",
    "NormalizedCriterion",
    "TerminologyAttribute",
    "TerminologyAuditRecord",
    "TerminologyCandidate",
    "TerminologyMapping",
    "TerminologyRelationship",
    "TerminologyRepository",
    "TerminologySuggestion",
    "TerminologyValue",
    "MappingSuggestionEngine",
    "audit_records_from_proposal",
    "create_verified_library",
    "create_working_library",
    "iter_eligibility_source_items",
    "normalize_with_mapping_proposals",
    "normalize_structured_value",
    "split_eligibility_criteria",
]

_EXPORT_MODULES = {
    "CodexJsonLLMProvider": "backend.criteria_processor.llm_providers",
    "CriteriaNormalizer": "backend.criteria_processor.normalizer",
    "EligibilityRuleExtractor": "backend.criteria_processor.rule_extractor",
    "EligibilitySourceItem": "backend.criteria_processor.models",
    "ExtractedCriterion": "backend.criteria_processor.models",
    "MappingEvidence": "backend.criteria_processor.models",
    "MappingExpansion": "backend.criteria_processor.models",
    "MappingProposal": "backend.criteria_processor.models",
    "MappingSelection": "backend.criteria_processor.models",
    "NormalizedCriterion": "backend.criteria_processor.models",
    "TerminologyAttribute": "backend.criteria_processor.models",
    "TerminologyAuditRecord": "backend.criteria_processor.models",
    "TerminologyCandidate": "backend.criteria_processor.models",
    "TerminologyMapping": "backend.criteria_processor.models",
    "TerminologyRelationship": "backend.criteria_processor.models",
    "TerminologyRepository": "backend.criteria_processor.terminology",
    "TerminologySuggestion": "backend.criteria_processor.models",
    "TerminologyValue": "backend.criteria_processor.models",
    "MappingSuggestionEngine": "backend.criteria_processor.mapping_suggestions",
    "audit_records_from_proposal": "backend.criteria_processor.mapping_suggestions",
    "create_verified_library": "backend.criteria_processor.terminology_updates",
    "create_working_library": "backend.criteria_processor.terminology_updates",
    "iter_eligibility_source_items": "backend.criteria_processor.segmentation",
    "normalize_with_mapping_proposals": "backend.criteria_processor.unified_normalization",
    "normalize_structured_value": "backend.criteria_processor.value_normalization",
    "split_eligibility_criteria": "backend.criteria_processor.segmentation",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
