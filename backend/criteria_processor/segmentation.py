"""Split registry eligibility prose into provenance-preserving source items."""

from __future__ import annotations

import re

from backend.criteria_processor.models import EligibilitySourceItem


def _clean_criterion(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(
        r"^(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)",
        "",
        cleaned,
    )
    return cleaned.strip()


def split_eligibility_criteria(
    eligibility_text: str,
) -> tuple[list[str], list[str], list[str]]:
    """Split registry text into inclusion, exclusion, and unheaded items."""

    inclusion: list[str] = []
    exclusion: list[str] = []
    other: list[str] = []
    current = other
    heading_pattern = re.compile(
        r"^\s*(?:\*{0,2})?"
        r"(inclusion|exclusion)(?:\s+criteria)?"
        r"\s*:?\s*(?:\*{0,2})?\s*$",
        re.IGNORECASE,
    )
    narrative_inclusion_heading = re.compile(
        r"\bmust meet all\b.*\bcriteria\b.*\bparticipate\b",
        re.IGNORECASE,
    )
    narrative_exclusion_heading = re.compile(
        r"\bmeets any\b.*\bcriteria\b.*\bexcluded\b",
        re.IGNORECASE,
    )
    item_pattern = re.compile(
        r"^\s*(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)"
    )

    for raw_line in (eligibility_text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if narrative_inclusion_heading.search(line):
            current = inclusion
            continue
        if narrative_exclusion_heading.search(line):
            current = exclusion
            continue
        heading = heading_pattern.match(line)
        if heading:
            current = (
                inclusion
                if heading.group(1).casefold() == "inclusion"
                else exclusion
            )
            continue

        cleaned = _clean_criterion(line)
        if not cleaned:
            continue
        if item_pattern.match(line) or not current:
            current.append(cleaned)
        elif current[-1].endswith((".", ";", ":")):
            current.append(cleaned)
        else:
            current[-1] = f"{current[-1]} {cleaned}"

    return inclusion, exclusion, other


def iter_eligibility_source_items(
    eligibility_text: str,
    *,
    base_path: str = "protocolSection.eligibilityModule.eligibilityCriteria",
) -> list[EligibilitySourceItem]:
    """Return headed inclusion/exclusion eligibility items with source paths."""

    inclusion, exclusion, other = split_eligibility_criteria(
        eligibility_text
    )
    rows: list[EligibilitySourceItem] = []
    for criterion_type, section_name, section_items in [
        ("Inclusion", "inclusion", inclusion),
        ("Exclusion", "exclusion", exclusion),
    ]:
        for index, source_text in enumerate(section_items, start=1):
            rows.append(
                EligibilitySourceItem(
                    criterion_type=criterion_type,
                    source_text=source_text,
                    source_path=f"{base_path}.{section_name}[{index}]",
                    source_index=index,
                )
            )
    return rows


def eligibility_source_items_from_sections(
    *,
    inclusion: list[str],
    exclusion: list[str],
    base_path: str = "protocolSection.eligibilityModule.eligibilityCriteria",
) -> list[EligibilitySourceItem]:
    """Build source items from already separated inclusion/exclusion text."""

    rows: list[EligibilitySourceItem] = []
    for criterion_type, section_name, section_items in [
        ("Inclusion", "inclusion", inclusion),
        ("Exclusion", "exclusion", exclusion),
    ]:
        for index, source_text in enumerate(section_items, start=1):
            cleaned = source_text.strip()
            if not cleaned:
                continue
            rows.append(
                EligibilitySourceItem(
                    criterion_type=criterion_type,
                    source_text=cleaned,
                    source_path=f"{base_path}.{section_name}[{index}]",
                    source_index=index,
                )
            )
    return rows
