"""Normalization-specific text canonicalization."""

from __future__ import annotations

import re


def normalize_whitespace(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_normalization_text(text: str | None) -> str:
    """Repair common source encodings and collapse insignificant whitespace."""

    cleaned = normalize_whitespace(text)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("â‰¥", ">=")
    cleaned = cleaned.replace("â‰¤", "<=")
    cleaned = cleaned.replace(">/=", ">=")
    cleaned = cleaned.replace("</=", "<=")
    return normalize_whitespace(cleaned)


def clean_attribute_key(text: str | None) -> str:
    """Build a deterministic, case-insensitive terminology lookup key."""

    cleaned = clean_normalization_text(text).casefold()
    cleaned = re.sub(r"[^a-z0-9\s\-/()]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
