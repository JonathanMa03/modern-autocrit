"""
Text utilities for eligibility criteria extraction.
"""

from __future__ import annotations

import re


def normalize_whitespace(text: str | None) -> str:
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def clean_text(text: str | None) -> str:
    text = normalize_whitespace(text)

    text = text.replace("\u00a0", " ")
    text = text.replace("≥", ">=")
    text = text.replace("≤", "<=")
    text = text.replace(">/=", ">=")
    text = text.replace("</=", "<=")

    return normalize_whitespace(text)


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s.strip()]


def chunk_text_by_words(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word chunks.

    This is intentionally simple for now. Later we can replace it with
    sentence-boundary-preserving chunking.
    """

    text = clean_text(text)
    words = text.split()

    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def clean_attribute_key(text: str | None) -> str:
    """
    Normalize extracted attributes for dictionary lookup.
    """

    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s\-/()]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text