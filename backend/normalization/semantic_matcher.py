from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)

from backend.normalization.terminology_repository import (
    TerminologyType,
)

from backend.normalization.terminology_service import (
    TerminologyService,
)


@dataclass
class SemanticMatch:
    raw_term: str
    canonical_term: str
    matched_text: str
    matched_source: str
    similarity: float
    rank: int

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticDecision:
    raw_term: str
    terminology_type: str
    decision: str
    suggested_term: str | None
    similarity: float | None
    matches: list[SemanticMatch]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "raw_term": self.raw_term,
            "terminology_type": (
                self.terminology_type
            ),
            "decision": self.decision,
            "suggested_term": (
                self.suggested_term
            ),
            "similarity": self.similarity,
            "matches": [
                match.to_dict()
                for match in self.matches
            ],
        }


class SemanticTerminologyMatcher:
    """
    Performs exact lookup first, followed by
    embedding-based semantic matching.
    """

    def __init__(
        self,
        terminology_type: TerminologyType,
        dictionary_path: str | Path | None = None,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        accept_threshold: float = 0.88,
        review_threshold: float = 0.70,
    ) -> None:
        if not 0 <= review_threshold <= 1:
            raise ValueError(
                "review_threshold must be between "
                "0 and 1."
            )

        if not 0 <= accept_threshold <= 1:
            raise ValueError(
                "accept_threshold must be between "
                "0 and 1."
            )

        if review_threshold > accept_threshold:
            raise ValueError(
                "review_threshold cannot exceed "
                "accept_threshold."
            )

        self.terminology_type = (
            terminology_type
        )

        self.terminology_service = (
            TerminologyService(
                terminology_type=terminology_type,
                dictionary_path=dictionary_path,
            )
        )

        self.model_name = model_name
        self.accept_threshold = (
            accept_threshold
        )
        self.review_threshold = (
            review_threshold
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        self.entries = (
            self.terminology_service.search_entries()
        )

        if not self.entries:
            raise ValueError(
                "The terminology dictionary contains "
                "no searchable entries."
            )

        corpus = [
            entry["text"]
            for entry in self.entries
        ]

        self.corpus_embeddings = (
            self.model.encode(
                corpus,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def suggest(
        self,
        raw_term: str,
        top_k: int = 3,
    ) -> list[SemanticMatch]:
        cleaned_term = str(raw_term).strip()

        if not cleaned_term:
            return []

        exact_match = (
            self.terminology_service.exact_lookup(
                cleaned_term
            )
        )

        if exact_match is not None:
            return [
                SemanticMatch(
                    raw_term=cleaned_term,
                    canonical_term=exact_match,
                    matched_text=cleaned_term,
                    matched_source="exact",
                    similarity=1.0,
                    rank=1,
                )
            ]

        query_embedding = self.model.encode(
            cleaned_term,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = np.dot(
            self.corpus_embeddings,
            query_embedding,
        )

        requested_count = min(
            max(top_k, 1),
            len(self.terminology_service.canonical_terms()),
        )

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        matches: list[SemanticMatch] = []
        seen_canonical_terms: set[str] = set()

        for raw_index in ranked_indices:
            index = int(raw_index)
            entry = self.entries[index]
            canonical = entry[
                "canonical_term"
            ]

            if canonical in seen_canonical_terms:
                continue

            seen_canonical_terms.add(
                canonical
            )

            matches.append(
                SemanticMatch(
                    raw_term=cleaned_term,
                    canonical_term=canonical,
                    matched_text=entry["text"],
                    matched_source=entry["source"],
                    similarity=float(
                        similarities[index]
                    ),
                    rank=len(matches) + 1,
                )
            )

            if (
                len(matches)
                >= requested_count
            ):
                break

        return matches

    def decide(
        self,
        raw_term: str,
        top_k: int = 3,
    ) -> SemanticDecision:
        matches = self.suggest(
            raw_term=raw_term,
            top_k=top_k,
        )

        if not matches:
            return SemanticDecision(
                raw_term=raw_term,
                terminology_type=(
                    self.terminology_type
                ),
                decision="reject",
                suggested_term=None,
                similarity=None,
                matches=[],
            )

        best_match = matches[0]
        score = best_match.similarity

        if score >= self.accept_threshold:
            decision = "accept"
        elif score >= self.review_threshold:
            decision = "review"
        else:
            decision = "reject"

        return SemanticDecision(
            raw_term=raw_term,
            terminology_type=(
                self.terminology_type
            ),
            decision=decision,
            suggested_term=(
                best_match.canonical_term
                if decision != "reject"
                else None
            ),
            similarity=score,
            matches=matches,
        )
    
    def rebuild_index(
        self,
    ) -> None:
        """
        Reloads the terminology dictionary and rebuilds
        the embedding index.
        """

        self.terminology_service.reload()

        self.entries = (
            self.terminology_service.search_entries()
        )

        if not self.entries:
            raise ValueError(
                "The terminology dictionary contains "
                "no searchable entries."
            )

        corpus = [
            entry["text"]
            for entry in self.entries
        ]

        self.corpus_embeddings = (
            self.model.encode(
                corpus,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )