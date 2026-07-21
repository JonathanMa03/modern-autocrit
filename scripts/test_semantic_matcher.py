from pprint import pprint

from backend.normalization.semantic_matcher import (
    SemanticTerminologyMatcher,
)


TEST_TERMS = {
    "attribute": [
        "ALT",
        "alanine liver enzyme",
        "aspartate liver enzyme",
        "patient performance score",
        "heart pumping efficiency",
        "blood platelet level",
        "kidney creatinine measurement",
        "completely unrelated concept",
    ],
    "disease": [
        "NSCLC",
        "non small cell lung carcinoma",
        "chronic lymphocytic leukemia",
        "CLL",
        "hepatitis b virus infection",
        "liver cell carcinoma",
        "small intestinal cancer",
        "completely unrelated disease",
    ],
    "entity": [
        "lab test",
        "laboratory measurement",
        "prior cancer therapy",
        "previous treatment history",
        "prescription medicine",
        "surgical intervention",
        "vital signs measurement",
        "completely unrelated entity",
    ],
}


def test_matcher(
    terminology_type: str,
    test_terms: list[str],
) -> None:
    print("\n" + "#" * 80)
    print(f"TERMINOLOGY TYPE: {terminology_type.upper()}")
    print("#" * 80)

    matcher = SemanticTerminologyMatcher(
        terminology_type=terminology_type,
        accept_threshold=0.88,
        review_threshold=0.70,
    )

    for term in test_terms:
        result = matcher.decide(
            raw_term=term,
            top_k=3,
        )

        print("\n" + "=" * 70)
        print(f"Raw term: {term}")

        pprint(
            result.to_dict(),
            sort_dicts=False,
        )


def main() -> None:
    for terminology_type, test_terms in (
        TEST_TERMS.items()
    ):
        test_matcher(
            terminology_type=terminology_type,
            test_terms=test_terms,
        )


if __name__ == "__main__":
    main()