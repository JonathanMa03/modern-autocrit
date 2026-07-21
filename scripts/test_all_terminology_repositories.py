from backend.normalization.terminology_repository import (
    TerminologyRepository,
)


TEST_CASES = {
    "attribute": {
        "ALT": "Alanine Aminotransferase (ALT)",
        "platelets": "Platelet Count",
        "ECOG": "ECOG Performance Status",
        "serum pregnancy test": "Pregnancy Test",
        "unknown attribute": None,
    },
    "disease": {
        "CLL": "Chronic Lymphocytic Leukemia",
        "HBV": "Hepatitis B Infection",
        "non-small-cell lung cancer": (
            "Non-Small Cell Lung Cancer"
        ),
        "human immunodeficiency virus": "HIV Infection",
        "unknown disease": None,
    },
    "entity": {
        "laboratory test": "Lab Test",
        "prior therapy": "Treatment History",
        "medicine": "Medication",
        "vitals": "Vital",
        "unknown entity": None,
    },
}


def test_repository(
    terminology_type: str,
    cases: dict[str, str | None],
) -> None:
    repository = TerminologyRepository(
        terminology_type=terminology_type,
    )

    print(f"\nTesting {terminology_type!r}")
    print(
        "Mapping count:",
        len(repository.alias_to_canonical),
    )
    print(
        "Canonical count:",
        len(repository.canonical_to_aliases),
    )

    for raw_term, expected in cases.items():
        actual = repository.exact_lookup(raw_term)

        assert actual == expected, (
            f"{terminology_type}: "
            f"{raw_term!r} returned {actual!r}; "
            f"expected {expected!r}"
        )

        print(
            f"PASS: {raw_term!r} -> {actual!r}"
        )


def main() -> None:
    for terminology_type, cases in (
        TEST_CASES.items()
    ):
        test_repository(
            terminology_type,
            cases,
        )

    print(
        "\nAll repository tests passed."
    )


if __name__ == "__main__":
    main()