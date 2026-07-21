from backend.normalization.terminology_repository import (
    TerminologyRepository,
)


def main() -> None:
    repository = TerminologyRepository(
        terminology_type="attribute",
    )

    test_terms = [
        "ALT",
        "alanine aminotransferase",
        "ALT(SGPT)",
        "ECOG",
        "platelets",
        "left ventricular ejection fraction",
        "unknown laboratory measurement",
    ]

    for term in test_terms:
        result = repository.exact_lookup(term)

        print(
            f"{term!r} -> {result!r}"
        )

    print("\nCanonical term count:")
    print(
        len(repository.canonical_terms())
    )

    print("\nAliases for Platelet Count:")
    print(
        repository.aliases_for(
            "Platelet Count"
        )
    )

    print("\nFirst 10 search entries:")

    for entry in (
        repository.search_entries()[:10]
    ):
        print(entry)


if __name__ == "__main__":
    main()