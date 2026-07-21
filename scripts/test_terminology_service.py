from backend.normalization.terminology_service import (
    TerminologyService,
)


def main() -> None:
    service = TerminologyService(
        terminology_type="attribute",
    )

    print(
        "Mapping count:",
        service.get_mapping_count(),
    )

    print(
        "Canonical count:",
        service.get_canonical_count(),
    )

    test_terms = [
        "ALT",
        "ECOG",
        "platelets",
        "LVEF",
        "unknown term",
    ]

    for term in test_terms:
        result = service.exact_lookup(term)

        print(
            f"{term!r} -> {result!r}"
        )


if __name__ == "__main__":
    main()