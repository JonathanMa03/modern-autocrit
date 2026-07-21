from collections import defaultdict

from backend.normalization.terminology_service import (
    TerminologyService,
)


TERMINOLOGY_TYPES = [
    "attribute",
    "disease",
    "entity",
]


def find_duplicates(
    terminology_type: str,
) -> dict[str, set[str]]:
    service = TerminologyService(
        terminology_type=terminology_type,
    )

    grouped: dict[str, set[str]] = defaultdict(
        set
    )

    for canonical in (
        service.repository
        .alias_to_canonical
        .values()
    ):
        normalized = " ".join(
            canonical
            .strip()
            .casefold()
            .split()
        )

        grouped[normalized].add(canonical)

    return {
        normalized: values
        for normalized, values in grouped.items()
        if len(values) > 1
    }


def main() -> None:
    found_any = False

    for terminology_type in TERMINOLOGY_TYPES:
        duplicates = find_duplicates(
            terminology_type
        )

        print(
            f"\n{terminology_type.upper()}"
        )

        if not duplicates:
            print(
                "No capitalization-only duplicates."
            )
            continue

        found_any = True

        for values in duplicates.values():
            print("Duplicate group:")

            for value in sorted(values):
                print(f"  - {value}")

    if found_any:
        raise SystemExit(
            "\nDuplicate canonical spellings found."
        )

    print(
        "\nAll canonical spellings are consistent."
    )


if __name__ == "__main__":
    main()