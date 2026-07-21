import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DICTIONARY_DIR = (
    PROJECT_ROOT
    / "config"
    / "dictionaries"
)

FILES = [
    "attribute_map.json",
    "disease_map.json",
    "entity_map.json",
]


def validate_dictionary(
    path: Path,
) -> None:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"{path.name} must contain a JSON object."
        )

    if not data:
        raise ValueError(
            f"{path.name} is empty."
        )

    for alias, canonical in data.items():
        if not isinstance(alias, str):
            raise TypeError(
                f"{path.name}: alias must be a string."
            )

        if not isinstance(canonical, str):
            raise TypeError(
                f"{path.name}: canonical value for "
                f"{alias!r} must be a string."
            )

        if alias != alias.strip():
            raise ValueError(
                f"{path.name}: alias has surrounding "
                f"whitespace: {alias!r}"
            )

        if canonical != canonical.strip():
            raise ValueError(
                f"{path.name}: canonical value has "
                f"surrounding whitespace: "
                f"{canonical!r}"
            )

        if alias != alias.lower():
            raise ValueError(
                f"{path.name}: alias must be lowercase: "
                f"{alias!r}"
            )

        if not alias:
            raise ValueError(
                f"{path.name}: empty alias found."
            )

        if not canonical:
            raise ValueError(
                f"{path.name}: empty canonical value "
                f"for alias {alias!r}."
            )

    print(
        f"PASS: {path.name} "
        f"({len(data)} mappings)"
    )


def main() -> None:
    for filename in FILES:
        path = DICTIONARY_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dictionary not found: {path}"
            )

        validate_dictionary(path)

    print(
        "\nAll terminology dictionaries are valid."
    )


if __name__ == "__main__":
    main()