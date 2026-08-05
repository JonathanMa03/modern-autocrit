"""Update terminology dictionaries from a Modern AutoCrit Excel output."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.services.file_manager import FileManager
from backend.utils.paths import ATTRIBUTE_MAP, DISEASE_MAP, ENTITY_MAP
from backend.utils.text_utils import clean_attribute_key


DEFAULT_OUTPUT = Path("outputs/modern_autocrit_output.xlsx")
DICTIONARY_DIR = ENTITY_MAP.parent
ORIGINAL_DIR = DICTIONARY_DIR / "original"
DICTIONARY_PATHS = {
    "attribute": ATTRIBUTE_MAP,
    "disease": DISEASE_MAP,
    "entity": ENTITY_MAP,
}
DISEASE_ENTITIES = {"diagnosis", "comorbidity"}


@dataclass(frozen=True)
class UpdateSummary:
    attribute_added: int
    disease_added: int
    entity_added: int
    rows_processed: int


def _text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    return result or None


def _first_text(row: pd.Series, *columns: str) -> str | None:
    for column in columns:
        if column in row.index:
            value = _text(row[column])
            if value:
                return value
    return None


def _add_if_new(mapping: dict[str, str], raw: str, canonical: str) -> bool:
    key = clean_attribute_key(raw)
    if not key or key in mapping:
        return False
    mapping[key] = canonical.strip()
    return True


def update_dictionaries(output_file: str | Path = DEFAULT_OUTPUT) -> UpdateSummary:
    """Learn non-destructive mappings from a pipeline output workbook."""
    path = Path(output_file)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline output was not found: {path}")

    dataframe = pd.read_excel(path)
    required = {"entity", "attribute"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Pipeline output is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    mappings = {
        name: FileManager.load_dictionary(dictionary_path)
        for name, dictionary_path in DICTIONARY_PATHS.items()
    }
    counts = {name: 0 for name in DICTIONARY_PATHS}

    for _, row in dataframe.iterrows():
        raw_entity = _first_text(row, "raw_entity", "entity")
        canonical_entity = _first_text(row, "canonical_entity", "entity")
        raw_attribute = _first_text(row, "raw_attribute", "attribute")
        canonical_attribute = _first_text(
            row,
            "canonical_attribute",
            "attribute",
        )

        if raw_entity and canonical_entity:
            counts["entity"] += int(
                _add_if_new(mappings["entity"], raw_entity, canonical_entity)
            )

        if not raw_attribute or not canonical_attribute:
            continue

        entity_key = clean_attribute_key(canonical_entity)
        dictionary_name = (
            "disease" if entity_key in DISEASE_ENTITIES else "attribute"
        )
        counts[dictionary_name] += int(
            _add_if_new(
                mappings[dictionary_name],
                raw_attribute,
                canonical_attribute,
            )
        )

    for name, dictionary_path in DICTIONARY_PATHS.items():
        FileManager.save_dictionary(mappings[name], dictionary_path)

    return UpdateSummary(
        attribute_added=counts["attribute"],
        disease_added=counts["disease"],
        entity_added=counts["entity"],
        rows_processed=len(dataframe),
    )


def reset_dictionaries() -> None:
    """Restore all dictionaries from the versioned original snapshots."""
    missing = [
        ORIGINAL_DIR / path.name
        for path in DICTIONARY_PATHS.values()
        if not (ORIGINAL_DIR / path.name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Original dictionary snapshot(s) missing: "
            + ", ".join(str(path) for path in missing)
        )

    for destination in DICTIONARY_PATHS.values():
        source = ORIGINAL_DIR / destination.name
        # Parse first so a damaged snapshot cannot replace a working dictionary.
        with source.open("r", encoding="utf-8") as file:
            json.load(file)
        shutil.copyfile(source, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update terminology dictionaries from pipeline Excel output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Pipeline workbook (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Restore the three dictionaries to their original repository versions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.reset:
            reset_dictionaries()
            print(f"Restored attribute, disease, and entity dictionaries from {ORIGINAL_DIR}")
            return 0

        summary = update_dictionaries(args.input)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {summary.rows_processed} output row(s).")
    print(f"Added {summary.attribute_added} attribute mapping(s).")
    print(f"Added {summary.disease_added} disease mapping(s).")
    print(f"Added {summary.entity_added} entity mapping(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
