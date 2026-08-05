"""Download ClinicalTrials.gov studies as pipeline-compatible XML files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import requests


API_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_OUTPUT_DIR = Path("data/raw/xml_trials")
DEFAULT_WARNING_THRESHOLD = 100
MAX_PAGE_SIZE = 1000


class ClinicalTrialsAPIError(RuntimeError):
    """Raised when ClinicalTrials.gov cannot complete a request."""


def _api_get(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests.Session()
    url = f"{API_BASE_URL}/{path.lstrip('/')}"

    try:
        response = client.get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        raise ClinicalTrialsAPIError(
            f"Could not reach ClinicalTrials.gov: {exc}"
        ) from exc

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        detail = f" Retry after {retry_after} seconds." if retry_after else ""
        raise ClinicalTrialsAPIError(
            "ClinicalTrials.gov API usage limit was reached (HTTP 429)."
            f"{detail} Try again later or request fewer trials."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ClinicalTrialsAPIError(
            f"ClinicalTrials.gov returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        ) from exc

    try:
        return response.json()
    except (requests.JSONDecodeError, json.JSONDecodeError) as exc:
        raise ClinicalTrialsAPIError(
            "ClinicalTrials.gov returned a response that was not valid JSON."
        ) from exc


def _add_text(parent: ET.Element, tag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        ET.SubElement(parent, tag).text = text


def study_to_xml(study: dict[str, Any]) -> ET.ElementTree:
    """Convert one v2 API study into the XML subset consumed by the pipeline."""
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {})
    eligibility = protocol.get("eligibilityModule", {})

    root = ET.Element("clinical_study")
    _add_text(root, "nct_id", identification.get("nctId"))
    _add_text(root, "brief_title", identification.get("briefTitle"))

    phases = design.get("phases") or []
    if phases:
        _add_text(root, "phase", ", ".join(map(str, phases)))

    for condition in conditions.get("conditions") or []:
        _add_text(root, "condition", condition)

    criteria_text = eligibility.get("eligibilityCriteria")
    if criteria_text:
        eligibility_element = ET.SubElement(root, "eligibility")
        criteria_element = ET.SubElement(eligibility_element, "criteria")
        _add_text(criteria_element, "textblock", criteria_text)

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def save_study_xml(
    study: dict[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    identification = (
        study.get("protocolSection", {}).get("identificationModule", {})
    )
    nct_id = str(identification.get("nctId", "")).strip().upper()
    if not nct_id:
        raise ClinicalTrialsAPIError("API study response did not contain an NCT ID.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{nct_id}.xml"
    study_to_xml(study).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )
    return output_path


def download_trial_xml(
    nct_id: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sleep_seconds: float = 0.25,
    session: requests.Session | None = None,
) -> Path:
    """Download one study by NCT ID and save pipeline-compatible XML."""
    normalized_id = nct_id.strip().upper()
    study = _api_get(f"studies/{normalized_id}", session=session)
    output_path = save_study_xml(study, output_dir)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return output_path


def search_studies(
    search_terms: str,
    number: int,
    *,
    session: requests.Session | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield up to ``number`` studies matching a v2 API search expression."""
    if not search_terms.strip():
        raise ValueError("Search terms cannot be empty.")
    if number < 1:
        raise ValueError("Number of trials must be at least 1.")

    remaining = number
    page_token: str | None = None

    while remaining:
        params: dict[str, Any] = {
            "query.term": search_terms,
            "pageSize": min(remaining, MAX_PAGE_SIZE),
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        payload = _api_get("studies", params=params, session=session)
        studies = payload.get("studies") or []
        if not studies:
            break

        for study in studies[:remaining]:
            yield study
            remaining -= 1
            if remaining == 0:
                return

        page_token = payload.get("nextPageToken")
        if not page_token:
            break


def download_search_results(
    search_terms: str,
    number: int,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    session: requests.Session | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for study in search_studies(search_terms, number, session=session):
        path = save_study_xml(study, output_dir)
        paths.append(path)
        print(f"Downloaded {path.name} ({len(paths)}/{number})")
    return paths


def download_xml_from_dataframe(
    df: pd.DataFrame,
    nct_column: str = "nct_id",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_trials: int = 10,
) -> list[Path]:
    nct_ids = df[nct_column].dropna().head(max_trials).tolist()
    return [download_trial_xml(str(nct_id), output_dir) for nct_id in nct_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search ClinicalTrials.gov and save matching studies as XML files "
            "compatible with the Modern AutoCrit pipeline."
        )
    )
    parser.add_argument(
        "search_terms",
        help='ClinicalTrials.gov search expression, e.g. "lung cancer AND recruiting".',
    )
    parser.add_argument(
        "number",
        type=int,
        help="Maximum number of trials to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation when the requested download is large.",
    )
    parser.add_argument(
        "--warning-threshold",
        type=int,
        default=DEFAULT_WARNING_THRESHOLD,
        help=(
            "Ask for confirmation at or above this many trials "
            f"(default: {DEFAULT_WARNING_THRESHOLD})."
        ),
    )
    return parser


def _confirm_large_download(number: int, threshold: int) -> bool:
    print(
        f"WARNING: You requested {number} trials. This may take time, use "
        "significant disk space, and trigger ClinicalTrials.gov API limits.",
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        print("Refusing non-interactive download; pass --yes to continue.", file=sys.stderr)
        return False
    answer = input("Continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.number < 1:
        print("ERROR: number must be at least 1.", file=sys.stderr)
        return 2
    if args.warning_threshold < 1:
        print("ERROR: --warning-threshold must be at least 1.", file=sys.stderr)
        return 2
    if (
        args.number >= args.warning_threshold
        and not args.yes
        and not _confirm_large_download(args.number, args.warning_threshold)
    ):
        print("Download cancelled.", file=sys.stderr)
        return 1

    try:
        paths = download_search_results(
            args.search_terms,
            args.number,
            args.output_dir,
        )
    except (ClinicalTrialsAPIError, ValueError) as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 1

    if len(paths) < args.number:
        print(
            f"WARNING: Only {len(paths)} matching trial(s) were available; "
            f"{args.number} were requested.",
            file=sys.stderr,
        )
    print(f"Saved {len(paths)} trial(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
