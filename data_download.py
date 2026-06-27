# Placeholder script for downloading from clinical trials
"""
Download ClinicalTrials.gov XML files for AutoCriteria.

AutoCriteria expects:
-input_file <directory_containing_trial_xml_files>
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


def download_trial_xml(
    nct_id: str,
    output_dir: str | Path = "data/raw/xml_trials",
    sleep_seconds: float = 0.25,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
    output_path = output_dir / f"{nct_id}.xml"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.write_text(response.text, encoding="utf-8")

    time.sleep(sleep_seconds)

    return output_path


def download_xml_from_dataframe(
    df: pd.DataFrame,
    nct_column: str = "nct_id",
    output_dir: str | Path = "data/raw/xml_trials",
    max_trials: int = 10,
) -> list[Path]:
    nct_ids = df[nct_column].dropna().head(max_trials).tolist()

    paths = []
    for nct_id in nct_ids:
        print(f"Downloading {nct_id}...")
        path = download_trial_xml(nct_id, output_dir=output_dir)
        paths.append(path)

    return paths