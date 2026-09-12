"""Retrieve protocol records from the ClinicalTrials.gov API v2."""

from __future__ import annotations

import re
from typing import Any

import requests


CTGOV_API_V2 = "https://clinicaltrials.gov/api/v2"
NCT_ID_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


def fetch_study(nct_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    normalized = nct_id.strip().upper()
    if not NCT_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Enter a valid NCT identifier such as NCT00006174.")
    response = requests.get(
        f"{CTGOV_API_V2}/studies/{normalized}",
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    if response.status_code == 404:
        raise ValueError(f"ClinicalTrials.gov has no study {normalized}.")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("ClinicalTrials.gov returned an invalid study record.")
    return payload


def eligibility_text_from_study(study: dict[str, Any]) -> str:
    protocol = study.get("protocolSection") or {}
    eligibility = protocol.get("eligibilityModule") or {}
    text = str(eligibility.get("eligibilityCriteria") or "").strip()
    if not text:
        identification = protocol.get("identificationModule") or {}
        nct_id = identification.get("nctId") or "the requested study"
        raise ValueError(f"No eligibility criteria were reported for {nct_id}.")
    return text
