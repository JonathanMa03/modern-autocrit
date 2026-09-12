"""Pure ClinicalTrials.gov API v2 study parser."""

from __future__ import annotations

from typing import Any, Iterable

from backend.ctg_parser.models import (
    ClinicalTrialsCandidate,
    ClinicalTrialsIntervention,
    ClinicalTrialsOutcome,
)


CTG_API_BASE = "https://clinicaltrials.gov/api/v2"


def _module(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    value = protocol.get(name, {})
    return value if isinstance(value, dict) else {}


def _date_value(module: dict[str, Any], key: str) -> str:
    value = module.get(key, {})
    if isinstance(value, dict):
        return str(value.get("date", "") or "")
    return str(value or "")


def parse_ctgov_study(
    study: dict[str, Any],
    *,
    matched_query_ids: Iterable[str] = (),
) -> ClinicalTrialsCandidate:
    """Normalize one ClinicalTrials.gov API v2 study record.

    This function is intentionally target-agnostic. It performs structural
    parsing only: registry JSON in, provenance-preserving trial metadata out.
    It does not search, filter, normalize criteria, call an LLM, or apply
    disease-specific rules.
    """

    protocol = study.get("protocolSection", {})
    identification = _module(protocol, "identificationModule")
    status = _module(protocol, "statusModule")
    sponsor = _module(protocol, "sponsorCollaboratorsModule")
    design = _module(protocol, "designModule")
    conditions = _module(protocol, "conditionsModule")
    arms = _module(protocol, "armsInterventionsModule")
    outcomes = _module(protocol, "outcomesModule")
    eligibility = _module(protocol, "eligibilityModule")
    contacts = _module(protocol, "contactsLocationsModule")

    nct_id = str(identification.get("nctId", "") or "").upper()
    if not nct_id:
        raise ValueError("ClinicalTrials.gov study record has no NCT ID.")

    intervention_rows = []
    for item in arms.get("interventions", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        intervention_rows.append(
            ClinicalTrialsIntervention(
                intervention_type=str(item.get("type", "") or ""),
                name=str(item.get("name", "") or ""),
                description=str(item.get("description", "") or ""),
                arm_labels=[
                    str(label)
                    for label in item.get("armGroupLabels", []) or []
                ],
            )
        )

    outcome_rows: list[ClinicalTrialsOutcome] = []
    for outcome_type, key in [
        ("primary", "primaryOutcomes"),
        ("secondary", "secondaryOutcomes"),
        ("other", "otherOutcomes"),
    ]:
        for item in outcomes.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            outcome_rows.append(
                ClinicalTrialsOutcome(
                    outcome_type=outcome_type,
                    measure=str(item.get("measure", "") or ""),
                    description=str(item.get("description", "") or ""),
                    time_frame=str(item.get("timeFrame", "") or ""),
                )
            )

    design_info = design.get("designInfo", {}) or {}
    masking_info = design_info.get("maskingInfo", {}) or {}
    enrollment = design.get("enrollmentInfo", {}) or {}
    lead_sponsor = sponsor.get("leadSponsor", {}) or {}
    location_rows = contacts.get("locations", []) or []
    countries = sorted(
        {
            str(location.get("country", "") or "")
            for location in location_rows
            if isinstance(location, dict) and location.get("country")
        }
    )
    arm_groups = [
        str(item.get("label", "") or "")
        for item in arms.get("armGroups", []) or []
        if isinstance(item, dict) and item.get("label")
    ]

    enrollment_count = enrollment.get("count")
    try:
        parsed_enrollment = (
            int(enrollment_count) if enrollment_count is not None else None
        )
    except (TypeError, ValueError):
        parsed_enrollment = None

    return ClinicalTrialsCandidate(
        nct_id=nct_id,
        brief_title=str(identification.get("briefTitle", "") or ""),
        official_title=str(identification.get("officialTitle", "") or ""),
        acronym=str(identification.get("acronym", "") or ""),
        conditions=[
            str(value) for value in conditions.get("conditions", []) or []
        ],
        keywords=[
            str(value) for value in conditions.get("keywords", []) or []
        ],
        interventions=intervention_rows,
        arm_groups=arm_groups,
        phases=[str(value) for value in design.get("phases", []) or []],
        overall_status=str(status.get("overallStatus", "") or ""),
        study_type=str(design.get("studyType", "") or ""),
        allocation=str(design_info.get("allocation", "") or ""),
        intervention_model=str(
            design_info.get("interventionModel", "") or ""
        ),
        primary_purpose=str(
            design_info.get("primaryPurpose", "") or ""
        ),
        masking=str(masking_info.get("masking", "") or ""),
        enrollment_count=parsed_enrollment,
        enrollment_type=str(enrollment.get("type", "") or ""),
        sex=str(eligibility.get("sex", "") or ""),
        minimum_age=str(eligibility.get("minimumAge", "") or ""),
        maximum_age=str(eligibility.get("maximumAge", "") or ""),
        healthy_volunteers=eligibility.get("healthyVolunteers"),
        eligibility_text=str(
            eligibility.get("eligibilityCriteria", "") or ""
        ),
        outcomes=outcome_rows,
        lead_sponsor=str(lead_sponsor.get("name", "") or ""),
        start_date=_date_value(status, "startDateStruct"),
        completion_date=_date_value(status, "completionDateStruct"),
        last_update_post_date=_date_value(
            status,
            "lastUpdatePostDateStruct",
        ),
        countries=countries,
        source_url=f"https://clinicaltrials.gov/study/{nct_id}",
        api_url=f"{CTG_API_BASE}/studies/{nct_id}",
        matched_query_ids=sorted(set(matched_query_ids)),
    )
