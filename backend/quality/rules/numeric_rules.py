from __future__ import annotations

import math
import re
from dataclasses import dataclass


NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
)

UNIT_PATTERN = re.compile(
    r"("
    r"mg/kg/day|"
    r"mg/kg|"
    r"mg/day|"
    r"mmol/l|"
    r"umol/l|"
    r"µmol/l|"
    r"cells/mm\^3|"
    r"cells/mm3|"
    r"/mm\^3|"
    r"/mm3|"
    r"x\s*10\^9/l|"
    r"10\^9/l|"
    r"mg/dl|"
    r"g/dl|"
    r"ng/ml|"
    r"pg/ml|"
    r"mmhg|"
    r"years?|"
    r"months?|"
    r"weeks?|"
    r"days?|"
    r"hours?|"
    r"bpm|"
    r"mcg|"
    r"µg|"
    r"ug|"
    r"mg|"
    r"kg|"
    r"mol/l|"
    r"g|"
    r"%|"
    r"uln"
    r")",
    flags=re.IGNORECASE,
)


@dataclass
class ParsedMeasurement:
    comparator: str | None
    number: float | None
    unit: str | None


def extract_first_number(text: str | None) -> float | None:
    if text is None:
        return None

    match = NUMBER_PATTERN.search(str(text))

    if match is None:
        return None

    raw = match.group(0).replace(",", "")

    try:
        value = float(raw)
    except ValueError:
        return None

    if not math.isfinite(value):
        return None

    return value


def extract_unit(text: str | None) -> str | None:
    if text is None:
        return None

    match = UNIT_PATTERN.search(str(text))

    if match is None:
        return None

    return match.group(1).lower().strip()


def extract_comparator(text: str | None) -> str | None:
    if text is None:
        return None

    value = str(text)

    comparator_patterns = [
        (r">=", ">="),
        (r"<=", "<="),
        (r">", ">"),
        (r"<", "<"),
        (r"\bat least\b", ">="),
        (r"\bminimum\b", ">="),
        (r"\bno more than\b", "<="),
        (r"\bat most\b", "<="),
        (r"\bmaximum\b", "<="),
    ]

    lowered = value.lower()

    for pattern, canonical in comparator_patterns:
        if re.search(pattern, lowered):
            return canonical

    return None


def parse_measurement(text: str | None) -> ParsedMeasurement:
    return ParsedMeasurement(
        comparator=extract_comparator(text),
        number=extract_first_number(text),
        unit=extract_unit(text),
    )


def normalize_attribute_name(attribute: str | None) -> str:
    if attribute is None:
        return ""

    value = str(attribute).lower().strip()
    value = re.sub(r"[^a-z0-9%]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def check_numeric_range(
    attribute: str | None,
    value: str | None,
) -> list[dict]:
    """
    Return deterministic numeric anomaly findings.

    These are screening rules, not clinical adjudication rules.
    """

    attribute_key = normalize_attribute_name(attribute)
    parsed = parse_measurement(value)

    if parsed.number is None:
        return []

    number = parsed.number
    unit = parsed.unit
    findings: list[dict] = []

    def add(
        rule_id: str,
        message: str,
        severity: str = "warning",
        suggested_action: str | None = None,
    ) -> None:
        findings.append(
            {
                "rule_id": rule_id,
                "severity": severity,
                "category": "numeric_range",
                "message": message,
                "suggested_action": suggested_action,
                "unit": unit,
                "confidence": 1.0,
            }
        )

    # Generic invalid values
    if number < 0:
        nonnegative_terms = {
            "age",
            "platelet",
            "neutrophil",
            "hemoglobin",
            "creatinine",
            "bilirubin",
            "albumin",
            "oxygen saturation",
            "ejection fraction",
            "body mass index",
            "bmi",
            "life expectancy",
        }

        if any(term in attribute_key for term in nonnegative_terms):
            add(
                "NUM_NEGATIVE_NONNEGATIVE_MEASURE",
                f"Negative value detected for '{attribute}': {value}.",
                severity="high",
                suggested_action="Verify the sign and extracted comparator.",
            )

    # Age
    if attribute_key == "age" or attribute_key.endswith(" age"):
        if number > 130:
            add(
                "NUM_AGE_TOO_HIGH",
                f"Age threshold appears implausibly high: {value}.",
                severity="high",
                suggested_action="Check whether the value or unit was extracted incorrectly.",
            )

    # Percentages
    percentage_terms = {
        "oxygen saturation",
        "ejection fraction",
        "lvef",
        "percentage",
    }

    if unit == "%" or any(
        term in attribute_key
        for term in percentage_terms
    ):
        if number < 0 or number > 100:
            add(
                "NUM_PERCENT_OUT_OF_RANGE",
                f"Percentage-like measurement is outside 0–100: {value}.",
                severity="high",
                suggested_action="Check the decimal point, unit, and source sentence.",
            )

    # ECOG
    if "ecog" in attribute_key:
        if number < 0 or number > 5:
            add(
                "NUM_ECOG_OUT_OF_RANGE",
                f"ECOG performance status is outside the expected 0–5 range: {value}.",
                severity="high",
                suggested_action="Verify the score extraction.",
            )

    # Karnofsky
    if "karnofsky" in attribute_key:
        if number < 0 or number > 100:
            add(
                "NUM_KARNOFSKY_OUT_OF_RANGE",
                f"Karnofsky score is outside the expected 0–100 range: {value}.",
                severity="high",
                suggested_action="Verify the score extraction.",
            )

    # BMI
    if attribute_key == "bmi" or "body mass index" in attribute_key:
        if number < 5 or number > 100:
            add(
                "NUM_BMI_SUSPICIOUS",
                f"BMI threshold appears unusual: {value}.",
                severity="warning",
                suggested_action="Review the source text and decimal placement.",
            )

    # Hemoglobin
    if "hemoglobin" in attribute_key and unit in {"g/dl", None}:
        if number > 30:
            add(
                "NUM_HEMOGLOBIN_TOO_HIGH",
                f"Hemoglobin value appears unusually high: {value}.",
                severity="high",
                suggested_action="Check whether the unit should be g/L rather than g/dL.",
            )

    # Creatinine
    if "creatinine" in attribute_key and "clearance" not in attribute_key:
        if unit == "mg/dl" and number > 50:
            add(
                "NUM_CREATININE_TOO_HIGH",
                f"Serum creatinine appears unusually high for mg/dL: {value}.",
                severity="warning",
                suggested_action="Review the unit and decimal point.",
            )

    # Platelets
    if "platelet" in attribute_key:
        if number > 10_000_000:
            add(
                "NUM_PLATELET_TOO_HIGH",
                f"Platelet threshold appears unusually high: {value}.",
                severity="warning",
                suggested_action="Check for misplaced zeros or incorrect units.",
            )

    # Absolute neutrophil count
    if (
        "absolute neutrophil" in attribute_key
        or attribute_key == "anc"
    ):
        if number > 1_000_000:
            add(
                "NUM_ANC_TOO_HIGH",
                f"Absolute neutrophil count appears unusually high: {value}.",
                severity="warning",
                suggested_action="Review the units and source sentence.",
            )

    # Dosage screening
    dosage_units = {
        "mg",
        "mg/day",
        "mg/kg",
        "mg/kg/day",
        "mcg",
        "µg",
        "ug",
        "g",
    }

    if unit in dosage_units:
        if unit in {"mg/kg", "mg/kg/day"} and number > 1000:
            add(
                "NUM_DOSE_MG_PER_KG_HIGH",
                f"Dose appears unusually large for {unit}: {value}.",
                severity="high",
                suggested_action="Check the decimal point, dose unit, and extraction.",
            )

        if unit in {"mg", "mg/day"} and number > 1_000_000:
            add(
                "NUM_DOSE_MG_HIGH",
                f"Dose appears unusually large: {value}.",
                severity="high",
                suggested_action="Check whether the value or unit was extracted incorrectly.",
            )

        if unit == "g" and number > 1000:
            add(
                "NUM_DOSE_GRAMS_HIGH",
                f"Dose appears unusually large in grams: {value}.",
                severity="high",
                suggested_action="Review the source text and unit.",
            )

    return findings


def check_unit_compatibility(
    attribute: str | None,
    value: str | None,
) -> list[dict]:
    attribute_key = normalize_attribute_name(attribute)
    parsed = parse_measurement(value)

    if parsed.number is None or parsed.unit is None:
        return []

    unit = parsed.unit
    findings: list[dict] = []

    expected_units: dict[str, set[str]] = {
        "age": {"year", "years", "month", "months"},
        "hemoglobin": {"g/dl"},
        "creatinine": {"mg/dl", "umol/l", "µmol/l"},
        "oxygen saturation": {"%"},
        "ejection fraction": {"%"},
        "lvef": {"%"},
        "body mass index": {"kg"},
    }

    matching_key = None

    for candidate in expected_units:
        if candidate in attribute_key:
            matching_key = candidate
            break

    if matching_key is None:
        return []

    allowed = expected_units[matching_key]

    if unit not in allowed:
        findings.append(
            {
                "rule_id": "UNIT_ATTRIBUTE_MISMATCH",
                "severity": "warning",
                "category": "unit_mismatch",
                "message": (
                    f"Unit '{unit}' may not match attribute "
                    f"'{attribute}' in value '{value}'."
                ),
                "suggested_action": "Review the source sentence and extracted unit.",
                "unit": unit,
                "confidence": 0.9,
            }
        )

    return findings