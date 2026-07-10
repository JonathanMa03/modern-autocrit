from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from backend.quality.rules.numeric_rules import (
    check_numeric_range,
    check_unit_compatibility,
)
from backend.quality.rules.terminology_rules import (
    check_bad_term,
    check_value_term,
)
from backend.schemas.anomaly_schema import AnomalyFinding


class AnomalyDetector:
    """
    Deterministic first-pass quality screening.

    Findings are review flags, not clinical conclusions.
    """

    def analyze_row(
        self,
        row: pd.Series,
        row_index: int,
    ) -> list[AnomalyFinding]:
        trial_id = str(row.get("trial_id", ""))
        entity = self._none_if_missing(
            row.get("canonical_entity", row.get("entity"))
        )
        attribute = self._none_if_missing(
            row.get(
                "canonical_attribute",
                row.get("attribute"),
            )
        )
        value = self._none_if_missing(
            row.get(
                "canonical_value",
                row.get("value"),
            )
        )

        raw_findings: list[dict] = []

        raw_findings.extend(
            check_numeric_range(
                attribute=attribute,
                value=value,
            )
        )
        raw_findings.extend(
            check_unit_compatibility(
                attribute=attribute,
                value=value,
            )
        )
        raw_findings.extend(
            check_bad_term(
                attribute=attribute,
                entity=entity,
            )
        )
        raw_findings.extend(
            check_value_term(value)
        )

        findings: list[AnomalyFinding] = []

        for finding in raw_findings:
            findings.append(
                AnomalyFinding(
                    trial_id=trial_id,
                    row_index=row_index,
                    category=finding["category"],
                    severity=finding["severity"],
                    entity=entity,
                    attribute=attribute,
                    value=value,
                    unit=finding.get("unit"),
                    message=finding["message"],
                    suggested_action=finding.get(
                        "suggested_action"
                    ),
                    rule_id=finding["rule_id"],
                    confidence=finding.get(
                        "confidence",
                        1.0,
                    ),
                )
            )

        return findings

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
    ) -> list[AnomalyFinding]:
        findings: list[AnomalyFinding] = []

        for row_index, row in df.iterrows():
            findings.extend(
                self.analyze_row(
                    row=row,
                    row_index=int(row_index),
                )
            )

        return findings

    @staticmethod
    def findings_to_dataframe(
        findings: Iterable[AnomalyFinding],
    ) -> pd.DataFrame:
        rows = [
            finding.model_dump()
            for finding in findings
        ]

        if not rows:
            return pd.DataFrame(
                columns=[
                    "trial_id",
                    "row_index",
                    "category",
                    "severity",
                    "entity",
                    "attribute",
                    "value",
                    "unit",
                    "message",
                    "suggested_action",
                    "rule_id",
                    "confidence",
                ]
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _none_if_missing(
        value,
    ) -> str | None:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        text = str(value).strip()

        return text or None