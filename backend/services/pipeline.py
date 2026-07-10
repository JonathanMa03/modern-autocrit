"""
Main extraction pipeline for Modern AutoCrit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.core.extractor import ModernAutoCritExtractor
from backend.core.normalizer import CriteriaNormalizer
from backend.services.clinicaltrials_service import ClinicalTrialsService
from backend.services.cost_monitor import CostMonitor
from backend.services.file_manager import FileManager
from backend.utils.logging_utils import get_logger
from backend.quality.anomaly_detector import AnomalyDetector


logger = get_logger(__name__)


@dataclass
class PipelineSummary:
    trials_processed: int
    criteria_extracted: int
    criteria_after_normalization: int
    criteria_after_deduplication: int
    output_file: Path
    total_cost_usd: float

    anomaly_findings: int = 0
    anomaly_high: int = 0
    anomaly_warnings: int = 0
    anomaly_info: int = 0

    anomaly_numeric: int = 0
    anomaly_units: int = 0
    anomaly_terminology: int = 0
    anomaly_missing_values: int = 0

    anomalies_file: Path | None = None


class ModernAutoCritPipeline:
    def __init__(
        self,
        extractor: ModernAutoCritExtractor,
        normalizer: CriteriaNormalizer,
        cost_monitor: CostMonitor,
    ):
        self.extractor = extractor
        self.normalizer = normalizer
        self.cost_monitor = cost_monitor

    def run(
        self,
        xml_directory: Path,
        output_excel: Path,
    ) -> PipelineSummary:
        logger.info("Starting Modern AutoCrit pipeline.")

        service = ClinicalTrialsService(xml_directory)

        all_criteria = []
        trials_processed = 0

        for xml_path in service.xml_files():
            trial = service.load_trial(xml_path)

            result = self.extractor.extract_from_trial_dict(trial)

            all_criteria.extend(result.criteria)
            trials_processed += 1

            logger.info(
                "%s finished with %d extracted criteria.",
                trial.get("trial_id", xml_path.stem),
                len(result.criteria),
            )

            if result.errors:
                for error in result.errors:
                    logger.warning(error)

        criteria_extracted = len(all_criteria)

        logger.info(
            "Extraction complete. %d total criteria extracted.",
            criteria_extracted,
        )

        if criteria_extracted == 0:
            empty_df = pd.DataFrame()
            FileManager.save_dataframe_excel(empty_df, output_excel)

            return PipelineSummary(
                trials_processed=trials_processed,
                criteria_extracted=0,
                criteria_after_normalization=0,
                criteria_after_deduplication=0,
                output_file=output_excel,
                total_cost_usd=self.cost_monitor.total_cost,
            )

        normalized_df = self.normalizer.normalize_many(all_criteria)
        criteria_after_normalization = len(normalized_df)

        deduped_df = self.normalizer.remove_exact_duplicates(normalized_df)
        anomaly_detector = AnomalyDetector()

        anomaly_findings = anomaly_detector.analyze_dataframe(
            deduped_df
        )

        anomalies_df = anomaly_detector.findings_to_dataframe(
            anomaly_findings
        )

        anomaly_count = len(anomalies_df)

        if anomalies_df.empty:
            anomaly_high = 0
            anomaly_warnings = 0
            anomaly_info = 0

            anomaly_numeric = 0
            anomaly_units = 0
            anomaly_terminology = 0
            anomaly_missing_values = 0
        else:
            severity_counts = (
                anomalies_df["severity"]
                .value_counts()
                .to_dict()
            )

            category_counts = (
                anomalies_df["category"]
                .value_counts()
                .to_dict()
            )

            anomaly_high = int(
                severity_counts.get("high", 0)
            )
            anomaly_warnings = int(
                severity_counts.get("warning", 0)
            )
            anomaly_info = int(
                severity_counts.get("info", 0)
            )

            anomaly_numeric = int(
                category_counts.get("numeric_range", 0)
            )
            anomaly_units = int(
                category_counts.get("unit_mismatch", 0)
            )
            anomaly_terminology = int(
                category_counts.get("terminology", 0)
            )
            anomaly_missing_values = int(
                category_counts.get("missing_value", 0)
            )



        criteria_after_deduplication = len(deduped_df)
        unmapped_terms = self.normalizer.find_unmapped_terms(deduped_df)

        FileManager.save_dataframe_excel(
            deduped_df,
            output_excel,
        )
        unmapped_output = output_excel.with_name(
            output_excel.stem + "_unmapped_terms.xlsx"
        )

        FileManager.save_dataframe_excel(
            unmapped_terms,
            unmapped_output,
        )

        anomalies_output = output_excel.with_name(
            output_excel.stem + "_anomalies.xlsx"
        )

        FileManager.save_dataframe_excel(
            anomalies_df,
            anomalies_output,
        )

        logger.info(
            "Anomaly screening produced %d finding(s).",
            len(anomalies_df),
        )

        logger.info("Saved pipeline output to %s", output_excel)

        return PipelineSummary(
            trials_processed=trials_processed,
            criteria_extracted=criteria_extracted,
            criteria_after_normalization=criteria_after_normalization,
            criteria_after_deduplication=criteria_after_deduplication,
            output_file=output_excel,
            total_cost_usd=self.cost_monitor.total_cost,

            anomaly_findings=anomaly_count,
            anomaly_high=anomaly_high,
            anomaly_warnings=anomaly_warnings,
            anomaly_info=anomaly_info,

            anomaly_numeric=anomaly_numeric,
            anomaly_units=anomaly_units,
            anomaly_terminology=anomaly_terminology,
            anomaly_missing_values=anomaly_missing_values,

            anomalies_file=anomalies_output,
        )

    @staticmethod
    def preview(
        output_excel: Path,
        n: int = 20,
    ) -> pd.DataFrame:
        return pd.read_excel(output_excel).head(n)