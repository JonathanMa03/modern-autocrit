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


logger = get_logger(__name__)


@dataclass
class PipelineSummary:
    trials_processed: int
    criteria_extracted: int
    criteria_after_normalization: int
    criteria_after_deduplication: int
    output_file: Path | None
    total_cost_usd: float


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
        criteria_after_deduplication = len(deduped_df)

        FileManager.save_dataframe_excel(
            deduped_df,
            output_excel,
        )

        logger.info("Saved pipeline output to %s", output_excel)

        return PipelineSummary(
            trials_processed=trials_processed,
            criteria_extracted=criteria_extracted,
            criteria_after_normalization=criteria_after_normalization,
            criteria_after_deduplication=criteria_after_deduplication,
            output_file=output_excel,
            total_cost_usd=self.cost_monitor.total_cost,
        )

    @staticmethod
    def preview(
        output_excel: Path,
        n: int = 20,
    ) -> pd.DataFrame:
        return pd.read_excel(output_excel).head(n)