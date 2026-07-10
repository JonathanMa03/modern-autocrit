"""
Modern extraction engine.

This replaces the monolithic AutoCriteria script with a reusable backend
component built around the modern OpenAI SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.core.parser import parse_extraction_response
from backend.core.prompts import build_extraction_prompt
from backend.schemas.criteria_schema import EligibilityCriterion
from backend.schemas.settings_schema import ExtractionSettings, LLMSettings
from backend.services.clinicaltrials_service import ClinicalTrialsService
from backend.services.cost_monitor import CostMonitor
from backend.services.llm.factory import create_llm_provider
from backend.utils.logging_utils import get_logger
from backend.utils.text_utils import chunk_text_by_words
from backend.services.llm.factory import create_llm_provider


logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    trial_id: str
    criteria: list[EligibilityCriterion]
    errors: list[str]


class ModernAutoCritExtractor:
    """
    Main extraction engine.
    """

    def __init__(
        self,
        llm_settings: LLMSettings | None = None,
        extraction_settings: ExtractionSettings | None = None,
        cost_monitor: CostMonitor | None = None,
    ):
        self.llm_settings = llm_settings or LLMSettings()
        self.extraction_settings = extraction_settings or ExtractionSettings()
        self.cost_monitor = cost_monitor or CostMonitor()

        self.provider = create_llm_provider(
            settings=self.llm_settings,
            cost_monitor=self.cost_monitor,
        )

    def extract_from_text(
        self,
        criteria_text: str,
        trial_id: str,
        criteria_type: str,
        phase: str | None = None,
        url: str | None = None,
        disease_context: str | None = None,
    ) -> list[EligibilityCriterion]:
        """
        Extract criteria from one inclusion or exclusion text block.
        """

        chunks = chunk_text_by_words(
            criteria_text,
            chunk_size=self.extraction_settings.chunk_size,
            overlap=self.extraction_settings.overlap,
        )

        all_criteria: list[EligibilityCriterion] = []

        for i, chunk in enumerate(chunks, start=1):
            logger.info(
                "Extracting %s criteria for %s, chunk %s/%s",
                criteria_type,
                trial_id,
                i,
                len(chunks),
            )

            logger.info(
                "Text length for %s %s chunk %d: %d chars",
                trial_id,
                criteria_type,
                i,
                len(chunk),
            )

            # logger.info("Chunk preview:\n%s", chunk[:1000])

            prompt = build_extraction_prompt(
                criteria_text=chunk,
                criteria_type=criteria_type,
                disease_context=disease_context,
            )

            response = self.provider.generate(
                prompt,
                model=self.llm_settings.model,
                temperature=self.llm_settings.temperature,
                max_output_tokens=self.llm_settings.max_output_tokens,
            )

            response_text = response.text


            # logger.info(
            #     "Raw model response for %s %s chunk %d:\n%s",
            #     trial_id,
            #     criteria_type,
            #     i,
            #     response_text[:2000],
            # )

            parsed = parse_extraction_response(
                response_text=response_text,
                trial_id=trial_id,
                criteria_type=criteria_type,
                phase=phase,
                url=url,
            )

            logger.info("Parsed %d criteria from chunk %d", len(parsed), i)

            all_criteria.extend(parsed)

        return all_criteria

    def extract_from_trial_dict(
        self,
        trial: dict,
    ) -> ExtractionResult:
        """
        Extract criteria from a trial dictionary returned by
        ClinicalTrialsService.load_trial().
        """

        trial_id = trial.get("trial_id", "")
        phase = trial.get("phase")
        url = trial.get("url")
        eligibility_text = trial.get("eligibility_text", "")
        conditions = trial.get("conditions", [])

        disease_context = ", ".join(conditions) if conditions else None

        errors: list[str] = []
        criteria: list[EligibilityCriterion] = []

        if not eligibility_text:
            return ExtractionResult(
                trial_id=trial_id,
                criteria=[],
                errors=["No eligibility text found."],
            )

        inclusion_text, exclusion_text = self.split_inclusion_exclusion(
            eligibility_text
        )

        logger.info(
            "%s eligibility lengths | full=%d | inclusion=%d | exclusion=%d",
            trial_id,
            len(eligibility_text),
            len(inclusion_text),
            len(exclusion_text),
        )

        logger.info("Inclusion preview:\n%s", inclusion_text[:500])
        logger.info("Exclusion preview:\n%s", exclusion_text[:500])

        try:
            if inclusion_text:
                criteria.extend(
                    self.extract_from_text(
                        criteria_text=inclusion_text,
                        trial_id=trial_id,
                        criteria_type="Inclusion",
                        phase=phase,
                        url=url,
                        disease_context=disease_context,
                    )
                )
        except Exception as exc:
            msg = f"Inclusion extraction failed for {trial_id}: {exc}"
            logger.exception(msg)
            errors.append(msg)

        try:
            if exclusion_text:
                criteria.extend(
                    self.extract_from_text(
                        criteria_text=exclusion_text,
                        trial_id=trial_id,
                        criteria_type="Exclusion",
                        phase=phase,
                        url=url,
                        disease_context=disease_context,
                    )
                )
        except Exception as exc:
            msg = f"Exclusion extraction failed for {trial_id}: {exc}"
            logger.exception(msg)
            errors.append(msg)

        return ExtractionResult(
            trial_id=trial_id,
            criteria=criteria,
            errors=errors,
        )

    def extract_from_xml_directory(
        self,
        xml_directory: Path,
    ) -> list[ExtractionResult]:
        """
        Extract criteria from every XML file in a directory.
        """

        service = ClinicalTrialsService(xml_directory)
        results: list[ExtractionResult] = []

        for xml_path in service.xml_files():
            trial = service.load_trial(xml_path)
            result = self.extract_from_trial_dict(trial)
            results.append(result)

        return results

    @staticmethod
    def split_inclusion_exclusion(
        eligibility_text: str,
    ) -> tuple[str, str]:
        """
        Split eligibility text into inclusion and exclusion sections.

        This is intentionally conservative and can be improved later.
        """

        lower_text = eligibility_text.lower()

        inclusion_markers = [
            "inclusion criteria:",
            "inclusion criteria",
        ]

        exclusion_markers = [
            "exclusion criteria:",
            "exclusion criteria",
        ]

        inclusion_start = -1
        exclusion_start = -1

        for marker in inclusion_markers:
            idx = lower_text.find(marker)
            if idx != -1:
                inclusion_start = idx
                break

        for marker in exclusion_markers:
            idx = lower_text.find(marker)
            if idx != -1:
                exclusion_start = idx
                break

        if inclusion_start == -1 and exclusion_start == -1:
            return eligibility_text, ""

        if inclusion_start != -1 and exclusion_start != -1:
            if inclusion_start < exclusion_start:
                return (
                    eligibility_text[inclusion_start:exclusion_start],
                    eligibility_text[exclusion_start:],
                )

            return (
                eligibility_text[inclusion_start:],
                eligibility_text[exclusion_start:inclusion_start],
            )

        if inclusion_start != -1:
            return eligibility_text[inclusion_start:], ""

        return "", eligibility_text[exclusion_start:]