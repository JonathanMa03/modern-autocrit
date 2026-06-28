from pathlib import Path

from backend.core.extractor import ModernAutoCritExtractor
from backend.core.normalizer import CriteriaNormalizer
from backend.schemas.settings_schema import OpenAISettings, ExtractionSettings
from backend.services.cost_monitor import CostMonitor
from backend.services.pipeline import ModernAutoCritPipeline


def main():
    xml_directory = Path("data/xml_trials")
    output_excel = Path("outputs/modern_autocrit_output.xlsx")

    cost_monitor = CostMonitor()

    openai_settings = OpenAISettings(
        model="gpt-4o-mini",
        temperature=0.0,
        max_output_tokens=4096,
    )

    extraction_settings = ExtractionSettings(
        chunk_size=200,
        overlap=50,
        enable_normalization=True,
        enable_semantic_deduplication=True,
    )

    extractor = ModernAutoCritExtractor(
        openai_settings=openai_settings,
        extraction_settings=extraction_settings,
        cost_monitor=cost_monitor,
    )

    normalizer = CriteriaNormalizer()

    pipeline = ModernAutoCritPipeline(
        extractor=extractor,
        normalizer=normalizer,
        cost_monitor=cost_monitor,
    )

    summary = pipeline.run(
        xml_directory=xml_directory,
        output_excel=output_excel,
    )

    print(summary)


if __name__ == "__main__":
    main()