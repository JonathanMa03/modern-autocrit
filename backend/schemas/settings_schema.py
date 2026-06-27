"""
Application settings models.

These classes define all user-configurable settings for the Modern
AutoCrit application. Settings are intended to be saved to and loaded
from a JSON configuration file.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class OpenAISettings(BaseModel):
    """Configuration for OpenAI API usage."""

    api_key: str = ""
    model: str = "gpt-5"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = 4096


class ExtractionSettings(BaseModel):
    """Extraction pipeline options."""

    chunk_size: int = 200
    overlap: int = 50

    enable_normalization: bool = True
    enable_semantic_deduplication: bool = True

    save_intermediate_outputs: bool = False


class CostSettings(BaseModel):
    """API cost monitoring."""

    enable_tracking: bool = True

    warning_threshold_usd: float = 5.00
    hard_limit_usd: Optional[float] = None


class DirectorySettings(BaseModel):
    """Filesystem locations."""

    output_directory: Path = Path("outputs")
    download_directory: Path = Path("data")
    log_directory: Path = Path("logs")


class DictionarySettings(BaseModel):
    """Dictionary files used during normalization."""

    entity_dictionary: Path = Path(
        "config/dictionaries/entity_map.json"
    )

    attribute_dictionary: Path = Path(
        "config/dictionaries/attribute_map.json"
    )

    disease_dictionary: Path = Path(
        "config/dictionaries/disease_map.json"
    )


class AppSettings(BaseModel):
    """
    Top-level application settings.

    This object is intended to be saved directly to
    config/default_settings.json.
    """

    openai: OpenAISettings = OpenAISettings()

    extraction: ExtractionSettings = ExtractionSettings()

    cost: CostSettings = CostSettings()

    directories: DirectorySettings = DirectorySettings()

    dictionaries: DictionarySettings = DictionarySettings()