"""
Centralized filesystem paths.

Every part of the application should import paths from here rather than
constructing directories manually.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_SETTINGS = CONFIG_DIR / "default_settings.json"

DICTIONARY_DIR = CONFIG_DIR / "dictionaries"

ENTITY_MAP = DICTIONARY_DIR / "entity_map.json"

ATTRIBUTE_MAP = DICTIONARY_DIR / "attribute_map.json"

DISEASE_MAP = DICTIONARY_DIR / "disease_map.json"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOG_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create required directories if they do not already exist.
    """

    DATA_DIR.mkdir(exist_ok=True)

    OUTPUT_DIR.mkdir(exist_ok=True)

    LOG_DIR.mkdir(exist_ok=True)

    DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)