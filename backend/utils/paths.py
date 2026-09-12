"""Filesystem paths used by the active browser application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS = CONFIG_DIR / "default_settings.json"
LOG_DIR = PROJECT_ROOT / "logs"


def ensure_directories() -> None:
    LOG_DIR.mkdir(exist_ok=True)
