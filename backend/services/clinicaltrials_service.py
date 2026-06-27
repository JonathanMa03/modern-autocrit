"""
ClinicalTrials.gov utilities.

Responsible for

    • downloading studies
    • loading XML studies
    • extracting eligibility criteria
    • extracting metadata

No LLM code belongs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup

from backend.utils.text_utils import clean_text


class ClinicalTrialsService:
    """
    Handles XML loading and metadata extraction.
    """

    def __init__(self, xml_directory: Path):
        self.xml_directory = Path(xml_directory)

    # ------------------------------------------------------------------
    # XML Discovery
    # ------------------------------------------------------------------

    def xml_files(self) -> list[Path]:
        """
        Return every XML file in the directory.
        """

        return sorted(self.xml_directory.glob("*.xml"))

    def iter_trials(self) -> Iterator[tuple[Path, BeautifulSoup]]:
        """
        Iterate through every XML trial.
        """

        for xml_file in self.xml_files():

            with open(xml_file, encoding="utf-8") as f:
                soup = BeautifulSoup(f, "xml")

            yield xml_file, soup

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def nct_id(soup: BeautifulSoup) -> str:

        tag = soup.find("nct_id")

        return tag.text.strip() if tag else ""

    @staticmethod
    def title(soup: BeautifulSoup) -> str:

        tag = soup.find("brief_title")

        return clean_text(tag.text) if tag else ""

    @staticmethod
    def phase(soup: BeautifulSoup) -> str:

        tag = soup.find("phase")

        return clean_text(tag.text) if tag else ""

    @staticmethod
    def condition(soup: BeautifulSoup) -> list[str]:

        return [
            clean_text(tag.text)
            for tag in soup.find_all("condition")
        ]

    @staticmethod
    def url(nct_id: str) -> str:

        return f"https://clinicaltrials.gov/study/{nct_id}"

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    @staticmethod
    def eligibility_text(soup: BeautifulSoup) -> str:
        eligibility = soup.find("eligibility")

        if eligibility is None:
            return ""

        criteria = eligibility.find("criteria")

        if criteria is None:
            return ""

        textblock = criteria.find("textblock")

        if textblock is None:
            return ""

        return clean_text(textblock.text)

    # ------------------------------------------------------------------
    # Trial Summary
    # ------------------------------------------------------------------

    def load_trial(self, xml_path: Path) -> dict:

        with open(xml_path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")

        nct = self.nct_id(soup)

        return {
            "trial_id": nct,
            "title": self.title(soup),
            "phase": self.phase(soup),
            "conditions": self.condition(soup),
            "url": self.url(nct),
            "eligibility_text": self.eligibility_text(soup),
        }