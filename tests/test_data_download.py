from pathlib import Path

import pytest

import data_download
from backend.services.clinicaltrials_service import ClinicalTrialsService


SAMPLE_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT12345678",
            "briefTitle": "Example Trial",
        },
        "designModule": {"phases": ["PHASE2"]},
        "conditionsModule": {"conditions": ["Lung Cancer"]},
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\nAge 18 years or older.\n"
                "Exclusion Criteria:\nActive infection."
            )
        },
    }
}


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = "response text"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise data_download.requests.HTTPError("request failed")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return next(self.responses)


def test_download_search_results_writes_pipeline_compatible_xml(tmp_path):
    session = FakeSession([FakeResponse({"studies": [SAMPLE_STUDY]})])

    paths = data_download.download_search_results(
        "lung cancer",
        1,
        tmp_path,
        session=session,
    )

    assert paths == [tmp_path / "NCT12345678.xml"]
    trial = ClinicalTrialsService(tmp_path).load_trial(paths[0])
    assert trial["trial_id"] == "NCT12345678"
    assert trial["title"] == "Example Trial"
    assert trial["phase"] == "PHASE2"
    assert trial["conditions"] == ["Lung Cancer"]
    assert "Age 18 years or older" in trial["eligibility_text"]


def test_search_studies_uses_pagination():
    second_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT87654321"}
        }
    }
    session = FakeSession(
        [
            FakeResponse({"studies": [SAMPLE_STUDY], "nextPageToken": "next"}),
            FakeResponse({"studies": [second_study]}),
        ]
    )

    studies = list(data_download.search_studies("cancer", 2, session=session))

    assert len(studies) == 2
    assert session.calls[1][1]["pageToken"] == "next"


def test_api_limit_has_clear_warning():
    session = FakeSession(
        [FakeResponse({}, status_code=429, headers={"Retry-After": "60"})]
    )

    with pytest.raises(data_download.ClinicalTrialsAPIError, match="usage limit"):
        list(data_download.search_studies("cancer", 1, session=session))


def test_large_noninteractive_download_requires_yes(monkeypatch, capsys):
    monkeypatch.setattr(data_download.sys.stdin, "isatty", lambda: False)
    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(data_download, "download_search_results", fake_download)

    result = data_download.main(["cancer", "100"])

    assert result == 1
    assert called is False
    assert "pass --yes" in capsys.readouterr().err
