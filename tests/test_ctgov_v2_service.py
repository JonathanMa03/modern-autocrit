from unittest.mock import Mock, patch

import pytest

from backend.services.ctgov_v2_service import (
    eligibility_text_from_study,
    fetch_study,
)


def test_fetch_study_uses_normalized_v2_study_endpoint():
    response = Mock(status_code=200)
    response.json.return_value = {"protocolSection": {}}
    response.raise_for_status.return_value = None
    with patch("backend.services.ctgov_v2_service.requests.get", return_value=response) as get:
        assert fetch_study("nct00006174") == {"protocolSection": {}}
    get.assert_called_once_with(
        "https://clinicaltrials.gov/api/v2/studies/NCT00006174",
        headers={"Accept": "application/json"},
        timeout=30.0,
    )


def test_fetch_study_rejects_invalid_identifier_without_request():
    with patch("backend.services.ctgov_v2_service.requests.get") as get:
        with pytest.raises(ValueError, match="valid NCT identifier"):
            fetch_study("06174")
    get.assert_not_called()


def test_extract_eligibility_text_from_v2_record():
    study = {
        "protocolSection": {
            "eligibilityModule": {"eligibilityCriteria": " Inclusion Criteria "}
        }
    }
    assert eligibility_text_from_study(study) == "Inclusion Criteria"
