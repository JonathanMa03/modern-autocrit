from backend.ctg_parser import parse_ctgov_study


def test_parse_ctgov_study_preserves_structured_fields():
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "nct00000001",
                "briefTitle": "Brief title",
                "officialTitle": "Official title",
                "acronym": "ABC",
            },
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2024-01"},
                "completionDateStruct": {"date": "2025-01"},
                "lastUpdatePostDateStruct": {"date": "2024-06-01"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Sponsor"}
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": "42", "type": "ESTIMATED"},
                "designInfo": {
                    "allocation": "RANDOMIZED",
                    "interventionModel": "PARALLEL",
                    "primaryPurpose": "TREATMENT",
                    "maskingInfo": {"masking": "DOUBLE"},
                },
            },
            "conditionsModule": {
                "conditions": ["Condition A"],
                "keywords": ["Keyword A"],
            },
            "armsInterventionsModule": {
                "armGroups": [{"label": "Arm A"}],
                "interventions": [
                    {
                        "type": "DRUG",
                        "name": "Drug A",
                        "description": "Drug description",
                        "armGroupLabels": ["Arm A"],
                    }
                ],
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Primary measure",
                        "description": "Primary description",
                        "timeFrame": "12 weeks",
                    }
                ],
                "secondaryOutcomes": [
                    {"measure": "Secondary measure"}
                ],
            },
            "eligibilityModule": {
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "80 Years",
                "healthyVolunteers": False,
                "eligibilityCriteria": "Inclusion Criteria:\n- Item 1",
            },
            "contactsLocationsModule": {
                "locations": [
                    {"country": "United States"},
                    {"country": "Canada"},
                    {"country": "United States"},
                ]
            },
        }
    }

    parsed = parse_ctgov_study(study, matched_query_ids=["q2", "q1", "q1"])

    assert parsed.nct_id == "NCT00000001"
    assert parsed.brief_title == "Brief title"
    assert parsed.conditions == ["Condition A"]
    assert parsed.interventions[0].name == "Drug A"
    assert parsed.interventions[0].arm_labels == ["Arm A"]
    assert [outcome.outcome_type for outcome in parsed.outcomes] == [
        "primary",
        "secondary",
    ]
    assert parsed.outcomes[0].time_frame == "12 weeks"
    assert parsed.enrollment_count == 42
    assert parsed.eligibility_text == "Inclusion Criteria:\n- Item 1"
    assert parsed.countries == ["Canada", "United States"]
    assert parsed.matched_query_ids == ["q1", "q2"]
    assert parsed.source_url.endswith("/NCT00000001")

