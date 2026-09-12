from webapp.server import _cleaned_item


def test_cleaned_item_prefers_numerical_value_and_exposes_review():
    row = _cleaned_item(
        {
            "criteria": "inclusion",
            "item": "Age 18 to 65 years",
            "entity_name": "Demographic",
            "attribute_name": "Age",
            "attribute_id": "age",
            "value_type": "Numerical",
            "numerical_value": "[18, 65]",
            "unit": "year",
            "confidence": 0.97,
            "mapping_decision": "existing_attribute",
            "mapping_review": {"final_decision": "CONFIRM_EXISTING"},
            "review_status": "pending",
        }
    )
    assert row["value"] == "[18, 65]"
    assert row["attribute"] == "Age"
    assert row["mapping_review"] == "CONFIRM_EXISTING"
