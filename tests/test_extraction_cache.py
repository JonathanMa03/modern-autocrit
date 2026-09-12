from backend.services.eligcrit_extraction import _cache_key


def test_cache_key_is_stable_for_identical_extraction_inputs():
    first = _cache_key(
        trial_id="NCT00006174",
        eligibility_text="Inclusion Criteria:\n- Age 1 to 8 years",
        model="gpt-4o-mini",
        library_sha256="abc",
    )
    second = _cache_key(
        trial_id="NCT00006174",
        eligibility_text="Inclusion Criteria:\n- Age 1 to 8 years",
        model="gpt-4o-mini",
        library_sha256="abc",
    )
    assert first == second
    assert first.startswith("NCT00006174_")


def test_cache_key_changes_with_text_model_or_library():
    base = dict(
        trial_id="NCT00006174",
        eligibility_text="Age 1 to 8 years",
        model="model-a",
        library_sha256="revision-a",
    )
    original = _cache_key(**base)
    assert _cache_key(**{**base, "eligibility_text": "Age 2 to 8 years"}) != original
    assert _cache_key(**{**base, "model": "model-b"}) != original
    assert _cache_key(**{**base, "library_sha256": "revision-b"}) != original
