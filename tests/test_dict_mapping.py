import json

import pandas as pd

import dict_mapping


def test_update_routes_terms_and_preserves_existing_mapping(tmp_path, monkeypatch):
    dictionary_dir = tmp_path / "dictionaries"
    dictionary_dir.mkdir()
    paths = {
        "attribute": dictionary_dir / "attribute_map.json",
        "disease": dictionary_dir / "disease_map.json",
        "entity": dictionary_dir / "entity_map.json",
    }
    paths["attribute"].write_text(
        json.dumps({"anc": "Absolute Neutrophil Count"}), encoding="utf-8"
    )
    paths["disease"].write_text("{}", encoding="utf-8")
    paths["entity"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dict_mapping, "DICTIONARY_PATHS", paths)

    output = tmp_path / "output.xlsx"
    pd.DataFrame(
        [
            {
                "entity": "Lab Test",
                "attribute": "ANC",
                "canonical_entity": "Lab Test",
                "canonical_attribute": "Wrong Replacement",
            },
            {
                "entity": "Diagnosis",
                "attribute": "NSCLC",
                "canonical_entity": "Diagnosis",
                "canonical_attribute": "Non-Small Cell Lung Cancer",
            },
            {
                "entity": "Vital Sign",
                "attribute": "Resting pulse",
                "canonical_entity": "Vital",
                "canonical_attribute": "Resting Pulse",
            },
        ]
    ).to_excel(output, index=False)

    summary = dict_mapping.update_dictionaries(output)

    assert summary.attribute_added == 1
    assert summary.disease_added == 1
    assert summary.entity_added == 3
    assert json.loads(paths["attribute"].read_text())["anc"] == (
        "Absolute Neutrophil Count"
    )
    assert json.loads(paths["attribute"].read_text())["resting pulse"] == (
        "Resting Pulse"
    )
    assert json.loads(paths["disease"].read_text())["nsclc"] == (
        "Non-Small Cell Lung Cancer"
    )


def test_reset_restores_all_snapshots(tmp_path, monkeypatch):
    active = tmp_path / "active"
    original = tmp_path / "original"
    active.mkdir()
    original.mkdir()
    paths = {}
    for name in ("attribute", "disease", "entity"):
        path = active / f"{name}_map.json"
        path.write_text('{"changed": "Changed"}', encoding="utf-8")
        (original / path.name).write_text(
            json.dumps({"original": name}), encoding="utf-8"
        )
        paths[name] = path

    monkeypatch.setattr(dict_mapping, "DICTIONARY_PATHS", paths)
    monkeypatch.setattr(dict_mapping, "ORIGINAL_DIR", original)

    dict_mapping.reset_dictionaries()

    for name, path in paths.items():
        assert json.loads(path.read_text()) == {"original": name}


def test_missing_required_columns_is_clear(tmp_path):
    output = tmp_path / "bad.xlsx"
    pd.DataFrame([{"entity": "Lab Test"}]).to_excel(output, index=False)

    try:
        dict_mapping.update_dictionaries(output)
    except ValueError as exc:
        assert "attribute" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
