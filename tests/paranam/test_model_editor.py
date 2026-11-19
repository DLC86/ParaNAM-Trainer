"""Model Editor: JSON metadata, control display order and safe file writes."""

import copy
import json
from pathlib import Path

import pytest

from nam.models.parametric._spec import ParamSpec
from paranam.model_editor_core import (
    apply_model_edits, editable_controls, load_model_file, write_model_file,
)


def sample_pnam(*, legacy=False, volume=True):
    params = [
        {
            "name": "Gain", "min": 0, "max": 10, "default": 5,
            "type": "continuous", "index": 1,
        },
        {
            "name": "Mode", "min": 0, "max": 2, "default": 1,
            "type": "switch", "index": 0,
            **(
                {"enum_names": ["Clean", "Crunch", "Lead"]}
                if legacy
                else {"options": [
                    {"name": "Clean", "value": 0, "index_sw": 2},
                    {"name": "Crunch", "value": 1, "index_sw": 0},
                    {"name": "Lead", "value": 2, "index_sw": 1},
                ]}
            ),
        },
    ]
    doc = {
        "version": "0.2.0",
        "architecture": "ParametricSlimmableContainer",
        "weights": [0.125, -0.25, 0.625],
        "metadata": {
            "loudness": -14.4, "gain": 0.6, "name": "Original",
            "modeled_by": "Old author", "custom_property": {"a": "b"},
        },
        "config": {
            "params": copy.deepcopy(params),
            "submodels": [
                {"name": size, "config": {"params": copy.deepcopy(params)},
                 "weight_offset": offset}
                for offset, size in ((0, "lite"), (100, "full"))
            ],
        },
    }
    if volume:
        doc["extensions"] = {
            "linear_volume": {
                "name": "Volume", "index": 2, "min": 0, "max": 10,
                "default": 10, "points": [[0, -60], [10, 0]],
            }
        }
    return doc


def test_editor_updates_only_display_metadata_and_all_hypera2_branches():
    doc = sample_pnam()
    original = copy.deepcopy(doc)
    rows = editable_controls(doc)
    assert [c["name"] for c in rows] == ["Mode", "Gain", "Volume"]
    assert sorted(row["key"] for row in rows if isinstance(row["key"], int)) == [0, 1]
    gain = next(c for c in rows if c["name"] == "Gain")
    mode = next(c for c in rows if c["name"] == "Mode")
    volume = next(c for c in rows if c["name"] == "Volume")
    gain["name"], gain["index"] = "Drive", 0
    mode["name"], mode["index"] = "Channel", 2
    volume["name"], volume["index"] = "Master", 1
    for option in mode["options"]:
        option["name"] += " v2"
        option["index_sw"] = (option["index_sw"] + 1) % 3

    edited = apply_model_edits(doc, {
        **doc["metadata"], "modeled_by": "New author", "input_z": 1_000_000,
        "gear_type": "pedal", "tone_type": "fuzz",
        "input_level_dbu": 12.5, "output_level_dbu": 7.0,
        "notes": "Editor test",
    }, rows)

    assert doc == original  # No in-place changes.
    assert edited["weights"] == doc["weights"]
    assert edited["config"]["submodels"][0]["weight_offset"] == 0
    assert edited["metadata"]["loudness"] == -14.4
    assert edited["metadata"]["custom_property"] == {"a": "b"}
    assert edited["metadata"]["gear_type"] == "pedal"
    assert edited["metadata"]["notes"] == "Editor test"
    assert edited["extensions"]["linear_volume"]["name"] == "Master"
    assert edited["extensions"]["linear_volume"]["points"] == [[0, -60], [10, 0]]
    for params in [
        edited["config"]["params"],
        *(branch["config"]["params"] for branch in edited["config"]["submodels"]),
    ]:
        assert [p["name"] for p in params] == ["Drive", "Channel"]
        assert [p["index"] for p in params] == [0, 2]
        assert params[0]["min"] == 0 and params[0]["max"] == 10
        spec = ParamSpec.from_dict(params[1])
        assert spec.enum_names == ("Clean v2", "Crunch v2", "Lead v2")
        assert spec.option_indices == (0, 1, 2)
        assert [o["value"] for o in params[1]["options"]] == [0, 1, 2]


def test_editor_refuses_normalization_changes_duplicate_names_and_neural_index_sw():
    doc = sample_pnam()
    original = editable_controls(doc)
    rows = copy.deepcopy(original)
    rows[0]["min"] = -10
    with pytest.raises(ValueError, match="min"):
        apply_model_edits(doc, doc["metadata"], rows)
    rows = copy.deepcopy(original)
    rows[0]["name"] = rows[1]["name"]
    with pytest.raises(ValueError, match="unique"):
        apply_model_edits(doc, doc["metadata"], rows)
    rows = copy.deepcopy(original)
    rows[0]["index"] = rows[1]["index"]
    with pytest.raises(ValueError, match="permutation"):
        apply_model_edits(doc, doc["metadata"], rows)
    rows = copy.deepcopy(original)
    switch = next(r for r in rows if r["type"] == "switch")
    switch["options"][0]["value"] = 10
    with pytest.raises(ValueError, match="neural values"):
        apply_model_edits(doc, doc["metadata"], rows)
    rows = copy.deepcopy(original)
    switch = next(r for r in rows if r["type"] == "switch")
    switch["options"][0]["index_sw"] = switch["options"][1]["index_sw"]
    with pytest.raises(ValueError, match="index_sw"):
        apply_model_edits(doc, doc["metadata"], rows)


def test_editor_migrates_legacy_switch_display_without_changing_numeric_values():
    doc = sample_pnam(legacy=True, volume=False)
    rows = editable_controls(doc)
    mode = next(row for row in rows if row["type"] == "switch")
    mode["options"][0]["name"] = "Off"
    mode["options"][0]["index_sw"] = 2
    mode["options"][2]["index_sw"] = 0
    changed = apply_model_edits(doc, doc["metadata"], rows)
    for params in (
        changed["config"]["params"],
        *(child["config"]["params"] for child in changed["config"]["submodels"]),
    ):
        switch = params[1]
        assert switch["enum_names"] == ["Off", "Crunch", "Lead"]
        assert [o["value"] for o in switch["options"]] == [0, 1, 2]
        assert [o["index_sw"] for o in switch["options"]] == [2, 1, 0]
        assert ParamSpec.from_dict(switch).enum_names == ("Off", "Crunch", "Lead")


def test_editor_save_copy_preserves_source_and_atomic_overwrite_keeps_backup(tmp_path):
    source = tmp_path / "Original.pnam"
    original = sample_pnam()
    source.write_text(json.dumps(original), encoding="utf-8")
    read = load_model_file(source)
    assert read == original
    copy_path = tmp_path / "Edited.pnam"
    changed = apply_model_edits(read, {**read["metadata"], "modeled_by": "New"}, editable_controls(read))
    assert write_model_file(source, copy_path, changed) == copy_path
    assert json.loads(source.read_text()) == original
    assert json.loads(copy_path.read_text())["metadata"]["modeled_by"] == "New"
    assert write_model_file(source, source, changed) == source
    assert json.loads((tmp_path / "Original.pnam.bak").read_text()) == original
    assert json.loads(source.read_text())["metadata"]["modeled_by"] == "New"
    with pytest.raises(ValueError, match="extension"):
        write_model_file(source, tmp_path / "Nope.nam", changed)


def test_editor_static_nam_metadata_preserves_submodel_loudness_and_weights():
    doc = {
        "architecture": "SlimmableContainer",
        "config": {"submodels": [
            {"model": {"metadata": {"loudness": -12, "gear_type": "amp"},
                       "weights": [1], "config": {}}},
            {"model": {"metadata": {"loudness": -13, "gear_type": "amp"},
                       "weights": [2], "config": {}}},
        ]},
        "metadata": {"loudness": -11, "gear_type": "amp"},
        "weights": [],
    }
    changed = apply_model_edits(doc, {"gear_make": "Fender", "gear_type": "amp"})
    assert changed["metadata"]["loudness"] == -11
    for child, db, weight in zip(changed["config"]["submodels"], [-12, -13], [1, 2]):
        model = child["model"]
        assert model["metadata"]["loudness"] == db
        assert model["metadata"]["gear_make"] == "Fender"
        assert model["weights"] == [weight]


def test_editor_keeps_child_specific_metadata_that_is_not_at_container_root():
    doc = {
        "architecture": "SlimmableContainer",
        "config": {"submodels": [
            {"model": {"metadata": {"gear_model": "One", "loudness": -18}}},
            {"model": {"metadata": {"gear_model": "Two", "loudness": -21}}},
        ]},
        "metadata": {"modeled_by": "Author"},
        "weights": [],
    }
    changed = apply_model_edits(doc, {"modeled_by": "Different author"})
    children = changed["config"]["submodels"]
    assert children[0]["model"]["metadata"]["gear_model"] == "One"
    assert children[1]["model"]["metadata"]["gear_model"] == "Two"
    assert children[0]["model"]["metadata"]["loudness"] == -18
    assert children[1]["model"]["metadata"]["loudness"] == -21
    assert all(
        item["model"]["metadata"]["modeled_by"] == "Different author"
        for item in children
    )


def test_editor_rejects_bad_model_files_and_invalid_metadata(tmp_path):
    path = tmp_path / "broken.pnam"
    path.write_text('{"metadata":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="structure"):
        load_model_file(path)
    path.write_text(json.dumps(sample_pnam()), encoding="utf-8")
    with pytest.raises(ValueError, match="ValidationError|greater"):
        apply_model_edits(load_model_file(path), {"input_z": -1})
