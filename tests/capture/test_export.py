import json as _json
from pathlib import Path as _Path

import pytest as _pytest

from nam.capture.export import (
    build_concat_learning_config as _build_concat_learning_config,
)
from nam.capture.export import build_concat_model_config as _build_concat_model_config
from nam.capture.export import build_data_config as _build_data_config
from nam.capture.export import (
    build_hyper_learning_config as _build_hyper_learning_config,
)
from nam.capture.export import build_hyper_model_config as _build_hyper_model_config
from nam.capture.export import concat_wavenet_channels as _concat_wavenet_channels
from nam.capture.export import update_data_json as _update_data_json
from nam.capture.export import (
    write_concat_training_configs as _write_concat_training_configs,
)
from nam.capture.export import (
    write_hyper_training_configs as _write_hyper_training_configs,
)
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.project import (
    CONCAT_LEARNING_CONFIG_FILENAME as _CONCAT_LEARNING_CONFIG_FILENAME,
)
from nam.capture.project import (
    CONCAT_MODEL_CONFIG_FILENAME as _CONCAT_MODEL_CONFIG_FILENAME,
)
from nam.capture.project import DATA_FILENAME as _DATA_FILENAME
from nam.capture.project import (
    HYPER_LEARNING_CONFIG_FILENAME as _HYPER_LEARNING_CONFIG_FILENAME,
)
from nam.capture.project import (
    HYPER_MODEL_CONFIG_FILENAME as _HYPER_MODEL_CONFIG_FILENAME,
)
from nam.capture.project import QAModel as _QAModel
from nam.capture.project import mark_captured as _mark_captured
from nam.capture.project import new_project as _new_project


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


def _project(**kwargs):
    defaults = dict(n_train=3, n_validation=2, seed=0)
    defaults.update(kwargs)
    return _new_project(_knobs(), **defaults)


def test_hyper_a2_configs_keep_project_controls_and_both_sizes(tmp_path):
    from nam.capture.export import (
        write_hyper_a2_training_configs,
        build_hyper_a2_learning_config,
    )
    from nam.models.parametric import HyperA2

    project = _project()
    paths = write_hyper_a2_training_configs(project, tmp_path)
    assert [p.name for p in paths] == ["model_hyper_a2.json", "learning_hyper_a2.json"]
    model_config = _json.loads(paths[0].read_text())
    assert model_config["net"]["name"] == "HyperA2"
    model = HyperA2.init_from_config(model_config["net"]["config"])
    assert model.param_names == ("Gain", "Tone")
    assert model.sizes == ("lite", "full")
    assert model.receptive_field == 6347
    learning = _json.loads(paths[1].read_text())
    assert learning == build_hyper_a2_learning_config(project)


def test_plan_display_indices_include_clean_volume_and_resumed_exports():
    from nam.capture.export import (
        apply_model_display_indices,
        build_hyper_a2_model_config,
    )
    from nam.capture.project import KnobModel

    project = _project()
    # Plan layout: Tone, Volume, Gain; neural training order: Tone, Gain.
    project.knobs = [
        project.knobs[1],
        KnobModel(name="Volume", min=0, max=10, step=0.5,
                  default=10, clean_volume=True),
        project.knobs[0],
    ]
    neural = build_hyper_a2_model_config(project)["net"]["config"]["params"]
    assert [(p["name"], p["index"]) for p in neural] == [("Tone", 0), ("Gain", 2)]

    # Old paused training jobs might still export param specs without indices.
    parent = [{"name": "Gain"}, {"name": "Tone"}]
    exported = {
        "architecture": "ParametricSlimmableContainer",
        "config": {
            "params": [dict(p) for p in parent],
            "submodels": [
                {"config": {"params": [dict(p) for p in parent]}}
                for _ in range(2)
            ],
        },
        "extensions": {"linear_volume": {"name": "Volume", "index": 0}},
    }
    apply_model_display_indices(
        exported, [knob.model_dump() for knob in project.knobs]
    )
    assert [(p["name"], p["index"]) for p in exported["config"]["params"]] == [
        ("Gain", 2), ("Tone", 0)
    ]
    for child in exported["config"]["submodels"]:
        assert child["config"]["params"] == exported["config"]["params"]
    assert exported["extensions"]["linear_volume"]["index"] == 1


def test_output_volume_hook_never_wraps_or_adds_a_neural_input():
    import pytest
    from nam.capture.output_gain import OutputGainExportHook

    native = {
        "architecture": "ParametricSlimmableContainer",
        "config": {"params": [{"name": "Gain", "index": 1}]},
        "weights": [0.25],
    }
    curve = {"name": "Volume", "index": 0, "schema_version": 1}
    hook = OutputGainExportHook(curve)
    result = hook.apply(native)
    assert result["architecture"] == "ParametricSlimmableContainer"
    assert result["config"]["params"] == [{"name": "Gain", "index": 1}]
    assert result["extensions"] == {"linear_volume": curve}
    assert result["weights"] == native["weights"]
    assert result is not native
    with pytest.raises(ValueError, match="Duplicate"):
        hook.apply(result)



def test_resumed_legacy_linear_volume_is_only_an_extension():
    from copy import deepcopy
    from nam.capture.export import apply_model_display_indices

    plan = [
        {"name": "Tone", "clean_volume": False},
        {"name": "Volume", "clean_volume": True},
        {"name": "Gain", "clean_volume": False},
    ]
    base = {
        "params": [{"name": "Gain"}, {"name": "Tone"}],
        "submodels": [
            {"config": {"params": [{"name": "Gain"}, {"name": "Tone"}]}}
            for _ in ("lite", "full")
        ],
    }
    original_curve = {"name": "Volume", "index": 0, "schema_version": 1}
    legacy = {
        "architecture": "ParametricOutputGain",
        "config": {
            "base_architecture": "ParametricSlimmableContainer",
            "base_config": deepcopy(base),
            "output_gain": deepcopy(original_curve),
        },
        "weights": [11.0, -5.0],
    }
    apply_model_display_indices(legacy, plan)
    assert legacy["architecture"] == "ParametricSlimmableContainer"
    assert "base_config" not in legacy["config"]
    assert "output_gain" not in legacy["config"]
    assert legacy["extensions"]["linear_volume"]["index"] == 1
    assert legacy["weights"] == [11.0, -5.0]
    assert [p["index"] for p in legacy["config"]["params"]] == [2, 0]
    assert all(
        branch["config"]["params"] == legacy["config"]["params"]
        for branch in legacy["config"]["submodels"]
    )
    # Re-applying to an already finalized file is safe and cannot duplicate Volume.
    apply_model_display_indices(legacy, plan, output_gain=original_curve)
    assert list(legacy["extensions"]) == ["linear_volume"]

    # Very old files that omitted their output stage can restore it solely
    # from the saved training recipe, never as an extra neural parameter.
    missing = {
        "architecture": "ParametricSlimmableContainer",
        "config": deepcopy(base),
        "weights": [11.0, -5.0],
    }
    apply_model_display_indices(missing, plan, output_gain=original_curve)
    assert missing["extensions"]["linear_volume"]["index"] == 1
    assert len(missing["config"]["params"]) == 2

    invalid = deepcopy(missing)
    invalid["config"]["params"][0]["name"] = "Volume"
    import pytest
    with pytest.raises(ValueError, match="Linear Volume"):
        apply_model_display_indices(invalid, plan)



def test_level_calibration_is_optional_and_exported_for_every_architecture(tmp_path):
    from nam.capture.export import build_hyper_a2_model_config
    from nam.capture.project import load_project, save_project
    from pydantic import ValidationError

    project = _project()
    for builder in (
        _build_concat_model_config,
        _build_hyper_model_config,
        build_hyper_a2_model_config,
    ):
        assert "metadata" not in builder(project)
    project.input_level_dbu = 18.0
    project.output_level_dbu = 12.5
    save_project(project, tmp_path)
    restored = load_project(tmp_path)
    assert restored.input_level_dbu == 18.0 and restored.output_level_dbu == 12.5
    for builder in (
        _build_concat_model_config,
        _build_hyper_model_config,
        build_hyper_a2_model_config,
    ):
        assert builder(restored)["metadata"] == {
            "input_level_dbu": 18.0,
            "output_level_dbu": 12.5,
        }
    invalid = project.model_dump()
    invalid["input_level_dbu"] = float("nan")
    with _pytest.raises(ValidationError):
        type(project).model_validate(invalid)


def test_build_data_config_includes_only_captured_entries():
    project = _project()
    train_entry = project.entries_for_split("train")[0]
    _mark_captured(train_entry, delay=7, qa=_QAModel())

    config = _build_data_config(project)

    assert config["type"] == "parametric"
    assert config["common"] == {"delay": 0}
    assert len(config["train"]) == 1
    assert config["validation"] == []
    assert config["train"][0]["y_path"] == train_entry.y_path

    # Pending entries never leak into either split.
    total_pending = len(project.pending_entries())
    assert total_pending == len(project.entries) - 1


def test_build_data_config_maps_paths_delay_and_window():
    project = _project()
    project.train_window.start_seconds = 1.0
    project.train_window.stop_seconds = 9.0
    project.train_window.ny = 4096
    project.validation_window.start_seconds = 2.0
    project.validation_window.stop_seconds = None
    project.validation_window.ny = 8192

    train_entry = project.entries_for_split("train")[0]
    validation_entry = project.entries_for_split("validation")[0]
    no_delay_entry = project.entries_for_split("train")[1]

    _mark_captured(train_entry, delay=15, qa=_QAModel())
    _mark_captured(validation_entry, delay=3, qa=_QAModel())
    _mark_captured(no_delay_entry, delay=None, qa=_QAModel())

    config = _build_data_config(project)

    train_by_path = {entry["y_path"]: entry for entry in config["train"]}
    validation_by_path = {entry["y_path"]: entry for entry in config["validation"]}

    train_row = train_by_path[train_entry.y_path]
    assert train_row["x_path"] == project.train_input
    assert train_row["delay"] == 15
    assert train_row["params"] == dict(train_entry.params)
    assert train_row["start_seconds"] == 1.0
    assert train_row["stop_seconds"] == 9.0
    assert train_row["ny"] == 4096

    validation_row = validation_by_path[validation_entry.y_path]
    assert validation_row["x_path"] == project.validation_input
    assert validation_row["delay"] == 3
    assert validation_row["start_seconds"] == 2.0
    assert validation_row["stop_seconds"] is None
    assert validation_row["ny"] == 8192

    no_delay_row = train_by_path[no_delay_entry.y_path]
    assert no_delay_row["delay"] == 0


def test_update_data_json_writes_file_matching_build_data_config(tmp_path: _Path):
    project = _project()
    entry = project.entries_for_split("train")[0]
    _mark_captured(entry, delay=4, qa=_QAModel())

    path = _update_data_json(project, tmp_path)

    assert path == tmp_path / _DATA_FILENAME
    assert path.is_file()
    with path.open() as fp:
        payload = _json.load(fp)
    assert payload == _build_data_config(project)


@_pytest.mark.parametrize(
    "builder,name",
    [
        (_build_concat_model_config, "ConcatWaveNet"),
        (_build_hyper_model_config, "HyperWaveNet"),
    ],
)
def test_build_model_config_shape_and_params_order(builder, name):
    project = _project()
    project.sample_rate = 48_000

    config = builder(project)

    assert config["net"]["name"] == name
    net_config = config["net"]["config"]

    params = net_config["params"]
    assert [p["name"] for p in params] == [knob.name for knob in project.knobs]
    for spec, knob in zip(params, project.knobs):
        assert spec["min"] == knob.min
        assert spec["max"] == knob.max
        assert spec["default"] == knob.default
        assert spec["step"] == knob.step
        assert spec["type"] == "continuous"
        assert "enum_names" not in spec
        assert "avoid_zero" not in spec

    assert net_config["sample_rate"] == 48_000.0
    assert isinstance(net_config["sample_rate"], float)

    assert net_config["head_scale"] == 0.01
    assert config["loss"]["val_loss"] == "esr"
    assert config["loss"]["mrstft_weight"] == 0.0005
    assert config["optimizer"]["weight_decay"] == 3.17e-07
    assert config["lr_scheduler"] == {
        "class": "ExponentialLR",
        "kwargs": {"gamma": 0.994},
    }


@_pytest.mark.parametrize(
    "builder", [_build_concat_model_config, _build_hyper_model_config]
)
def test_build_model_config_omits_sample_rate_when_unset(builder):
    project = _project()
    assert project.sample_rate is None

    config = builder(project)

    assert "sample_rate" not in config["net"]["config"]


def test_concat_model_config_derives_sizes_and_channels():
    project = _project()

    config = _build_concat_model_config(project)
    net_config = config["net"]["config"]

    # 8 + 2 per knob past the first; the two-knob project is the narrowest useful case.
    assert _concat_wavenet_channels(len(project.knobs)) == 10
    assert config["optimizer"]["lr"] == 0.003
    for layer in net_config["layers"]:
        assert layer["channels"] == 10
        assert "input_size" not in layer
        assert "condition_size" not in layer


def test_concat_wavenet_channels_matches_reference_five_knob_run():
    assert _concat_wavenet_channels(5) == 16


def test_hyper_model_config_template_and_hypernet():
    project = _project()

    config = _build_hyper_model_config(project)
    net_config = config["net"]["config"]

    assert config["optimizer"]["lr"] == 0.002
    assert net_config["hypernet"] == {
        "hidden_sizes": [16],
        "activation": "LeakyReLU",
        "selector": {"exclude_suffixes": ["_conv.weight"]},
    }
    # The knobs act through the hypernetwork, so the template stays a stock audio-in
    # channels_8 WaveNet regardless of knob count.
    for layer in net_config["layers"]:
        assert layer["channels"] == 8
        assert layer["input_size"] == 1
        assert layer["condition_size"] == 1


def test_build_model_config_is_loadable_by_its_net():
    from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
    from nam.models.parametric import HyperWaveNet as _HyperWaveNet

    project = _project()

    assert (
        _ConcatWaveNet.init_from_config(
            _build_concat_model_config(project)["net"]["config"]
        )
        is not None
    )
    assert (
        _HyperWaveNet.init_from_config(
            _build_hyper_model_config(project)["net"]["config"]
        )
        is not None
    )


def test_concat_model_config_silence_anchors_build_a_module():
    from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
    from nam.train.parametric import (
        _ParametricLightningModule as _Module,
    )
    from nam.train.parametric import _ParametricLossConfig as _LossConfig

    config = _build_concat_model_config(_project())

    loss_config = _LossConfig.init_from_config(config["loss"])
    assert loss_config.silence_anchor is not None
    assert loss_config.silence_anchor_ramp is not None

    net = _ConcatWaveNet.init_from_config(config["net"]["config"])
    # The ramp anchor rejects a net that cannot take a param trajectory at construction,
    # so this also pins that the generated pair is legal for the net it ships with.
    assert _Module(net, loss_config=loss_config) is not None


def test_hyper_model_config_has_no_ramp_anchor():
    # HyperWaveNet cannot be handed a param trajectory, so the ramp anchor would raise.
    assert "silence_anchor_ramp" not in _build_hyper_model_config(_project())["loss"]


@_pytest.mark.parametrize(
    "builder,gradient_clip_val",
    [
        (_build_concat_learning_config, 0.5),
        (_build_hyper_learning_config, 1.0),
    ],
)
def test_build_learning_config_matches_its_architecture(builder, gradient_clip_val):
    project = _project()

    config = builder(project)

    assert config["trainer"]["accelerator"] == "auto"
    assert config["trainer"]["max_epochs"] == 200
    assert config["trainer"]["gradient_clip_val"] == gradient_clip_val
    assert config["train_dataloader"]["batch_size"] == 32
    assert config["train_dataloader"]["num_workers"] == 0
    assert config["val_dataloader"]["batch_size"] == 32
    assert config["torch_compile"] == {"enabled": True, "mode": "reduce-overhead"}


@_pytest.mark.parametrize(
    "writer,model_filename,learning_filename,model_builder,learning_builder",
    [
        (
            _write_concat_training_configs,
            _CONCAT_MODEL_CONFIG_FILENAME,
            _CONCAT_LEARNING_CONFIG_FILENAME,
            _build_concat_model_config,
            _build_concat_learning_config,
        ),
        (
            _write_hyper_training_configs,
            _HYPER_MODEL_CONFIG_FILENAME,
            _HYPER_LEARNING_CONFIG_FILENAME,
            _build_hyper_model_config,
            _build_hyper_learning_config,
        ),
    ],
)
def test_write_training_configs_writes_both_files(
    tmp_path: _Path,
    writer,
    model_filename,
    learning_filename,
    model_builder,
    learning_builder,
):
    project = _project()

    paths = writer(project, tmp_path)

    assert paths == [tmp_path / model_filename, tmp_path / learning_filename]
    for path in paths:
        assert path.is_file()

    with (tmp_path / model_filename).open() as fp:
        assert _json.load(fp) == model_builder(project)
    with (tmp_path / learning_filename).open() as fp:
        assert _json.load(fp) == learning_builder(project)


def test_both_architectures_export_side_by_side(tmp_path: _Path):
    project = _project()

    _write_concat_training_configs(project, tmp_path)
    _write_hyper_training_configs(project, tmp_path)

    written = {path.name for path in tmp_path.iterdir()}
    assert written == {
        _CONCAT_MODEL_CONFIG_FILENAME,
        _CONCAT_LEARNING_CONFIG_FILENAME,
        _HYPER_MODEL_CONFIG_FILENAME,
        _HYPER_LEARNING_CONFIG_FILENAME,
    }
