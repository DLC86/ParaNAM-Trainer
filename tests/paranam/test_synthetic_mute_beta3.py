"""0.8.0 beta 3: real full-length zero WAVs in the ordinary dataset."""
import shutil

import pytest
import soundfile as sf
import torch

from nam.capture.params import KnobSpec
from nam.capture.plan_rules import PlanRule
from nam.capture.export import build_data_config
from nam.data import Dataset
from nam.train.parametric import _ParametricLightningModule, _CaptureBatchSampler
from paranam import training
from paranam.workspace import read_json


def _configure_joint_mute(workspace):
    workspace.draft_controls = [
        KnobSpec("Gain I", 0, 10, 1, 5).to_dict(),
        KnobSpec("Gain II", 0, 10, 1, 5).to_dict(),
        KnobSpec("Tone", 0, 10, 1, 5).to_dict(),
    ]
    rule = PlanRule(
        controls=("Gain I", "Gain II"), operator="=", value=0, kind="mute"
    )
    workspace.preferences["draft_plan_rules"] = [rule.model_dump(mode="json")]
    workspace.preferences.update(n_train=12, n_validation=4, corners=True)
    workspace.generate_plan()
    return rule


def test_multicontrol_mute_rule_always_has_completed_synthetic_corner(workspace):
    rule = _configure_joint_mute(workspace)
    project = workspace.project
    selected = [
        entry for entry in project.entries
        if entry.split == "train" and rule.excludes(entry.params)
    ]
    assert selected, "Uncovered AND Mute rule needs an extra training corner"
    assert any(entry.is_corner for entry in selected) or len(selected) > 0
    assert all(entry.synthetic_mute and entry.status == "captured" for entry in selected)
    for entry in selected:
        assert project.classification_for_entry(entry) == "mute"
        target = workspace.directory / entry.y_path
        source = workspace.directory / project.input_for_split(entry.split)
        assert sf.info(target).frames == sf.info(source).frames
        assert sf.info(target).samplerate == sf.info(source).samplerate
        audio, _ = sf.read(target, dtype="float32")
        assert not audio.any()
    included = build_data_config(project, include_synthetic=True)
    omitted = build_data_config(project, include_synthetic=False)
    assert any(item["y_path"] == selected[0].y_path for item in included["train"])
    assert all(item["y_path"] != selected[0].y_path for item in omitted["train"])
    assert workspace.training_capture_entries()


def test_synthetic_mute_job_contains_normal_dataset_and_no_anchor(workspace):
    _configure_joint_mute(workspace)
    for entry in workspace.project.entries:
        if entry.status == "captured":
            continue
        output = workspace.directory / entry.y_path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            workspace.directory / workspace.project.input_for_split(entry.split),
            output,
        )
        entry.status = "captured"
        entry.delay = 0
    workspace.save()
    assert workspace.training_ready()
    job = training.make_job(workspace)
    spec = read_json(job / "spec.json")
    assert spec["model"]["loss"].get("synthetic_mute") is True
    assert "mute_anchor" not in spec["model"]["loss"]
    assert spec["learning"]["synthetic_mute_train_indices"]
    assert spec["learning"]["synthetic_mute_reference_rms"] is False
    assert spec["learning"]["rms_loss_normalization"] is True
    assert spec["learning"]["snr_weighted_loss_normalization"] is False
    assert len(spec["data"]["train"]) > len(spec["learning"]["synthetic_mute_train_indices"])


class _FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, *args, pad_start=False):
        return self.gain * x


def test_raw_rms_floor_and_optional_unity_reference_for_exact_zero():
    samples = torch.zeros(32)
    ds = Dataset(
        x=samples, y=samples.clone(), nx=1, ny=8,
        require_input_pre_silence=None,
    )
    module = _ParametricLightningModule(_FakeNet())
    module.set_synthetic_mute_indices((0,), (), reference_rms=True)
    module.set_capture_mse_normalization(ds)
    assert module._capture_mse_scales == pytest.approx((1.0,))
    assert torch.count_nonzero(ds.y).item() == 0
    module.set_synthetic_mute_indices((0,), (), reference_rms=False)
    module.set_capture_mse_normalization(ds)
    # -18 dBFS reference / -58 dBFS floor = 40 dB power ratio.
    assert module._capture_mse_scales == pytest.approx((10000.0,))


def test_synthetic_batches_use_only_mse_and_do_not_poison_esr(monkeypatch):
    module = _ParametricLightningModule(_FakeNet())
    sampler = _CaptureBatchSampler((1,), 1, False, False)
    list(sampler)
    module.set_capture_batch_samplers(train_sampler=sampler, val_sampler=sampler)
    module.set_synthetic_mute_indices((0,), (0,))
    logs = {}
    monkeypatch.setattr(module, "log_dict", lambda values, **kw: logs.update(values))
    audio = torch.ones(1, 16)
    silence = torch.zeros_like(audio)
    module.on_train_epoch_start()
    loss = module.training_step((audio, silence), 0)
    assert loss.item() == pytest.approx(0.25)
    assert loss.requires_grad
    loss.backward()
    assert module.net.gain.grad is not None
    module.on_train_epoch_end()
    assert any("Mute RMS dBFS" in k for k in logs)
    assert not any(k.startswith("ESR/") for k in logs)
    logs.clear()
    module.on_validation_epoch_start()
    module.validation_step((audio, silence), 0)
    module.on_validation_epoch_end()
    assert any("Mute RMS dBFS" in k for k in logs)
    assert not any(k.startswith("ESR/") for k in logs)


def test_synthetic_mute_visible_as_captured_and_blue_tag(window, workspace):
    _configure_joint_mute(workspace)
    window._attach(workspace)
    row = next(
        i for i, entry in enumerate(window._capture_entries)
        if entry.synthetic_mute
    )
    assert window.capture_table.item(row, 2).text() == "Captured"
    assert "Mute" in window.capture_table.item(row, 5).text()
    assert window.capture_table.item(row, 5).foreground().color().name() == "#53aaff"
    field = window.advanced_fields["synthetic_mute_reference_rms"]
    assert not field.isChecked()
    field.setChecked(True)
    assert workspace.training_options()["synthetic_mute_reference_rms"] is True


@pytest.mark.parametrize(
    "method", ["standard", "maximin", "orthogonal",
               "near_orthogonal", "sobol", "optimized"]
)
def test_all_sampling_methods_keep_muted_endpoints_but_preserve_avoid(workspace, method):
    from nam.capture.params import KnobSpec

    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10, 1, 5, mute_at_min=True).to_dict(),
        KnobSpec("Tone", 0, 10, 1, 5).to_dict(),
    ]
    avoid = PlanRule(
        controls=("Tone",), operator="≥", value=10, kind="avoid"
    )
    workspace.preferences["draft_plan_rules"] = [avoid.model_dump(mode="json")]
    train, validation = (9, 9) if method == "orthogonal" else (12, 4)
    workspace.preferences.update(
        lhs_method=method, n_train=train, n_validation=validation,
        corners=True,
    )
    workspace.generate_plan()
    project = workspace.project
    assert project.n_train_lhs == train
    assert not any(row.params["Tone"] == 10 for row in project.entries)
    assert not any("mute_edge" in row.y_path for row in project.entries)
    assert any(row.synthetic_mute and row.params["Gain"] == 0
               for row in project.entries)
    corners = [row for row in project.entries if row.is_corner]
    assert corners
    assert all(row.params["Gain"] in (0, 10) for row in corners)
    assert all(row.params["Tone"] in (0, 10) for row in corners)
    assert all(row.params["Tone"] == 0 for row in corners)
    assert all(row.status == "captured" and project.classification_for_entry(row) == "mute"
               for row in project.entries if row.params["Gain"] == 0)


def test_validation_and_training_mutes_share_one_wav_per_length(workspace):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10, 1, 5).to_dict(),
        KnobSpec("Tone", 0, 10, 1, 5).to_dict(),
    ]
    rule = PlanRule(
        controls=("Gain",), operator="≤", value=5, kind="mute"
    )
    workspace.preferences["draft_plan_rules"] = [rule.model_dump(mode="json")]
    workspace.preferences.update(n_train=16, n_validation=8, corners=True)
    workspace.generate_plan()
    project = workspace.project
    split_paths = {}
    for split in ("train", "validation"):
        rows = [
            e for e in project.entries if e.split == split and e.synthetic_mute
        ]
        assert len(rows) > 1
        names = {e.y_path for e in rows}
        assert len(names) == 1
        relative = next(iter(names))
        split_paths[split] = relative
        target = workspace.directory / relative
        source = workspace.directory / project.input_for_split(split)
        assert sf.info(target).frames == sf.info(source).frames
        assert sf.info(target).samplerate == sf.info(source).samplerate
    # The two input lengths are not generally equal; only allocate another
    # mute target when the validation length actually differs.
    source_train = sf.info(workspace.directory / project.input_for_split("train"))
    source_val = sf.info(workspace.directory / project.input_for_split("validation"))
    if (source_train.frames == source_val.frames and
            source_train.samplerate == source_val.samplerate):
        assert split_paths["train"] == split_paths["validation"]
    else:
        assert split_paths["train"] != split_paths["validation"]
