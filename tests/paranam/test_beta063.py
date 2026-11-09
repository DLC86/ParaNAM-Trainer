import math

import pytest
import torch

from nam.capture.params import KnobSpec, SwitchSpec, capture_recommendations
from nam.train.parametric import _EpochMetrics
from paranam import training
from paranam.workspace import Workspace, read_json


@pytest.mark.parametrize(
    "knobs,expected", [(1, 8), (2, 8), (3, 12), (5, 20), (6, 24), (9, 24)]
)
def test_validation_recommendations_per_joint_switch_state(knobs, expected):
    controls = [KnobSpec(f"K{i}", 0, 10) for i in range(knobs)]
    controls += [
        SwitchSpec("A", ("Off", "On")),
        SwitchSpec("B", ("One", "Two", "Three")),
    ]
    assert capture_recommendations(controls) == (16 * knobs * 6, expected * 6, 6)
    assert capture_recommendations([SwitchSpec("A", ("Off", "On"))]) == (2, 0, 2)


def test_validation_default_does_not_overwrite_saved_plans(tmp_path):
    ws = Workspace.create(tmp_path / "Rig", "Rig")
    assert ws.preferences["n_validation"] == 12
    ws.preferences["n_validation"] = 6
    ws.save()
    assert Workspace(ws.directory).preferences["n_validation"] == 6


def test_worst_is_per_complete_setting_and_does_not_weight_loud_captures():
    metrics = _EpochMetrics()
    # Quiet setting has ESR=1, loud setting ESR=.01. Pooling across settings
    # would hide the quiet failure; the maximum batch ESR would also be wrong.
    metrics.update(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 1.0]]), {}, 0)
    metrics.update(torch.tensor([[-1.0, 1.0]]), torch.tensor([[1.0, 1.0]]), {}, 0)
    metrics.update(torch.tensor([[9.0, 9.0]]), torch.tensor([[10.0, 10.0]]), {}, 1)
    assert metrics.compute()["ESR"] == pytest.approx(0.505)
    assert metrics.worst_esr() == 1
    ungrouped = _EpochMetrics()
    ungrouped.update(torch.ones(1, 2), torch.ones(1, 2), {})
    assert ungrouped.worst_esr() is None


def test_beta_publication_keeps_draft(monkeypatch):
    import json
    import runpy
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from paranam import __version__

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("GH_REPO", "DLC86/ParaNAM-Trainer")
    monkeypatch.setenv("RELEASE_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "source")
    monkeypatch.setenv("KEEP_DRAFT", "true")
    release = dict(
        draft=True,
        prerelease=True,
        target_commitish="source",
        tag_name="v0.6.3-beta",
        assets=[
            dict(
                name=f"ParaNAM-Trainer-{__version__}-" + suffix,
                state="uploaded",
                size=100,
            )
            for suffix in (
                "Windows-x64-Setup.exe",
                "Linux-amd64.deb",
                "Linux-x86_64.tar.gz",
                "macOS-arm64.dmg",
                "macOS-x86_64.dmg",
            )
        ],
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda args, **kw: json.dumps(
            release
            if args[-1].endswith("/123")
            else dict(full_name="DLC86/ParaNAM-Trainer")
        ),
    )
    edits = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: edits.append(a))
    monkeypatch.setitem(sys.modules, "build", SimpleNamespace(VERSION=__version__))
    monkeypatch.setattr(sys, "argv", ["publish_native.py", "finalize"])
    runpy.run_path(str(root / "packaging/publish_native.py"), run_name="__main__")
    assert not edits
