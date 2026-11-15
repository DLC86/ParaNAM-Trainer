"""Knob edit dialog order and mutually exclusive semantic roles."""

import pytest
from PySide6.QtWidgets import QFormLayout

from nam.capture.params import KnobSpec
from paranam.widgets import ControlDialog


def test_knob_options_follow_requested_order_and_labels(qapp):
    dialog = ControlDialog()
    form = dialog.pages.widget(0).layout()
    assert isinstance(form, QFormLayout)
    expected = (
        dialog.minimum,
        dialog.maximum,
        dialog.step,
        dialog.default,
        dialog.sampling_bias,
        dialog.gain,
        dialog.clean_volume,
        dialog.mute_at_min,
        dialog.mute_at_max,
    )
    assert form.rowCount() == len(expected)
    assert [
        form.itemAt(row, QFormLayout.ItemRole.FieldRole).widget()
        for row in range(form.rowCount())
    ] == list(expected)
    assert dialog.gain.text() == "Gain/Drive control"
    assert dialog.clean_volume.text() == "Linear volume control"
    assert dialog.mute_at_min.text() == "Mute at minimum"
    assert dialog.mute_at_max.text() == "Mute at maximum"
    assert "exclude" in dialog.mute_at_min.toolTip()
    assert "exclude" in dialog.mute_at_max.toolTip()
    label = form.itemAt(4, QFormLayout.ItemRole.LabelRole).widget()
    assert label.text() == "Capture density"
    dialog.close()


def test_gain_and_linear_volume_are_mutually_exclusive(qapp):
    dialog = ControlDialog()
    assert dialog.gain.isEnabled() and dialog.clean_volume.isEnabled()
    dialog.gain.click()
    assert dialog.gain.isChecked() and not dialog.clean_volume.isEnabled()
    assert not dialog.clean_volume.isChecked()
    dialog.gain.click()
    assert dialog.clean_volume.isEnabled()
    dialog.clean_volume.click()
    assert dialog.clean_volume.isChecked() and not dialog.gain.isEnabled()
    assert not dialog.gain.isChecked()
    assert not dialog.mute_at_min.isEnabled()
    assert not dialog.mute_at_max.isEnabled()
    dialog.clean_volume.click()
    assert dialog.gain.isEnabled()
    assert dialog.mute_at_min.isEnabled()
    assert dialog.mute_at_max.isEnabled()
    dialog.close()


@pytest.mark.parametrize("role", ["is_gain", "clean_volume"])
def test_existing_knob_role_disables_incompatible_checkbox(qapp, role):
    knob = KnobSpec("Volume", 0, 10, **{role: True})
    dialog = ControlDialog(control=knob)
    if role == "is_gain":
        assert dialog.gain.isChecked()
        assert not dialog.clean_volume.isEnabled()
    else:
        assert dialog.clean_volume.isChecked()
        assert not dialog.gain.isEnabled()
    dialog.close()


def test_schema_rejects_both_roles_even_without_the_dialog():
    with pytest.raises(ValueError, match="Gain/Drive"):
        KnobSpec("Volume", 0, 10, is_gain=True, clean_volume=True)


def test_mute_endpoints_roundtrip_and_implicitly_exclude_captures(qapp):
    dialog = ControlDialog()
    dialog.name.setText("Reverse Gain")
    dialog.mute_at_min.setChecked(True)
    dialog.mute_at_max.setChecked(True)
    dialog._accept()
    knob = dialog.result_control
    assert knob.mute_at_min and knob.mute_at_max
    restored = KnobSpec.from_dict(knob.to_dict())
    assert restored.mute_at_min and restored.mute_at_max
    assert not restored.avoid_zero and not restored.avoid_max
    specs = restored.to_param_spec()
    assert specs.avoid_zero and specs.avoid_max
    with pytest.raises(ValueError, match="Linear volume"):
        KnobSpec("Volume", 0, 10, clean_volume=True, mute_at_max=True)


def test_legacy_avoid_flags_survive_edit_without_enabling_mute_anchor(qapp):
    old = KnobSpec("Drive", 0, 10, avoid_zero=True, avoid_max=True)
    dialog = ControlDialog(control=old)
    assert not dialog.mute_at_min.isChecked()
    assert not dialog.mute_at_max.isChecked()
    dialog._accept()
    new = dialog.result_control
    assert new.avoid_zero and new.avoid_max
    assert not new.mute_at_min and not new.mute_at_max
    dialog.close()


def test_unchecking_mute_reenables_endpoint_without_legacy_avoid(qapp):
    dialog = ControlDialog(control=KnobSpec("Gain", 0, 10, mute_at_min=True))
    dialog.mute_at_min.setChecked(False)
    dialog._accept()
    knob = dialog.result_control
    assert not knob.mute_at_min
    assert not knob.to_param_spec().avoid_zero
    dialog.close()
