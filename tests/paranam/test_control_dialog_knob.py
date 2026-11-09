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
        dialog.avoid_zero,
        dialog.avoid_max,
    )
    assert form.rowCount() == len(expected)
    assert [
        form.itemAt(row, QFormLayout.ItemRole.FieldRole).widget()
        for row in range(form.rowCount())
    ] == list(expected)
    assert dialog.gain.text() == "Gain/Drive control"
    assert dialog.clean_volume.text() == "Linear volume control"
    assert dialog.avoid_zero.text() == "Avoid min"
    assert dialog.avoid_max.text() == "Avoid max"
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
    assert not dialog.avoid_zero.isEnabled()
    assert not dialog.avoid_max.isEnabled()
    dialog.clean_volume.click()
    assert dialog.gain.isEnabled()
    assert dialog.avoid_zero.isEnabled()
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
