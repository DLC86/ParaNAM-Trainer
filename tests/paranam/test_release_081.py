"""0.8.1 UI regressions: live capture recommendations and macOS form geometry."""

from nam.capture.params import KnobSpec


def test_sampling_combo_updates_recommended_counts_immediately(window, workspace):
    workspace.draft_controls = [
        KnobSpec(name, 0, 10, step=1).to_dict()
        for name in ("Gain", "Tone", "Drive")
    ]
    window._attach(workspace)
    window.refresh_recommendation(True)
    assert window.lhs_method.currentData() == "sobol"
    assert window.n_train.value() == 32
    window.lhs_method.setCurrentIndex(window.lhs_method.findData("standard"))
    assert window.n_train.value() == 48
    assert window.n_validation.value() == 12
    assert workspace.preferences["n_train"] == 48
    assert workspace.preferences["lhs_method"] == "standard"
    assert "Suggested: 48 sampled training" in window.recommendation.text()


def test_capture_counts_and_metadata_have_safe_layout_width(window, workspace):
    window._attach(workspace)
    assert window.n_train.minimumWidth() >= 132
    assert window.n_validation.minimumWidth() >= 132
    # Qt stylesheets may override the nominal 40 px minimum (e.g. to 38 px
    # with Qt 6.8 on Linux). Check actual legibility instead of this unstable
    # style-derived number.
    for field in (window.n_train, window.n_validation):
        assert field.minimumHeight() >= field.fontMetrics().height() + 16
        assert field.sizeHint().height() >= field.fontMetrics().height() + 16
    assert "Model format:" in window.model_format.text()
    assert window.custom_label.isReadOnly()
    assert window.custom_label.sizeHint().height() > 0
