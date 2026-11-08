Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
from nam.capture.params import KnobSpec, SwitchSpec
from paranam.widgets import ControlDialog


def test_latency_combines_integer_and_fractional_samples():
    from types import SimpleNamespace
    from nam.capture.project import CaptureEntryModel, QAModel
    from nam.capture.session import CaptureSession
    from paranam.wizard import WizardWindow

    entry = CaptureEntryModel(index=0, split="train", params={}, y_path="test.wav")
    assert WizardWindow._capture_metrics(entry)[0] == "—"
    entry.delay = 4221
    assert WizardWindow._capture_metrics(entry)[0] == "4221 samples"
    entry.qa = QAModel(subsample_shift=-0.201)
    assert WizardWindow._capture_metrics(entry)[0] == "4221.201 samples"
    entry.qa.subsample_shift = 0.201
    assert WizardWindow._capture_metrics(entry)[0] == "4220.799 samples"
    entry.delay = None
    entry.qa.loopback_delay = 4221
    assert WizardWindow._capture_metrics(entry)[0] == "4221 samples"
    # Exercise the capture engine's actual sign convention, not a UI-only sum.
    entry.delay, shift = CaptureSession._alignment(
        SimpleNamespace(peak_delay=4247.201, delay=4221), True, 26.0
    )
    entry.qa.subsample_shift = shift
    assert WizardWindow._capture_metrics(entry)[0] == "4221.201 samples"


def test_capture_modes_keep_independent_batch_defaults(window, tmp_path):
    from paranam.workspace import Workspace

    ws = Workspace.create(tmp_path / "Defaults", "Defaults")
    window._attach(ws)
    assert window.batch_size.value() == 16
    window.batch_size.setValue(4)
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.batch_size.value() == 16
    assert ws.training_batch_size() == 16
    window.batch_size.setValue(8)
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("parametric"))
    assert window.batch_size.value() == 4
    window.capture_kind.setCurrentIndex(window.capture_kind.findData("static"))
    assert window.batch_size.value() == 8
    assert not hasattr(window, "static_size")


def test_gpu_button_reflects_installed_component(window, monkeypatch):
    from paranam import gpu_runtime

    if not hasattr(window, "gpu_setup_button"):
        return  # macOS uses Metal, without the optional NVIDIA component.
    monkeypatch.setattr(gpu_runtime, "ready", lambda: False)
    window.refresh_gpu_button()
    assert window.gpu_setup_button.text() == "Add NVIDIA GPU support"
    assert not window.gpu_setup_button.isHidden()
    monkeypatch.setattr(gpu_runtime, "ready", lambda: True)
    window.refresh_gpu_button()
    assert window.gpu_setup_button.isHidden()


def test_capture_list_expands_with_window(window, workspace, qapp):
    from paranam.widgets import RoutingDiagram

    window._attach(workspace)
    window.navigate(2)
    window.resize(1100, 850)
    window.show()
    qapp.processEvents()
    assert not window.findChildren(RoutingDiagram)
    original = window.capture_table.height()
    window.resize(1100, 1050)
    qapp.processEvents()
    assert window.capture_table.height() >= original + 100


def test_loopback_stays_enabled_when_mono_device_is_discovered(window, monkeypatch):
    from types import SimpleNamespace
    from paranam import wizard

    def device(channels):
        return SimpleNamespace(
            host_api="Test",
            name=f"Device {channels}",
            max_input_channels=channels,
            max_output_channels=channels,
        )

    monkeypatch.setattr(
        wizard,
        "capabilities",
        lambda info: SimpleNamespace(
            inputs=[f"Input {i}" for i in range(info.max_input_channels)],
            outputs=[f"Output {i}" for i in range(info.max_output_channels)],
            buffers=[512, 2048],
            default_buffer=2048,
            note="Test",
        ),
    )
    assert window.loop_enabled.isChecked()
    window._populate_audio_controls(device(1))
    assert window.loop_enabled.isChecked()
    window._populate_audio_controls(device(2))
    assert window.loop_enabled.isChecked()
    assert window.buffer.currentData() == 2048
    window.loop_enabled.setChecked(False)
    window._populate_audio_controls(device(2))
    assert not window.loop_enabled.isChecked()


def test_seven_pages_and_back_navigation_keep_drafts(window, workspace):
    window._attach(workspace)
    assert window.pages.count() == 7
    window.navigate(1)
    window.mode.setCurrentIndex(1)
    window.epochs.setValue(75)
    window.back()
    assert window.pages.currentIndex() == 0
    window.next()
    assert window.mode.currentData() == "active_learning"
    assert workspace.preferences["epochs"] == 75
    window.next()
    assert window.pages.currentIndex() == 2
    assert not window.next_button.isEnabled()
    assert not window.train_button.isEnabled()
    window.back()
    assert window.pages.currentIndex() == 1
    assert window.n_train.value() == 2


def test_switch_dialog_can_change_type_count_labels_and_default(qapp):
    dialog = ControlDialog(control=SwitchSpec("Mode", ("A", "B", "C"), 2))
    assert dialog.pages.currentIndex() == 1
    dialog.count.setValue(4)
    dialog.positions.item(3, 0).setText("Boost")
    dialog.switch_default.setCurrentIndex(3)
    dialog._accept()
    assert dialog.result_control.enum_names == ("A", "B", "C", "Boost")
    assert dialog.result_control.default == 3
    dialog.close()


def test_switch_coverage_recommendations_and_signal_selection(window, workspace):
    workspace.draft_controls = [
        KnobSpec("Gain", 0, 10).to_dict(),
        SwitchSpec("Mode", ("A", "B", "C")).to_dict(),
        SwitchSpec("Bright", ("Off", "On")).to_dict(),
    ]
    window._attach(workspace)
    window.refresh_recommendation(True)
    assert window.n_train.value() == 90
    assert window.n_validation.value() == 48
    assert "6 Complex Switch" in window.recommendation.text()
    assert not window.custom_button.isEnabled()
    window.signal_mode.setCurrentIndex(window.signal_mode.findData("custom"))
    assert window.custom_button.isEnabled()
    assert workspace.preferences["input_mode"] == "custom"


def test_capture_diagram_shows_numeric_and_named_switch_values(window, workspace):
    workspace.draft_controls.append(
        SwitchSpec("Mode", ("Off", "Clean", "Lead")).to_dict()
    )
    workspace.preferences.update(n_train=6, n_validation=3)
    workspace.generate_plan()
    window._attach(workspace)
    window.navigate(2)
    window.select_capture()
    assert window.scene.values["Mode"] in ("Off", "Clean", "Lead")
    assert isinstance(window.scene.values["Gain"], float)
    assert not window.record_button.isEnabled()
    window.rig_ready.setChecked(True)
    assert window.record_button.isEnabled()


def test_sidebar_handles_missing_custom_input_without_losing_the_project(
    window, workspace, monkeypatch
):
    window._attach(workspace)
    window.navigate(1)
    window.signal_mode.setCurrentIndex(window.signal_mode.findData("custom"))
    messages = []
    monkeypatch.setattr(
        window, "notify_error", lambda error: messages.append(str(error))
    )
    window.navigate(2)
    assert window.pages.currentIndex() == 1
    assert len(messages) == 1 and "custom" in messages[0]
    assert window.workspace is workspace


def test_capture_rows_record_the_displayed_setting_after_route_reordering(
    window, workspace, monkeypatch
):
    from nam.capture.project import CaptureEntryModel, KnobModel
    from paranam import wizard

    workspace.project.knobs = [
        KnobModel.from_knob_spec(KnobSpec("Gain", 0, 10, default=0))
    ]
    original = [
        CaptureEntryModel(
            index=i, split="train", params={"Gain": value}, y_path=f"captures/{i}.wav"
        )
        for i, value in enumerate([10, 0, 5])
    ]
    workspace.project.entries = original
    window._attach(workspace)
    assert window._capture_entries[0] is original[1]
    assert window.scene.values["Gain"] == 0
    from nam.capture.audio import DeviceInfo
    from paranam.audio_devices import Capabilities

    monkeypatch.setattr(
        wizard,
        "capabilities",
        lambda info: Capabilities(
            ("Return", "Loopback"), ("Send", "Loopback"), (2048,)
        ),
    )
    window.device.addItem("Test interface", DeviceInfo(0, "Test", "ASIO", 2, 2, 48000))
    window.device.setCurrentIndex(window.device.count() - 1)
    window.rig_ready.setChecked(True)
    recorded = []

    class Session:
        def capture_entry(self, capture, **kwargs):
            recorded.append(capture)

    monkeypatch.setattr(window, "save_audio", lambda: None)
    monkeypatch.setattr(window, "_get_session", lambda: Session())
    monkeypatch.setattr(wizard, "timebase_problem", lambda *args: None)
    monkeypatch.setattr(window, "run_audio", lambda action, done: action(None, None))
    window.record()
    assert recorded == [original[1]]
    assert workspace.project.entries == original

