"""Avoid/Unstable replacements vanish when their source becomes Mute or is recorded.

There is never an on-disk delete: recorded replacement WAVs are archived as
unused recordings, and other concurrently planned replacements are retained.
"""
from nam.capture.params import KnobSpec
from nam.capture.project import (
    CaptureProject, classify_capture, load_project, new_project, save_project,
)


def _project():
    return new_project(
        [KnobSpec("Gain", 0, 10, 1, 5), KnobSpec("Tone", 0, 10, 1, 5)],
        n_train=12, n_validation=4, lhs_method="sobol", seed=25,
    )


def test_avoid_to_mute_removes_only_its_replacement():
    original = _project()
    first, second = original.entries_for_split("train")[:2]
    avoided, added_a = classify_capture(original, first, "avoid")
    assert len(added_a) == 1
    assert added_a[0].replaces_avoided_capture == first.y_path
    # A second Unstable capture has its own independent replacement.
    avoided, added_b = classify_capture(avoided, second, "avoid")
    assert len(added_b) == 1
    assert added_b[0].replaces_avoided_capture == second.y_path
    before = len(avoided.entries)

    restored, created = classify_capture(avoided, first, "mute")
    assert created == []
    assert len(restored.entries) == before - 1
    assert first.params not in restored.avoided_settings
    assert first.params in restored.muted_settings
    assert not any(e.y_path == added_a[0].y_path for e in restored.entries)
    assert any(e.y_path == added_b[0].y_path for e in restored.entries)
    assert second.params in restored.avoided_settings
    assert restored.classification_for_entry(
        next(e for e in restored.entries if e.y_path == first.y_path)
    ) == "mute"


def test_successful_recapture_removes_replacement_and_keeps_recorded_take(tmp_path):
    project = _project()
    source = project.entries_for_split("validation")[0]
    avoided, added = classify_capture(project, source, "avoid")
    assert len(added) == 1
    replacement = added[0]
    # The extra capture may already have been physically recorded.
    replacement.status = "captured"
    replacement.delay = 0
    wav = tmp_path / replacement.y_path
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"recording must never be deleted")

    # Provenance survives normal project serialization and reloading.
    save_project(avoided, tmp_path)
    reloaded = load_project(tmp_path)
    row = next(e for e in reloaded.entries if e.y_path == source.y_path)
    replacement_row = next(e for e in reloaded.entries if e.y_path == replacement.y_path)
    assert replacement_row.replaces_avoided_capture == source.y_path
    row.status = "captured"  # happens after a successful physical capture
    reloaded.clear_capture_classification(row)

    assert reloaded.classification_for_entry(row) is None
    assert not reloaded.avoided_settings
    assert not any(e.y_path == replacement.y_path for e in reloaded.entries)
    assert replacement.y_path in reloaded.unused_recordings
    assert reloaded.unused_recordings[replacement.y_path]["context"] is None
    assert wav.read_bytes() == b"recording must never be deleted"


def test_original_recording_archive_is_not_discarded_by_switch_to_mute():
    project = _project()
    source = project.entries_for_split("train")[0]
    source.status, source.delay = "captured", 0
    avoided, added = classify_capture(project, source, "avoid")
    assert source.y_path in avoided.unused_recordings
    updated, created = classify_capture(avoided, source, "mute")
    assert created == []
    assert added[0].y_path not in {e.y_path for e in updated.entries}
    assert source.y_path in updated.unused_recordings
    assert updated.muted_settings == [source.params]
    assert not updated.avoided_settings


def test_untracked_legacy_dynamic_entries_are_not_removed():
    project = _project()
    source = project.entries_for_split("train")[0]
    avoided, added = classify_capture(project, source, "avoid")
    assert len(added) == 1
    legacy = next(e for e in avoided.entries if e.y_path == added[0].y_path)
    legacy.replaces_avoided_capture = None
    converted, _ = classify_capture(avoided, source, "mute")
    # No guessing based on ordering, filename prefixes, or knob positions.
    assert any(e.y_path == legacy.y_path for e in converted.entries)
