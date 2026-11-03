# ParaNAM Trainer 0.5.3

- Recent projects use two properly measured lines: a larger bold name and a
  smaller regular path, without the stylesheet clipping or restyling the text.
  The project selector is taller and fills the available Projects-page space.
- Opening or creating a project automatically opens Plan.
- Sidebar pages are Projects, Plan, Capture, Train, Queue and Compare.
  Settings is pinned below them with a gear icon. The two sidebar slogans are removed.
- The bold ParaNAM Trainer title and `VERSION 0.5.3` share a left edge.
  The TTS logo spans the complete title/version block.
- Audio routing, buffer and current level calibration are configured globally in
  Settings, including before opening a project. Switching projects does not
  restore their former hardware setup. Existing installations migrate once from
  the last project; subsequent changes use only global settings.
- Dataset calibration remains historical: send, return, capture output level and
  Instrument reference used for capture are saved with the project, and each new
  job saves its own `capture_calibration` snapshot. Changing Settings does not
  alter these stored values. Capture projects may still contain routing history
  from a recording; it is not the source of the application's current settings.
- Static Capture contains only the model Name and batch-capture actions. Other
  metadata is edited in Train, where Name is hidden for Static and remains visible
  for Parametric. Editing shared static metadata supersedes old exposed per-capture
  overrides while preserving unexposed/custom metadata.
- Increasing window height grows both the queue table and the chart/log area.

The existing training recipes, nested Full/Lite outputs, input signals and
playback levels remain unchanged. Physical recording/listening still needs
verification on an actual audio interface.
