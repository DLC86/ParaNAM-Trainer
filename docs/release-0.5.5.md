# ParaNAM Trainer 0.5.5

- Added the optional **A2 — same cadence per optimizer update** learning-rate
  preset for parametric training. It initializes the audio LR to 0.004 and
  hypernetwork LR to 0.0004; both remain editable. The reference is one bundled
  standard input, official 10-second/final-nine-second split, batch 16, NY 8192
  and decay 0.994. The scheduler applies a discrete reduction after each
  reference-epoch number of optimizer updates, independently of capture count,
  current batch and NY. Normal per-epoch scheduling remains available.
- A header-only preview counts complete batches for each capture, including
  trimming, delay, receptive field and dropped partial batches. It shows total
  training audio, updates per epoch, equivalent decay and both final rates for
  the selected epoch budget. This matches a reference cadence, not an automatic
  selection of an optimal learning rate or training duration. Saved jobs retain
  their own scheduler and resume state.
- Restored Compare's original dial design. Parametric dials are slightly larger
  than Input/Output, inside a neutral grey amp-front panel. Values have no visible
  input box and remain editable. Double-click resets every dial to its original
  default, including after changing Full/Lite while preserving current settings.
- Model size and the new Output mode selector align with Input/Output labels and
  dial centres. Output modes are Raw, Calibrated (default, falling back to Raw
  without the required references), and Normalized (loudness metadata, otherwise
  Raw). They use the plugin's native calculations and can change during playback
  with a short gain transition.
- Export best NAM/PNAM also works while a job is paused or stopped. It loads the
  independent best Full/Lite checkpoints on CPU in the background, preserving
  metadata, calibration, ESR and resume files. At least one validated epoch is
  required. Running jobs retain their safe-boundary export behavior.
- Every newly generated capture plan includes the exact default control
  combination as a required training point within the requested count. Other
  Latin-hypercube points are constructed around its occupied strata, then
  deduplicated on the capture grid. Validation excludes defaults and corners.
  Each switch combination retains coverage and its own default-knob anchor.
  Existing recordings are retained until a plan is explicitly regenerated.

Windows x64, Linux x86-64 DEB/portable archive, and macOS Apple Silicon/Intel
installers remain self-contained. GPU support installed previously is reused.
The independent Full/Lite validation and held-out training-audio ESR introduced
in 0.5.4 remain in place.
