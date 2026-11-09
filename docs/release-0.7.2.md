# ParaNAM Trainer 0.7.2 preview

## Capture planning

### Four Latin Hypercube Sampling methods

The **Plan** page now offers four reproducible ways to distribute training
captures across the control space:

- **Standard** — the original seeded LHS, unchanged and still the default.
- **Maximin** — chooses the candidate LHS design with the greatest minimum
  separation between its points.
- **Orthogonal** — strength-2 orthogonal-array LHS, for sample counts and
  dimensions satisfying its mathematical constraints.
- **Near-orthogonal** — minimizes correlations between sampled dimensions
  without the stricter Orthogonal sample-count requirements.

The plan's seed and selected method are saved with the project. The default
anchor, sampling bias, control-step quantization, categorical switches and
corner captures can reduce the ideal properties of an initial LHS design.
Existing plans and recorded captures are not silently regenerated.

### Alternative validation selection: farthest from train

The **Validation selection** setting in Plan offers two methods:

- **Independent LHS (current)** — the existing held-out, independently seeded
  LHS split, still selected by default.
- **Farthest from train (maximin)** — first builds the training LHS, then
  greedily picks validation combinations from a reproducible, quantized
  candidate pool. Each new point maximizes its distance from the nearest
  training/default/corner point or previously selected validation point.
  Controls are normalized by their range; Simple Switch states are treated
  as categories and balanced, while Complex Switch combinations retain
  separate coverage.

This optional validation method probes less densely covered regions of the
control space. It is **not** an unbiased estimate of average accuracy over
all possible settings. Changing the selection requires regenerating the plan
and capturing the newly planned settings. Old projects retain Independent LHS.

### Updated capture-count recommendations

Recommended training LHS density rises from **15 to 16 captures per knob or
Simple Switch dimension**, with separate coverage for Complex Switch
combinations. The validation-count recommendation is unchanged. Existing
saved capture counts remain as they were.

## Training

### Alternative checkpoint selection for HyperA2

**Advanced Training Settings → Checkpoint selection** now provides three
independent Full/Lite checkpoint selection rules:

- **Lowest Validation ESR (current)** — the original behavior, kept as default.
- **Validation within 5%, then lowest Train ESR** — among checkpoints within
  5% of that size's final lowest Validation ESR, select the lowest Train ESR
  measured on held-out audio from training-control combinations. Checkpoint
  candidates that cannot win are pruned.
- **Combined ESR** — choose the checkpoint minimizing
  `alpha × Validation ESR + (1 − alpha) × Train ESR`.
  The configurable **Checkpoint alpha** ranges from **0 to 1** and defaults
  to **0.75**. It is enabled only for Combined ESR.

Full and Lite can be selected from different epochs, including in exports
and resumable snapshots. Neither optional criterion changes gradient
updates, the optimizer or learning-rate schedule. Existing parametric jobs
keep their original selection setting; static training is unchanged.

## Improvements and fixes

### Selected text colors in Captures and Queue

Selecting a row now preserves the original text colors throughout the
selected row, rather than replacing them with white. Running text stays
green, completed/captured text cyan and warning text yellow. The row's
selection background is explicitly rendered in neutral grey to avoid
system-specific blue highlights on Windows, Linux or macOS.

### Clean Volume: silence means no detected test signal

A muted minimum-volume recording is now recognized by the **absence of
the played timing-test signal** in the amplifier return, rather than
an arbitrary RMS/peak noise-floor threshold. Background hiss does not
by itself mean the test signal was detected.

- A verified silent minimum no longer produces a misleading
  peak-versus-RMS non-linearity warning.
- Verified muted minima map to **exactly zero output gain**, not the
  measured background-noise RMS.
- Earlier captures with saved raw timing recordings can be rechecked
  and obsolete warnings cleared without re-recording.
- Non-silent recordings retain the existing 0.20 dB discrepancy check.
- Loopback timing verification remains intact; missing expected timing
  signals at other knob settings still fail.

### Windows auto-update preserves installer choices

Removed the empty `/TASKS=` argument from the Windows auto-updater.
Inno Setup can now retain the user's previous installer task selections
instead of forcibly clearing them during a silent update. This includes
the desktop shortcut and optional NVIDIA GPU support. NVIDIA remains an
opt-in choice for new manual installations; existing GPU libraries are
not removed by an update that does not reinstall them.

## Compatibility

- Existing projects, captures and exported models remain supported.
- Independent validation LHS and the original lowest-Validation-ESR
  checkpoint rule remain the defaults.
- HyperA2 Full/Lite and static A2 model architectures are unchanged.
- Selecting new validation points requires a new capture plan; changing
  checkpoint selection requires a new training instance with matching
  settings for later resume.

**Full Changelog:** https://github.com/DLC86/ParaNAM-Trainer/compare/v0.7.1...v0.7.2
