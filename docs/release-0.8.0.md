# ParaNAM Trainer 0.8.0

Changes since **0.7.7**.

## Rules and capture planning

- Renamed **Avoid rules** to **Rules**. Each rule is now either **Avoid** (unsafe or unstable settings) or **Mute** (a region known to produce silence). Rules support independent operators and thresholds for every selected knob/switch, with AND semantics. Existing Avoid rules remain compatible.
- Virtual Mute Anchors train silence at combinations matching Mute rules (e.g. Gain I = 0 AND Gain II = 0, or Bass = 0 AND Mid = 0 AND Treble = 0), with unconstrained controls randomized. Muted regions are excluded from real capture planning without requiring a separate network or player-format modification.
- Mute and Unstable capture tags appear in blue instead of warning yellow; the column is now **Tags/Warnings**. Unstable replacements are required to be distinct from the classified setting and all other reserved plan points.
- The Capture page shows a white travel segment between the previously recorded and currently selected knob positions, pulsing transparently in sync with the tile highlight. At the dimmest frame only the underlying red travel arc remains; the effect stops when **My rig matches…** is confirmed.

## Training queue and projects

- A queue job can be right-clicked to change its total epoch target, including after training has completed. Resume may continue from the saved checkpoint when the target exceeds completed epochs; the target cannot be reduced below the completed count. Existing checkpoints and captured audio remain intact.
- Retrain opens an editable, independent snapshot without adding another entry to the Projects list.
- Duplicate project names use consecutive numeric suffixes: copying a project ending in ` (2)` produces ` (3)`, not ` (2) (2)`.

## Noise-aware training losses

- **RMS loss normalization and SNR-weighted loss normalization are enabled by default** for new parametric training presets. Saved choices in existing projects are preserved.
- RMS normalization still affects **MSE only**, using the capture power and the existing adjustable RMS exponent.
- SNR uses the pre-blip raw recording noise estimate, applying a capped **linear signal-reliability factor** (minimum 10%) to MSE. It also attenuates MRSTFT with a more conservative **square-root reliability factor** (minimum 20%). This prevents very quiet captures from receiving nearly zero gradient while downweighting especially noisy spectra.
- Imported or legacy recordings without a raw pre-blip return are still usable: their SNR contribution is left at unity and the missing measurement is explicitly recorded rather than fabricated.
- These weights affect training, not the player/model format. Validation metrics remain unweighted.

## Distribution

- Build version **0.8.0** for Windows x64, Linux (.deb and .tar.gz), macOS ARM64 and macOS x86_64. Existing draft release is retagged as `v0.8.0`, its five native assets replaced, and the release stays **draft/prerelease**, not published.
