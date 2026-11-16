# ParaNAM Trainer 0.8.0

**Complete changes since 0.7.6**, including everything introduced in 0.7.7 and 0.8.0.

## New and updated in 0.8.0

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

- Build version **0.8.0** for Windows x64, Linux (.deb and .tar.gz), macOS ARM64 and macOS x86_64. The existing 0.7.7 release draft is renamed and retagged as **0.8.0**; its five installer assets are replaced after successful native builds and verification. The release stays **draft/prerelease** until explicitly published.

## Changes introduced in 0.7.7 (included in 0.8.0)

The following earlier improvements are also included. Where 0.8.0 changes a default or algorithm (notably RMS/SNR normalization and Rules), the 0.8.0 behavior above supersedes the historical description below.

## Capture planning and controls

- **New sampling methods: Sobol and Optimized LHS.** Scrambled Sobol provides low-discrepancy multidimensional coverage, with the strongest balance for power-of-two capture counts; arbitrary counts are supported. Optimized LHS uses SciPy's centered-discrepancy optimization (`random-cd`) to improve low-dimensional projections. Both are selectable on the **Plan** page, reproducible from the saved seed and available for training and independently sampled validation.
- **Plan interface:** the former “LHS design” selector is now labelled **Capture sampling method**, and the training count is labelled **Training captures** to include both LHS and Sobol designs.
- **Sobol is now the default for new projects:** Standard LHS, Maximin LHS, Orthogonal LHS, Near-orthogonal LHS and Optimized LHS remain selectable. Existing projects keep their saved method and capture plans unchanged. New two-knob projects start with **32 sampled training + 8 validation** captures.
- **Lower Sobol training recommendations and automatic 25% validation:** 2 knobs → 32 training / 8 validation; 3 → 32 / 8; 4 → 64 / 16; 5 → 64 / 16; 6 → 64 / 16. Beyond six knobs, training grows by 32 for every two more controls (7–8 → 96 / 24, 9–10 → 128 / 32), staying below the previous 16-per-control training heuristic. For **all sampling methods** (Sobol and every LHS variant), suggested validation is now one quarter of training, rounded to the nearest integer with half values rounded up. Editing the Training captures field automatically recalculates Validation captures (for example, 32 → 64 updates 8 → 16), while Validation remains independently editable. Grid limits, reserved corners/defaults and Complex Switch coverage can reduce the achievable ratio; switch-only plans keep their separate-audio validation policy. The optional Default and Corner captures are additional.
- **Sobol Simple/Complex Switch strategies:** newly added switches default to **Simple Switch**. With 2–4 joint Simple states, one continuous-knob Sobol sequence is partitioned into consecutive power-of-two blocks, dividing rather than multiplying the training budget. With 5+ states, balanced categorical combinations favor pairwise state coverage alongside Sobol knob settings. Complex Switches still multiply capture totals by their number of states, use separately scrambled reproducible Sobol designs and avoid reusing knob settings across states when grid capacity permits. LHS strategies remain unchanged.
- **Capture classifications stay visible:** **Mark as mute** and **Mark as unstable** leave every marked capture in the table, annotated as **Mute** or **Unstable** in **Warnings**. Marked WAVs are excluded from training until re-recorded; a successful new recording clears that mark. The redundant **Undo** button is removed. Existing project marks are preserved across reopening and plan regeneration.
- **Restored and extended Avoid Rules:** a dedicated **Avoid Rules** section is visible on the Plan screen. Rules can now specify different comparison operators and numeric thresholds per selected knob (for example, **Gain > 8 AND Bias > 7**), optionally combined with switch positions. Only combinations matching every condition are excluded. Saved legacy single-threshold rules remain compatible, and rule conditions survive control renaming. Rules are edited only in Plan, not inferred or modified during Capture.
- **No generic replacement for Mute:** marking an ordinary training or validation capture as **Mute** creates an exact virtual Mute Anchor without scheduling another recording. Muted corners may still add the nearest missing boundary-transition captures, one control at a time. **Unstable** continues to exclude and replace the rejected recording with a fresh valid Sobol/LHS point in the same split.

- **Optional extra Default capture:** the new **Include default as additional capture** checkbox on Plan is enabled by default. When enabled, exactly one full default-setting capture is added to training **outside** the requested LHS/Sobol count; the sampling design itself is left unanchored. When disabled, no default is forced or included as a separate capture: the requested sampler points are used instead (with the exact default excluded from fallback selections). Complex Switch combinations still receive one global default, not one extra per state. Legacy saved plans retain their original anchor/counts until a Plan edit is applied; they are never regenerated automatically.
- **Mute at minimum / Mute at maximum:** each option now combines the training-only Mute Anchor with automatic exclusion of the corresponding silent endpoint from all new training, validation and corner captures. The nearest usable knob position replaces muted corners, so no muted recording is required. Both options work independently for normal and reversed controls. Legacy Avoid min/max settings remain honored for saved projects without silently enabling Mute Anchor. If the additional Default capture is enabled, its setting cannot be a muted endpoint.
- **Linear Volume capture calibration:** other numeric controls are held at 10 (or the nearest physically reachable position), while switches use their highest position. Avoid thresholds and capture-plan rules do not restrict the volume sweep; the Linear Volume control is still measured across its entire range. Training captures otherwise use that volume control at maximum.

- **Planning and Retrain regressions:** added coverage for optional default capture on/off with Standard LHS, Sobol and Optimized LHS; Simple/Complex Switch combinations; mute/avoid thresholds; grid capacity; legacy Plan settings; and restoring the job's original recorded captures after subsequent project changes.

## Parametric training and loss normalization

- **Mute Anchor:** when a knob has an enabled mute endpoint, parametric training adds a training-only silence constraint using real input audio, the selected knob held at its mute endpoint and randomized settings for the other controls. HyperA2 Full and Lite can learn the endpoint without an extra muted recording. Learned silence is approximate, not mathematically guaranteed zero.
- **Mute Anchor performance controls:** new **Mute Anchor Interval** (default 1 optimizer step), **Mute Anchor Batch Size** (default 2) and **Mute Anchor NY** (default 512) at the bottom of Advanced training settings. An interval of N calculates the anchor once per N optimizer steps and multiplies its loss weight by N on those steps. Existing defaults retain the previous per-step behavior.
- **Optional per-capture RMS loss normalization (introduced in 0.7.7; enabled by default in 0.8.0):** disabled by default in 0.7.7 for newly created training. When enabled, adjusts MSE and MSE_FFT for each capture relative to **−18 dBFS RMS**, using a **−60 dBFS** floor. The reference audio, validation ESR and exported inference gain are not normalized by this option.
- **RMS loss normalization exponent:** Advanced setting adjustable from **0.01 to 2**, default **2**. Exponent 0.01 gives minimal RMS compensation, 1 weights by the RMS amplitude ratio and 2 preserves the previous squared-ratio implementation. Disable RMS to remove compensation completely. Previously saved exponent-0 settings migrate to RMS disabled with SNR preserved, retaining the same effective training weights.
- **Independent SNR weighted loss normalization (introduced in 0.7.7; enabled by default and revised in 0.8.0):** disabled by default in 0.7.7 and selectable **with or without RMS normalization**. It estimates noise from the saved raw amp-return recording's pre-blip silence and downweights noisy captures in the training MSE/MSE_FFT. With RMS off it applies only the SNR reliability factor; with both enabled the original combined formula was preserved in 0.7.7, before the revised MSE/MRSTFT weighting in 0.8.0. No extra captures or playback silence are needed.
- **Normalization reliability and compatibility:** corrected raw-manifest lookup, preamble noise measurement, and scaling of noise estimates to match dataset output gain. Noise measurements are frozen along with training-queue captures. The normalization settings are carried into saved jobs and checked when resuming incompatible checkpoints.

## Application, queue and installer fixes

- **Training queue:** failed jobs display **Retry** and continue through the existing checkpoint recovery flow; frozen queue datasets retain per-capture SNR measurements.
- **Retrain restores the original plan:** clicking Retrain creates an independent editable copy of the selected job's recorded capture plan and frozen audio, rather than rebuilding captures using the current source project's Plan settings. Later unrecorded active-learning suggestions are omitted; the old job and current source project remain unchanged. Retrain audio files are physically independent copies, not writable hardlinks to the original job's immutable WAV cache.
- **Numeric controls:** fixed runaway repeated +/− increments in advanced training spin boxes.
- **Windows GPU runtime:** support extended-length paths when extracting optional NVIDIA/PyTorch runtime files.
- **Build and CI safeguards (introduced in 0.7.7):** draft 0.7.7 native installers were built for Windows, Linux, macOS Apple Silicon and macOS Intel. The draft-update workflow requires passing tests for the exact source revision, prevents obsolete workflows from replacing newer release assets and performs targeted regression and frozen-app checks before finalizing the draft. Release remains a draft until explicitly published.

## Compatibility

- No changes to the player-facing `.pnam` architecture, control schema or inference requirements. Existing players remain compatible.
- Existing captured projects are not silently regenerated, and existing mute endpoints are unchanged unless explicitly enabled.

**Full Changelog:** https://github.com/DLC86/ParaNAM-Trainer/compare/v0.7.6...v0.8.0
