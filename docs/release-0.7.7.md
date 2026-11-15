# ParaNAM Trainer 0.7.7

Changes since **0.7.6**.

## Capture planning and controls

- **New sampling methods: Sobol and Optimized LHS.** Scrambled Sobol provides low-discrepancy multidimensional coverage, with the strongest balance for power-of-two capture counts; arbitrary counts are supported. Optimized LHS uses SciPy's centered-discrepancy optimization (`random-cd`) to improve low-dimensional projections. Both are selectable on the **Plan** page, reproducible from the saved seed and available for training and independently sampled validation.
- **Plan interface:** the former “LHS design” selector is now labelled **Capture sampling method**, and the training count is labelled **Training captures** to include both LHS and Sobol designs.
- **Existing sampling choices preserved:** Standard, Maximin, Orthogonal and Near-orthogonal LHS still work; Standard remains the default. The new methods go through the existing control-grid quantization, duplicate removal, default-setting anchor, avoidance rules and Simple/Complex Switch planning. Saved projects retain their selected method and existing captures are not regenerated automatically.
- **Mute at minimum / Mute at maximum:** new independent rig-control settings for knobs that mute at either endpoint, including reversed controls. They are not the same as Avoid min/max, and do not require recording a muted example.
- **Linear Volume capture calibration:** other numeric controls are held at 10 (or the nearest physically reachable position), while switches use their highest position. Avoid thresholds and capture-plan rules do not restrict the volume sweep; the Linear Volume control is still measured across its entire range. Training captures otherwise use that volume control at maximum.

## Parametric training and loss normalization

- **Mute Anchor:** when a knob has an enabled mute endpoint, parametric training adds a training-only silence constraint using real input audio, the selected knob held at its mute endpoint and randomized settings for the other controls. HyperA2 Full and Lite can learn the endpoint without an extra muted recording. Learned silence is approximate, not mathematically guaranteed zero.
- **Mute Anchor performance controls:** new **Mute Anchor Interval** (default 1 optimizer step), **Mute Anchor Batch Size** (default 2) and **Mute Anchor NY** (default 512) at the bottom of Advanced training settings. An interval of N calculates the anchor once per N optimizer steps and multiplies its loss weight by N on those steps. Existing defaults retain the previous per-step behavior.
- **Optional per-capture RMS loss normalization:** disabled by default for newly created training. When enabled, adjusts MSE and MSE_FFT for each capture relative to **−18 dBFS RMS**, using a **−60 dBFS** floor. The reference audio, validation ESR and exported inference gain are not normalized by this option.
- **RMS loss normalization exponent:** new Advanced setting adjustable from **0 to 2**, default **2**. Exponent 0 applies no RMS compensation, 1 weights by the RMS amplitude ratio and 2 preserves the previous squared-ratio implementation. This enables less aggressive weighting of quiet captures without changing the learning-rate settings.
- **Optional SNR weighted loss normalization:** disabled by default and available when RMS normalization is enabled. It estimates the noise floor from the existing pre-blip silence in each saved raw amp-return recording, then downweights noisy captures in the training loss. No extra silence is inserted into the capture playback and no extra captures are needed.
- **Normalization reliability and compatibility:** corrected raw-manifest lookup, preamble noise measurement, and scaling of noise estimates to match dataset output gain. Noise measurements are frozen along with training-queue captures. The normalization settings are carried into saved jobs and checked when resuming incompatible checkpoints.

## Application, queue and installer fixes

- **Training queue:** failed jobs display **Retry** and continue through the existing checkpoint recovery flow; frozen queue datasets retain per-capture SNR measurements.
- **Numeric controls:** fixed runaway repeated +/− increments in advanced training spin boxes.
- **Windows GPU runtime:** support extended-length paths when extracting optional NVIDIA/PyTorch runtime files.
- **Build and CI safeguards:** draft 0.7.7 native installers are built for Windows, Linux, macOS Apple Silicon and macOS Intel. The draft-update workflow requires passing tests for the exact source revision, prevents obsolete workflows from replacing newer release assets and performs targeted regression and frozen-app checks before finalizing the draft. Release remains a draft until explicitly published.

## Compatibility

- No changes to the player-facing `.pnam` architecture, control schema or inference requirements. Existing players remain compatible.
- Existing captured projects are not silently regenerated, and existing mute endpoints are unchanged unless explicitly enabled.

**Full Changelog:** https://github.com/DLC86/ParaNAM-Trainer/compare/v0.7.6...v0.7.7
