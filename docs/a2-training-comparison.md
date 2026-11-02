# A2 static and HyperA2 training comparison

Audit: 2 November 2026. Official NAM reference:
[`e680deaf4ed1d28ea260e7267680b9f10fed24b8`](https://github.com/sdatkinson/neural-amp-modeler/tree/e680deaf4ed1d28ea260e7267680b9f10fed24b8).
ParaNAM baseline: 0.4.1 (`ae646b0`). This document describes 0.4.2 after the
requested alignment with the official A2 recipe.

## Differences removed

| Area | Previous ParaNAM HyperA2 | ParaNAM 0.4.2 A2 recipe |
| --- | --- | --- |
| Optimizer | AdamW inherited from the shared active-learning trainer | Adam by default; AdamW remains an explicit user option |
| Default LR / batch | 0.002 / 32 | 0.004 / 16, matching A2; existing saved preferences are respected |
| Conditional weights | Low-rank deltas (rank 2), convolution kernels shared across settings | Full deltas for every weight tensor; no excluded convolution kernels |
| Branch execution | Sequential Full and Lite forwards | Differentiable joint PackedWaveNet execution for capture-homogeneous batches |
| Branch loss | Mean of Full and Lite losses | Sum of branch losses, as in official packed A2 |
| Gradient handling | Clipping at 1.0 and an inherited per-parameter nonfinite-gradient scan | Official Lightning behavior with no additional clipping or scan |
| Validation | Usually 8192-sample windows in batches | Entire validation recording, batch size 1; different per-capture lengths are supported |
| Incomplete training batch | Retained | Dropped, matching the official recipe; insufficient input produces a clear error |
| Best branch weights | One combined best epoch | Independent best Full and best Lite checkpoints, matching official packed A2 |

Defaults also match exponential LR gamma 0.994, training NY 8192, MSE plus MRSTFT
weight 0.0005, weight decay 3.17e-7, float32, LeakyReLU and head scale 0.01.
The audio branches retain the official 3/8 channels, kernels, dilations and RF
6347. Optional Mel losses, silence anchors and scheduler warmup are disabled.
The hypernetwork remains necessary to map controls to weights; its hidden layer
is `[16]`. Full/Lite names describe audio-network sizes, independently of the
hypernetwork's full versus low-rank weight-generation mode.

## Intentional remaining differences

* **RMS normalization is disabled**, as explicitly requested. Official NAM scales
  training/validation outputs to -18 dBFS RMS internally and compensates at export.
  ParaNAM keeps capture samples and their inter-setting output levels unchanged.
* **Full is the primary ESR reference**, as requested. It selects the main
  checkpoint and is displayed in the UI. Lite still has its own best checkpoint.
* Multiple control settings require a conditional weight generator and coverage
  across settings. Capture-homogeneous batches avoid regenerating weights for
  every audio window; audio windows and batch order remain shuffled.
* Validation pools error/target energy within each recording and size, then
  averages equally across control settings. This prevents loud settings from
  masking errors on quiet ones. Per-setting Full ESR is the direct comparator
  for an A2 Full static model trained/evaluated at that setting.
* Input and validation files follow the requested capture workflow. Comparisons
  against external trainers need the same audio, alignment and validation data.
* Installers disable torch.compile to avoid requiring an external compiler.
  cuDNN benchmarking is enabled as a runtime optimization.

No claim of equal ESR after a fixed epoch count follows solely from this
alignment: learning the control-to-weight mapping remains an additional task.

## Optimizer, resume, logs and metadata

Adam/AdamW is selectable under Advanced training settings and saved separately
for static/parametric modes. A changed optimizer requires a new run. Legacy
configs lacking an optimizer class retain their actual original interpretation
(static Adam, parametric AdamW), architecture and loss recipe when resumed.
New training uses the aligned A2 recipe. Existing custom learning settings are
preserved; the reference defaults are LR 0.004 and batch 16.

Both sizes train on every batch. Each epoch logs Lite ESR, Full ESR and their
mean, labelled with the epoch. The main UI and checkpoint use Full ESR.

Export metadata includes:

* `training.validation_esr`: Full ESR (primary).
* `training.validation_esr_full` and `training.validation_esr_lite`: scores of
  the actual exported best Full and Lite branches.
* `training.epoch_full` and `training.epoch_lite`: their corresponding epochs,
  which can differ. `training.epoch` refers to Full.
* `training.validation_esr_mean`, `training.validation_reference` and `optimizer`.

Static nested branches carry their own score and epoch too. Custom metadata and
calibration remain preserved. When resuming a legacy mean-ranked run, Full-based
ranking restarts rather than relabelling a historical mean as a measured Full ESR.

## Verification and performance scope

Joint execution is checked against the prior capture-homogeneous separate path,
with nonzero deltas for all tensors in full and low-rank generator modes. Tests
compare output, gradients, RNG state and state-dict keys. The execution cache
adds no optimizer/export parameters and is rebuilt after checkpoint loading.
Trainer exports with all convolution weights conditioned were also compared
against the existing plugin's C++ Full/Lite engine.

An initial packing-only CUDA probe on an RTX 3070 Laptop GPU used NY 8192,
rank 2, hidden `[16]` and cuDNN benchmarking:

| Batch | Separate branches | Packed branches |
| --- | ---: | ---: |
| 16 | 57.01 ms | 56.43 ms |
| 32 | 107.51 ms | 104.41 ms |

This probe preceded the full-weight recipe alignment and excluded MRSTFT,
optimizer and validation overhead. It shows a modest packing-only gain, not an
end-to-end speed prediction for the new defaults or another GPU.

## Official source references

* [core.py](https://github.com/sdatkinson/neural-amp-modeler/blob/e680deaf4ed1d28ea260e7267680b9f10fed24b8/nam/train/core.py): normalization, loaders, best-branch export.
* [lightning_module.py](https://github.com/sdatkinson/neural-amp-modeler/blob/e680deaf4ed1d28ea260e7267680b9f10fed24b8/nam/train/lightning_module.py): losses, optimizer, metrics, checkpoints.
* [A2 configuration](https://github.com/sdatkinson/neural-amp-modeler/blob/e680deaf4ed1d28ea260e7267680b9f10fed24b8/nam/train/_resources/config_model_packed.json).
