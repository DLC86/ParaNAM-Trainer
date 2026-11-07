# Training controls

## Log and ESR curves

Each epoch updates one live progress line. Once validation finishes, that line
contains Validation ESR (Lite, Full and their mean), followed by Train ESR
(Lite and Full). Both are evaluated with fixed end-of-epoch weights, without
gradient updates. Validation ESR uses only separate validation captures. Train
ESR uses only the final nine seconds in training captures, matching the official
NAM v3 validation interval. Each capture contributes
equally to its split's average, independently of its signal level. Train ESR is
not the error on the audio used to update weights during the epoch.

Train ESR calls NAM's `nam.models.losses.esr` directly on each complete held-out
recording (`ny=None`, batch size 1, `pad_start=False`). Dataset context and delay
alignment follow the same NAM code as official static validation. It then
averages the per-capture ESRs equally. Validation ESR retains the existing
separate seven-second validation recordings; the two metrics therefore use
different clip boundaries. Train ESR follows the official static trainer's
measurement, whereas Validation ESR also tests unseen control combinations.

Static and parametric jobs use the official NAM v3 training interval: start at
10 seconds and stop 9 seconds before the end. Neither copy of the validation
audio enters gradient training. This layout also applies to TTS input v11;
custom inputs must retain the same initial/final structure. Validation captures
continue to use their whole separate validation recording. Queue shows separate
**Best Validation ESR (Full)** and **Best Train ESR (Full)** cards; the minima
can occur at different epochs. Best checkpoint selection and export continue to
use Validation ESR only.

Jobs made before this split change require a new run, since their weights have
already learned the reserved segments. Existing recordings can be reused.
The details viewer retains the entire log instead of only its last 18,000 bytes;
repeated progress updates replace the current line. Diagnostic messages remain.
Long lines scroll horizontally so an epoch does not wrap into multiple rows.

The chart below the log has **Full ESR chart** and **Lite ESR chart** tabs. Each
shows Validation ESR (solid blue) and Train ESR (dashed orange) for its own model
size at each completed epoch, on a labelled logarithmic axis. The horizontal axis
is fixed to the job's planned total epochs, including before the first epoch;
adding points does not change their horizontal spacing. The initial logarithmic
Y range is 0.005–0.5 for both charts. Values inside that range do not rescale it;
values outside expand the necessary side in logarithmic decades to include them. Lower values are better. Missing or
nonfinite values break a curve rather than becoming a misleading point. History
is saved with progress and checkpoints and survives pause/resume. Each static
capture or ensemble member has its own displayed history.

## Advanced training settings

The base **Learning Rate** and **Hypernetwork Learning Rate** are independently
editable. Beside each respective **decay** field, select **per epoch** or **per 62 optimizer updates
(A2 cadence)** for its editable decay. These choices replace the former presets.
Changing a cadence leaves initial LR and decay values unchanged. The experimental
Cosine schedules from the draft 0.6.3 beta have been removed in 0.7.0; those beta
Cosine jobs require a new training. Ordinary cadence jobs remain resumable.

The 0.7.0 preview recommends `min(max(8, 4 × dimensions), 24)` held-out validation captures
per joint Complex Switch combination. Dimensions include neural knobs and Simple
Switches; clean volume is measured separately. New-project default validation count is 12; saved
plans are unchanged. ESR_worst is the maximum complete-setting validation ESR,
reported beside the mean for Full/Lite. It does not change best-checkpoint selection.

New HyperA2 projects use LR 0.004/0.001, decay 0.994/0.998497 per 62 optimizer
updates, batch 16 and 50 epochs. Static models default to 200 epochs and retain
per-epoch scheduling. Switching capture type preserves separate epoch budgets.

For legacy presets, the editor computes an equivalent average decay. If the old
reduction interval was I updates and its multiplier was gamma, the displayed
62-update multiplier is `gamma ** (62/I)`. Dataset-normalized linear decay maps
to per epoch with the same gamma. Sqrt normalization uses the complete optimizer
updates per training epoch after cropping and validation exclusion. Old control
cadences use their original control factor and A2 Slow's division by sqrt(2).
The equivalent rates match their average logarithmic slope; their discrete
reduction instants need not coincide. No decay schedule guarantees generalization.

Unchanged legacy editor settings retain their stored identity. Resume uses the
job's original frozen scheduler configuration, optimizer state and both saved
counters, including fractional cadence intervals. Retrain and newly queued jobs
use the two-choice representation and equivalent decay. Capture, architecture
and evaluation-protocol compatibility checks still apply. Follow is not a
selectable mode; editing either group's cadence/decay makes it independent.

In static capture mode, hypernetwork-specific fields and their labels are hidden,
including the parametric recipe selector, LR/preset/decay, weight mode/rank,
convolution conditioning and Full/Lite parametric loss reduction. Common static
training settings remain visible. Switching back restores the saved parametric
values.

**Hypernetwork weight mode** selects Full or Low rank independently of
**Condition dilated convolution kernels on knobs**. All four combinations are supported.
**Rank (low-rank mode)** is editable from 1 to 8 when Low rank is selected;
the effective rank also depends on each matrix's dimensions. When dilated
convolutions are not conditioned, their base weights remain trainable and shared
across settings; their biases may still be conditioned. **Full/Lite loss reduction** independently selects Sum or Mean
for training; it does not change the displayed validation ESR reduction.

**Gradient clipping** enables norm clipping; **Maximum gradient norm** specifies
its positive limit. **Check NaN/Inf gradients** skips an entire optimizer update
when any gradient is nonfinite, preserving weights and optimizer moments.
Disabling the latter skips the scan completely. The settings are independent:
neither a low LR nor clipping guarantees the absence of nonfinite gradients.

The current HyperA2 preset disables both protections (the stored clipping limit
remains 1.0). HyperWaveNet enables both, with clipping norm 1.0; static A2 keeps
its existing defaults. Older saved projects retain their own choices. Preferences are saved separately
for static and parametric modes and recorded in each job's spec. Changing a
recipe, an LR or gradient controls requires a new run. Jobs using the reserved
validation-audio protocol can resume with matching settings and optimizer state.

## Presets

Selecting a preset restores its advanced settings and batch size. Values can
then be customized and remain saved in the project.
All differences between the two presets are visible controls. A preset does not
activate additional hidden model behavior. Hidden-layer size, activation, audio
topologies, validation and compilation settings are common to both recipes.

| Setting | HyperA2 | HyperWaveNet (Phillip Self) |
| --- | --- | --- |
| Base LR | 0.004 | 0.002 |
| Hypernetwork LR | 0.001 | 0.002 |
| Optimizer | Adam | AdamW |
| Batch size | 16 | 32 |
| Generated deltas | Low rank, rank 2; dilated convolution kernels excluded | Low rank, rank 2; dilated convolution kernels excluded |
| Hypernetwork hidden layer / activation | 16 / LeakyReLU | 16 / LeakyReLU |
| Full/Lite branch loss reduction | Sum | Mean |
| Clipping / NaN/Inf protection | Disabled / disabled (limit stored as 1.0) | 1.0 / enabled |
| NY / MRSTFT weight | 8192 / 0.0005 | 8192 / 0.0005 |
| LR decay / weight decay | 0.994 / 3.17e-7 | 0.994 / 3.17e-7 |
| Audio LR schedule | Every 62 optimizer updates, decay 0.994 | Manual, per epoch |
| Hypernetwork LR schedule | Every 62 optimizer updates, decay 0.998497 | Manual, per epoch |
| cuDNN benchmark / deterministic algorithms | Disabled / disabled | Disabled / disabled |

HyperA2 uses the explicit 0.6.2 defaults above. Old jobs retain their saved settings on resume.

The HyperWaveNet settings were verified against Phillip Self's
[capture exporter](https://github.com/phillipmself/neural-amp-modeler-parametric/blob/a3684411076be69dab53505df14922487697f2cf/nam/capture/export.py)
and [hypernetwork defaults](https://github.com/phillipmself/neural-amp-modeler-parametric/blob/a3684411076be69dab53505df14922487697f2cf/nam/models/parametric/_hypernet.py).
This is a recipe adaptation to ParaNAM's independent Full/Lite branches, not a
claim that it reproduces the single-branch trainer. Both presets retain whole
validation recordings at batch 1, independent best-size checkpoints, unnormalized
capture levels, and compilation disabled for a self-contained installer. The
upstream exporter enables compilation and uses validation batches of 32.

## Export while training

**Export best PNAM** (parametric) or **Export best NAM** (static) becomes available after validated Full and Lite checkpoints
exist. Choose a destination; the training worker handles the request at a batch
boundary and automatically continues. Export takes some processing time, but
requires no pause/resume or restart. An export error is reported in the UI and
does not fail training.

The result is **one model file** combining Full from its
best Full checkpoint and Lite from its own best Lite checkpoint. They may come
from different epochs; each score and epoch is recorded in export metadata.
Parametric training exports a **`.pnam` HyperA2** retaining both branches'
hypernetworks, knob/switch definitions and live parameter controls. It is not
baked to fixed knob values. Static training exports a **`.nam` SlimmableContainer**
for the currently training named capture. The button label, file dialog and file
extension follow the training type automatically. Calibration and user metadata are
retained. The snapshot does not modify live weights, optimizer state or training
RNG, and writes atomically to the selected destination.


## Reproducible comparisons

Advanced training settings include **cuDNN benchmark (CUDA)** (default on) and
**Deterministic training** (default off) for both static and parametric models.
Disabling benchmark prevents timing-based convolution algorithm selection; it
alone does not guarantee deterministic arithmetic. Deterministic training forces
benchmark off, configures cuBLAS before the worker probes CUDA, and requests strict
PyTorch deterministic algorithms. Unsupported operations fail with an error instead
of silently falling back to nondeterministic training. It can reduce performance.

The existing seed is 938, offset for ensemble members/active-learning rounds.
Deterministic mode also isolates the batch shuffle from model initialization and
seeds it by epoch, including checkpoint resume. Existing default jobs retain their
original shuffling and backend settings. Changing either backend option requires
a new training instance; existing checkpoints resume with their saved choices.

Use this mode to compare settings with the same recordings, split, seed, batch size
and training budget. Different architectures have different initial weights even
with the same seed. Verify promising configurations over several seeds and datasets;
repeatability does not itself improve generalization. Bitwise reproducibility is
limited to supported operations within the same hardware/software environment,
not across PyTorch/CUDA versions or CPU/GPU/Metal backends. See the
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.10/notes/randomness.html).
