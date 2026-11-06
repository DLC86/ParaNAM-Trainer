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
editable and have separate scheduler selectors. New HyperA2 projects use
**Dataset-normalized** for both, starting at 0.004 and 0.0004 with decay 0.994.
All other HyperA2 advanced values retain the requested 06/11 02:52 recipe.
Batch size defaults to 16 and the epoch budget to 100.

Let `S` be the number of complete optimizer updates in one project training
epoch and `R` the bundled static A2 reference (62 updates: standard input,
NY 8192, batch 16, initial 10 seconds and final 9 seconds excluded).
The two new presets replace the control-count-based choices:

| Preset | Discrete reduction interval | Example: S = 100 × R |
| --- | --- | --- |
| Dataset-normalized | `R × (S/R) = S` | 6,200 updates |
| Dataset-normalized √ | `R × sqrt(S/R)` | 620 updates |

Each reduction multiplies the selected group's LR by 0.994. Linear normalization
therefore applies one reduction per complete dataset epoch. For 100 epochs,
initial rates 0.004/0.0004 become approximately 0.002191/0.0002191 when both groups
use linear normalization. The square-root variant decays faster than the linear
variant for datasets larger than the reference, and slower for smaller datasets.
The audio and hypernetwork can select different modes, or **Manual / per-epoch
decay**, whose decay value remains editable separately for each group.
**Follow learning-rate preset** shares the main group's cadence and decay.
Choosing a preset resets only that group's initial rate and decay; initial rates
remain editable. Hypernetwork options apply only to parametric training.
Static A2 scheduling is unchanged.

The accounting reads WAV headers and mirrors the real training loader: cropping,
initial/final validation exclusion, receptive-field context, delay alignment,
NY and complete batches dropped separately for each capture. Validation captures
do not enter S. Thus captures with different useful durations, batch sizes or NY
are accounted for through optimizer updates, rather than raw WAV duration or the
number of knobs/switches. The preview reports each group's interval, reference,
complete updates per epoch, ratio and final LR. Empty/incomplete training batches
produce a clear error rather than a guessed schedule.

Fractional intervals retain their remainder: reduction k occurs at
`ceil(k × interval)`. Both independent counters are saved in checkpoints and
restored on resume. Generated jobs freeze the calculated interval; scheduler
normalization is not recomputed mid-training. Changing a preset requires a new
training instance. No schedule guarantees better validation ESR or generalization;
normalization trades faster decay for a comparable progression per dataset epoch.

The earlier update/control presets are restored under consistent names:

| Current name | Earlier name | Reduction interval |
| --- | --- | --- |
| Update-based | Original A2 cadence | 62 updates |
| Control-normalized | Controls-scaled A2 | `62 × max(1, knobs) × 2^switches` |
| Control-normalized / √2 | A2 Slow | `62 × max(1, knobs) × 2^switches / sqrt(2)` |

They are independently selectable for the audio network and hypernetwork,
alongside both Dataset-normalized choices, Manual and hypernetwork Follow.
The serialized IDs remain `a2_updates`, `a2_controls` and `a2_slow`: loading a
saved project does not rename/migrate its effective settings or reset initial
rates. Resume retains the job's frozen interval and both checkpoint counters,
including jobs prepared with earlier control-count formulas. Existing capture,
architecture and evaluation-protocol compatibility checks still apply.
The 0.6.0 exclusion has been removed. Resuming a paused queue can therefore
continue its earlier paused instance before newly retrained jobs. Retrain restores
the effective frozen advanced configuration, taking precedence over stale editor
preferences. New-project defaults remain Dataset-normalized for both groups.

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
| Hypernetwork LR | 0.0004 | 0.002 |
| Optimizer | Adam | AdamW |
| Batch size | 16 | 32 |
| Generated deltas | Low rank, rank 2; dilated convolution kernels excluded | Low rank, rank 2; dilated convolution kernels excluded |
| Hypernetwork hidden layer / activation | 16 / LeakyReLU | 16 / LeakyReLU |
| Full/Lite branch loss reduction | Mean | Mean |
| Clipping / NaN/Inf protection | Disabled / disabled (limit stored as 1.0) | 1.0 / enabled |
| NY / MRSTFT weight | 8192 / 0.0005 | 8192 / 0.0005 |
| LR decay / weight decay | 0.994 / 3.17e-7 | 0.994 / 3.17e-7 |
| Audio LR schedule | Dataset-normalized, every S updates | Manual, per epoch |
| Hypernetwork LR schedule | Dataset-normalized, every S updates | Follow |
| cuDNN benchmark / deterministic algorithms | Disabled / disabled | Enabled / disabled |

HyperA2 retains the advanced fields of the saved 06/11 02:52 training,
with the subsequently requested dataset-normalized audio/hypernetwork schedules
and its preset label changed to HyperA2. Old jobs retain their saved labels, files and original scheduler settings on resume.

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
