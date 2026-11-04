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
size at each completed epoch, on a labelled logarithmic axis. Lower values are better. Missing or
nonfinite values break a curve rather than becoming a misleading point. History
is saved with progress and checkpoints and survives pause/resume. Each static
capture or ensemble member has its own displayed history.

## Advanced training settings

The base **Learning Rate** and **Hypernetwork Learning Rate** can be set
independently. Both use the same selected LR schedule. Hypernetwork controls
apply only to parametric training.

The optional **A2 — same cadence per optimizer update** learning-rate preset
starts at audio LR 0.004 and hypernetwork LR 0.0004. Both rates remain editable.
It uses one bundled standard capture as reference, with the official 10-second /
final-nine-second split, NY 8192 and batch 16: 62 complete updates per reference
epoch. A step-based scheduler multiplies both rates by 0.994 every 62 optimizer
updates. Its counter is stored in checkpoints and restored on resume; it is not
reset at parametric epoch boundaries. Changing Batch Size or NY changes how many
updates a project epoch contains, without changing that reference cadence.

The **A2 — n× slower (knobs + switches)** variant retains the same reference,
initial rates and 0.994 reduction factor, but applies a reduction every
`62 × n` optimizer updates, where `n` is the number of model knobs plus switches.
A switch counts as one control regardless of how many positions it has. With
two controls this is one reduction every 124 updates; one control gives the
original cadence. It does not use the number of captured combinations.
Both the audio and hypernetwork LR follow the slower schedule. The preview and
saved job specification show the multiplier, actual update interval, reduction
count and final rates. Resume restores the scheduler counter; changing the
preset requires a new training instance.

Control-count scaling is an optional heuristic for optimization, not a guarantee
of better validation or generalization. A large capture set and a long epoch
budget can still make the final rates very small; inspect the preview before
starting a run. Manual and the original A2 cadence remain selectable.

The preview reads WAV headers and counts actual complete batches separately for
each capture, including context, trimming and alignment. It shows audio duration,
updates per epoch, the average equivalent per-epoch multiplier and final rates for
the chosen epoch budget. A large parametric dataset can undergo many reference
epochs inside one project epoch, so keeping 200 project epochs can produce a very
small final LR. This preset matches a cadence; it does not automatically choose
an optimal duration or guarantee equal accuracy. Manual per-epoch decay is the
default and remains available. Rate preferences and the calculated schedule are
frozen into each independent job.

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

The defaults retain 0.4.3 behavior: both protections enabled for HyperA2, with
clipping norm 1.0; both disabled for static A2. Preferences are saved separately
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
| Generated deltas | Full, all tensors | Low rank, rank 2; dilated convolution kernels excluded |
| Hypernetwork hidden layer / activation | 16 / LeakyReLU | 16 / LeakyReLU |
| Full/Lite branch loss reduction | Sum | Mean |
| Clipping / NaN/Inf protection | 1.0 / enabled | 1.0 / enabled |
| NY / MRSTFT weight | 8192 / 0.0005 | 8192 / 0.0005 |
| LR decay / weight decay | 0.994 / 3.17e-7 | 0.994 / 3.17e-7 |

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
