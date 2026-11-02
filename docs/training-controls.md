# Training controls in 0.4.4

## Log and ESR curves

Each epoch updates one live progress line. Once validation finishes, that line
contains Lite ESR, Full ESR (the primary checkpoint reference), and their mean.
The details viewer retains the entire log instead of only its last 18,000 bytes;
repeated progress updates replace the current line. Diagnostic messages remain.
Long lines scroll horizontally so an epoch does not wrap into multiple rows.

The chart below the log shows Full and Lite validation ESR for each completed
epoch, on a labelled logarithmic axis. Lower values are better. Missing or
nonfinite values break a curve rather than becoming a misleading point. History
is saved with progress and checkpoints and survives pause/resume. Each static
capture or ensemble member has its own displayed history.

## Advanced training settings

The base **Learning Rate** and **Hypernetwork Learning Rate** can be set
independently. Both use the selected exponential LR decay. Hypernetwork controls
apply only to parametric training.

**Gradient clipping** enables norm clipping; **Maximum gradient norm** specifies
its positive limit. **Check NaN/Inf gradients** skips an entire optimizer update
when any gradient is nonfinite, preserving weights and optimizer moments.
Disabling the latter skips the scan completely. The settings are independent:
neither a low LR nor clipping guarantees the absence of nonfinite gradients.

The defaults retain 0.4.3 behavior: both protections enabled for HyperA2, with
clipping norm 1.0; both disabled for static A2. Preferences are saved separately
for static and parametric modes and recorded in each job's spec. Changing a
recipe, an LR or gradient controls requires a new run. Existing 0.4.3 jobs can
resume with matching original settings and optimizer state.

## Presets

Selecting a preset restores its advanced settings and batch size. Values can
then be customized and remain saved in the project.

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

**Export best NAM** becomes available after validated Full and Lite checkpoints
exist. Choose a destination; the training worker handles the request at a batch
boundary and automatically continues. Export takes some processing time, but
requires no pause/resume or restart. An export error is reported in the UI and
does not fail training.

The result is **one standard `.nam` SlimmableContainer** combining Full from its
best Full checkpoint and Lite from its own best Lite checkpoint. They may come
from different epochs; each score and epoch is recorded in export metadata.
Parametric weights are baked at the project's default knob/switch values: this
fixed-setting `.nam` does not expose live parameter controls. Static batches
export the currently training named capture. Calibration and user metadata are
retained. The snapshot does not modify live weights, optimizer state or training
RNG, and writes atomically to the selected destination.
