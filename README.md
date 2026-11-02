# ParaNAM Trainer

A desktop wizard for parametric **HyperA2 Full + Lite inside one nested `.pnam` model**
and batches of independent static **A2 Full + Lite nested `.nam` models**. Based on DLC86's `Knobs+Switch` trainer branch,
with the established recording, alignment and model code retained.

## Install and launch

The 0.4.3 interface uses a #bf0000 red accent, neutral gray surfaces, the bundled
Michroma title font from NAM On Steroids, and a TTS logo linking to the shop.

**[Download the 0.4.3 development preview](https://github.com/DLC86/ParaNAM-Trainer/releases/tag/v0.4.3-preview)**
(sign in with an account that can access this private repository).

Download the package for your computer from this repository's installer artifacts
or releases. End users do not need Python, Anaconda, a CUDA toolkit, build tools,
or a terminal. Python, Qt, PyTorch, audio libraries and both capture signals are
included. A normal audio-interface driver is still required.

| Computer | Package | Training acceleration |
| --- | --- | --- |
| Windows 10/11 x64 | Setup `.exe` | CPU included; optional NVIDIA download |
| macOS Apple Silicon | `.dmg`, drag the app to Applications | Apple Metal / CPU |
| macOS Intel | `.dmg`, drag the app to Applications | CPU |
| Ubuntu/Debian-compatible Linux x64 | `.deb`, open in the graphical package installer | CPU included; optional NVIDIA download |
| Other compatible Linux x64 desktops | Portable `.tar.gz` | CPU included; optional NVIDIA download |

Linux packages target Ubuntu 22.04 or newer and the corresponding system ABI.
The desktop package installer resolves standard OS audio/display libraries. There is
one installer per OS/architecture. Windows offers an unchecked NVIDIA option during
setup; Linux offers the same choice on first launch. Both can manage it later from
Training settings, using **Add NVIDIA GPU support**, shown only while the component
is missing. Windows setup shows GPU preparation in an additional progress bar in the
installer itself, with no separate setup dialog or second confirmation.
CPU training and offline capture work without downloading anything.
The optional engine downloads compatible, checksum-verified PyTorch/NVIDIA wheels
from official sources into a per-user folder, retaining all their runtime dependencies
and license files. Matching installed components are reused and interrupted downloads
can resume. No CUDA toolkit, Python installation or terminal is required. A compatible
NVIDIA graphics driver is still required for GPU execution; otherwise Auto uses CPU.
macOS uses Apple Metal where supported and does not offer CUDA. Builds are unsigned until signing credentials are
configured; macOS may require allowing the app through its normal security UI.

From 0.2.0 onward the downloaded GPU engine lives outside the application folder
and survives app updates. Future builds reuse its verified wheel checksums even
when download URLs or size metadata change, checking the installed engine without
downloading or extracting it again. A genuinely different required engine needs
its matching GPU component. The old bundled 0.1.0 CUDA edition is not migrated.

The **ParaNAM native installers** workflow builds on each target OS, tests the
application, freezes it, smoke-tests the frozen bundle, and uploads native packages.
Apple Silicon and Intel builds are separate because PyTorch's supported binaries
are different. Intel uses the last official Intel-compatible PyTorch 2.2.2.

## Guided workflow

1. **Project & routing.** Create/open a project or pick a recent one. Choose your
   interface and channels. Clean timing loopback is enabled by default.
   Test the connection and optionally enter
   send/return dBu calibration. Channel selectors use driver labels, and buffer
   choices come from the driver: 2048 samples is selected when available, otherwise
   its highest supported value. When an API exposes no names or sizes, the UI says
   so and uses its driver-managed buffer instead of offering invented choices.
2. **Controls & plan.** Choose Parametric or Static batch; both always train Full
   and Lite together. Enter model metadata: name, author, gear type, manufacturer, model,
   tone type and notes. Existing custom metadata is preserved. In Parametric, add/edit knobs and named multi-position switches, choose
   Normal or Active learning, and select Standard or Custom training input. Capture
   counts include all switch combinations; corners are repeated for every state.
3. **Capture your rig.** Follow the graphic knob/switch settings, confirm the rig
   matches, and record each entry. The list tracks completed/pending captures and
   quality warnings in separate resizable **Latency**, **Peak (dBFS)** and **Warnings**
   columns. Latency includes subsample correction. Static mode instead shows named
   batch items with editable per-model metadata; each records training and fixed
   validation automatically without changing the rig between them. Training unlocks after the entire plan is captured.
   Pending captures are automatically ordered to reduce manual changes: first the
   number of controls to move, then the total knob rotation. Identical settings
   stay together, including separate training/validation recordings. On resuming,
   the route starts from the last recorded setting; active-learning suggestions
   follow the same rule. A deterministic nearest-neighbour route plus bounded
   2-opt improves the route without changing capture IDs or training data order.
4. **Train & export.** Set epochs, batch size and automatic/CPU execution. See current
   epoch, best Full validation ESR, elapsed time and an estimate based on completed epoch
   durations. Static batch size defaults to 16, matching the official NAM trainer;
   both static and parametric modes default to batch 16. Optimizer (Adam by default,
   or AdamW), Batch Size, Learning Rate, LR Decay, NY, Fit MRSTFT, MRSTFT Weight and
   Weight Decay are inside **Advanced training settings**, initially collapsed.
   Both modes follow the A2 recipe: LR 0.004, decay 0.994, NY 8192, MRSTFT enabled
   with weight 0.0005, and weight decay 3.17e-7. Existing saved custom values are
   retained. New HyperA2 runs use 0.1 times the selected LR for the hypernetwork
   (0.0004 at the default audio-weight LR of 0.004), gradient norm clipping at 1.0
   and nonfinite-gradient protection. This fixes the unstable all-weight updates
   introduced in 0.4.2; see [the diagnosis](docs/a2-training-comparison.md#hypernetwork-stability-fix).
   Start a new run to use this recipe; resuming an old run retains its saved recipe.
   Validation uses each full recording with batch 1. RMS normalization
   is disabled, preserving capture output levels across control settings.
   Changing the recipe requires a new run; batch size and device can be changed
   when resuming.
   Epoch progress bars are also retained in the training details log. Parametric export is `models/round-N/ParaNAM_HyperA2.pnam`, containing both
   Full and Lite with Full as the playback default. Static batches export one standard
   A2 Full + Lite `SlimmableContainer` `.nam` per named item under `models/<run>/`, with its metadata and
   send/return dBu calibration. Channel selectors use driver labels, and buffer
   choices come from the driver: 2048 samples is selected when available, otherwise
   its highest supported value. When an API exposes no names or sizes, the UI says
   so and uses its driver-managed buffer instead of offering invented choices.

**Back** and the step sidebar let you revisit settings. Drafts are saved. A changed
capture plan is confirmed before recordings are archived into `history/`; they are
preserved, not deleted or silently relabelled. A saved training checkpoint may only
resume against the same captured data, control definitions and model settings.
Batch size and execution device may be changed before resuming, for example after
a GPU-memory error.

## Included input signals

* Standard: original [TONE3000 T3K-sweep-v3.wav](https://www.tone3000.com/T3K-sweep-v3.wav),
  mono, 48 kHz, **190 seconds**. It is included in the installer, with no runtime download.
* Validation: unmodified [validation.wav from the upstream training-inputs-v1 release](https://github.com/phillipmself/neural-amp-modeler-parametric/releases/tag/training-inputs-v1),
  mono, 48 kHz, **7 seconds**, always included and always used for validation.
* Custom: choose one local mono 48 kHz WAV for **training only**. The validation file
  remains fixed. Applying a different signal requires a new capture plan.

The files' sources, sample counts and SHA-256 hashes are in
[paranam/assets/manifest.json](paranam/assets/manifest.json). Recordings are still
made separately for training and validation through the actual rig. Switch-only
projects validate the same categorical states with the independent validation audio.

## Guided level calibration and capture output

Open **Calibrate send and return** in Level calibration. The tone starts only when
**Start sine** is pressed; **Stop sine** and closing the dialog stop the output.
Its default is 1000 Hz at 0 dBFS peak. Frequency is adjustable from 20 to 20000 Hz;
output level accepts negative dBFS values or `-inf` for silence.

1. Disconnect the gear and speakers. Drive the cable that will feed the gear,
   including any reamp box, and measure its RMS voltage with a suitable meter.
   Enter the voltage and apply send calibration: `send_dBu = 20 log10(V_RMS / 0.775)`.
2. Stop the tone, then connect that same measured cable directly to the selected
   return input, bypassing the gear. Restart the tone and confirm the direct patch.
   The app measures digital RMS, expresses it as sine-equivalent peak dBFS, and
   calculates `return_dBu = send_dBu - measured_dBFS`. Clipping blocks calibration
   and prompts the user to lower input gain. Silence/noise is not accepted as a sine.
3. Apply return calibration. Send and return values support three decimal places.

The selected sine level is also the playback attenuation for **both training and
validation captures**, including the main send's timing marks. Original training
reference WAVs stay unchanged: the model learns the attenuated physical rig, and
its input dBu metadata describes that effective send setting. Do not extrapolate
send calibration back to the unattenuated DAC level. The separate timing loopback
retains its normal reference level for reliable timing; raw audio stores both
playback references and the applied attenuation. Muted capture playback is refused.

Changing attenuation invalidates send calibration and requires a new capture plan;
previous recordings and their calibration are archived, preventing mixed levels
in one model. Return calibration remains valid while the input gain is unchanged.

## Capture timing

Each recording is aligned independently using the clean loopback when enabled.
The **Latency** column shows one fractional sample value, recovering the delay as
the stored integer label minus the correction already applied to the recording.
NAM's one-sample safety factor is already included by the latency calibrator.
Loopback alignment retains a conservative onset margin (initially four samples
before the response peak, widened for the interface when needed). Static and
parametric training use these prepared labels without subtracting the margin again.
Amp-return versus loopback differences and changes in latency between recordings
do not generate discrepancy errors. Timing marks before and after the entire input
verify clock stability. A change of **0.1 sample or more**, including fractional
changes below one sample, or unreliable timing detection rejects the recording.
The item stays pending, is excluded from training, and must be recorded again; the
app suggests increasing the audio-interface buffer. Diagnostic raw audio remains
available. Replaced accepted audio is archived rather than silently reused.

## Pause, stop and resume

**Pause & save** stops at the next train/validation batch boundary, without
waiting for the epoch to finish. The incomplete epoch is discarded. Resume reloads
the optimizer, scheduler, loop counters and RNG from the last complete epoch and
restarts the interrupted epoch. If paused before completing the first epoch, it
restarts from the saved initial state. A running GPU operation must reach the
batch boundary; it is not interrupted inside a CUDA kernel.

An atomic rolling checkpoint is saved initially and after every complete epoch,
so unexpected training errors and process interruption also leave a recovery point.
Fix the error and choose **Resume training**; batch size may be lowered after OOM,
or the execution device changed. Older jobs with a complete native Lightning
checkpoint can be recovered even if they have no ParaNAM resume file. Corrupt or
partial-epoch checkpoints are not used. The window can close safely after Pause.

**Stop** cancels at the next batch boundary, retaining captured WAVs and previous
saved runs. It does not publish an incomplete current model. Details remain in the
training log. ETA covers the current round's training, not future manual captures.

## GPU training throughput

New projects default to batch **32**, matching the reference HyperWaveNet recipe;
existing projects keep their chosen batch. The log and training status show the
actual device, batch size and batches per epoch. CUDA data transfers use pinned
host memory. Both Full and Lite are still trained and exported together.

A controlled RTX 3070 Laptop microbenchmark found HyperA2 at batch 32 processed
about **5 times** as many windows per second as batch 4. At the same batch 32,
Full+Lite cost about **1.5 times** HyperWaveNet. These are isolated GPU measurements,
not a promised epoch time on other GPUs. See [measurement details](docs/paranam-performance.md).

## Active learning

Active learning uses independently seeded **HyperA2** ensemble members, evaluates
prediction disagreement in **both Full and Lite**, and recommends high-uncertainty
settings from a reproducible candidate pool. Suggestions exclude every existing
training/validation setting and respect knob steps and switch labels. After a round,
choose **Capture suggested settings**, record the new batch, then train the next
round. Every published model remains nested HyperA2; no ConcatLSTM surrogate is
published. Capture-count recommendations and uncertainty selection are heuristics,
not guarantees of accuracy.

## Development

For contributors only (end users use the installer):

```sh
python -m pip install -e '.[test]'
python -m paranam
python -m pytest tests/paranam -q
python -m pip install pyinstaller Pillow
python packaging/build.py --flavor CPU
```

Run the build on its target OS. Windows builds additionally use Inno Setup 6+;
macOS uses hdiutil; Linux uses dpkg-deb. GitHub Actions provisions these tools.
Windows/Linux bundles must be built from CPU PyTorch. The build resolves and pins
the optional GPU download manifest against the official CPU/CUDA wheel closures.
Native CI verifies the frozen CPU application and optional external GPU engine;
macOS builds use `--flavor Metal`. AVX2 nam-cpu-trainer integration is deferred.

The original trainer documentation is preserved in
[README_UPSTREAM.md](README_UPSTREAM.md). See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for credits and bundled components.

## Full/Lite validation and training audit

Every epoch logs Lite ESR, Full ESR and their mean. The best checkpoint and UI
use **Full ESR**. Export metadata `training.validation_esr` refers to Full;
`validation_esr_full` and `validation_esr_lite` record both values at the exported
branches, alongside their respective epochs and optimizer. Static nested children also carry
their own ESR. Changing optimizer requires a new training run.

HyperA2 evaluates both audio branches together in a differentiable packed layout.
The [training audit](docs/a2-training-comparison.md) documents the official A2
comparison, remaining differences and the measured scope of the optimization.
