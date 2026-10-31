# ParaNAM Trainer

A desktop wizard for parametric **HyperA2 Full + Lite inside one nested `.nam` model**
and batches of independent static **A2 Full or Lite** models. Based on DLC86's `Knobs+Switch` trainer branch,
with the established recording, alignment and model code retained.

## Install and launch

**[Download the 0.2.0 development preview](https://github.com/DLC86/ParaNAM-Trainer/releases/tag/v0.2.0-preview)**
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
Training settings. CPU training and offline capture work without downloading anything.
The optional engine downloads compatible, checksum-verified PyTorch/NVIDIA wheels
from official sources into a per-user folder, retaining all their runtime dependencies
and license files. Matching installed components are reused and interrupted downloads
can resume. No CUDA toolkit, Python installation or terminal is required. A compatible
NVIDIA graphics driver is still required for GPU execution; otherwise Auto uses CPU.
macOS uses Apple Metal where supported and does not offer CUDA. Builds are unsigned until signing credentials are
configured; macOS may require allowing the app through its normal security UI.

The **ParaNAM native installers** workflow builds on each target OS, tests the
application, freezes it, smoke-tests the frozen bundle, and uploads native packages.
Apple Silicon and Intel builds are separate because PyTorch's supported binaries
are different. Intel uses the last official Intel-compatible PyTorch 2.2.2.

## Guided workflow

1. **Project & routing.** Create/open a project or pick a recent one. Choose your
   interface and channels. The connection diagram shows the reamp/rig/return path
   and optional clean timing loopback. Test the connection and optionally enter
   input/output dBu calibration.
2. **Controls & plan.** Choose Parametric or Static batch (A2 Full by default, Lite
   selectable). Enter model metadata: name, author, gear type, manufacturer, model,
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
   epoch, best validation ESR, elapsed time and an estimate based on completed epoch
   durations. Export is always `models/round-N/ParaNAM_HyperA2.nam`, containing both
   Full and Lite with Full as the playback default. Static batches export one standard
   WaveNet A2 `.nam` per named item under `models/<run>/`, with its metadata and
   input/output dBu calibration.

**Back** and the step sidebar let you revisit settings. Drafts are saved. A changed
capture plan is confirmed before recordings are archived into `history/`; they are
preserved, not deleted or silently relabelled. A saved training checkpoint may only
resume against the same captured data, control definitions and training settings.

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

## Capture timing

Each recording is aligned independently using the clean loopback when enabled.
Amp-return versus loopback differences and changes in latency between recordings
do not generate discrepancy errors. Timing marks before and after the entire input
verify clock stability. A change of **0.1 sample or more**, including fractional
changes below one sample, or unreliable timing detection rejects the recording.
The item stays pending, is excluded from training, and must be recorded again; the
app suggests increasing the audio-interface buffer. Diagnostic raw audio remains
available. Replaced accepted audio is archived rather than silently reused.

## Pause, stop and resume

**Pause & save** finishes the current epoch, then saves model weights, optimizer,
scheduler, loop counters and random-generator state. The window can close safely
once saved. Open the project after restarting the computer and choose **Resume
training**. Closing during training offers to pause safely first.

**Stop** cancels at the next training batch boundary, retaining the captured WAVs and
previous saved runs. It does not publish an incomplete current model. Detailed logs
remain in the project's training folder. ETA covers training in the current round,
not the time needed to manually capture future active-learning suggestions.

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
