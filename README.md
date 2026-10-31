# ParaNAM Trainer

A desktop wizard for capturing a parametric rig and training **HyperA2 Full + Lite
inside one nested `.nam` model**. Based on DLC86's `Knobs+Switch` trainer branch,
with the established recording, alignment and model code retained.

## Install and launch

Download the package for your computer from this repository's installer artifacts
or releases. End users do not need Python, Anaconda, a CUDA toolkit, build tools,
or a terminal. Python, Qt, PyTorch, audio libraries and both capture signals are
included. A normal audio-interface driver is still required.

| Computer | Package | Training acceleration |
| --- | --- | --- |
| Windows 10/11 x64 | Setup `.exe` | CPU or separate self-contained NVIDIA CUDA edition |
| macOS Apple Silicon | `.dmg`, drag the app to Applications | Apple Metal / CPU |
| macOS Intel | `.dmg`, drag the app to Applications | CPU |
| Ubuntu/Debian-compatible Linux x64 | `.deb`, open in the graphical package installer | CPU or separate self-contained NVIDIA CUDA edition |
| Other compatible Linux x64 desktops | Portable `.tar.gz` | CPU / CUDA, subject to OS library compatibility |

Linux packages target Ubuntu 22.04 or newer and the corresponding system ABI.
The desktop package installer resolves standard OS audio/display libraries. CUDA
editions bundle the NVIDIA runtime used by PyTorch; they use the normal compatible
NVIDIA display driver, not a separately installed CUDA toolkit. The CPU edition
works without an NVIDIA GPU. Builds are unsigned until signing credentials are
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
2. **Controls & plan.** Add/edit knobs and named multi-position switches, choose
   Normal or Active learning, and select Standard or Custom training input. Capture
   counts include all switch combinations; corners are repeated for every state.
3. **Capture your rig.** Follow the graphic knob/switch settings, confirm the rig
   matches, and record each entry. The list tracks completed/pending captures and
   quality warnings. Training unlocks after the entire plan is captured.
4. **Train & export.** Set epochs, batch size and automatic/CPU execution. See current
   epoch, best validation ESR, elapsed time and an estimate based on completed epoch
   durations. Export is always `models/round-N/ParaNAM_HyperA2.nam`, containing both
   Full and Lite with Full as the playback default.

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
`--flavor CUDA` must be built from an environment containing the official CUDA
PyTorch wheel; choosing the flag alone does not add GPU libraries.

The original trainer documentation is preserved in
[README_UPSTREAM.md](README_UPSTREAM.md). See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for credits and bundled components.
