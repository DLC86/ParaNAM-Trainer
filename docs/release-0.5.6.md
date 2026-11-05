# ParaNAM Trainer 0.5.6

- Retrain reopens the source project on Train with the selected stopped, paused
  or finished instance’s settings. It does not start training automatically or
  duplicate the project. Projects can be opened while another instance trains.
- Compare includes an optional monitor IR loader: the six original Tone3000
  preview IRs (Ampeg, Celestion, Eminence, Mesa, Plate, Spring) and custom mono/
  stereo WAVs. Files are loaded/resampled off the audio thread, normalized using
  Tone3000’s method, and processed by partitioned FFT convolution with a short
  crossfade on changes. IR adds 256 samples latency (5.33 ms at 48 kHz) and affects
  Gear/Model monitoring, leaving the physical reamp send unchanged. Default: Off.
  Original assets, source revision, hashes and MIT license are bundled.
- The model control panel has a black background. Discrete controls use slide
  switches with a compact fixed width, names aligned to knob labels, and position
  labels below each switch aligned to knob values. Control labels are one point larger.
- Play is grey while idle and red while playing.
- Parametric model controls sit above Input, Model size, Output mode and Output,
  in that order. Typed knob/gain values commit on Enter or focus loss.
- Input Instrument and stereo Monitor meters flank the gain controls, with peak
  dBFS labels below. Instrument measures raw input before gain; Monitor measures
  actual output after gain/IR. Peaks and clipping/dropout flags aggregate across
  each reporting interval rather than only its final audio callback. Peak labels
  have enough width and height to show complete dBFS readings. The redundant
  Input/Monitor Peak line is removed; clipping/dropout warnings remain.
- Input/output routing and buffer choices are remembered per interface (driver
  API and device name), including disabled loopback channel choices. Disconnecting
  an interface does not replace its saved selections; selecting it again restores
  the available channels even if PortAudio assigned it a different index.
- Queue adds Name between Order and Project, using metadata Name (the first
  model for a static batch).
- The slower A2 learning-rate preset now uses `knobs × 2^switches`: 2 knobs and
  no switch = 2×; with one switch = 4×; with two = 8×. Switch position counts are
  irrelevant; switch-only models use a knob multiplier of one. Preview and newly
  created jobs use the same formula. Existing jobs resume their saved scheduler.
- Added the separate **A2 Slow** learning-rate preset with factor
  `(knobs × 2^switches) / √2`. Fractional update cadence is retained rather than
  rounded: two knobs and no switches average 87.68124 optimizer updates between
  reductions. The scheduler counter and interval survive checkpoint resume.
  Initial rates and decay match the other A2 cadence presets. Existing presets,
  new-project defaults and saved scheduler states are preserved.
- New projects default to HyperWaveNet, the slower A2 LR preset and 100 epochs.
  Batch size is 32; initial audio/hypernetwork LR is 0.004/0.0004. Existing project
  and training-instance settings are retained. Static training keeps its A2
  advanced defaults; the new-project epoch budget is 100 for either capture type.

Self-contained installers: Windows x64, Linux x86-64 (DEB and portable archive),
macOS Apple Silicon and Intel. Existing optional NVIDIA runtimes are reused.
