# ParaNAM Trainer 0.5.9

- Added unexpected-shutdown recovery. Empty, truncated, null-filled or structurally
  invalid state no longer needs to block application startup. Damaged files are
  preserved; safe JSON backups are restored and training progress can be rebuilt
  from complete checkpoints, including ESR history, optimizer/scheduler state in
  the untouched checkpoint, and the last complete epoch.
- Mutable JSON state keeps one previous valid backup. Important writes and
  resumable checkpoints are flushed before replacement. One previous valid
  backup is retained for each mutable checkpoint path, including fixed best-size slots. Frequent live batch updates stay
  lightweight; a live worker is protected from checkpoint replacement.
- A visible recovery notice and modeless Review recovery window explain repairs.
  Users can retry, restore a JSON backup or explicitly rebuild a damaged queue
  index from recent projects. Rebuilt unfinished jobs are held for review rather
  than launched automatically. Essential unrecoverable configuration is not reset
  silently, and trained instances do not silently restart when checkpoints are lost.
  No recovery action deletes recordings or training directories.
- HyperA2 now reproduces the advanced settings saved in the 06/11 02:52 training:
  Adam; audio/hypernetwork LR 0.004/0.0004; decay 0.994; audio A2 cadence and separate
  control-scaled hypernetwork cadence; NY 8192; MRSTFT enabled at 0.0005; weight
  decay 3.17e-7; low rank 2; dilated convolution kernels excluded; Mean Full/Lite
  training loss. Clipping, NaN/Inf scanning, cuDNN benchmark and deterministic
  algorithms are off; the inactive clipping limit remains 1.0.
- New projects use that HyperA2 preset with batch 16 and 100 epochs. Existing
  projects and saved training instances retain their configuration. The
  HyperWaveNet preset remains separately selectable.
- Includes the Settings meters and white-noise connection test from 0.5.8.

See [state recovery](state-recovery.md) and [training controls](training-controls.md).

Installers: Windows x64, Linux x64 DEB/portable archive, macOS Apple Silicon/Intel.
