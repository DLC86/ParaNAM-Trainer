# ParaNAM Trainer 0.6.2

- Capture advances to the next recording with its progress bar reset to 0%.
  Only controls whose positions changed pulse slowly toward light grey. Selecting
  a capture manually or confirming that the rig matches stops the pulse.
- Capture switches now use the same compact slide-switch drawing as Compare,
  retaining Capture's tile backgrounds and colors. Up to two knobs can be marked
  Gain/Drive; corner planning crosses both gain ranges with the other controls'
  corners and switch combinations. The corners checkbox has a clearer name.
- Compare opens without a project. Its model controls keep a fixed minimum gap
  and wrap into as many columns as fit. Models and IRs are shown in two adjacent
  lists at the bottom. The Instrument meter has the width of one Monitor channel.
- Numeric fields and knob values accept either a decimal point or comma.
  Typed model knob values continue to apply on Enter or loss of focus.
- Replaced both LR preset menus with independent cadence selectors beside each
  learning rate: **per epoch** and **per 62 optimizer updates (A2 cadence)**.
  Both decay multipliers remain editable. Legacy editor values are represented
  by an equivalent average decay; newly queued/Retrain jobs use that representation.
  **Existing jobs remain resumable:** resume preserves their frozen original
  scheduler configuration, optimizer state and checkpoint counters. Equivalent
  average decay does not imply identical discrete reduction instants for a new job.
  Normal recording/architecture/evaluation compatibility checks still apply.
- New parametric projects and the HyperA2 preset use Adam, batch 16, LR 0.004,
  hypernetwork LR 0.001, decay 0.994/0.998497 per 62 updates, MRSTFT enabled,
  NY 8192, MRSTFT weight 0.0005, weight decay 0.000000317, rank-2 low-rank
  hypernetwork deltas, shared dilated convolution kernels and summed Full/Lite
  losses. Clipping, NaN/Inf gradient checks, cuDNN benchmark and deterministic
  training default off. HyperWaveNet is unchanged except cuDNN benchmark is off.
  Existing projects retain their settings; hypernetwork fields stay hidden in static mode.
- Default epoch budgets are 50 for parametric and 200 for static training.
  Switching capture type remembers each budget separately.
- Queue now allows every row to move up or down, including running and finished
  jobs. The running process remains associated with its instance. Removed the
  paused-job batch/device editors; above-list actions align right except Start/Pause
  queue. Renamed the actions to **Pause selected training** and **Cancel selected training**.
- Elapsed time advances with a monotonic clock, independently of worker updates.
  Elapsed and remaining time retain seconds after one hour.
- Added **Open project folder** for the selected project on the Projects page.

Installers: Windows x64, Linux x64 DEB/portable archive, macOS Apple Silicon/Intel.
