# ParaNAM Trainer 0.6.1

- Restored the original update/control cadences and use this fixed order in both
  LR selectors: A2 Cadence, A2 Slow, A2 Slower, A2 Dataset-normalized √,
  A2 Dataset-normalized, then Manual. Follow is no longer a selectable mode.
  Intervals are respectively `62`, `62 × F / sqrt(2)`, `62 × F`,
  `62 × sqrt(S/62)`, and `S` optimizer updates, where
  `F = max(1, knobs) × 2^switches` and S is complete updates per training epoch.
  A2 Slow keeps its previous formula; it does not use sqrt(F). Initial LR values
  remain editable and new-project defaults remain A2 Dataset-normalized.
- LR Decay and Hypernetwork LR Decay are editable with every preset. Selecting a
  cadence keeps the entered decay; preview and training use each group's value
  rather than forcing 0.994. The default remains 0.994. Values 0 and 1 are supported.
  Legacy Follow configurations display their explicit effective cadence and decay;
  unchanged legacy checkpoint identities are preserved, while any separate edit
  creates independently scheduled settings.
- Removed the 0.6.0 block on jobs using the earlier preset IDs. IDs, effective
  settings, frozen scheduler intervals, optimizer state and checkpoint counters
  are preserved on resume. No training is automatically restarted or migrated.
  Normal capture, architecture and evaluation-protocol checks remain in force.
- Fixed the paused-queue/Retrain path: starting a queue can again resume the older
  paused instance instead of being blocked before a newly retrained job can run.
  Retrain restores the effective instance settings rather than stale editor values.
- Full and Lite ESR charts now start with the same fixed logarithmic range:
  0.005 at the bottom, 0.5 at the top. The range remains unchanged while scores fit
  inside it; only an exceeded side expands by logarithmic decades. Empty charts,
  zero ESR and nonfinite scores remain safe. The X axis still spans all planned epochs.
- Includes the state recovery, None audio routes, static-mode field visibility,
  and verified GitHub update notifications introduced in previous releases.

Installers: Windows x64, Linux x64 DEB/portable archive, macOS Apple Silicon/Intel.
