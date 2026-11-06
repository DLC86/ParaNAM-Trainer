# ParaNAM Trainer 0.6.1

- Restored all previous learning-rate cadences with descriptive names:
  Update-based (original A2, every 62 updates), Control-normalized
  (`62 × max(1, knobs) × 2^switches`), and Control-normalized / √2
  (the former A2 Slow). These are independently selectable for the audio network
  and hypernetwork alongside Dataset-normalized, Dataset-normalized √, Manual
  and hypernetwork Follow. New-project defaults remain Dataset-normalized.
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
