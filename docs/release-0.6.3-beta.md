# ParaNAM Trainer 0.6.3 beta

This beta is prepared on `beta/0.6.3-cosine-validation`. The release remains a
draft; the public 0.6.2 release and main branch are unchanged.

- Added **Cosine (whole budget)** independently for the audio network and
  hypernetwork. It warms up linearly from zero to the configured initial/peak LR
  over the first 2% of optimizer updates, then follows a cosine curve to 1% of
  that LR at the end of the epoch budget. The budget uses actual complete
  training batches after validation exclusion, context and per-capture drop-last.
  Every rate is computed directly from the base LR and saved step counter.
  Both budgets and counters are frozen in the job/checkpoint and restored on
  pause/resume, without re-estimating the schedule. Existing epoch/A2 cadences
  remain available. The decay field is inactive for a group using cosine.
- Increased recommended held-out validation captures to
  `min(max(8, 4 × knobs), 24)` per joint switch combination. The default
  `n_validation` for new projects is 12. Saved plans keep their previous counts;
  captures remain distinct from training settings. Switch-only models retain
  their existing behavior because there are no unrecorded control settings to
  hold out after all switch states have been captured. Coarse knob grids must
  have enough distinct settings for both train and validation counts.
- Added **ESR_worst**, the highest ESR of a complete held-out control setting.
  It pools audio energies within each setting and takes the maximum across
  settings, so quiet failures cannot be hidden by louder recordings. Logged
  alongside mean ESR for Full, Lite and the combined validation score, and saved
  in progress/history/checkpoints. It reuses validation predictions without an
  extra inference pass. Missing capture identities are not reported as a false
  worst-setting score; nonfinite setting scores remain visible as failures.
- Best-checkpoint selection continues to use mean Full validation ESR. Existing
  jobs retain their original scheduler and validation plan on resume. Use a new
  training instance to try cosine; changing the number of held-out settings
  requires updating the capture plan and recording those captures.

Draft installers: Windows x64, Linux x64 DEB/portable archive, macOS Apple Silicon/Intel.
