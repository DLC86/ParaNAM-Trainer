# ParaNAM Trainer 0.6.0

**Upgrade warning:** the original A2, control-scaled A2 and A2 Slow learning-rate
presets have been removed for both the audio network and hypernetwork. Training
instances using these presets **cannot resume after updating to 0.6.0**, including
previously queued instances. Keep the previous app version to finish those runs,
or use Retrain to start a new instance with a new preset. Recordings, training
history and checkpoints are preserved; best validated checkpoints remain exportable.


- Replaced the new-project learning-rate choices with Dataset-normalized and
  Dataset-normalized √, independently available for the audio network and the
  hypernetwork. Both keep discrete A2 decay: multiply LR by 0.994 every
  `62 × S/62` or `62 × sqrt(S/62)` optimizer updates, where S counts complete
  training updates per dataset epoch. Validation captures/audio are excluded.
- Header-based accounting mirrors the real loader, including per-capture
  drop-last batches, cropping, context, alignment, NY and batch size. The preview
  shows actual updates, the reference ratio, interval and final rates.
- HyperA2 and new projects default to linear normalization for both independent
  groups, LR 0.004/0.0004. Other requested 06/11 02:52 advanced values are retained,
  along with batch 16 and 100 epochs. HyperWaveNet and static scheduling are unchanged.
- Removed all older A2 scheduler choices rather than retaining Legacy entries.
  Older projects select Dataset-normalized for a new training. Existing jobs
  using removed schedules are held and cannot launch/resume. No checkpoint or
  recording is deleted. New scheduler counters and fractional intervals survive resume.
- Hide all hypernetwork-specific settings and labels in static capture mode;
  switching back restores the saved parametric configuration.
- Full and Lite ESR charts keep the X axis fixed to the selected job's total
  planned epochs, including hover coordinates and before the first validation.
- Added None to every audio input/output routing selector. The choice is saved
  per interface and preserved on reconnect. Unassigned Settings meters show zero,
  test noise feeds only assigned outputs, and required routes are checked before
  capture, calibration or playback.
- Added a background GitHub Latest release check at startup. The update notice
  offers Download and install, Release notes and Dismiss. Dismiss persists for that
  release; a blinking clickable LED beside VERSION remains while an update is
  available. Downloads are verified with published size and SHA-256. Installation
  waits for safe app shutdown/pause and then reopens the application. Windows uses
  the silent native installer, macOS updates the app bundle, Debian uses the system
  installer authorization, and Linux portable replaces its app folder with rollback
  preservation. Source checkouts open the downloaded native package instead.
- Includes the automatic state recovery, safe backups, recovery review dialog and
  blackout resilience introduced in 0.5.9.

Normalization does not guarantee better generalization. See
[training controls](training-controls.md) for formulas and behavior.

Installers: Windows x64, Linux x64 DEB/portable archive, macOS Apple Silicon/Intel.
