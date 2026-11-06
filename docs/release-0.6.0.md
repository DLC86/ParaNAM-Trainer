# ParaNAM Trainer 0.6.0

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
- Old saved projects, queue jobs and checkpoint scheduler counters are not
  migrated. Their original A2/control-based schedules remain supported and appear
  as Legacy entries when loading those settings. Fractional intervals and both
  independent scheduler states survive resume.
- Includes the automatic state recovery, safe backups, recovery review dialog and
  blackout resilience introduced in 0.5.9.

Normalization does not guarantee better generalization. See
[training controls](training-controls.md) for formulas and behavior.

Installers: Windows x64, Linux x64 DEB/portable archive, macOS Apple Silicon/Intel.
