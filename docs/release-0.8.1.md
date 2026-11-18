# ParaNAM Trainer 0.8.1

Maintenance release on top of 0.8.0. No change to the .pnam model format.

- **Fresh projects with synthetic Mute captures:** timing-margin auto-calibration recognizes generated silent WAVs as belonging to the current capture plan. Physical delay labels remain protected; synthetic targets stay at zero.
- **Capture sampling method:** switching Sobol/LHS options immediately refreshes the recommended and selected training/validation capture counts, without requiring another Plan edit.
- **macOS Plan page:** Training/Validation count editors have room for digits and step buttons; form rows and long path/metadata labels avoid vertically clipped or overlapping text.
- **Train ESR:** held-out Train ESR excludes silent synthetic Mute recordings from its zero-energy denominator, as Validation ESR already does; synthetic captures still contribute to training MSE.
- **Queue epoch extension:** after increasing total epochs, the queue immediately displays the last completed epoch of the resumed member instead of resetting to zero. Checkpoints and optimizer state are preserved.

Draft preview: keep unpublished until all five native packages have been verified.
