# ParaNAM Trainer 0.8.0 beta 3

Experimental preview. This is a **draft**, not a public release. Built from
`feature/synthetic-mute-captures-080-beta3` without changing `main`.

## New in beta 3

- **Synthetic Mute Captures** is a fourth optional training mode alongside Mute
  Anchor, Virtual mute captures and Hypernetwork output-weight constraint.
  Selected Sobol, validation and corner settings which satisfy a Mute endpoint,
  Mute rule or previously marked exact Mute point receive an *ordinary* WAV
  target containing exact zero, the same number of frames and sample rate as
  that split's normal DI input. No separate anchor forward, scheduling or extra
  loss weight is introduced.
- The Capture screen shows those generated targets as **Captured** with a blue
  **Mute** tag in the **Tags/Warnings** column. They need no physical recording.
  Avoid/Unstable rules continue excluding settings; Mute rules are not Avoids.
- Each Mute rule is guaranteed at least one *training* corner or Sobol point
  satisfying the full AND combination. When the existing training plan has
  none, an extra synthetic Mute corner is appended.
- Synthetic Mute batches use the standard waveform **MSE**, not MRSTFT, and
  retain the existing full-length recording and capture-grouped batch sampler.
  No substitute training steps are imposed. Real recordings keep their normal
  MSE/MRSTFT loss and original RMS/SNR weighting.
- **Use -18 dBFS RMS reference for synthetic Mute captures** (default ON)
  assigns *only the normalization reference* a synthetic -18 dBFS RMS, so the
  MSE weighting is unity; the WAV itself remains exactly silent. Uncheck to
  test the actual zero target power with the ordinary RMS normalization floor.
- The global **RMS normalization floor is now -58 dBFS**, instead of -60 dBFS,
  for *all* recordings. With the optional reference disabled and exponent 2,
  a zero-target capture has a 10000x MSE normalization factor.
- Synthetic Mute captures contribute **no Train/Validation ESR** (a zero
  denominator is undefined). Their predicted output energy is recorded
  separately as **Mute RMS dBFS**. Real validation captures still select
  checkpoints normally.
- Full-size and Lite HyperA2 branches receive the same synthetic supervision;
  generated PNAM files retain the existing player format.

This beta is an experiment intended to measure mute depth and any effect on
accuracy near mute boundaries. Existing Mute Anchor and beta 2 methods remain
selectable without changed semantics (apart from the global -58 dBFS RMS floor).
A training job using another mute method cannot resume as Synthetic Mute
Captures; start a new job to compare.

## Downloads

The draft will contain five native installer assets after all platform builds
and frozen-package smoke tests succeed: Windows, Linux .deb and .tar.gz, macOS
Apple Silicon and Intel. Keep the draft unpublished.
