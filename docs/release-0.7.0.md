# ParaNAM Trainer 0.7.0 preview

- Complex Switch keeps the existing plan for each joint switch state. Simple
  Switch adds one discrete Latin-hypercube dimension, contributing to suggested
  capture counts like one knob. Only actual switch positions are captured. Both
  modes use the same categorical neural-model encoding.
- Knobs can concentrate captures in the lower or upper half of their range.
  A mirrored exponential sampling transform places about 73% of initial draws
  in the selected half before capture-grid rounding and duplicate removal.
  Physical knob values and model normalization remain unchanged.
- Avoid max excludes the upper endpoint from LHS and corner captures, mirroring
  Avoid zero. Clean output volume requires its maximum as the reference and
  therefore cannot use Avoid max or Gain/Drive.
- Clean output volume trains the neural effect at maximum volume and measures
  a separate attenuation curve with short validation-signal recordings. Nine
  initial positions are checked at intermediate points; additional measurements
  are requested where needed, up to 65 curve positions. A second control context
  checks that attenuation is independent of the other controls. Waveform/taper
  failures preserve recordings and block training until corrected or replanned.
  Calibration is the first phase of the saved plan, including any adaptive extra
  points. Neural captures are blocked until it passes, including manual selection.
  Maximum-volume references are recorded first. A silent minimum is measured as
  a noise-floor clip, without requiring timing alignment or a loopback. Audible
  captures retain timing checks; an explicitly configured loopback must work.
  The 0.25 dB target is checked at measured positions, not a guarantee between
  arbitrary physical pot positions. At least one other neural control is required.
- Clean volume is stored in a ParametricOutputGain wrapper and applied after
  the neural model using a monotone PCHIP curve and 5 ms smoothing. Compare,
  NAM Parametric Plugin 0.4.5 preview and NeuralAudio 0.1.4 preview support this
  format. Older players reject it; volume-only changes do not regenerate neural
  weights. Static export in the new plugin includes the selected volume gain.
- Adds **Truncated (Phillip Self)**: the original 38-second inputTrunc.wav plus
  the bundled NAM standard first 10 and last 9 seconds (57 seconds total).
  The original 38 seconds remain unchanged and form the training segment;
  the validation segments stay excluded from fitting. Separate held-out captures
  continue to use the existing short validation signal. Source:
  https://github.com/phillipmself/neural-amp-modeler-parametric/releases/tag/training-inputs-v1
- LR cadence selectors now sit beside their respective decay fields, with
  aligned numeric fields and transparent row containers.
- Held-out validation recommendation is min(max(8, 4 × dimensions), 24) per
  Complex Switch state; Simple Switch dimensions count alongside knobs. The
  initial validation count is 12. ESR_worst reports the worst complete held-out
  setting alongside mean ESR for Full/Lite.
- Removes the draft 0.6.3 beta Cosine schedules. Existing ordinary-cadence jobs
  retain resume compatibility; beta Cosine jobs require a new training.

Released as a preview on main; Latest remains unchanged. Existing projects and
recordings are not silently regenerated when new planning features are selected.
