# ParaNAM Trainer 0.7.7

- **Rig controls:** optional **Mute at minimum** and **Mute at maximum** flags, independent of capture-avoidance thresholds. Use for controls that mute the device at either endpoint, including reversed controls.
- **HyperA2 training:** Mute Anchor regularization uses real training audio, a selected control pinned to its mute endpoint, and randomized settings for all other controls; Full and Lite learn to suppress their output. No muted recording is required.
- **Advanced training settings:** Mute Anchor Interval, Batch Size and NY default to 1 step, 2 examples and 512 samples. On the selected optimizer steps, the loss weight is multiplied by the interval.
- **Parametric training:** **RMS loss normalization** is off by default. Enable it to weight each capture's MSE relative to a −18 dBFS RMS reference with a −60 dBFS floor. With RMS enabled, **SNR weighted loss normalization** can further reduce the MSE weight of noisy captures using the pre-blip silence in the raw recordings.
- **Training queue:** SNR noise measurements follow the frozen captures. Failed jobs show **Retry** and use the existing checkpoint recovery path.
- **Fixes:** Fixed repeating +/− clicks in numeric training settings and NVIDIA runtime extraction on long Windows paths.
- **Compatibility:** this is a training-only loss. Exported PNAM architecture and parameter schema are unchanged; existing players remain compatible. Learned mute is near-silence rather than a guarantee of mathematically exact zero.
- **Linear Volume calibration:** every other numeric control is parked at 10 (or its nearest physically reachable value), switches at their highest position, and capture rules/Avoid thresholds are ignored for the volume sweep. The volume knob is still measured across its entire range.
- Existing projects retain their previous Mute Anchor behavior unless a mute endpoint is explicitly enabled.

**Full Changelog:** https://github.com/DLC86/ParaNAM-Trainer/compare/v0.7.6...v0.7.7
