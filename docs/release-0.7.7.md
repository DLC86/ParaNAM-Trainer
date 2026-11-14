# ParaNAM Trainer 0.7.7

- **Rig controls:** optional **Mute at minimum** and **Mute at maximum** flags, independent of capture-avoidance thresholds. Use for controls that mute the device at either endpoint, including reversed controls.
- **HyperA2 training:** Mute Anchor regularization uses real training audio, a selected control pinned to its mute endpoint, and randomized settings for all other controls; Full and Lite learn to suppress their output. No muted recording is required.
- **Compatibility:** this is a training-only loss. Exported PNAM architecture and parameter schema are unchanged; existing players remain compatible. Learned mute is near-silence rather than a guarantee of mathematically exact zero.
- Existing projects retain their previous behavior unless a mute endpoint is explicitly enabled.

**Full Changelog:** https://github.com/DLC86/ParaNAM-Trainer/compare/v0.7.6...v0.7.7
