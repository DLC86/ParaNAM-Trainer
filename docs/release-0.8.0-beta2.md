# ParaNAM Trainer 0.8.0 beta 2

Experimental draft build based on the 0.8.0 development branch. This build is **not a stable release**.

## Three selectable mute training methods

Under **Advanced training settings → Mute training method**, choose one method:

- **Mute Anchor (audio forward)**: original Mute Anchor; adds a separate audio forward with real guitar input and zero output target (default, retains existing behavior).
- **Virtual mute captures (zero target)**: substitutes 1 in 20 ordinary training minibatches with the same guitar input conditioned on a sampled muted control combination and an all-zero output target. Uses a dedicated quadratic silence loss; bypasses RMS, SNR and MRSTFT normalization for synthetic silent targets. No extra audio forward. This trades 5% of normal audio training batches for virtual mute supervision.
- **Hypernetwork output-weight constraint**: on each training step, samples a muted combination and directly regularizes all conditioned parameters of the final WaveNet head-rechannel layer. No additional audio forward, only a lightweight hypernetwork computation. Requires the final output stage to be conditioned, which the included HyperA2 model supports.

All three methods support mute-at-min/max, explicitly tagged mute points and AND-based mute rules. The existing mute settings (interval, batch size, NY) remain applicable to the audio-forward Mute Anchor option. Capture planning and the `.pnam` architecture/serialization remain unchanged.

### Testing guidance

These methods are experimental and not yet proven equivalent in audio quality. Compare the muted residual level in dBFS, ESR on non-mute settings, transition behavior near mute boundaries, gradient stability and wall-clock step time. A network-based constraint cannot guarantee bit-exact zero for every unsampled setting.
