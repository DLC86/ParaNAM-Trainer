# ParaNAM Trainer 0.8.0 beta 3

**Experimental preview / draft release.** This build is from
`feature/synthetic-mute-captures-080-beta3` and does not change `main`.

## Synthetic Mute Captures — the only mute training mode

- The old **Mute Anchor**, **Virtual Mute** and **Hypernetwork weight constraint**
  choices have been retired from the UI and from newly created training jobs.
  There is no mute-mode selector and no mute-anchor interval, batch size or NY.
  Existing saved controls/recipes containing those keys can still be opened;
  new jobs always use dataset-level Synthetic Mute Captures.
- Every Sobol or LHS sample, held-out validation point, and normal Corner matching
  **Mute at minimum / maximum**, a complete **Mute Rule** or a manually marked
  exact Mute combination becomes a completed **Captured** row with a blue
  **Mute** tag in the **Tags/Warnings** column. No physical recording is needed.
  The model receives an ordinary training DI and an all-zero target of the same
  length and sample rate. Synthetic captures count within the requested training,
  validation and real-extrema corner totals.
- All six capture samplers — Standard LHS, Maximin LHS, Orthogonal LHS,
  Near-orthogonal LHS, Sobol and Optimized LHS — follow the same rules.
  **Avoid** still excludes unstable combinations from planning and uses the
  previous replacement and corner logic. It never turns them into Mute.
- Corners use real control minima/maxima, not the old adjacent-to-mute step.
  An additional training corner is generated **only if a multi-control Mute
  rule has no already-planned real corner** satisfying its complete AND
  condition. Existing sampled corners are reused whenever possible.
- All synthetic Mute entries for the training split reuse **one shared silent
  WAV**, while validation reuses it where the length and sample rate match or
  generates a second, validation-sized silent WAV. Physical recordings are
  never overwritten by these shared target files.
- Mute batches use waveform MSE without an extra forward or replacement-step
  schedule. MRSTFT is omitted only for their exact-zero targets; the ordinary
  loss is unchanged for actual captures. Train and validation ESR omit mute
  entries and report their predicted residual as a separate **Mute RMS dBFS**.
  Both Full and Lite HyperA2 branches are covered without player-format changes.

## RMS normalization and training defaults

- **Use −18 dBFS RMS reference for synthetic Mute captures** now defaults
  **OFF**. With the default off, true-zero mute WAVs are normalized using the
  global **−58 dBFS RMS floor**. Turning it on assigns a reference power
  equivalent to −18 dBFS for unity RMS loss weighting, without touching the
  actual silent target.
- The suggested parametric training length is now **100 epochs**.
- The default loss settings are **RMS ON**, **SNR OFF**, and
  **gradient clipping ON**. User-customized saved settings remain editable;
  newly created projects use the new recommendations.

## Compare

- Model entries discovered from **project model folders** and from
  **completed training exports** can be removed from the Compare **Models**
  list without deleting their NAM/PNAM files on disk. Removal hides the entry;
  manually loading the file again restores it.

## Installation

The draft release will be updated with five native installers after the
test suite, packaging and platform smoke tests complete: Windows, Linux DEB,
Linux TAR.GZ, macOS Apple Silicon and macOS Intel. Keep this release as a
**draft** (unpublished).
