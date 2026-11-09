# ParaNAM Trainer 0.7.2 preview

## Capture and Queue list selection

The Captures and Queue tables preserve their per-cell foreground colors when
selected. Running (green), captured/completed (cyan), warning (yellow) and
other status text no longer becomes white on selection. The row selection
background remains visible.

## Selectable LHS sampling

Plan now exposes four reproducible Latin-hypercube designs:

- **Standard**: the existing seeded Latin hypercube; retains older behavior.
- **Maximin**: bounded best-of LHS search maximizing the minimum Euclidean
  distance between any pair of *unit-design* points.
- **Orthogonal**: true strength-2 orthogonal-array LHS using SciPy. It requires
  n = p² samples per split and Complex Switch combination, for a prime p,
  and at most p+1 sampled dimensions. Examples: 9, 25, 49 or 121 settings;
  the count can vary based on the number of dimensions.
- **Near-orthogonal**: bounded best-of LHS search minimizing the sum of
  squared Pearson correlations between the sampled dimensions, without the
  orthogonal-array count restriction.

Standard remains the default for existing projects. The selected LHS design
is stored in the project and is reapplied whenever the plan is regenerated.
Training and validation use separate deterministic streams. Simple Switches
are sampled as categorical LHS dimensions; Complex Switches use independent
plans for every joint switch state. Existing default-setting anchors,
no-duplicate capture-grid quantization, corner captures and volume controls
still apply.

These are properties of the initial unit-cube sample: selecting an explicit
default setting, per-knob capture density, Simple Switch discretization,
quantization and duplicate resolution can weaken ideal LHS design properties.
Selecting a different method does not silently modify existing recordings:
applying a changed plan follows the normal capture-plan regeneration flow.

## Recommended number of LHS captures

The planning recommendation is now **16 training LHS captures per sampled knob
or Simple Switch dimension**, multiplied by the number of Complex Switch
combinations. Validation recommendations keep their prior formula. Saved
capture counts and plans are preserved on upgrade.
