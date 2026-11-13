# Knobs and switches in the capture app

The **Controls** tab replaces **Knobs**. Choose **Add control**, then **Knob** or
**Switch**. A knob keeps its range, step, Avoid zero and Gain/Drive settings. For a
switch, enter its name, number of positions (at least two), and a distinct name
for every position, in physical switch order. Use its **positions — edit** button
to change these later. Positions are categorical: intermediate positions are
independent states, not interpolated knob values.

## Exported switch option metadata

ParaNAM Trainer 0.7.3 exports switches using the legacy
`enum_names` array, in neural-value order, in the parent and both Full/Lite
submodels. For example, `["Clean", "Crunch", "Lead"]` assigns values 0, 1 and
2 respectively. A numeric `default` of 1 selects Crunch. The control-level
`index` still orders knobs, switches and Linear Volume.

The reader also accepts experimental `options` with separate neural
`value` and visual `index_sw` fields. The trainer preserves them in internal
project data but does not write this format into exported 0.7.3 models.

## Capture counts

Let K be the number of continuous knobs plus Simple Switch dimensions, and S be
the number of distinct Complex Switch combinations. The baseline heuristics are
**16 × K × S** training captures and
**min(max(8, 4 × K), 24) × S** validation captures.

The app now calculates the *actual* capture grid from each knob's min/max,
step, Avoid min, Avoid max and any discrete switch positions. It reduces the
recommendations automatically if the grid is too small for the baseline. The
default setting is part of training even when it is off-grid. Validation avoids
the default, all corner settings and every training setting; the recommendation
reserves enough room for those constraints.

For example, a knob spanning 0–10 in steps of 0.5 with Avoid min enabled has
20 usable positions per switch state. With a two-position Complex Switch the
suggestion is **32 training + 8 validation captures total** (16 + 4 per
state), rather than 32 + 16. A coarser grid may have no feasible held-out
settings; in that case the suggested validation count is zero. Use a finer
step or wider range to make held-out control validation possible.

The **Training LHS captures** and **Validation captures** numeric fields are
bounded dynamically: they cannot exceed the realizable unique combinations.
Raising either split reduces the permitted maximum of the other. The limit is
checked per Complex Switch state, including uneven totals and corners, rather
than merely multiplying the number of switch positions. Changing control
settings updates these limits. An already generated plan is never silently
regenerated, and its capture progress is preserved.

Corner captures repeat the existing continuous-knob corner design for **every**
Complex Switch combination. Duplicates of existing training captures are
skipped. The app displays the corner count before planning. These counts remain
heuristics and cannot guarantee model accuracy; complex rigs may benefit from
more captures or active learning.

With **switches only**, capture every categorical combination once. There are
no different control settings to hold out, but the main trainer still validates
the same states using the separate bundled validation audio. The older capture
app exposes zero held-out control validation settings for this case.

The initial plan groups captures by Complex Switch state in declared position
order to reduce switching, then sweeps the knobs within each group. Existing
generated plans keep their stored counts and progress. After changing control
names, types or switch positions, regenerate the plan before adding corners.

## Project and model files

Existing knob-only projects still load. The project retains its historical `knobs`
JSON key for compatibility, now containing both control types; switches carry
`type: "switch"`, `enum_names` and a default position index. Capture parameters
store the exact position names, including numeric-looking names such as `"1"`.

ConcatWaveNet, HyperWaveNet, HyperA2 and active-learning model configurations retain
switch types, all labels and the default index. Switch labels containing characters
that Windows disallows in filenames are escaped in filenames; the original labels
remain unchanged in project, dataset and model metadata. Existing input/output level
calibration metadata is preserved.
