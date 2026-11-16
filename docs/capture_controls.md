# Knobs and switches in the capture app

The **Controls** tab replaces **Knobs**. Choose **Add control**, then **Knob** or
**Switch**. A knob keeps its range, step and Gain/Drive settings. **Mute at minimum** and **Mute at maximum** each teach the network silence at the selected endpoint **and** exclude that position from new capture plans (including validation and corners). Legacy Avoid min/max flags in existing projects remain supported without enabling mute training. For a
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
step, automatically excluded mute endpoints, legacy Avoid min/max flags and any discrete switch positions. It reduces the
recommendations automatically if the grid is too small for the baseline. The Plan checkbox **Include default as additional capture** is enabled by
default for newly generated plans. It adds one training capture at the complete
default combination **in addition to** the requested LHS/Sobol count, rather
than changing any of the sampler's points. With the checkbox off, the default
is not added as a special training capture and the actual LHS/Sobol samples
are used (the exact default combination is excluded). Existing project plans
are left unchanged until Plan is edited and applied. Validation avoids the
default when reserved, all corner settings and every training setting; the
recommendation reserves enough room for those constraints.

For example, a knob spanning 0–10 in steps of 0.5 with Mute at minimum enabled has
20 usable positions per switch state. With a two-position Complex Switch the
suggestion is **32 training + 8 validation captures total** (16 + 4 per
state), rather than 32 + 16. A coarser grid may have no feasible held-out
settings; in that case the suggested validation count is zero. Use a finer
step or wider range to make held-out control validation possible.

The **Training captures** and **Validation captures** numeric fields are
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

## Discovering mute and unstable settings during capture

In the **Capture** screen, select a pending or already recorded parametric
training/validation setting and use **Mark as mute** or **Avoid this setting**.
These are the only two classifications: *Avoid* includes unstable, dangerous
or otherwise unwanted combinations. The selected point is removed from the
active training/validation set and stored in the project; its WAV, if recorded,
is retained on disk but excluded from training. A discarded sampled
LHS/Sobol point is replaced with a fresh, distinct setting in the same split.
Other recorded settings and the original sampled points are not regenerated.
Complex Switch states remain fixed when replacing a sampled setting.

A point marked **Mute** becomes an **exact virtual silence anchor**. The
training network learns that *complete* setting without a WAV; marking
Gain I=0 and Gain II=0 does **not** claim that either Gain alone mutes the amp.
Known mute endpoints defined in Controls continue to generate their broader
Mute Anchor conditions as before. Virtual points are training-only data; the
exported model and players require no new fields.

For a muted **Corner**, the planner checks the nearest playable step along
each continuous control presently at a range endpoint. If no training
capture already covers the step along that axis, it adds the corner setting
with that one control moved to the adjacent step. This creates a small
boundary set for mute transitions, without blindly filling the entire
corner grid. A marked *Avoid* Corner is replaced by a fresh sampled
training point; no adjacent Corner is presumed safe. The conditions and replacements survive saving, reopening
and plan regeneration. **Undo last mark** restores the most recent annotation
and any still-unrecorded additions; it refuses to remove audio acquired in
those additions. Switch-only projects cannot apply this replacement workflow
because there are no distinct spare states to select.

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
