# ParaNAM Trainer 0.5.0 workflow

## Project and capture

Project & routing selects driver-reported channels for the reamp send, gear
return, optional timing loopback, Instrument input, and Monitor out L/R.
Instrument and return must be different inputs; send and the two monitor channels
must be three different outputs for Play. Capture still uses its original send,
return and optional loopback route. Instrument/monitor settings do not alter
recorded capture levels or the timing checks.

Controls & plan retains knobs, switches, capture coverage and the existing
training/active-learning choices. Model metadata/defaults for static captures
have moved to Train & export; per-capture static overrides remain available.
The 0.4.4 recipes, gradient safeguards, independent hypernetwork controls and
Full/Lite ESR/export behavior are retained.

## Training instances

Every Train click creates a new instance. It copies the click-time settings and
metadata and freezes the referenced audio. The queue runs one worker at a time.
Start queue starts waiting jobs in their displayed order. A running job does not
lock the project editor or audio interface: another project can be opened,
modified, captured and added to the queue while training continues.

Training queue owns Start, Pause, Stop, Resume, ordering, removal, intermediate
best-model export, progress, ESR charts and the training log. Selecting a row
shows that instance's history, including completed jobs. Moving a waiting row
changes its execution order. Removing it hides the instance without deleting its
models, recordings or checkpoints, and does not automatically re-import it.

Pause takes effect at the next safe batch boundary. Resume uses that instance's
last complete epoch, including after closing/restarting the app or after a
recoverable failure. The open project's current settings have no role in resume.
Resume batch size/device can be changed in the queue after e.g. an out-of-memory
error; validation retains batch size one. Before the first complete epoch,
resume restarts the same instance from its initial configuration.

The scheduler continues to other waiting jobs when an instance ends or is
paused/stopped. Closing the app holds the queue and pauses its active worker
before exiting; restarting the app requires Start queue/Resume. A missing
project drive marks that instance unavailable without blocking other projects.

The small queue index lives in application data. Instance files and a
content-addressed audio store live under each original project's
`training/queue-data`, keeping large data on the project's drive. The app never
hardlinks mutable original recordings: cache copies are made first, and immutable
instances share those copies where the filesystem permits. Exports use distinct
instance folders under `models`, so rerunning the same project cannot overwrite
an earlier instance's result. Existing jobs from known/opened projects are listed
without replacing their old checkpoints or exports.

For active learning, a finished round shows **Capture next active-learning
round**. This opens an editable project branch containing the suggested captures.
Its audio files are real copies: recording there cannot modify an older training
instance. Capture the additions and press Train to enqueue the next round.

## Calibration and Play

Send/return calibration is unchanged. To calibrate Instrument, stop the sine,
move the same measured cable directly to the configured instrument input,
select Instrument in the calibration dialog, confirm the direct connection,
restart the sine and apply instrument calibration. A clean, unclipped sine is
required. The instrument's 0 dBFS reference is:

`instrument_level_dbu = measured_voltage_dbu - measured_sine_peak_dbfs`.

The instrument calibration is separate from model capture metadata and stored in
the project. Changing its selected input/driver invalidates this reference.
Keep the interface input gain at its calibrated setting.

Play discovers exported models from the queue/project and can load additional
`.nam`/`.pnam` files. It uses the plugin's native CPU DSP, with Full/Lite selection,
live parametric controls (including multiple switches), and Input/Output gains.
Static models need only the two level controls. Playback has a separate,
driver-reported buffer selector; lowering it reduces playing latency without
changing capture settings.

Audio is explicitly started/stopped. The instrument feeds both the model and the
real gear; Gear/Model chooses only which source reaches Monitor L/R. Both paths
keep processing during comparison, and source/gain transitions are smoothed.
Only send and the selected monitors receive output; the timing-loopback output
is silent during Play. The same mono signal goes to the two monitor channels.
Disable interface direct monitoring and set real gear controls to match the
displayed model settings when comparing. Stop Play before recording/calibrating
with the interface.

With calibration available, model input gain is the instrument interface's dBu
reference minus the model's input reference. Model output gain is the model's
output reference minus the calibrated return reference, matching the real return
in digital units. The reamp send also compensates the instrument reference and
capture-output attenuation. Output volume remains authentic: there is no RMS
normalization. Missing calibration references use raw gain for that part of the
route and are identified in the Play screen.

## Validation scope

Tests cover independent static/parametric instances, editing/capturing another
project during training, queue order/removal/recovery, real worker completion and
checkpoint resume, instrument calibration and unclipped-sine requirements,
feedback-route rejection, output channel isolation, and Full/Lite native DSP
against Python references with multiple switch combinations. Installers verify
native playback from the frozen app using actual trained exports.
Physical converter calibration and listening still require the user's hardware.
