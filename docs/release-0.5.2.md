# ParaNAM Trainer 0.5.2

1. The current audio driver, channels, buffer and level calibration are remembered
   across restarts and normal project opens/creation. Existing recorded calibration
   is retained in capture metadata; current playback references are separate.
   Applying a changed capture output/calibration uses the existing plan/history flow.
2. **Input signal** precedes Rig controls. TONE3000 Standard, bundled **TTS input
   v11** (mono, 48 kHz, 300 seconds) and Custom are available. The fixed repository
   validation WAV remains unchanged. TTS is installed into a project when selected,
   and its identity is preserved in queued instances and editable Retrain copies.
3. Model type choices are **Parametric** and **Static**.
4. Static mode hides Capture plan & learning method.
5. Capture progress reaches 100% for each recorded signal, including both signals
   of a static capture. The status identifies training versus validation.
6. A recording locks capture selection, while the table can still scroll and its
   columns can still be resized.
7. **Retrain** is between Resume and Export best NAM/PNAM. It opens an editable
   capture/configuration copy for a fresh training run, including started, paused,
   stopped, failed and completed jobs. New jobs waiting in the queue are excluded.
   Unrecorded active-learning suggestions are omitted from this copy.
8. ESR chart and Training log fill the space below the compact queue summary.
9. Relative knob dragging retains the value on mouse release. Parametric model
   knobs use the Capture your rig ring, body and pointer appearance.
10. Input and Output both span **−20 to +20 dB**, with 0 dB at the center. Output
    defaults to 0 dB.
11. **Play** has the same rounded appearance as Gear/Model with a smaller size/font.
12. The sidebar uses Michroma and custom routing, knob, capture, network, queue and
    guitar icons instead of step numbers.
13. Recent projects show larger bold names above their paths.
14. Selected checkbox indicators show a red checkmark on the neutral background.
15. Final `.nam` and `.pnam` basenames use the exported Name metadata. Invalid OS
    filename characters are replaced; static names that collide after replacement
    use separate capture folders to avoid overwriting models.
16. Model, Full/Lite size and playback buffer can be changed during Play. The app
    ramps down, applies the change and automatically restarts listening; reopening
    the audio stream can briefly interrupt sound. Full/Lite changes retain knob
    positions; a different model loads its own defaults. Model size is beside I/O.
17. Back retains its normal button width in Play & Compare.
18. Project, input/model load and model export pickers share the last visited folder.

The training recipes, nested Full/Lite format and relative output levels are
unchanged. Hardware listening and physical converter calibration still require
verification with an actual interface; automated tests use native model rendering
and simulated audio streams.
