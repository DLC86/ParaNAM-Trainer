## ParaNAM Trainer 0.7.4

- **Plan:** Rig controls now use the same control cards as Capture. Cards wrap onto new rows, can be reordered by dragging, and have a red × for removal. Use the grey dashed + card to add a knob. Input signal now follows Rig controls.
- **Control settings:** Edit settings inline with automatic saving. Switch positions use the classic editable list with Add position, Delete, Move up and Move down buttons. Reordering keeps the default on the same named position.
- **Plan and Capture:** The control cards and the dashed Add tile share the same dimensions, min/max labels have more horizontal clearance, and the knob arc and pointer span exactly 300°. Knobs show their minimum and maximum. The red arc starts at zero when zero is in range, or at the minimum otherwise, and extends to the current position.
- **Capture and Train:** Capture keeps the audio stream open between takes. Train metadata stays editable while recording and edits are saved when the capture finishes. The Train icon now changes from grey to red as a single glyph.
- **Queue:** Column widths follow their contents. Validation Full ESR is now Val. ESR.
- **Window:** ParaNAM remembers the previous window geometry, and main panels expand horizontally with the window.
- **Export:** Parametric switches use `options` with a stable neural `value` and a separate display `index_sw`, including the Full/Lite branches. Static NAM containers and their submodels use format version `0.7.0`.

- **Capture plans:** Training and validation count fields honor the real capacity of the control grid, preserving the latest main safeguards.
