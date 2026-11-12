## ParaNAM Trainer 0.7.3

- **Static capture:** choose **Record** or **Load WAV** for a named capture. Import the aligned mono training and validation returns as two PCM WAVs or as one WAV with training immediately followed by validation. Use the project sample rate and match the selected reference lengths exactly. Imported returns keep their level and are labelled as not timing-verified.
- **Settings:** instrument-input calibration is global and stays in place when opening projects. Historical capture and job calibration snapshots remain intact. ASIO devices have a **Settings** button above **Refresh** to open the selected driver's control panel.
- **Train:** remembers the last author for projects that do not already specify one. **Input impedance** accepts Ω, kΩ and MΩ with either decimal separator. Export writes the optional numeric **input_z** in ohms.
- **Plan and Capture:** Avoid min also skips nonzero minimum positions. Capture diagrams show larger values and only the selected switch label. Every cell in the running capture row is bright green.
- **Recording integrity:** reported stream loss aborts capture immediately. A stopped or stalled stream also fails the capture instead of accepting a partial return.
- **PNAM compatibility:** ParametricSlimmableContainer still includes Full and Lite. This release writes legacy switch **enum_names** in neural order, including exports from resumed jobs. Existing indexed files remain readable; **index_sw** output is deferred.
