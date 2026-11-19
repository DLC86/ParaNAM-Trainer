# Model Editor

The **Editor** page is available from the left navigation, immediately above **Settings**. It does not require an open capture project.

## Loading and saving

1. Choose **Load NAM / PNAM…** and select an existing `.nam` or `.pnam` file.
2. Edit Author, Manufacturer, Gear model, Gear type, Tone type, input/output level (dBu), input impedance, or Notes. Gear and tone type use the same choices as Train's **Model metadata**.
3. Choose **Overwrite file** to update the loaded model (and retain the previous version in `<filename>.bak`), or **Save copy as…** to write a new file without changing the original.
4. The Editor validates numerical metadata and the model's JSON structure before saving. Saves use a temporary file, flush/fsync, and atomic rename. Overwriting an existing copy target also makes a `.bak` backup.

The Editor preserves weights, model configuration, computed loudness/gain, unexposed metadata, IR routing, and the model's name. It never retrains, bakes or converts the model.

## Parametric models

Only when a `.pnam` is loaded, the **Parametric controls** panel appears.

- **Control name:** change the displayed knob/switch name. Updated consistently in all Full/Lite parameter descriptions; the neural parameter position remains unchanged.
- **index:** control display order, 0-based. Moving a control changes only this index, never the order of neural arrays or weights. A Linear Volume extension, if present, is included in the display order.
- **Switch option name:** rename one switch position. The underlying numeric **value** is preserved.
- **index_sw:** change the physical/display order of the switch options. Each option retains its original neural `value`, and index_sw remains a 0-based permutation.
- **Knob min/max:** shown as read-only values. They define the signed min/max input normalization used by the trained network; changing them without adjusting the neural network could change the sound, so the Editor deliberately does not offer this unsafe operation.

For multi-size parametric models, edits to neural parameter descriptions are mirrored into their Full/Lite branches. Legacy `enum_names` arrays remain in the original neural-value order for compatibility; modern `options` records preserve each value while carrying the new index_sw.

If a model's branches have incompatible parameter descriptions or validation fails, the Editor refuses control edits and does not write an invalid file.

## Compare layout

On the **Compare** page, Play & Compare, the explanatory text, Playback buffer, Model size and Output mode now share one left column. Instrument and Monitor L/R meters occupy a separate aligned column to the right of this entire block.
