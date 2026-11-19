# ParaNAM Trainer 0.8.4 — Compare layout and asset management

## Play & Compare layout

- The parametric control panel reserves the full height of a **single row of knobs/switches**, even when the selected model has no editable controls or no model is loaded.
- The **Instrument** and **Monitor L/R** VU meters have moved to the **upper-right corner**, aligned vertically with the **Play & Compare** title.
- **Input** and **Output** level knobs sit at the same height as the **Play / Gear / Model** action block, with Input anchored to the left and Output to the right.
- **Model size** and **Output mode** each occupy a dedicated row immediately **below Playback buffer**, rather than sharing a row with the gain controls.
- Both the **Models** and **Impulse responses** lists reserve enough height to display at least **ten rows** before scrolling.

## Clear list buttons

- Added **Clear list** for Models: removes all listed models, including those automatically discovered in project output folders or the training queue, while leaving the actual model files untouched. Previously cleared models stay hidden after refresh and application restart; manually importing a model makes it available again.
- Added **Clear list** for Impulse responses: removes only imported/custom IRs. The **six bundled Tone3000 IRs** and the **Off** selection always remain. Actual IR files are never deleted.
- Existing single-item and multiselection **Remove** actions retain their previous behavior.

## Included changes

Contains all fixes and features included in **0.8.3** and prior releases, including the Capture arc and control-transition refinements, Linear Volume waveform analysis, macOS Return monitoring, and live editing of the epoch budget.

*Draft release for verification; not automatically published.*
