# ParaNAM Trainer 0.8.4 — Compare layout and asset management

## Play & Compare layout

- The parametric control panel reserves the full height of a **single row of knobs/switches**, even when the selected model has no editable controls or no model is loaded.
- The **Play & Compare** title, explanatory copy, **Playback buffer**, **Model size** and **Output mode** form one complete left-hand header block. The **Instrument** and **Monitor L/R** VU meters occupy a separate column to the right of that entire block.
- **Input** and **Output** level knobs sit at the same height as the **Play / Gear / Model** action block, with Input anchored to the left and Output to the right.
- **Model size** and **Output mode** each occupy a dedicated row immediately **below Playback buffer**, rather than sharing a row with the gain controls.
- Both the **Models** and **Impulse responses** lists reserve enough height to display at least **ten rows** before scrolling.
- The **Parametric controls** panel displays a prominent **No model loaded** message whenever no model is selected.
- Model files can be loaded by dragging **.nam / .pnam** files into the Models panel; **.wav** files can be dropped into Impulse responses.

## Clear list buttons

- Added **Clear list** for Models: removes all listed models, including those automatically discovered in project output folders or the training queue, while leaving the actual model files untouched. Previously cleared models stay hidden after refresh and application restart; manually importing a model makes it available again.
- Added **Clear list** for Impulse responses: removes only imported/custom IRs. The **six bundled Tone3000 IRs** and the **Off** selection always remain. Actual IR files are never deleted.
- Existing single-item and multiselection **Remove** actions retain their previous behavior.

## Static training defaults and options

- **RMS loss normalization** is **OFF by default** for new static training configurations. Parametric training keeps its existing ON default. Explicit values saved in existing projects and training snapshots remain unchanged.
- **Checkpoint selection** and **Checkpoint alpha** are completely hidden in Static training's Advanced options, since those controls apply to the parametric checkpoint-selection workflow. Parametric Checkpoint selection remains visible; the alpha field is shown only for Combined ESR mode.

## Model Editor

- Added a standalone **Editor** page immediately above Settings in the left navigation, accessible even without an open capture project.
- Load **.nam** and **.pnam** files and edit **Author, Manufacturer, Gear model, Gear type, Tone type, Input level (dBu), Output level (dBu), Input impedance and Notes**, with the same options used in Train's metadata form.
- Save by **overwriting the existing model** or **creating a named copy**. Replacing an existing file creates a `.bak` copy; writes are atomic and preserve neural weights, computed loudness and gain, and unexposed metadata.
- Parametric models additionally support changing the **display order** of controls (index), **switch option display order** (index_sw), and renaming controls and switch options without changing neural parameter positions or option numeric values. Changes propagate to Full/Lite control descriptions.
- The Parametric controls panel expands to use the available window height. Model files can also be loaded by dragging **.nam / .pnam** files onto Model metadata.
- **Knob min/max cannot be edited**: these values define network input normalization and modifying them would change the sound.
- Compare selector labels now match the **Playback buffer** label size, and the **Model size / Output mode** row is fully visible.
- Expanded Qt and model-format regression tests, and updated the existing Compare/sidebar tests for the new layout.

## Capture and training queue

- Capture control transitions now display the preceding value and arrow at the same size as the current value.
- The Queue columns are ordered **Name, Project, Type, Status, Epoch, ESR (Validation/Train), Created**.
- **Type** now identifies the training family as **Parametric** or **A2 static**. **ESR (Validation/Train)** shows the Full-model validation and training ESR values side by side.

## Linear Volume capture planning

- Newly regenerated Linear Volume plans use a centered fixed rig context: ordinary controls are set to **5**, while controls marked **Gain/Drive** are set to **0**.
- If a preferred value is a mute point or is prohibited by a Mute or Avoid rule, the planner uses the nearest permitted step instead.
- Existing plans preserve their recorded context unchanged. The new behavior applies only when a plan is regenerated.

## Regression and release verification

- Added and updated regression coverage for capture planning, Capture value transitions, Queue metrics, static-training defaults, Editor metadata preservation and the revised Play & Compare layout.
- The draft is rebuilt from one exact source commit across Windows, Linux, macOS Intel and macOS Apple Silicon; each native package is smoke-tested before it replaces its previous draft asset.

*Draft release for verification; not automatically published.*
