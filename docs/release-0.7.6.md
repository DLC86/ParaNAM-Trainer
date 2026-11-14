# ParaNAM Trainer 0.7.6

- **Plan → Rig controls** adds **Plan rules**. Select controls and use **Avoid all** to skip settings where every selected control meets the condition.
- Knobs have editable **Avoid <** and **Avoid >** thresholds. Older projects carry their endpoint exclusions forward as equivalent thresholds. Linear Volume calibration honors these exclusions and keeps a fixed rig setting throughout the volume sweep.
- Plan edits reuse compatible recorded takes. Other takes stay in the project folder and can be reused if they match a later plan.
- **Capture** highlights the controls that differ from the last recorded take, including after selecting another row or returning to the page. Confirm **My rig matches…** to stop the highlighting.

- Scrolling over numeric value fields no longer changes their values accidentally.
- Panel dividers are aligned across the Trainer interface.
- A dimmed overlay with an animated spinner appears as soon as a project starts loading. It also blocks input during longer calculations and stays visible briefly between consecutive operations to avoid flicker.
- Projects can be renamed or duplicated from the Projects page. Duplicates receive the first available numbered suffix.
