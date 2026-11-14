# ParaNAM Trainer 0.7.5

## Knob values in Plan, Capture and Compare

- **Positive values on bipolar knobs** now display a leading `+` when the control's range crosses zero (minimum below zero and maximum above zero).
- The rule applies to the knob readouts in **Plan**, **Capture** and **Compare**, to the small **minimum and maximum** labels beside the knobs in Plan and Capture, and to the **Control settings** column of the Capture table.
- Zero, negative values and strictly unipolar controls retain their previous notation. Only the displayed text changes; numeric parameter values, model files and audio processing are unaffected.

## Main window

- **Updated TTS branding:** the Trainer now uses the original, user-supplied **256×256 transparent PNG without resizing, recompression or color changes**, preserving the artwork's smooth gradients. The same source is used for the interface, window/taskbar icon and native installer icons across Windows, macOS and Linux.
- **Project header:** the open project's name is displayed **larger and in white**, while the separator dot and the **Automatically saved** status keep their previous size and muted color.
- **Full-width grey separators** now extend from the border of the left navigation panel to the right edge of the window. The project name, pages and Back/Next controls retain their 30px side padding, preserving the resizable Capture table.

## Plan, Capture, Queue and Compare

- Plan control cards follow the pointer during drag-and-drop, while the other cards animate into position. Hover highlights control cards and both the border and icon of the add card.
- Selecting a control skips recalculating capture recommendations. Capture grid limits account for both marked Gain/Drive controls.
- Queue fits its columns on first opening and then preserves their widths. Queue and Capture reuse existing table cells, and Compare keeps the model selector intact when its contents are unchanged.
- Scrolling over closed dropdowns no longer changes their selection.

## Update notification

- **GitHub-style Markdown release notes** are rendered directly in the update window, including headings, bold and italic text, lists, links, code blocks and tables.
- **Resizable notes area:** when the update dialog grows or shrinks, the release notes panel changes height while the version notification at the top stays fixed.
- **Visible Windows installer:** an update started from the notification opens the normal installer interface rather than silently hiding the installation steps. Existing verification and safe shutdown checks are preserved.

## Automatic cleanup of old installers

- On startup, the Trainer removes **all completed or abandoned updater download folders**, including installers left behind by **earlier releases**, not just the most recent update.
- This clears downloaded setup executables, DMGs, Debian packages, portable archives and their associated temporary files, logs and manifests.
- **Safety:** an updater directory whose installation result is still pending is left untouched. Files locked by the operating system are skipped and tried again on a later launch. Training projects, recordings, exports and unrelated files are never targeted.

## Availability

This release is prepared as a **draft**. All five native installers are built from the same source commit and verified before the draft assets are updated. The draft is **not published automatically**.
