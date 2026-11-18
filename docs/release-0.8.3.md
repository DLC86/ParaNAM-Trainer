# ParaNAM Trainer 0.8.3

## Capture control transitions

- The white guide segment on a changed knob's red arc now pulses between **50% and 75% opacity** instead of fully transparent and solid white. The red transition remains visible underneath.
- While a control differs from the previous recorded capture, its tile shows the **previous value → new value**. The previous value and arrow are smaller, **bold** and a slightly darker light grey; the prominent red current value remains horizontally centered, with no overlap. Works with knobs and switches.
- Both guidance elements appear only on controls that changed and disappear immediately after confirming **My rig matches the control positions shown above**; manually selecting another capture updates the comparison.
- Tiny knob minimum/maximum labels are hidden in **Capture** to reduce visual clutter; they remain visible in **Plan**.

## Linear Volume calibration

- Linear Volume is now checked with a gain-matched waveform null test rather than an arbitrary 0.20 dB difference between peak and RMS. The default maximum residual is **−40 dB** relative to the level-matched signal, with a repeatability/noise check and an inconclusive result when the signal-to-noise ratio is insufficient.
- The maximum-volume reference automatically records two consecutive passes in the **same capture operation**, with a brief silent gap. The auxiliary repeat is not an additional Plan/Capture row.
- The waveform comparison checks whether the volume control alters tone, distortion or dynamics, even when the equipment itself is already highly distorted (for example, a fuzz). The ordinary volume attenuation curve is preserved.
- The automatic repeat file is kept with its reference recording and cleaned up when the associated capture is cleared. Legacy references lacking a repeat can be recaptured to establish the new noise baseline.

## Capture plans and compatibility

- Train, Validation, Corner and Gain captures now list controls in the project's declared order in filenames and **Control settings**, including when Linear Volume is excluded from the sampling dimensions.
- Previously recorded WAV filenames are not renamed. Existing projects can reopen and regenerate compatible plans without re-recording captures solely because the old filename listed Volume last.

## Monitor Return on macOS

- Removes the ASIO-only guard that previously blocked Return monitoring during capture on macOS.
- The non-ASIO/CoreAudio recording callback now routes the live Return to the selected Monitor L/R physical outputs without modifying Send or clean-loopback paths. Invalid/conflicting monitor channels are rejected.

## Change total epochs while training

- **Change total epochs…** is available for running and starting queue jobs, not only paused/completed ones.
- A new total is written atomically and adopted by the running PyTorch Lightning fit loop at batch boundaries, without resetting optimizer, scheduler, completed epochs, ESR history, or checkpoints.
- Supports extending and reducing a run, provided the new target is not below the already started/completed epoch, and supports both parametric and static jobs, including multiple members. Updated targets persist across pause/resume and reopening the app.
- Legacy and interrupted training jobs retain their completed-member and checkpoint handling.

## Retrain and update checks

- Incorporates the preceding 0.8.2 retrain-plan compatibility fixes and the cumulative updater release-notes changes already integrated into main.

**Draft preview:** release assets are intended for verification before publication.
