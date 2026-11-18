# ParaNAM Trainer 0.8.2

Retrain now preserves the capture plan of the selected job when opening projects made with older trainer versions. Previously, a legacy Avoid limit could disappear during migration. Accepting the resulting plan update could replace recorded captures at the lowest audible setting with synthetic Mute targets. The fix also applies when retraining a job that was itself created with Retrain.

To repeat an earlier training run with its original captures, select the queue entry that still contains those recordings and choose Retrain. Existing jobs keep their saved datasets.

Resume now recognizes a frozen job even when that job was originally created through Retrain. If 0.8.1 already converted its old Avoid limits into equivalent rules during a failed Resume, 0.8.2 can verify the original checkpoint using the saved control settings and the current recordings. A mismatch still blocks Resume.

## Linear Volume: waveform comparison

The 0.20 dB peak/RMS-attenuation difference warning has been replaced by a gain-normalized waveform null test with a -40 dB residual target. This detects changes caused by the Volume control (including coloration, compression or added saturation) even when the modeled device is already heavily distorted, such as a fuzz. The result is checked against recording repeatability and background noise; measurements with insufficient SNR are reported as inconclusive rather than certified as linear.

The maximum-volume calibration point automatically plays the validation signal **twice in one continuous capture**, separated by a short silent gap. The second pass is stored as an internal auxiliary WAV for the repeatability estimate. It does not create another Plan/Capture entry and requires no manual action. The ordinary volume level sweep and exported RMS/PCHIP attenuation curve are unchanged.

For previously recorded maximum-volume references that do not contain an automatic repeat pass, the waveform check is inconclusive until that reference is recaptured. Dismissing a warning still permits completing calibration at the user's discretion.

## Capture control order and existing WAV compatibility

Train, Validation, Corner and Linear Volume calibration captures now use the rig's declared control order both in generated WAV filenames and in the Capture table's **Control settings** column. Fixed controls such as Linear Volume are no longer moved to the end merely because the sampling core processes them separately.

Existing projects do not rename any WAVs. The Capture table displays their controls in the correct order even if the saved parameter dictionary was ordered differently. When a compatible plan is regenerated, recorded captures are matched by split and control name/value pairs rather than filename formatting, so old WAVs with the former suffix order are reused without re-recording.

## macOS Monitor Return and live training epochs

- **Monitor Return on macOS/CoreAudio:** Capture no longer rejects Return monitoring on non-ASIO devices. The non-ASIO duplex callback now sends the live Return input to the selected Monitor L/R outputs, independently of Send and optional clean-loopback channels, with channel-overlap validation. ASIO monitoring continues to use its existing persistent-stream path.
- **Change total epochs during training:** The Queue context-menu command is now available while a job is running. New limits are communicated atomically to the worker and applied to Lightning's active fit loop at a batch boundary. Completed epochs, optimizer and LR scheduler state, ESR history, checkpoint paths and job identity are preserved. The same mechanism supports static and multi-member training; members completed under an earlier shorter target receive their remaining epochs.
- **Safety and persistence:** The new target cannot be set below an epoch already started or completed. Edits survive pause/resume and app restarts. An active worker remains the sole writer of progress.json, avoiding progress counter resets from GUI edits. No new queue job or recapture is needed.
