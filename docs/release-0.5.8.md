# ParaNAM Trainer 0.5.8

- Settings now keeps one duplex audio stream open while the page is visible.
  Return and Loopback return meters update continuously, including when the
  connection test is stopped. Send meters show the actual test output.
- Removed the periodic PortAudio reinitialization while a selected interface is
  available. Windows USB hardware notifications trigger a refresh when needed;
  explicit Refresh remains available. This avoids repeatedly unloading/reloading
  ASIO drivers and the recurring ASIO4ALL loading window.
- Test connection toggles white noise on Send and Loopback output simultaneously,
  using the Capture / validation output setting as its dBFS peak ceiling. Both
  outputs receive the same noise, including when clean-loopback capture is disabled.
  Other outputs remain silent. Stop test silences the outputs while input metering
  continues. Changing the level updates the signal without reopening the stream.
- The connection test no longer runs capture blips, timing stabilization, latency
  measurement, or connection validation. Capture's separate alignment checks are
  unchanged.
- Leaving Settings, changing routing, calibrating levels, starting Capture/Compare,
  or closing the app stops the Settings stream before another owner opens audio.
  Driver failures are displayed and are not retried by the UI timer.
- Includes all features and training settings from 0.5.7. Existing projects,
  recordings, queue entries, and training checkpoints remain compatible.

## Packages

Windows x64 installer with optional NVIDIA support, macOS Apple Silicon/Intel disk
images, and Linux x64 DEB/portable archive. Installers include the application and
its training, audio, and playback dependencies.
