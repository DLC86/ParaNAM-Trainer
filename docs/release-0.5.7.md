# ParaNAM Trainer 0.5.7

Built on the latest 0.5.6 `main`, retaining independent LR schedules, deterministic
training, reverb/cabinet IRs and all other existing changes.

- Plan's Rig controls table grows with its rows. Page widths and entry widths stay
  bounded on large displays, while capture and queue tables retain vertical growth.
- Settings calibration fields align in one form. Interface selection survives a
  refresh or disconnect without choosing a different interface. Reconnection restores
  the same device by driver API/name, with its saved channels and buffer size.
- Parametric capture requires a separate clean loopback. Static capture can still
  use an optional loopback.
- Test connection shows four horizontal meters (Send, Return, Loopback Send and
  Loopback Return), filling the available width to the right of the button, with
  labels on the left and dBFS readings on the right. Status and instructions are below the meters. Meter
  updates run outside audio callbacks.
- Compare uses a Strat-style guitar icon; Settings uses a toothed cog icon.
- Queue's best ESR metric headings omit `(FULL)`; their scores still refer to Full.
  Every column, including the final one, is independently resizable. Table headers
  have subtle vertical dividers.
- Start queue becomes Pause queue while processing. It pauses the active instance
  and suspends scheduling, so no successor starts. Start queue resumes that instance
  before the waiting jobs; the paused instance association survives app restarts.
- ESR charts show a vertical hover cursor snapped to the nearest completed epoch
  and display its Validation/Train values instead of the last epoch's values.
- ETA updates during an epoch from recent batch timings, rather than average epoch
  duration. It includes measured validation and Train-ESR evaluation overhead after
  the first completed validation. Waiting time while paused is excluded. The log
  shows training speed in `it/s`.
- Compare has two square, mutually exclusive input transports: live instrument and
  recorded audio. Eight original TONE3000 preview WAVs are bundled for offline
  auditioning, with upstream license, pinned source URLs and checksums. The selected
  recording loops through both the model and the real gear; changing source/file
  safely stops and restarts the stream.
- Entry fields and selectors have black backgrounds, flat arrow/button areas and
  explicit white plus/minus symbols. The version caption uses normal weight.
- Custom mono/stereo IRs have no fixed duration cap and retain their complete tail.
  Longer IRs require proportionally more memory and convolution work.

## Packages

Windows x64 installer with optional NVIDIA support, macOS Apple Silicon/Intel disk
images, and Linux x64 DEB/portable archive. Capture/validation inputs, preview WAVs,
IRs, fonts, UI icons and native playback runtime are included in each package.
