# Unexpected-shutdown recovery

Training progress, queue state and project preferences are JSON files. An abrupt
power loss can leave a cached file empty, truncated or filled with null bytes.
An atomic rename alone does not guarantee that the data has reached storage.

Mutable JSON state retains one previous valid `.bak` document. Important state
changes are flushed before atomic replacement. Live running batch fractions stay
lightweight and atomic; durable metadata is saved at epoch boundaries and status
changes. The complete resumable checkpoint is flushed before replacement and
keeps one previous valid `.ckpt.bak` per mutable checkpoint path, including
fixed best-size slots. Best-size checkpoints are also written atomically and
flushed. This does not create an unlimited series of snapshots.

When damaged JSON is encountered, ParaNAM tries a validated backup. For training
progress, it first reconstructs epoch count, best ESR, Train/Validation ESR history
and resume information from a complete checkpoint loaded on CPU. It never takes
an optimizer step during recovery. A damaged latest checkpoint can fall back to
its previous valid backup or a complete legacy epoch checkpoint. Archive checksums
are checked before loading. The notice identifies the recovered epoch; incomplete
or damaged newer work cannot be resumed. A live worker's lock prevents replacing
its checkpoint during recovery.

Files being replaced by recovery are preserved as `.corrupt-<timestamp>-<id>`.
Recordings, project folders and model/checkpoint directories are never deleted.
If a damaged progress file cannot be recovered safely, its instance is held for
manual recovery instead of silently restarting training from scratch. Multi-model
runs require their progress backup to identify completed members safely.

The application opens with a recovery notice and a modeless **Review recovery**
window. It lists file paths, actions and preserved originals. Available actions:

- **Retry recovery** after reconnecting a drive or fixing permissions.
- **Restore JSON backup…** to select a valid backup of the affected state file.
- **Rebuild queue…** when the queue index has no usable backup. This requires
  confirmation, scans readable recent projects, honors removal markers and holds
  unfinished recovered instances as Interrupted for review. The old queue order
  may not be recoverable; review it before resuming.
- **Open folder** to inspect preserved files. Closing the report leaves the app
  usable and the recovery notice available.

Corrupt preferences or project definitions are not silently replaced by defaults.
Unresolved essential files remain in place until the user chooses a solution.
Replaceable cache/pointer files may be reset, with the original preserved and the
action reported. Read-only failures can show recovered data in memory and explain
why it could not be saved. Recovery never starts the queue automatically.

Storage hardware/filesystem failure can affect several copies. Backups and flushing
reduce the risk, but do not guarantee recovery of unsaved work or damaged audio.
