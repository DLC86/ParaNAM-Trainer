"""Commit every ParaNAM checkpoint before dependent progress metadata is saved."""

import functools
import os
from pathlib import Path
import tempfile

from nam.capture.project import replace_file


def install_durable_checkpoint_io(trainer):
    if getattr(trainer, "_paranam_durable_checkpoint_io", False):
        return
    original = trainer.save_checkpoint

    @functools.wraps(original)
    def save(path, *args, **kwargs):
        if not getattr(trainer, "is_global_zero", True):
            return original(path, *args, **kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        try:
            original(temporary, *args, **kwargs)
            replace_file(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    trainer.save_checkpoint = save
    trainer._paranam_durable_checkpoint_io = True
