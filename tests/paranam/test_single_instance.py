"""GUI uniqueness does not interfere with training workers or update helpers."""

import os
import subprocess
import sys

from paranam.single_instance import GuiInstanceGuard


def test_two_gui_guards_cannot_coexist_and_release_cleanly():
    first, second = GuiInstanceGuard(), GuiInstanceGuard()
    try:
        assert first.acquire()
        assert not second.acquire()
        first.close()
        assert second.acquire()
    finally:
        first.close()
        second.close()


def test_gui_lock_is_cross_process_and_recovers_after_exit(tmp_path):
    if sys.platform == "win32":
        # CI verifies cross-process Win32 mutex separately via bundled smoke.
        return
    program = (
        "from paranam.single_instance import GuiInstanceGuard; "
        "import sys; g=GuiInstanceGuard(); "
        "sys.exit(0 if g.acquire() else 19)"
    )
    first = GuiInstanceGuard()
    try:
        assert first.acquire()
        blocked = subprocess.run([sys.executable, "-c", program], timeout=20)
        assert blocked.returncode == 19
    finally:
        first.close()
    released = subprocess.run([sys.executable, "-c", program], timeout=20)
    assert released.returncode == 0
