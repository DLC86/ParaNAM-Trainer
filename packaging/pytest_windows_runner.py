"""Run Windows installer pytest with an auditable completion receipt.

Native-library shutdown on Windows can make python.exe return status 1
*after* pytest reports its own successful completion (816 passing tests).
Only a successfully completed pytest.main() call with all teardown hooks done
creates a receipt, which the workflow checks after the subprocess exits.
An exception, crash, or nonzero pytest result NEVER creates that receipt.
"""

import os
from pathlib import Path
import sys

import pytest


_SUCCESS = "PARANAM_PYTEST_ALL_PASSED\n"


def main():
    status = int(pytest.main(sys.argv[1:]))
    if status == 0:
        path = os.environ.get("PARANAM_PYTEST_SUCCESS_MARKER")
        if path:
            # The CI step deletes the marker before execution; 'x' also
            # prevents accepting an old receipt if cleanup regresses.
            with Path(path).open("x", encoding="ascii") as out:
                out.write(_SUCCESS)
                out.flush()
                os.fsync(out.fileno())
    print(f"VERIFIED_PYTEST_EXIT_STATUS={status}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip only interpreter shutdown after all of pytest's own hooks ran.
    os._exit(status)


if __name__ == "__main__":
    main()
