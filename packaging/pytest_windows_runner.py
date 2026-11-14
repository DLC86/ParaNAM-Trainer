"""Run the native Windows installer smoke suite, preserving pytest's true outcome.

The 0.7.7 Windows job completed all 816 tests successfully and printed the
pytest summary, but python.exe exited 1 during interpreter shutdown.  PyTorch,
Qt and native audio libraries have finalizers outside pytest's own lifecycle.
The call to pytest.main() still runs pytest_sessionfinish, pytest_unconfigure,
fixtures, and failure-reporting hooks, and returns a real nonzero status if
any part of the suite fails.  Bypass *only* Python interpreter shutdown after
that fully completed, verified test run.
"""

import os
import sys

import pytest


def main():
    status = int(pytest.main(sys.argv[1:]))
    print(f"VERIFIED_PYTEST_EXIT_STATUS={status}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid nondeterministic native finalizers once all pytest hooks are done.
    os._exit(status)


if __name__ == "__main__":
    main()
