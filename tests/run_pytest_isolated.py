"""Run each collected pytest case in a fresh Python process.

Useful for isolating native-extension / Qt teardown crashes that occur only
when a group of otherwise-passing GUI tests share one interpreter.
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tests/run_pytest_isolated.py <pytest target> [pytest args...]")
        return 2

    target, *pytest_args = sys.argv[1:]
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", target, "--collect-only", "-q", "-o", "addopts="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(collect.stdout, flush=True)
    if collect.returncode:
        return collect.returncode

    nodeids = [
        line.strip()
        for line in collect.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith(("=", "<"))
    ]
    if not nodeids:
        print(f"No pytest cases collected from {target}", flush=True)
        return 2

    failures = []
    for index, nodeid in enumerate(nodeids, 1):
        print(f"\\n[{index}/{len(nodeids)}] Isolated test: {nodeid}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", nodeid, *pytest_args],
            check=False,
        )
        if result.returncode:
            failures.append((nodeid, result.returncode))
            print(
                f"FAILED process: exit code {result.returncode}: {nodeid}",
                flush=True,
            )

    if failures:
        print("\\nIsolated test process failures:", flush=True)
        for nodeid, code in failures:
            print(f"  exit {code}: {nodeid}", flush=True)
        return 1

    print(f"\\nAll {len(nodeids)} isolated test processes passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
