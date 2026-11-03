"""Compile the pinned plugin DSP on the build host; end users receive the library."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def build():
    def run(*args):
        subprocess.run([str(a) for a in args], cwd=ROOT, check=True)

    run("git", "submodule", "update", "--init", "external/playback-runtime")
    runtime = ROOT / "external" / "playback-runtime"
    run(
        "git",
        "-C",
        runtime,
        "submodule",
        "update",
        "--init",
        "external/nam_core_parametric",
        "external/AudioDSPTools",
    )
    run(
        "git",
        "-C",
        runtime / "external" / "nam_core_parametric",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "Dependencies/eigen",
        "Dependencies/AudioDSPTools",
    )
    run(
        "cmake",
        "-S",
        ROOT / "native",
        "-B",
        ROOT / "build" / "playback",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    run(
        "cmake",
        "--build",
        ROOT / "build" / "playback",
        "--config",
        "Release",
        "--parallel",
        "2",
    )
    run(
        "cmake",
        "--install",
        ROOT / "build" / "playback",
        "--config",
        "Release",
        "--prefix",
        ROOT / "paranam" / "native",
    )


if __name__ == "__main__":
    build()
