"""Build a native installer on its target OS. End users never run this script."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.5.0"


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", default="CPU", choices=("CPU", "Metal"))
    parser.add_argument("--skip-freeze", action="store_true")
    args = parser.parse_args()
    from PIL import Image

    image = Image.open(ROOT / "paranam" / "assets" / "TTSLogo.png").convert("RGBA")
    image.save(
        ROOT / "packaging" / "icon.ico",
        sizes=[(16, 16), (32, 32), (48, 48), (256, 256)],
    )
    if sys.platform == "darwin":
        image.save(ROOT / "packaging" / "icon.icns")
    image.resize((256, 256), Image.Resampling.LANCZOS).save(ROOT / "packaging" / "icon.png")
    installers = ROOT / "installers"
    installers.mkdir(exist_ok=True)
    if not args.skip_freeze:
        from native_engine import build as build_playback

        build_playback()
        if sys.platform in ("win32", "linux"):
            import torch

            if torch.version.cuda is not None:
                raise RuntimeError(
                    "Unified installers must be built from a CPU PyTorch environment"
                )
            run(sys.executable, "packaging/make_gpu_manifest.py")
        run(
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "packaging/ParaNAM.spec",
        )
    bundle = ROOT / "dist" / "ParaNAM Trainer"
    arch = platform.machine().lower()
    if sys.platform == "win32":
        executable = bundle / "ParaNAM Trainer.exe"
    elif sys.platform == "darwin":
        executable = (
            ROOT
            / "dist"
            / "ParaNAM Trainer.app"
            / "Contents"
            / "MacOS"
            / "ParaNAM Trainer"
        )
    else:
        executable = bundle / "ParaNAM Trainer"
    # Test the frozen bundle, not the developer's Python environment.
    smoke = ROOT / "build" / "bundle-smoke.json"
    from validate_bundle import (
        clean_environment,
        validate_training,
        validate_static,
        validate_installer_configuration,
    )

    environment = clean_environment()
    result = subprocess.run(
        [str(executable), "--self-test", str(smoke)],
        env=environment,
        timeout=180,
    )
    if smoke.exists():
        print(smoke.read_text(encoding="utf-8"), flush=True)
    if (
        result.returncode != 0
        or not smoke.exists()
        or not json.loads(smoke.read_text())["ok"]
    ):
        raise RuntimeError("Frozen application smoke test failed")
    validate_installer_configuration(executable, ROOT / "build")
    validate_training(executable, ROOT / "build")
    validate_static(executable, ROOT / "build")
    if sys.platform == "win32":
        compiler = shutil.which("ISCC")
        if not compiler:
            candidates = [
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
                / f"Inno Setup {version}"
                / "ISCC.exe"
                for version in (7, 6)
            ]
            compiler = next((str(path) for path in candidates if path.exists()), None)
        if not compiler:
            raise RuntimeError(
                "Build host needs Inno Setup 6+. The application bundle is already in dist/."
            )
        run(
            compiler,
            f"/DBuildRoot={bundle}",
            f"/DOutputRoot={installers}",
            f"/DFlavor={args.flavor}",
            "packaging/windows.iss",
        )
    elif sys.platform == "darwin":
        stage = ROOT / "build" / ("dmg-" + arch)
        stage.mkdir(exist_ok=True)
        app = stage / "ParaNAM Trainer.app"
        if app.exists():
            raise RuntimeError("DMG staging folder exists; use a clean build directory")
        shutil.copytree(ROOT / "dist" / "ParaNAM Trainer.app", app, symlinks=True)
        (stage / "Applications").symlink_to("/Applications")
        run(
            "hdiutil",
            "create",
            "-volname",
            "ParaNAM Trainer",
            "-srcfolder",
            stage,
            "-ov",
            "-format",
            "UDZO",
            installers / f"ParaNAM-Trainer-{VERSION}-macOS-{arch}.dmg",
        )
    else:
        package = ROOT / "build" / ("deb-" + args.flavor.lower())
        target = package / "opt" / "paranam-trainer"
        if target.exists():
            raise RuntimeError(
                "Debian staging folder exists; use a clean build directory"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, target, symlinks=True, copy_function=os.link)
        desktop = package / "usr" / "share" / "applications"
        desktop.mkdir(parents=True, exist_ok=True)
        (desktop / "paranam-trainer.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=ParaNAM Trainer\n"
            'Exec="/opt/paranam-trainer/ParaNAM Trainer"\nIcon=paranam-trainer\n'
            "Categories=AudioVideo;Audio;\nTerminal=false\n",
            encoding="utf-8",
        )
        icons = package / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        icons.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "packaging" / "icon.png", icons / "paranam-trainer.png")
        control = package / "DEBIAN"
        control.mkdir()
        (control / "control").write_text(
            f"Package: paranam-trainer\nVersion: {VERSION}\nArchitecture: amd64\n"
            "Maintainer: DLC86\nSection: sound\nPriority: optional\n"
            "Depends: libc6 (>= 2.35), libasound2, libpulse0, libgl1, libegl1, libopengl0, "
            "libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, "
            "libxcb-render-util0, libxcb-xinerama0, libxcb-xkb1, libxcb-shape0, libfontconfig1, libdbus-1-3\n"
            "Description: Guided parametric amp capture and HyperA2 training\n"
            " Bundles Python, Qt, PyTorch and the capture/training dependencies.\n",
            encoding="utf-8",
        )
        run(
            "dpkg-deb",
            "-Zxz",
            "-z3",
            "--root-owner-group",
            "--build",
            package,
            installers / f"ParaNAM-Trainer-{VERSION}-Linux-amd64.deb",
        )
        if args.flavor == "CPU":
            # Portable folder for Linux distributions that do not use Debian packages.
            run(
                "tar",
                "-C",
                ROOT / "dist",
                "-czf",
                installers / f"ParaNAM-Trainer-{VERSION}-Linux-x86_64.tar.gz",
                "ParaNAM Trainer",
            )


if __name__ == "__main__":
    main()
