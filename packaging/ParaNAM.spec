# Build on the target OS; all Python/Qt/Torch dependencies live inside this bundle.
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

root = Path(SPECPATH).parent
data = [(str(root / "LICENSE"), "licenses"),
        (str(root / "CREDITS.md"), "licenses"),
        (str(root / "nam/_dependencies/auraloss/LICENSE"), "licenses/auraloss"),
        (str(root / "THIRD_PARTY_NOTICES.md"), "licenses")]
for package in ("nam", "paranam", "pytorch_lightning", "lightning_fabric", "librosa", "scipy", "torchmetrics"):
    data += collect_data_files(package)
# Lightning checks these paths on disk before importing checkpoint version info.
for package in ("pytorch_lightning", "lightning_fabric"):
    data += collect_data_files(package, includes=["__about__.py", "__version__.py"], include_py_files=True)
binaries = []
binaries += collect_dynamic_libs("paranam")
runtime = root / "external" / "playback-runtime"
for path, target in (
    (runtime / "LICENSE", "licenses/playback-plugin"),
    (runtime / "THIRD-PARTY-NOTICES.md", "licenses/playback-plugin"),
    (runtime / "external/nam_core_parametric/LICENSE", "licenses/playback-nam-core"),
    (runtime / "external/AudioDSPTools/LICENSE", "licenses/playback-audio-dsp"),
    (runtime / "external/nam_core_parametric/Dependencies/eigen/COPYING.MPL2", "licenses/playback-eigen"),
):
    data.append((str(path), target))
for package in ("torch", "scipy", "numpy"):
    binaries += collect_dynamic_libs(package)
try:
    data += collect_data_files("_sounddevice_data")
    binaries += collect_dynamic_libs("_sounddevice_data")
except Exception:
    pass
if sys.platform.startswith("linux"):
    import subprocess
    # Linux sounddevice wheels use the system PortAudio shared library. Ship it.
    listing = subprocess.check_output(["ldconfig", "-p"], text=True)
    paths = [line.split("=>")[-1].strip() for line in listing.splitlines() if "libportaudio.so.2" in line]
    if not paths:
        raise RuntimeError("Build host is missing libportaudio2")
    binaries.append((paths[0], "."))
    try:
        binaries += collect_dynamic_libs("nvidia")
        data += collect_data_files("nvidia", includes=["**/LICENSE*", "**/license*"])
    except Exception:
        pass
hidden = collect_submodules("nam.models") + collect_submodules("paranam")
hidden += ["nam.train.parametric", "sounddevice", "wavio", "filelock",
           "scipy.special._ufuncs_cxx", "pydantic.deprecated.decorator"]
analysis = Analysis([str(root / "packaging" / "launcher.py")], pathex=[str(root)],
                    binaries=binaries, datas=data, hiddenimports=hidden,
                    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "notebook",
                              "pytest", "tensorboard", "transformers", "triton"],
                    runtime_hooks=[str(root / "packaging" / "runtime.py")], noarchive=False)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="ParaNAM Trainer",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          icon=str(root / "packaging" / "icon.ico") if sys.platform == "win32" and (root / "packaging" / "icon.ico").exists() else None)
bundle = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name="ParaNAM Trainer")
if sys.platform == "darwin":
    app = BUNDLE(bundle, name="ParaNAM Trainer.app", bundle_identifier="com.dlc86.paranamtrainer",
                 icon=str(root / "packaging" / "icon.icns"),
                 info_plist={"CFBundleShortVersionString":"0.6.2", "CFBundleVersion":"0.6.2",
                             "NSMicrophoneUsageDescription":"Record your rig and timing loopback to create an amp model.",
                             "NSHighResolutionCapable":True})

