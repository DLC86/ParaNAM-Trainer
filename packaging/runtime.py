"""Initialize native-library and cache paths before frozen imports."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "win32":
    os.environ.setdefault("SD_ENABLE_ASIO", "1")
    cache = (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ParaNAM Trainer" / "cache"
    )
else:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "paranam"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
