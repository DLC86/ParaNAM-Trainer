"""Resolve official wheels at build time; no pip or resolver is needed by users."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def main():
    if sys.platform not in ("win32", "linux"):
        return
    from paranam.gpu_runtime import platform_key, validate_url, manifest_id

    reports = {}
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    for flavor in ("cpu", "cu128"):
        output = build / f"torch-{flavor}-report.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--only-binary=:all:",
                "--report",
                str(output),
                "torch==2.10.0",
                "--index-url",
                f"https://download.pytorch.org/whl/{flavor}",
            ],
            check=True,
        )
        reports[flavor] = json.loads(output.read_text(encoding="utf-8"))
    base = {
        item["metadata"]["name"].lower(): item["metadata"]["version"]
        for item in reports["cpu"]["install"]
    }
    wheels = []
    for item in reports["cu128"]["install"]:
        name = item["metadata"]["name"].lower()
        info = item["download_info"]
        if name != "torch" and base.get(name) == item["metadata"]["version"]:
            continue  # Identical dependencies already shipped in the CPU application.
        digest = info["archive_info"]["hashes"]["sha256"]
        url = info["url"]
        validate_url(url)
        filename = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
        if not filename.endswith(".whl"):
            raise RuntimeError("GPU components must be prebuilt wheels")
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as response:
            size = int(response.headers.get("Content-Length", "0"))
        wheels.append(
            dict(
                name=name,
                version=item["metadata"]["version"],
                url=url,
                filename=filename,
                sha256=digest,
                size=size,
            )
        )
    manifest = dict(
        schema=1,
        platform=platform_key(),
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        torch="2.10.0+cu128",
        cuda="12.8",
        wheels=sorted(wheels, key=lambda x: x["name"]),
    )
    manifest["id"] = manifest_id(manifest)
    target = ROOT / "paranam" / "assets" / f"gpu-{platform_key()}.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"GPU manifest: {len(wheels)} official wheels, {sum(w['size'] for w in wheels):,} bytes"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
