"""Upload tested native packages directly to a release draft, avoiding CI storage."""

import json
import os
from pathlib import Path
import subprocess
import sys


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def main():
    repository = os.environ["GH_REPO"]
    release_id = os.environ["RELEASE_ID"]
    assert release_id.isdecimal()
    repo = json.loads(gh("api", f"repos/{repository}"))
    release = json.loads(gh("api", f"repos/{repository}/releases/{release_id}"))
    assert repo["full_name"].casefold() == repository.casefold(), "Repository mismatch"
    assert release["draft"] and release["prerelease"], "Expected a draft preview"
    target = release["target_commitish"]
    if target != os.environ["GITHUB_SHA"]:
        resolved = json.loads(gh("api", f"repos/{repository}/commits/{target}"))["sha"]
        assert resolved == os.environ["GITHUB_SHA"], "Source mismatch"
    tag = release["tag_name"]
    if sys.argv[1] == "upload":
        packages = sorted(Path("installers").glob("*"))
        assert packages and all(
            p.is_file() and p.stat().st_size < 2**31 for p in packages
        )
        subprocess.run(
            ["gh", "release", "upload", tag, *map(str, packages), "--clobber"],
            check=True,
        )
    elif sys.argv[1] == "finalize":
        from build import VERSION

        prefix = f"ParaNAM-Trainer-{VERSION}-"
        expected = {
            prefix + suffix
            for suffix in (
                "Windows-x64-Setup.exe",
                "Linux-amd64.deb",
                "Linux-x86_64.tar.gz",
                "macOS-arm64.dmg",
                "macOS-x86_64.dmg",
            )
        }
        assert {
            a["name"] for a in release["assets"]
        } == expected, "Missing native packages"
        assert all(
            a["state"] == "uploaded" and a["size"] > 0 for a in release["assets"]
        )
        if os.environ.get("KEEP_DRAFT", "false") == "true":
            print("All native packages verified; release remains in draft.")
        else:
            subprocess.run(["gh", "release", "edit", tag, "--draft=false"], check=True)
    else:
        raise ValueError("Choose upload or finalize")


if __name__ == "__main__":
    main()
