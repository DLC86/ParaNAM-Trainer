"""Move or copy a project together with its project-local references."""

import json
from pathlib import Path
import shutil
import tempfile

from nam.capture.project import atomic_write_json


def project_path(source, name):
    if (
        not name or name.endswith((".", " ")) or name in (".", "..")
        or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name)
        or name.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
    ):
        raise ValueError("Choose a valid project and folder name.")
    return source.with_name(name)


def _occupied(target, *, exclude=None):
    for path in target.parent.iterdir():
        if exclude is not None and path.name == exclude.name:
            continue
        if path.name.casefold() == target.name.casefold():
            return True
        try:
            metadata = json.loads((path / "capture_project.json").read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(metadata, dict) and str(metadata.get("name", "")).casefold() == target.name.casefold():
            return True
    return False


def rebase_paths(value, source, target):
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(source):
            return str(target / path.relative_to(source))
    elif isinstance(value, list):
        return [rebase_paths(item, source, target) for item in value]
    elif isinstance(value, dict):
        return {key: rebase_paths(item, source, target) for key, item in value.items()}
    return value


def _project_updates(root, source, target, name):
    project_file = root / "capture_project.json"
    files = [project_file, root / "paranam.json"]
    for filename in ("spec.json", "progress.json", "instance.json", "current.json",
                     "paranam.json", "export-request.json", "export-result.json"):
        files.extend((root / "training").rglob(filename))
    for path in dict.fromkeys(files):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Cannot safely update project file: {path}")
        original = json.loads(path.read_bytes())
        updated = rebase_paths(original, source, target)
        if path == project_file:
            updated["name"] = name
        if updated != original:
            yield path.relative_to(root), original, updated


def change_project(source, name, *, duplicate=False, queue_store=None):
    source = Path(source)
    if source.is_symlink() or not (source / "capture_project.json").is_file():
        raise ValueError("Select a valid project folder.")
    source = source.resolve()
    if duplicate:
        index = 2
        while _occupied(project_path(source, f"{name} ({index})")):
            index += 1
        name = f"{name} ({index})"
    target = project_path(source, name)
    if target.name == source.name and not duplicate:
        if json.loads((source / "capture_project.json").read_bytes()).get("name") == name:
            return source
    if _occupied(target, exclude=None if duplicate else source):
        raise ValueError("A project or folder with that name already exists here.")

    updates = list(_project_updates(source, source, target, name))
    if duplicate:
        with tempfile.TemporaryDirectory(prefix=".paranam-copy-", dir=source.parent) as staging:
            copy = Path(staging) / "project"
            shutil.copytree(source, copy, symlinks=True)
            for path, _, updated in updates:
                atomic_write_json(copy / path, updated)
            training = copy / "training"
            for spec in (*training.glob("*/spec.json"),
                         *training.glob("queue-data/instances/*/spec.json")):
                if not spec.parent.resolve().is_relative_to(copy.resolve()):
                    continue
                marker = spec.with_name("queue-removed.json")
                if not marker.exists() and not marker.is_symlink():
                    atomic_write_json(marker, {"duplicated_project": True})
            if _occupied(target):
                raise ValueError("The destination appeared while copying. Try Duplicate again.")
            copy.rename(target)
        return target

    with queue_store.lock:
        from .recovery import require_resolved
        from .workspace import read_json

        queue = read_json(queue_store.path, {"version": 1, "entries": []})
        require_resolved(queue_store.path)
        updated_queue = rebase_paths(queue, source, target)
        for entry in updated_queue["entries"]:
            if entry.get("project") == str(target):
                entry["name"] = name
        moved = target != source
        if moved:
            if _occupied(target, exclude=source):
                raise ValueError("The destination appeared while renaming. Choose another name.")
            source.rename(target)
        written = []
        try:
            for path, original, updated in updates:
                atomic_write_json(target / path, updated)
                written.append((path, original))
            if updated_queue != queue:
                atomic_write_json(queue_store.path, updated_queue)
        except Exception:
            for path, original in reversed(written):
                atomic_write_json(target / path, original)
            if moved:
                target.rename(source)
            raise
    return target
