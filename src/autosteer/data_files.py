"""The template data the generator reads, pinned by a committed file list.

The templates (~25 MB of Parquet, see autosteer.dataset_io) live in data/ of the repository (or AUTOSTEER_DATA_DIR,
else ~/.autosteer/data for an installed package). `data_files.json` next to this module lists every file the generator release expects
there, with its size and sha256, and where the data came from. `autosteer data verify` checks a data directory
against it; write_list() re-pins it after a deliberate change to the data. The sha256s are those of the files as
written by the pinned pandas and pyarrow; a rebuild with other versions can differ in bytes though not in content.

`manifest.json` in the data directory is not listed: the sampler rebuilds it from the templates when the set of
element directories changes (sampler/manifest.py).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

LIST_PATH = Path(__file__).with_name("data_files.json")
DERIVED = {"manifest.json"}


def load_list() -> dict:
    return json.loads(LIST_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(data_dir: Path) -> dict[str, dict]:
    """{relative path: {size, sha256}} for every file under data_dir except derived ones."""
    out = {}
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            p = Path(root) / name
            rel = p.relative_to(data_dir).as_posix()
            if rel in DERIVED or "__pycache__" in rel:
                continue
            out[rel] = {"size": p.stat().st_size, "sha256": sha256(p)}
    return out


def verify(data_dir: Path) -> dict:
    """Compare data_dir with the pinned list. Returns {ok, missing, changed, extra, checked}."""
    want = {f["path"]: f for f in load_list()["files"]}
    have = scan(data_dir) if data_dir.is_dir() else {}
    missing = sorted(set(want) - set(have))
    changed = sorted(p for p in set(want) & set(have)
                     if (have[p]["size"], have[p]["sha256"]) != (want[p]["size"], want[p]["sha256"]))
    extra = sorted(set(have) - set(want))
    return {"ok": not missing and not changed, "missing": missing, "changed": changed, "extra": extra,
            "checked": len(want), "data_dir": str(data_dir)}


def write_list(data_dir: Path, source: dict) -> dict:
    """Write data_files.json from the files now in data_dir (used when the template data is re-pinned)."""
    files = scan(data_dir)
    doc = {
        "about": "Files expected in the autosteer data directory (AUTOSTEER_DATA_DIR); see autosteer/data_files.py.",
        "source": source,
        "total_files": len(files),
        "total_bytes": sum(f["size"] for f in files.values()),
        "files": [{"path": p, **files[p]} for p in sorted(files)],
    }
    LIST_PATH.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


