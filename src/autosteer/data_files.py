"""The template data the generator reads, pinned by a committed file list.

The templates (~25 MB of Parquet, see autosteer.dataset_io) live in data/ of the repository (or AUTOSTEER_DATA_DIR,
else ~/.autosteer/data for an installed package). `data_files.json` next to this module lists every file the generator release expects
there, with its size and sha256, and the STEER-generation commit the files come from. `autosteer data verify`
checks a data directory against it; `autosteer data build` recreates one from a STEER-generation checkout (git
archive of the pinned commit, whose datasets are pickles), converts it to Parquet, removes the duplicated example
seeds and seed candidates (autosteer.data_cleanup), and then verifies it. The sha256s are those of the files as
written by the pinned pandas and pyarrow; a rebuild with other versions can differ in bytes though not in content.

`manifest.json` in the data directory is not listed: the sampler rebuilds it from the templates when the set of
element directories changes (sampler/manifest.py).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
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


def build(data_dir: Path, source_repo: Path, commit: str | None = None) -> dict:
    """Extract `data/` of the pinned commit of a STEER-generation checkout into data_dir, then verify."""
    commit = commit or load_list()["source"]["commit"]
    tar = subprocess.run(["git", "-C", str(source_repo), "archive", "--format=tar", commit, "data"],
                         check=True, capture_output=True).stdout
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            if not m.name.startswith("data/") or not m.isfile():
                continue
            m.name = m.name[len("data/"):]
            t.extract(m, data_dir, filter="data")
    # the commit holds pickles; the pinned files are their Parquet conversion, cleaned of duplicated seeds
    from autosteer import data_cleanup, dataset_io
    report = dataset_io.convert(data_dir)
    if report["failed"]:
        raise RuntimeError(f"could not convert: {sorted(report['failed'])}")
    data_cleanup.fix_duplicate_seeds(data_dir)
    data_cleanup.fix_duplicate_templates(data_dir)
    data_cleanup.remove_non_elements(data_dir)
    return verify(data_dir)
