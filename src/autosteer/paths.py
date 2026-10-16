"""Where autoSTEER's files are.

The data directory (templates, seeds, review records) is decided when it is first needed, not when a module is
imported, so the CLI's --data-dir applies everywhere: set_data_dir(), else AUTOSTEER_DATA_DIR (STEER_DATA_DIR is the
old name), else data/ of the repository checkout, else ~/.autosteer/data (an installed package without the
repository: copy the repository's data/ there, or set AUTOSTEER_DATA_DIR).
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[1]
REPO_DATA_DIR = REPO_DIR / "data"
DATA_KINDS = ("example_seeds", "generated_seeds", "templates")

_override: Path | None = None


def default_data_dir() -> Path:
    """data/ of the repository when it holds the templates, else ~/.autosteer/data."""
    return REPO_DATA_DIR if (REPO_DATA_DIR / "templates").is_dir() else Path("~/.autosteer/data").expanduser()


def set_data_dir(path: str | os.PathLike) -> Path:
    """Use `path` as the data directory from now on, in this process and in processes it starts (a reloader)."""
    global _override
    _override = Path(path).expanduser().resolve()
    os.environ["AUTOSTEER_DATA_DIR"] = str(_override)
    return _override


def data_dir() -> Path:
    if _override is not None:
        return _override
    env = os.environ.get("AUTOSTEER_DATA_DIR") or os.environ.get("STEER_DATA_DIR")
    return Path(env).expanduser().resolve() if env else default_data_dir()


def kind_dir(kind: str) -> Path:
    """The folder of one kind of data: example_seeds, generated_seeds or templates."""
    if kind not in DATA_KINDS:
        raise ValueError(f"unknown kind of data {kind!r}; expected one of {', '.join(DATA_KINDS)}")
    return data_dir() / kind


def templates_dir() -> Path:
    return data_dir() / "templates"


def review_dir() -> Path:
    return data_dir() / "review"
