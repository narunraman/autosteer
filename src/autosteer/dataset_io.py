"""Datasets on disk as Parquet: one file per table plus dataset.json, instead of a pickled dict.

A dataset folder holds ``dataset.json`` (format version, the non-table attributes, and how to read each table back)
and ``<table>.parquet`` for every DataFrame attribute (questions, options, example_questions, ...). Text columns are
read back as object columns, as they were pickled. A column whose values mix types (true/false next to numbers next
to text, or lists) is stored as JSON text and decoded on load, so every value keeps its Python type.

Older folders that only have ``dataset.pkl`` still load; ``autosteer data convert`` rewrites them.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT = "autosteer-dataset/1"
JSON_NAME = "dataset.json"


def saved_file(folder: str | os.PathLike) -> Path | None:
    """The file that marks a saved dataset in folder (dataset.json), or None."""
    path = Path(folder) / JSON_NAME
    return path if path.exists() else None


def _plain(v):
    """A JSON-safe copy of v (numpy scalars and arrays to Python, NaN to None)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out, text, as_json, nulls = df.copy(), [], [], {}
    for c in df.columns:
        if df[c].dtype != object and not pd.api.types.is_string_dtype(df[c]):
            continue
        vals = [v for v in df[c].tolist() if not _is_missing(v)]
        if all(isinstance(v, str) for v in vals):
            text.append(c)
        else:
            as_json.append(c)
            out[c] = [None if _is_missing(v) else json.dumps(_plain(v), ensure_ascii=False) for v in df[c].tolist()]
        # which missing marker the pickle had (None or NaN), so it comes back the same
        kinds = {("none" if v is None else "nan") for v in df[c].tolist() if _is_missing(v)}
        if kinds:
            nulls[c] = "nan" if kinds == {"nan"} else "none"
    # pandas 3 reads column names back as its string type; the pickles have plain object names
    return out, {"text": text, "json": as_json, "nulls": nulls, "object_names": df.columns.dtype == object}


def _decode(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    for c in info.get("text", []) + info.get("json", []):
        if c not in df.columns:
            continue
        null = np.nan if info.get("nulls", {}).get(c) == "nan" else None
        if c in info.get("json", []):
            vals = [null if _is_missing(v) else json.loads(v) for v in df[c].tolist()]
        else:
            vals = [null if _is_missing(v) else v for v in df[c].tolist()]
        df[c] = pd.Series(vals, index=df.index, dtype=object)
    if info.get("object_names"):
        df.columns = pd.Index(list(df.columns), dtype=object)
    return df


def save(attributes: dict, folder: str | os.PathLike) -> None:
    """Write a dataset's attributes into folder; tables become Parquet files, the rest goes into dataset.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables, plain = {}, {}
    for key, value in attributes.items():
        if isinstance(value, pd.DataFrame):
            frame, info = _encode(value)
            frame.to_parquet(folder / f"{key}.parquet.tmp", index=not isinstance(frame.index, pd.RangeIndex))
            os.replace(folder / f"{key}.parquet.tmp", folder / f"{key}.parquet")
            tables[key] = info
        else:
            plain[key] = _plain(value)
    for stale in folder.glob("*.parquet"):
        if stale.stem not in tables:
            stale.unlink()
    doc = {"format": FORMAT, "attributes": plain, "tables": tables}
    tmp = folder / (JSON_NAME + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, folder / JSON_NAME)  # written last: its time stamp says the dataset changed


def load(folder: str | os.PathLike) -> dict:
    """The attributes saved in folder, from dataset.json and the Parquet files."""
    folder = Path(folder)
    if (folder / JSON_NAME).exists():
        doc = json.loads((folder / JSON_NAME).read_text(encoding="utf-8"))
        if doc.get("format") != FORMAT:
            raise ValueError(f"{folder / JSON_NAME}: unknown format {doc.get('format')!r}")
        attrs = dict(doc.get("attributes", {}))
        for key, info in doc.get("tables", {}).items():
            attrs[key] = _decode(pd.read_parquet(folder / f"{key}.parquet"), info)
        return attrs
    raise FileNotFoundError(f"No saved dataset ({JSON_NAME}) in {folder}")
