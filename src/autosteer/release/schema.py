"""Arrow schema of the released datasets (STEER and STEER-ME) and table-level validation.

The fields and their order are those of autosteer.schemas.DatasetRow (one row per question part); this module
adds the Parquet types and the checks that need the whole table (unique ids, duplicate questions).
"""
from __future__ import annotations

import pyarrow as pa

from autosteer.schemas.records import DATASET_FIELDS, DatasetRow

GRADING = ("answer", "consistency")

SCHEMA = pa.schema([
    ("element", pa.string()),
    ("question_id", pa.string()),
    ("base_id", pa.string()),
    ("part", pa.int32()),
    ("template_id", pa.string()),
    ("question_text", pa.string()),
    ("options", pa.list_(pa.string())),
    ("grading", pa.string()),
    ("correct_index", pa.int32()),
    ("consistency_group", pa.string()),
    ("consistency_rule", pa.string()),
    ("type", pa.string()),
    ("domain", pa.string()),
    ("perspective", pa.string()),
    ("tags", pa.list_(pa.string())),
    ("difficulty", pa.int32()),
    ("setting", pa.string()),
    ("module", pa.string()),
    ("generator", pa.string()),
    ("seed", pa.int64()),
])
assert tuple(SCHEMA.names) == DATASET_FIELDS, "SCHEMA and DatasetRow disagree"

REQUIRED = [f for f in SCHEMA.names if f not in ("correct_index", "consistency_group", "consistency_rule",
                                                  "perspective")]


def to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pydict({f: [r[f] for r in rows] for f in SCHEMA.names}, schema=SCHEMA)


def validate(table: pa.Table) -> list[str]:
    """Human-readable violations (empty list = valid): per-row rules of DatasetRow plus table-level checks."""
    errs: list[str] = []
    if not table.schema.equals(SCHEMA, check_metadata=False):
        return ["schema mismatch"]
    rows = table.to_pylist()
    bad = {}
    for r in rows:
        for f in REQUIRED:
            if r[f] is None:
                bad.setdefault(f"null {f}", r["question_id"])
        try:
            DatasetRow(**r)
        except Exception as e:  # noqa: BLE001 - pydantic ValidationError; report the first message per kind
            msg = str(e).splitlines()[2].strip() if len(str(e).splitlines()) > 2 else str(e)
            bad.setdefault(msg, r["question_id"])
    errs += [f"{msg} (e.g. {qid})" for msg, qid in bad.items()]
    ids = [r["question_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append(f"{len(ids) - len(set(ids))} duplicate question_id")
    # parts of one base: 0..k-1, one grading, one consistency group
    by_base: dict[tuple, list[dict]] = {}
    for r in rows:
        by_base.setdefault((r["element"], r["base_id"]), []).append(r)
    sigs = set()
    for key, parts in by_base.items():
        if sorted(p["part"] for p in parts) != list(range(len(parts))):
            errs.append(f"{key[0]}/{key[1]}: parts are not 0..k-1")
        if len({(p["grading"], p["consistency_group"], p["template_id"], p["seed"]) for p in parts}) != 1:
            errs.append(f"{key[0]}/{key[1]}: parts disagree on grading, consistency group, template or seed")
        sig = (key[0],) + tuple((p["question_text"], tuple(sorted(p["options"])))
                                for p in sorted(parts, key=lambda p: p["part"]))
        if sig in sigs:
            errs.append(f"{key[0]}/{key[1]}: duplicate question (same texts and options as another base)")
        sigs.add(sig)
    return errs
