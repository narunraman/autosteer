"""Clean-up of the example seeds and seed candidates carried over from STEER-generation (October 2026).

    python -m autosteer.data_cleanup DATA_DIR [--dry-run]

Run by `autosteer data build` after the conversion to Parquet.

Some example_seeds / generated_seeds datasets hold the same question many times over (micro_utility's 794 example
seeds are 15 questions), candidates that are copies of the element's own seeds, and options joined to the wrong
questions (up to 121 per question, all numbered 0). For each dataset this keeps the first copy of every distinct
question (text, domain, type, level, perspective), drops candidates whose text is one of the element's example
seeds, renumbers the questions 0..n-1, and keeps a
question's options only when they are plausibly its own (at most 12, and its id was not shared); otherwise the
question gets the empty option row that option-free candidates have. Only example_questions / example_options
(and validation_df ids) change. The fixed dataset is saved as Parquet, whichever format it was read from.

Templates (fix_duplicate_templates): a template that repeats an earlier one exactly (every part, its options and
its metadata) is deleted; question ids are kept, so template ids do not change. The released data used such repeats
to balance a field (independence_risk's domains); the sampler balances it instead (SAMPLING_BALANCE). Copies that
only the sampler's load-time corrections make are dropped when the templates are loaded (template_fixes).
"""
import argparse
from pathlib import Path

import pandas as pd

from autosteer import dataset_io

KEY = ["question_text", "domain", "type", "difficulty_level", "tags"]
MAX_OPTIONS = 12


def fix(attrs: dict, seed_texts: set | None) -> tuple[dict, str]:
    q, o = attrs["example_questions"], attrs["example_options"]
    shared_ids = set(q.loc[q.duplicated(["question_id", "sub_id"], keep=False), "question_id"])
    keep = q.drop_duplicates(KEY)
    if seed_texts is not None:
        keep = keep[~keep["question_text"].isin(seed_texts)]
    per = o.groupby("question_id").size()
    new_q, new_o = [], []
    for new_id, (_, row) in enumerate(keep.iterrows()):
        old = row["question_id"]
        r = row.copy(); r["question_id"] = new_id; new_q.append(r)
        own = o[o["question_id"] == old]
        if old not in shared_ids and 0 < per.get(old, 0) <= MAX_OPTIONS:
            own = own.copy(); own["question_id"] = new_id; new_o.extend(r2 for _, r2 in own.iterrows())
        else:
            new_o.append(pd.Series({"question_id": new_id, "sub_id": int(row["sub_id"]), "option_id": 0,
                                    "option_text": "", "correct_answer": True}))
    nq = pd.DataFrame(new_q, columns=q.columns).reset_index(drop=True).astype(q.dtypes.to_dict())
    no = pd.DataFrame(new_o, columns=o.columns).reset_index(drop=True).astype(o.dtypes.to_dict())
    out = dict(attrs, example_questions=nq, example_options=no)
    v = attrs.get("validation_df")
    if isinstance(v, pd.DataFrame) and len(v):
        out["validation_df"] = v.iloc[0:0].copy()  # ids changed; the solve check is re-run on demand
    return out, f"{len(q)} -> {len(nq)} questions, {len(o)} -> {len(no)} option rows"


def fix_duplicate_seeds(data_dir, dry_run: bool = False) -> list[str]:
    """Fix every broken example_seeds / generated_seeds dataset under data_dir; one line per dataset changed."""
    root = Path(data_dir)
    load = lambda kind, el: dataset_io.load(root / kind / el)
    done = []
    for kind in ("example_seeds", "generated_seeds"):  # seeds first: candidates are compared with the fixed seeds
        if not (root / kind).is_dir():
            continue
        for d in sorted(p for p in (root / kind).iterdir() if dataset_io.saved_file(p)):
            attrs = load(kind, d.name)
            q, o = attrs["example_questions"], attrs["example_options"]
            seeds = None
            if kind == "generated_seeds" and dataset_io.saved_file(root / "example_seeds" / d.name):
                seeds = set(load("example_seeds", d.name)["example_questions"]["question_text"])
            per = o.groupby("question_id").size()
            broken = (q.duplicated(KEY).any() or q.duplicated(["question_id", "sub_id"]).any()
                      or (len(per) and per.max() > MAX_OPTIONS) or (seeds is not None and q["question_text"].isin(seeds).any()))
            if not broken:
                continue
            fixed, msg = fix(attrs, seeds)
            done.append(f"{kind}/{d.name}: {msg}")
            if not dry_run:
                dataset_io.save(fixed, d)
                (d / dataset_io.PICKLE_NAME).unlink(missing_ok=True)  # the fixed copy is the Parquet one
    return done


def _copy_ids(q, o) -> set:
    """question_ids of the templates that repeat an earlier one (lower id) exactly."""
    from autosteer.sampler.template_fixes import exact_copies
    return exact_copies(q, o)


def fix_duplicate_templates(data_dir, dry_run: bool = False) -> list[str]:
    """Delete duplicated templates under data_dir/templates; one line per element changed. Elements whose templates
    are written in code (their data directory is not read) are left as they are."""
    from autosteer.filler_tools import code_templates

    root = Path(data_dir) / "templates"
    done = []
    if not root.is_dir():
        return done
    for d in sorted(p for p in root.iterdir() if dataset_io.saved_file(p)):
        if code_templates.has_code_templates(d.name):
            continue
        attrs = dataset_io.load(d)
        q, o = attrs.get("questions"), attrs.get("options")
        if not isinstance(q, pd.DataFrame) or q.empty or "question_id" not in q.columns:
            continue
        drop = _copy_ids(q, o)
        if not drop:
            continue
        n = q["question_id"].nunique()
        nq = q[~q["question_id"].isin(drop)].reset_index(drop=True)
        no = o[~o["question_id"].isin(drop)].reset_index(drop=True) if isinstance(o, pd.DataFrame) else o
        done.append(f"templates/{d.name}: {n} -> {nq['question_id'].nunique()} templates")
        if not dry_run:
            dataset_io.save(dict(attrs, questions=nq, options=no), d)
            (d / dataset_io.PICKLE_NAME).unlink(missing_ok=True)
    return done


# template directories of the source that are not elements: removed from a built data directory
NOT_ELEMENTS = ["backward_induction copy"]  # a second, unvetted wording of backward_induction (58 of 600 pass its audit)


def remove_non_elements(data_dir, dry_run: bool = False) -> list[str]:
    import shutil
    done = []
    for name in NOT_ELEMENTS:
        d = Path(data_dir) / "templates" / name
        if d.is_dir():
            done.append(f"templates/{name}: removed")
            if not dry_run:
                shutil.rmtree(d)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for line in (fix_duplicate_seeds(a.data_dir, a.dry_run) + fix_duplicate_templates(a.data_dir, a.dry_run)
                 + remove_non_elements(a.data_dir, a.dry_run)):
        print(line)


if __name__ == "__main__":
    main()
