"""Export STEER and STEER-ME from the generator: Parquet per element plus a manifest, to a local directory.

    autosteer export --out out/export-2026-10-03

Which elements: every element of web/content/elements.json whose status is not `held` and that the sampler can
sample (a registered filler and templates in the data directory). Others are listed in the manifest with the
reason they were skipped.

How many: cells are the (type, domain, difficulty) combinations of the element's test templates. Each cell gets
--per-cell questions (default 20); if that totals less than --floor (default 200), the floor is spread evenly
over the cells instead (the first floor % cells cells, in sorted order, get one more). A question counts once
however many parts (rows) it has, so a consistency group counts once.
The rows' `difficulty` is the relative level within the element (1 = its easiest; autosteer.sampler.difficulty);
cells, seeds and the sampler use the raw level, which maps one to one onto it.

Each question i of a cell is drawn with its own seed, derive_seed(export seed, element, type, domain,
difficulty, i), so (element, template_id, seed) in a row reproduces it (ElementSampler.sample_question) and the
export does not depend on element order or worker count. Draws that raise, that fail row_problems() or that
repeat an earlier question of the element (same content hash, or the same texts containing numbers with other
options) are skipped and counted; the next index is tried, up to --max-attempts-per-question times the cell's target.

few_shot (--few-shot-per-type, default 5): that many questions per question type, from templates that never serve
test (few_shot_plan: the element's example templates, or test templates held out of test without removing a test
cell), and never a question that is also in test.

Writes <out>/<steer|steer_me>/<element>/{test,few_shot}.parquet and <out>/manifest.json. Refuses a dirty checkout and a
data directory that does not match data_files.json unless told otherwise. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

DATASET_DIRS = {"STEER": "steer", "STEER-ME": "steer_me"}
DEFAULT_SEED = 20261003


def cell_targets(cells: list[tuple], per_cell: int = 20, floor: int = 200) -> dict[tuple, int]:
    """Questions per cell: per_cell each, or the floor spread evenly when per_cell * cells < floor."""
    cells = sorted(cells)
    if not cells:
        return {}
    if per_cell * len(cells) >= floor:
        return {c: per_cell for c in cells}
    base, extra = divmod(floor, len(cells))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(cells)}


def _pool(sampler, element: str, pool: str):
    """The pool's question frame, or None when the element has no such pool or it is empty."""
    try:
        q, _ = sampler._pool_frames(sampler._load_template(element), pool)
    except ValueError:
        return None
    return q if len(q) else None


def _cell_label(element: str, cell: tuple) -> str:
    """`type|domain|difficulty` of a cell for the export manifest, with the relative difficulty level (the dataset's
    `difficulty`; cells are keyed internally by the raw level)."""
    from autosteer.sampler.difficulty import to_relative

    qtype, domain, level = cell
    return f"{qtype}|{domain}|{to_relative(element, level)}"


def element_cells(sampler, element: str, pool: str = "questions", exclude=frozenset()) -> list[tuple]:
    """Sorted (type, domain, raw public difficulty) combinations present in the element's templates (minus
    `exclude`, template row ids of the pool). The rows carry the relative level of the raw one (one per raw level,
    so the cells are the same); seeds are derived from the raw level."""
    from autosteer.sampler.template_fixes import public_difficulty

    q = _pool(sampler, element, pool)
    if q is None:
        return []
    q = q[~q["question_id"].isin(list(exclude))]
    combos = {(str(t), str(d), int(public_difficulty(element, lvl)))
              for t, d, lvl in q[["type", "domain", "difficulty_level"]].itertuples(index=False)}
    return sorted(combos)


def _signatures(q) -> dict[int, tuple]:
    """Template row id -> its question texts in part order (the unfilled template text)."""
    sig: dict[int, list] = {}
    for qid, _, text in q.sort_values(["question_id", "sub_id"])[["question_id", "sub_id", "question_text"]].itertuples(
            index=False):
        sig.setdefault(int(qid), []).append(str(text))
    return {k: tuple(v) for k, v in sig.items()}


def few_shot_plan(sampler, element: str, seed: int, max_held_out: int = 5, protect=frozenset()) -> dict:
    """Which templates the few_shot split is drawn from; they never serve the test split.

    Per question type of the test templates:
    - `examples`: the element's few-shot example templates of that type (the generator's `examples` pool) whose
      text is not also a test template's text;
    - otherwise `held_out`: test templates of that type are held out of test. Templates with identical text move
      together. Up to max(1, min(max_held_out, groups // 10)) text groups per type, each time the group whose
      cells keep the most templates (ties broken in an order seeded by (seed, element, type)); a group is only
      held out if every (type, domain, difficulty) cell of test keeps at least one template, so test cells and
      sizes do not change;
      Cells in `protect` (cells whose test sample fell short with templates held out) lose no templates;
    - `none`: neither is possible (the type's templates are each the only one of some cell).

    Returns {"types": {type: {"source", "pool", "templates": [row ids]}}, "held_out": sorted test row ids,
    "cells": the test cells that lost templates}.
    """
    import random

    from autosteer.sampler.generator import derive_seed
    from autosteer.sampler.template_fixes import public_difficulty

    q = _pool(sampler, element, "questions")
    ex = _pool(sampler, element, "examples")
    test_sig = _signatures(q)
    test_texts = set(test_sig.values())
    first = q.sort_values(["question_id", "sub_id"]).drop_duplicates("question_id")
    t_type = {int(i): str(t) for i, t in first[["question_id", "type"]].itertuples(index=False)}
    t_cell = {int(i): (str(t), str(d), int(public_difficulty(element, lvl)))
              for i, t, d, lvl in first[["question_id", "type", "domain", "difficulty_level"]].itertuples(index=False)}
    ex_by_type: dict[str, list[int]] = {}
    if ex is not None:
        ex_first = ex.sort_values(["question_id", "sub_id"]).drop_duplicates("question_id")
        ex_sig = _signatures(ex)
        for i, t in ex_first[["question_id", "type"]].itertuples(index=False):
            if ex_sig[int(i)] not in test_texts:
                ex_by_type.setdefault(str(t), []).append(int(i))

    cell_count = collections.Counter(t_cell.values())
    groups: dict[tuple, list[int]] = {}
    for i, s in test_sig.items():
        groups.setdefault(s, []).append(i)
    held: set[int] = set()
    plan: dict[str, dict] = {}
    for qtype in sorted(set(t_type.values())):
        if ex_by_type.get(qtype):
            plan[qtype] = {"source": "examples", "pool": "examples", "templates": sorted(ex_by_type[qtype])}
            continue
        mine = sorted({test_sig[i] for i, t in t_type.items() if t == qtype and i not in held})
        random.Random(derive_seed(seed, element, "held_out", qtype)).shuffle(mine)
        want, taken = max(1, min(max_held_out, len(mine) // 10)), []
        rank = {sig: k for k, sig in enumerate(mine)}
        while len(taken) < want:
            # the group whose cells keep the most templates (ties: seeded order), so small cells stay varied
            best, best_key = None, None
            for sig in mine:
                if groups[sig][0] in held:
                    continue
                need = collections.Counter(t_cell[i] for i in groups[sig])
                if any(c in protect for c in need):
                    continue
                left = min(cell_count[c] - n for c, n in need.items())
                if left >= 1 and (best_key is None or (left, -rank[sig]) > best_key):
                    best, best_key = sig, (left, -rank[sig])
            if best is None:
                break
            ids = groups[best]
            cell_count.subtract(collections.Counter(t_cell[i] for i in ids))
            held.update(ids)
            taken.extend(i for i in ids if t_type[i] == qtype)
        plan[qtype] = ({"source": "held_out", "pool": "questions", "templates": sorted(taken)} if taken
                       else {"source": "none", "pool": None, "templates": []})
    return {"types": plan, "held_out": sorted(held), "cells": sorted({t_cell[i] for i in held})}


class _Collector:
    """Accepts sampled questions for one split: drops failing, duplicate and near-duplicate questions."""

    def __init__(self, generator: str, avoid_keys=(), avoid_texts=()):
        self.generator = generator
        self.rows: list[dict] = []
        self.keys: set[str] = set()
        self.avoid_keys, self.avoid_texts = set(avoid_keys), set(avoid_texts)
        self.texts: set[tuple] = set()
        self.rejected = collections.Counter()
        self.errors = collections.Counter()

    def draw(self, fn) -> bool:
        from autosteer.release.rows import build_rows, row_problems

        try:
            q = fn()
        except Exception as e:  # noqa: BLE001 - every failure is counted
            self.errors[f"{type(e).__name__}: {str(e)[:160]}"] += 1
            self.rejected["sampler_error"] += 1
            return False
        r = build_rows(q, self.generator)
        probs = row_problems(q, r)
        if probs:
            for p in probs:
                self.rejected[p] += 1
            return False
        key = r[0]["base_id"]
        texts = tuple(x["question_text"] for x in r)
        if key in self.keys or key in self.avoid_keys:
            self.rejected["duplicate_question"] += 1
            return False
        if any(ch.isdigit() for t in texts for ch in t) and (texts in self.texts or texts in self.avoid_texts):
            # same filled numbers in the text but other options: the same question with other distractors.
            # (Texts without numbers, e.g. "Which would you choose?" with the numbers in the options, repeat.)
            self.rejected["same_text_other_options"] += 1
            return False
        self.keys.add(key)
        self.texts.add(texts)
        self.rows.extend(r)
        return True


def _draw_test(sampler, element, seed, per_cell, floor, max_factor, generator, held):
    """Test questions of every cell from the templates not in `held`: (collector, cells, targets, per cell)."""
    import random

    from autosteer.sampler.generator import derive_seed

    cells = element_cells(sampler, element, exclude=held)
    targets = cell_targets(cells, per_cell, floor)
    test = _Collector(generator)
    per_cell_out = {}
    for cell in sorted(targets):
        qtype, domain, level = cell
        ids = [i for i in sampler.template_ids(element, type=qtype, domain=domain, difficulty_level=level)
               if i not in held]
        want, got, i = targets[cell], 0, 0
        limit = max_factor * want + 20
        while got < want and i < limit:
            qseed = derive_seed(seed, element, qtype, domain, level, i)
            i += 1
            # the template is chosen as sample_question chooses it (same RNG), from the test templates only
            tid = ids[random.Random(derive_seed("template", qseed)).randrange(len(ids))]
            got += test.draw(lambda: sampler.sample_question(element, qseed, template_id=tid))
        per_cell_out["|".join(map(str, cell))] = {"target": want, "questions": got, "draws": i}
    return test, cells, targets, per_cell_out


def export_element(element: str, seed: int, per_cell: int, floor: int, max_factor: int, generator: str,
                   few_shot_per_type: int = 0, max_held_out: int = 5) -> dict:
    """Generate one element's rows. Returns {element, rows, few_shot_rows, stats} (test rows sorted by cell, then
    draw index; few_shot rows by type, then draw index). few_shot_per_type = 0 exports test only, from all
    test templates (the behaviour before the few_shot split)."""
    warnings.filterwarnings("ignore")
    import random

    import numpy as np

    np.seterr(all="ignore")
    from autosteer.sampler import ElementSampler
    from autosteer.sampler.generator import derive_seed

    t0 = time.time()
    sampler = ElementSampler()
    protect: set = set()
    while True:
        plan = few_shot_plan(sampler, element, seed, max_held_out, frozenset(protect)) if few_shot_per_type else None
        test, cells, targets, per_cell_out = _draw_test(sampler, element, seed, per_cell, floor, max_factor, generator,
                                                        frozenset(plan["held_out"]) if plan else frozenset())
        # a cell that fell short after losing templates to few_shot keeps them all
        short = {c for c in targets if per_cell_out["|".join(map(str, c))]["questions"] < targets[c]}
        lost = short & set(plan["cells"]) if plan else set()
        if not lost:
            break
        protect |= lost
    stats = {
        "cells": len(cells),
        "target": sum(targets.values()),
        "questions": len(test.keys),
        "rows": len(test.rows),
        "draws": sum(c["draws"] for c in per_cell_out.values()),
        "shortfall": sum(targets.values()) - len(test.keys),
        "rejected": dict(sorted(test.rejected.items())),
        "sampler_errors": dict(test.errors.most_common(5)),
        "short_cells": {_cell_label(element, c): per_cell_out["|".join(map(str, c))] for c in sorted(targets)
                        if per_cell_out["|".join(map(str, c))]["questions"] < targets[c]},
    }
    if plan:
        stats["protected_cells"] = [_cell_label(element, c) for c in sorted(protect)]
    few_rows: list[dict] = []
    if plan:
        few = _Collector(generator, avoid_keys=test.keys, avoid_texts=test.texts)
        per_type = {}
        for qtype, p in plan["types"].items():
            tids = list(p["templates"])
            random.Random(derive_seed(seed, element, "few_shot", qtype)).shuffle(tids)
            got, i = 0, 0
            limit = max_factor * few_shot_per_type + 20 if tids else 0
            while got < few_shot_per_type and i < limit:
                qseed = derive_seed(seed, element, "few_shot", qtype, i)
                tid = tids[(got + i) % len(tids)]  # round robin over the templates, shifted on rejects
                i += 1
                got += few.draw(lambda: sampler.sample_question(element, qseed, template_id=tid, pool=p["pool"]))
            per_type[qtype] = {"source": p["source"], "templates": len(tids), "questions": got, "draws": i}
        few_rows = few.rows
        stats["few_shot"] = {
            "per_type": few_shot_per_type,
            "questions": len(few.keys),
            "rows": len(few.rows),
            "target": few_shot_per_type * sum(1 for p in plan["types"].values() if p["templates"]),
            "held_out_templates": len(plan["held_out"]),
            "sources": sorted({p["source"] for p in plan["types"].values()}),
            "types": per_type,
            "rejected": dict(sorted(few.rejected.items())),
            "sampler_errors": dict(few.errors.most_common(5)),
        }
    stats["seconds"] = round(time.time() - t0, 1)
    return {"element": element, "rows": test.rows, "few_shot_rows": few_rows, "stats": stats}


def select_elements(requested: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    from autosteer.release.rows import catalog
    from autosteer.sampler import ElementSampler

    sampleable = set(ElementSampler()._sampleable)
    chosen, skipped = [], {}
    for name, e in sorted(catalog().items()):
        if e.get("status") == "held":
            skipped[name] = "held (web/content/elements.json)"
        elif name not in sampleable:
            skipped[name] = "no generator (no filler or no templates)"
        else:
            chosen.append(name)
    for name in sorted(sampleable - set(catalog())):
        skipped[name] = "not listed in web/content/elements.json"
    if requested:
        bad = [e for e in requested if e not in chosen]
        if bad:
            raise SystemExit("not exportable: " + ", ".join(f"{e} ({skipped.get(e, 'unknown element')})" for e in bad))
        chosen = [e for e in chosen if e in requested]
    return chosen, skipped


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from autosteer import data_files
    from autosteer.paths import data_dir
    DATA_PATH = data_dir()
    from autosteer.release.rows import element_info, generator_id, git_state
    from autosteer.release.schema import to_table, validate

    out = Path(args.out).expanduser()
    git = git_state()
    if git.get("dirty") and not args.allow_dirty:
        raise SystemExit("refusing to export from a checkout with uncommitted changes to tracked files "
                         f"({', '.join(git['dirty_files'][:5])}); commit them or pass --allow-dirty")
    data_check = data_files.verify(Path(DATA_PATH))
    if not data_check["ok"] and not args.allow_unverified_data:
        raise SystemExit(f"data directory {DATA_PATH} does not match data_files.json "
                         f"({len(data_check['missing'])} missing, {len(data_check['changed'])} changed); "
                         "run `autosteer data verify`")
    generator = generator_id()
    elements, skipped = select_elements(args.elements)
    out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {len(elements)} elements with {generator} (seed {args.seed}) -> {out}", flush=True)

    results: dict[str, dict] = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(export_element, e, args.seed, args.per_cell, args.floor, args.max_attempts_per_question,
                            generator, args.few_shot_per_type, args.max_held_out): e for e in elements}
        for fut in cf.as_completed(futs):
            e = futs[fut]
            try:
                results[e] = fut.result()
            except Exception as ex:  # noqa: BLE001
                results[e] = {"element": e, "rows": [], "few_shot_rows": [], "stats": {"fatal": f"{type(ex).__name__}: {ex}",
                                                                   "traceback": traceback.format_exc()[-2000:]}}
            st = results[e]["stats"]
            print(f"  {e:40s} q={st.get('questions', 0):5d}/{st.get('target', 0):<5d} rows={st.get('rows', 0):5d} "
                  f"draws={st.get('draws', 0):5d} few_shot={st.get('few_shot', {}).get('questions', 0):3d} "
                  f"{st.get('fatal', '')}", flush=True)

    per_element = {}
    for e in elements:
        res, info = results[e], element_info(e)
        st = res["stats"]
        rec = {"benchmark": info["benchmark"], "setting": info["setting"], "module": info["module"], **st}
        if res["rows"]:
            table = to_table(res["rows"])
            errs = validate(table)
            path = out / DATASET_DIRS[info["benchmark"]] / e / "test.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path, compression="zstd")
            rec.update({"file": str(path.relative_to(out)), "sha256": _sha256(path), "schema_errors": errs})
        else:
            errs = ["no rows"]
            rec["schema_errors"] = errs
        if res.get("few_shot_rows"):
            table = to_table(res["few_shot_rows"])
            few_errs = validate(table)
            path = out / DATASET_DIRS[info["benchmark"]] / e / "few_shot.parquet"
            pq.write_table(table, path, compression="zstd")
            rec["few_shot"].update({"file": str(path.relative_to(out)), "sha256": _sha256(path),
                                    "schema_errors": few_errs})
            errs = errs + [f"few_shot: {x}" for x in few_errs]
        few = st.get("few_shot")
        short = st.get("shortfall") or (few is not None and few["questions"] < few["target"])
        rec["status"] = ("fail" if errs or "fatal" in st else "short" if short else "ok")
        per_element[e] = rec

    totals = collections.defaultdict(lambda: {"elements": 0, "questions": 0, "rows": 0,
                                              "few_shot_questions": 0, "few_shot_rows": 0})
    for rec in per_element.values():
        t = totals[rec["benchmark"]]
        t["elements"] += 1
        t["questions"] += rec.get("questions", 0)
        t["rows"] += rec.get("rows", 0)
        t["few_shot_questions"] += rec.get("few_shot", {}).get("questions", 0)
        t["few_shot_rows"] += rec.get("few_shot", {}).get("rows", 0)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": generator,
        "git": git,
        "seed": args.seed,
        "rule": {"per_cell": args.per_cell, "floor": args.floor, "cell": ["type", "domain", "difficulty"],
                 "split": "test", "pool": "questions",
                 "few_shot": {"per_type": args.few_shot_per_type, "max_held_out_per_type": args.max_held_out,
                              "templates": "the examples pool where it has the type (texts not in test), else test "
                                           "templates held out of test (see few_shot_plan)"}},
        "command": " ".join(["autosteer export", *map(str, args.argv)]),
        "data_dir": str(DATA_PATH),
        "data_files": {"source": data_files.load_list()["source"], "verified": data_check["ok"]},
        "totals": dict(totals),
        "elements": per_element,
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    bad = sorted(e for e, r in per_element.items() if r["status"] != "ok")
    print(f"done: {len(elements) - len(bad)} ok, {len(bad)} not ok {bad}; manifest {out / 'manifest.json'}")
    return 1 if any(per_element[e]["status"] == "fail" for e in bad) else 0


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True, help="local output directory")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"export seed (default {DEFAULT_SEED})")
    ap.add_argument("--elements", nargs="*", help="subset of elements (default: all exportable)")
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--floor", type=int, default=200)
    ap.add_argument("--max-attempts-per-question", type=int, default=20,
                    help="give up on a cell after this many draws per wanted question")
    ap.add_argument("--few-shot-per-type", type=int, default=5,
                    help="few_shot questions per question type (0: no few_shot split, test from all templates)")
    ap.add_argument("--max-held-out", type=int, default=5,
                    help="at most this many template texts per type held out of test for few_shot")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--allow-dirty", action="store_true", help="export from a checkout with uncommitted changes")
    ap.add_argument("--allow-unverified-data", action="store_true",
                    help="export although the data directory does not match data_files.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autosteer export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args(argv)
    args.argv = list(argv if argv is not None else sys.argv[1:])
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
