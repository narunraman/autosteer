"""Merge templates that repeat a text in another setting into one text with several settings (October 2026).

For every template folder and pool: variants (the former templates) that share their text (every part, spaces
collapsed), domain and other text columns become one text; its id is the smallest variant id, and every variant keeps
its id (autosteer.template_settings). Variants may differ in type, difficulty_level and tags, in their explanation
(worked examples) and in which option is correct (backward_induction: the setting decides the answer). Option rows
of templates that no longer exist are dropped (nothing reads them). Variants must
have the same option texts, except in elements whose filler always supplies its own options (the stored option rows
are placeholders there and are never read): those keep the first variant's rows.

Texts that repeat with a different domain are left separate (a labelling inconsistency, not a setting); they are
listed. Writes the folders, then checks that loading each one gives back the original tables (with a text_id column),
up to the replaced placeholder options.

    uv run python scripts/migrations/merge_template_settings.py [--data DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autosteer import template_settings as ts
from autosteer.datasets import Dataset
from autosteer.paths import data_dir

POOLS = {'questions': 'options', 'example_questions': 'example_options'}
# Variants of one text that are the same in everything (setting, explanation, options) are kept once. They were
# separate templates that differed only in placeholder option rows. Variants that published rows name are all kept
# (narunraman/steer and steer_me, generator autosteer@0.1.0+4bf3f71: test rows of welfare_theorem_1, few-shot rows
# of enforceability), so every published template_id still resolves; else the smallest id.
PUBLISHED_DUPLICATES = {('welfare_theorem_1', 'questions'): {23081},
                        ('enforceability', 'example_questions'): {0, 1, 8, 10}}


def own_option_elements(folders: list[str]) -> set[str]:
    """Elements whose filler returned options of its own on every probe (40 seeds, both pools)."""
    from autosteer.sampler import ElementSampler
    from autosteer.sampler import generator as G
    seen = collections.defaultdict(set)
    orig = G.ElementSampler._call_filler

    def spy(self, element_name, metadata):
        out = orig(self, element_name, metadata)
        seen[element_name].add(isinstance(out, dict) and 'options_list' in out)
        return out

    G.ElementSampler._call_filler = spy
    try:
        s = ElementSampler(seed=0)
        for el in folders:
            if el not in s._sampleable:
                continue
            for seed in range(40):
                for pool in ('questions', 'examples'):
                    try:
                        s.sample_question(el, seed, pool=pool)
                    except Exception:  # noqa: BLE001 - only the fills that ran count
                        pass
    finally:
        G.ElementSampler._call_filler = orig
    return {el for el, v in seen.items() if v == {True}}


def plan(ds: Dataset, pool: str, own_options: bool, keep=frozenset()) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The pool's questions with text_id set (and placeholder options aligned), and a report."""
    q, o = getattr(ds, pool), getattr(ds, POOLS[pool])
    q = q.copy(); o = o.copy() if o is not None else o
    orphans = 0
    if o is not None and len(o):   # option rows of templates that no longer exist (never read): dropped
        has_template = o['question_id'].astype(int).isin(set(q['question_id'].astype(int)))
        orphans = int((~has_template).sum())
        o = o[has_template].reset_index(drop=True)
    parts, opts = ts._parts(q), ts._option_rows(o)
    groups = collections.defaultdict(list)
    for qid, rows in parts.items():
        groups[ts.text_key(rows, opts.get(qid), with_options=not own_options)].append(int(qid))
    text_of, replaced = {}, 0
    for ids in groups.values():
        ids.sort()
        for i in ids:
            text_of[i] = ids[0]
        if own_options and o is not None and len(ids) > 1:
            head = opts.get(ids[0])
            for i in ids[1:]:
                vo = opts.get(i)
                same = vo is not None and head is not None and len(vo) == len(head) and (
                    vo[['sub_id', 'option_id', 'option_text']].astype(str).to_numpy()
                    == head[['sub_id', 'option_id', 'option_text']].astype(str).to_numpy()).all()
                if not same:   # unused placeholders: take the text's rows
                    o = o[o['question_id'].astype(int) != i]
                    if head is not None:
                        h = head.copy(); h['question_id'] = i
                        o = pd.concat([o, h], ignore_index=True)
                    replaced += 1
    # variants of one text that are the same in everything: one is kept
    dropped = []
    if len(q):
        parts, opts = ts._parts(q), ts._option_rows(o)
        by_text = collections.defaultdict(list)
        for i, t in text_of.items():
            by_text[t].append(i)
        for t, ids in by_text.items():
            if len(ids) < 2:
                continue
            same_as = collections.defaultdict(list)
            for i in ids:
                rows, vo = parts[i], opts.get(i)
                sig = (tuple(map(tuple, rows.drop(columns=['question_id']).astype(str).to_numpy())),
                       () if vo is None else tuple(map(tuple, vo.drop(columns=['question_id']).astype(str).to_numpy())))
                same_as[sig].append(i)
            for members in same_as.values():
                if len(members) > 1:
                    kept = {i for i in members if i in keep} or {min(members)}
                    dropped += [i for i in members if i not in kept]
        if dropped:
            gone = set(dropped)
            q = q[~q['question_id'].astype(int).isin(gone)]
            if o is not None:
                o = o[~o['question_id'].astype(int).isin(gone)]
            survivors = collections.defaultdict(list)
            for i, t in text_of.items():
                if i not in gone:
                    survivors[t].append(i)
            text_of = {i: min(v) for v in survivors.values() for i in v}
    q['text_id'] = q['question_id'].astype(int).map(text_of)
    if o is not None:
        o = o.sort_values(['question_id', 'sub_id', 'option_id'], kind='stable').reset_index(drop=True)
    # same text, other domain: left separate, listed
    by_text = collections.defaultdict(set)
    for qid, rows in ts._parts(q).items():
        by_text[tuple(ts._norm(t) for t in rows['question_text'])].add(text_of[int(qid)])
    domain_conflicts = [sorted(v) for v in by_text.values() if len(v) > 1]
    rep = {'variants': len(parts), 'texts': len(set(text_of.values())), 'placeholder_options_replaced': replaced,
           'identical_variants_dropped': sorted(dropped),
           'orphan_option_rows_dropped': orphans,
           'same_text_other_domain': domain_conflicts}
    return q, o, rep


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    a = a.reset_index(drop=True); b = b.reset_index(drop=True)
    return all(a[c].astype(str).tolist() == b[c].astype(str).tolist() for c in a.columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', default=None)
    args = ap.parse_args()
    warnings.filterwarnings('ignore'); np.seterr(all='ignore')
    root = Path(args.data) if args.data else data_dir()
    folders = sorted(p for p in (root / 'templates').iterdir() if (p / 'dataset.json').exists())
    own = own_option_elements([p.name for p in folders])
    report, bad = {}, []
    for folder in folders:
        ds = Dataset.load(folder)
        if any('text_id' in getattr(ds, pool).columns for pool in POOLS if getattr(ds, pool) is not None):
            report[folder.name] = 'already merged'
            continue
        before = {}
        reps = {}
        for pool in POOLS:
            if getattr(ds, pool) is None or getattr(ds, pool).empty:
                continue
            before[pool] = None
            q, o, reps[pool] = plan(ds, pool, folder.name in own, PUBLISHED_DUPLICATES.get((folder.name, pool), frozenset()))
            setattr(ds, pool, q); setattr(ds, POOLS[pool], o)
            if pool == 'questions' and reps[pool]['identical_variants_dropped'] and len(ds.validation_df):
                v = ds.validation_df
                ds.validation_df = v[~v['question_id'].astype(int).isin(reps[pool]['identical_variants_dropped'])]
        report[folder.name] = reps
        with tempfile.TemporaryDirectory() as tmp:
            ds.save(tmp)
            back = Dataset.load(tmp)
            for pool in POOLS:
                if pool not in before:
                    continue
                q0 = getattr(ds, pool).drop(columns='text_id').sort_values(['question_id', 'sub_id']).reset_index(drop=True)
                o0 = getattr(ds, POOLS[pool]).sort_values(['question_id', 'sub_id', 'option_id']).reset_index(drop=True)
                q1, o1 = getattr(back, pool).drop(columns='text_id'), getattr(back, POOLS[pool])
                if not same(q1, q0) or not same(o1, o0):
                    bad.append((folder.name, pool, 'questions' if not same(q1, q0) else 'options'))
            if not args.dry_run and not any(b[0] == folder.name for b in bad):
                for f in folder.glob('*.parquet'):
                    f.unlink()
                for f in Path(tmp).iterdir():
                    shutil.copy2(f, folder / f.name)
    totals = collections.Counter()
    for r in report.values():
        if isinstance(r, dict):
            for p in r.values():
                totals['variants'] += p['variants']; totals['texts'] += p['texts']
                totals['placeholder_options_replaced'] += p['placeholder_options_replaced']
                totals['orphan_option_rows_dropped'] += p['orphan_option_rows_dropped']
                totals['identical_variants_dropped'] += len(p['identical_variants_dropped'])
                totals['same_text_other_domain'] += len(p['same_text_other_domain'])
    print(dict(totals), 'own-option elements:', sorted(own))
    print('round trip failed:', bad or 'none', '(nothing written for those)' if bad else '')
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
