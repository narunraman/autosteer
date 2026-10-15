"""Template texts and their settings.

A template folder stores each template text once. A text can be filled in several settings (add_sub's stories as
integer, large or decimal numbers; mult_div's with positive, negative or mixed signs at three levels; a worked example
whose explanation depends on the type), and before October 2026 each setting was a separate template with a copy of
the text. On disk now:

- ``questions`` / ``options`` (and ``example_questions`` / ``example_options``): one row per part of each text
  (question_id = the text's id) and the text's options;
- ``settings`` (and ``example_settings``): one row per variant, a text in one setting: ``question_id`` (the text),
  ``variant_id`` (the variant's id: the id validation, the published datasets and the API's ``template_id`` use, so
  the ids of the separate templates are kept), ``type``, ``difficulty_level``, ``tags``, and where variants of a
  text differ in them, ``explanation`` (one per part) and ``correct`` (the options' correct_answer flags in
  (sub_id, option_id) order, when the setting decides the answer, as in backward_induction).

``Dataset.load`` expands this into the table the rest of the code reads, one row per variant and part with the
variant's id as question_id plus a ``text_id`` column, and ``Dataset.save`` collapses it again. Rows without a
text_id (added in developer mode) join an existing text when they are the same text in another setting, so a text
is never stored twice.
"""
from __future__ import annotations

import pandas as pd

SETTING_KEYS = ['type', 'difficulty_level', 'tags']
TABLES = {'questions': ('options', 'settings'), 'example_questions': ('example_options', 'example_settings')}
_OPTION_COLS = ['question_id', 'sub_id', 'option_id', 'option_text', 'correct_answer']


def _norm(text) -> str:
    return ' '.join(str(text).split())


def _parts(q: pd.DataFrame) -> dict:
    """question_id -> its rows sorted by sub_id."""
    return {qid: g.sort_values('sub_id', kind='stable') for qid, g in q.groupby('question_id', sort=False)}


def _option_rows(o: pd.DataFrame) -> dict:
    if o is None or o.empty:
        return {}
    return {qid: g.sort_values(['sub_id', 'option_id'], kind='stable') for qid, g in o.groupby('question_id', sort=False)}


def text_key(rows: pd.DataFrame, opts: pd.DataFrame | None, with_options: bool = True) -> tuple:
    """What two variants of one text share: every part's text (spaces collapsed) and domain, any other text column
    except explanation, and (unless with_options is False) the options' texts."""
    other = [c for c in rows.columns if c not in SETTING_KEYS + ['question_id', 'sub_id', 'question_text', 'explanation',
                                                                   'text_id']]
    key = (tuple(rows['sub_id'].astype(int)), tuple(_norm(t) for t in rows['question_text']),
           tuple(tuple(rows[c].astype(str)) for c in sorted(other)))
    if with_options:
        key += (() if opts is None or opts.empty else
                tuple(zip(opts['sub_id'].astype(int), opts['option_id'].astype(int), opts['option_text'].astype(str))),)
    return key


def assign_text_ids(q: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """q with every row's text_id set: rows that have one keep it; a question without one joins the text it repeats
    (same text_key) or starts its own text (text_id = its question_id)."""
    q = q.copy()
    if 'text_id' not in q.columns:
        q['text_id'] = pd.NA
    missing = q['text_id'].isna()
    if not missing.any():
        q['text_id'] = q['text_id'].astype(int)
        return q
    parts, opts = _parts(q), _option_rows(o)
    known = {}
    for qid, rows in parts.items():
        if rows['text_id'].notna().all():
            known.setdefault(text_key(rows, opts.get(qid)), int(rows['text_id'].iloc[0]))
    new = {}
    for qid in sorted(int(x) for x in q.loc[missing, 'question_id'].unique()):
        k = text_key(parts[qid], opts.get(qid))
        new[qid] = known.setdefault(k, qid)
    q.loc[missing, 'text_id'] = q.loc[missing, 'question_id'].astype(int).map(new)
    q['text_id'] = q['text_id'].astype(int)
    return q


def collapse(q: pd.DataFrame, o: pd.DataFrame, pool: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(texts, text options, settings) of a pool's expanded tables (see the module docstring)."""
    q = assign_text_ids(q, o)
    parts, opts = _parts(q), _option_rows(o)
    text_cols = [c for c in q.columns if c not in SETTING_KEYS + ['text_id']]
    texts, text_opts, settings = [], [], []
    for tid, members in q.groupby('text_id', sort=True):
        ids = sorted(int(x) for x in members['question_id'].unique())
        if tid not in ids:
            raise ValueError(f"text {tid}: its id is not one of its variants {ids[:5]}")
        head = parts[tid]
        k = text_key(head, opts.get(tid), with_options=False)
        head_opts = opts.get(tid)
        expl = {i: tuple(parts[i]['explanation'].astype(str)) for i in ids} if 'explanation' in q.columns else {}
        per_setting_expl = len(set(expl.values())) > 1
        for i in ids:
            rows = parts[i]
            if text_key(rows, None, with_options=False) != k:
                raise ValueError(f"text {tid}: variant {i} has a different text, domain or part structure")
            if rows[SETTING_KEYS].astype(str).drop_duplicates().shape[0] != 1:
                raise ValueError(f"variant {i}: parts with different settings")
            vo = opts.get(i)
            same_texts = (vo is None and head_opts is None) or (
                vo is not None and head_opts is not None and len(vo) == len(head_opts)
                and (vo[['sub_id', 'option_id', 'option_text']].astype(str).to_numpy()
                     == head_opts[['sub_id', 'option_id', 'option_text']].astype(str).to_numpy()).all())
            if not same_texts:
                raise ValueError(f"text {tid}: variant {i} has other options")
            correct = None
            if vo is not None and list(vo['correct_answer'].astype(bool)) != list(head_opts['correct_answer'].astype(bool)):
                correct = [bool(x) for x in vo['correct_answer']]
            r0 = rows.iloc[0]
            settings.append({'question_id': tid, 'variant_id': i, 'type': str(r0['type']),
                             'difficulty_level': int(r0['difficulty_level']), 'tags': str(r0['tags']),
                             'explanation': list(rows['explanation']) if per_setting_expl else None,
                             'correct': correct})
        t = head[text_cols].copy()
        t['question_id'] = tid
        if per_setting_expl:
            t['explanation'] = None
        texts.append(t)
        if head_opts is not None:
            ho = head_opts[_OPTION_COLS].copy()
            ho['question_id'] = tid
            text_opts.append(ho)
    texts_df = pd.concat(texts, ignore_index=True) if texts else q[text_cols].iloc[0:0]
    opts_df = pd.concat(text_opts, ignore_index=True) if text_opts else (o.iloc[0:0] if o is not None else None)
    settings_df = pd.DataFrame(settings, columns=['question_id', 'variant_id'] + SETTING_KEYS + ['explanation', 'correct'])
    for c in ('explanation', 'correct'):
        if settings_df[c].isna().all():
            settings_df = settings_df.drop(columns=c)
    return texts_df, opts_df, settings_df


def expand(texts: pd.DataFrame, text_opts: pd.DataFrame, settings: pd.DataFrame, columns: list[str],
           pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The expanded (one row per variant and part) questions and options of a pool; `columns` orders the question
    columns as before (text_id last)."""
    s = settings.rename(columns={'question_id': 'text_id'})
    t = texts.rename(columns={'question_id': 'text_id'})
    has_expl = 'explanation' in s.columns
    q = s.drop(columns=[c for c in ('correct', 'explanation') if c in s.columns]).merge(t, on='text_id', how='left')
    q = q.rename(columns={'variant_id': 'question_id'})
    if has_expl:
        per = {int(r.variant_id): r.explanation for r in s.itertuples() if isinstance(r.explanation, (list, tuple))}
        if per:
            q['explanation'] = [per[int(v)][int(sub)] if int(v) in per else e
                                for v, sub, e in zip(q['question_id'], q['sub_id'], q['explanation'])]
    q = q.sort_values(['question_id', 'sub_id'], kind='stable').reset_index(drop=True)
    first = [c for c in columns if c in q.columns]
    q = q[first + [c for c in q.columns if c not in first and c != 'text_id'] + ['text_id']]
    q = q.astype({'question_id': int, 'sub_id': int, 'difficulty_level': int, 'text_id': int})
    if text_opts is None:
        return q, text_opts
    o = s[['text_id', 'variant_id'] + (['correct'] if 'correct' in s.columns else [])].merge(
        text_opts.rename(columns={'question_id': 'text_id'}), on='text_id', how='inner')
    o = o.sort_values(['variant_id', 'sub_id', 'option_id'], kind='stable').reset_index(drop=True)
    if 'correct' in o.columns:
        flags = []
        for vid, g in o.groupby('variant_id', sort=False):
            c = g['correct'].iloc[0]
            flags += list(c) if isinstance(c, (list, tuple)) else list(g['correct_answer'])
        o['correct_answer'] = flags
        o = o.drop(columns='correct')
    o = o.drop(columns='text_id').rename(columns={'variant_id': 'question_id'})
    o = o[_OPTION_COLS + [c for c in o.columns if c not in _OPTION_COLS]]
    return q, o.astype({'question_id': int, 'sub_id': int, 'option_id': int, 'correct_answer': bool})
