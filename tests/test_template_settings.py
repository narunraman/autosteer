"""Template texts and their settings (autosteer.template_settings): the stored format round-trips, rows added in
developer mode join the text they repeat, every template folder stores each text once, and the sampler draws a text
before one of its settings."""
import collections
import tempfile
from pathlib import Path

import pandas as pd

from autosteer import template_settings as ts
from autosteer.datasets import Dataset, reindex
from autosteer.paths import templates_dir


def _toy():
    q = pd.DataFrame({
        'question_id': [0, 1, 2, 3], 'sub_id': [0, 0, 0, 0],
        'question_text': ['Add {a} and {b}.', 'Add {a} and {b}.', 'Sell {a} of {b}.', 'Add  {a} and {b}.'],
        'domain': ['numbers'] * 4, 'type': ['integer', 'large', 'integer', 'decimal'],
        'difficulty_level': [0, 2, 0, 1], 'tags': ['add', 'add', 'sub', 'add']})
    o = pd.DataFrame({'question_id': [0, 0, 1, 1, 2, 2, 3, 3], 'sub_id': [0] * 8, 'option_id': [0, 1] * 4,
                      'option_text': ['x', 'y'] * 4, 'correct_answer': [True, False, False, True, True, False,
                                                                        True, False]})
    return q, o


def test_new_rows_join_the_text_they_repeat_and_round_trip():
    q, o = _toy()
    ds = Dataset('toy', questions=q.assign(text_id=[0, 0, 2, pd.NA]), options=o)
    with tempfile.TemporaryDirectory() as tmp:
        ds.save(tmp)
        settings = pd.read_parquet(Path(tmp) / 'settings.parquet')
        texts = pd.read_parquet(Path(tmp) / 'questions.parquet')
        back = Dataset.load(tmp)
    assert len(texts) == 2 and sorted(settings['variant_id']) == [0, 1, 2, 3]
    assert back.questions['text_id'].tolist() == [0, 0, 2, 0]           # variant 3 joined text 0 (spaces collapsed)
    assert back.questions['type'].tolist() == q['type'].tolist()
    # variant 1's answer key differs from the text's: stored with its setting and restored
    assert back.options['correct_answer'].tolist() == o['correct_answer'].tolist()


def test_reindex_renumbers_text_ids_with_question_ids():
    q, o = _toy()
    q = q[q['question_id'] != 0].assign(text_id=[1, 2, 1])
    ds = reindex(Dataset('toy', questions=q, options=o[o['question_id'] != 0]))
    assert ds.questions['question_id'].tolist() == [0, 1, 2] and ds.questions['text_id'].tolist() == [0, 1, 0]


def test_template_folders_store_each_text_once():
    """No two texts of a folder's pool share their text, domain and options (texts repeated with another domain
    are a known labelling difference, left separate)."""
    for folder in sorted(Path(templates_dir()).iterdir()):
        if not (folder / 'dataset.json').exists():
            continue
        ds = Dataset.load(folder)
        for pool, opt in (('questions', 'options'), ('example_questions', 'example_options')):
            q = getattr(ds, pool)
            if q is None or q.empty:
                continue
            assert 'text_id' in q.columns, (folder.name, pool)
            parts, opts = ts._parts(q), ts._option_rows(getattr(ds, opt))
            seen = collections.defaultdict(set)
            for qid, rows in parts.items():
                seen[ts.text_key(rows, opts.get(qid), with_options=False)].add(int(rows['text_id'].iloc[0]))
            assert all(len(v) == 1 for v in seen.values()), (folder.name, pool)


def test_sampler_draws_a_text_then_a_setting():
    from autosteer.sampler import ElementSampler
    s = ElementSampler(seed=0)
    q = s._load_template('add_sub').questions
    n_texts = q['text_id'].nunique()
    settings = q.drop_duplicates('question_id').groupby('text_id').size()
    got = collections.Counter()
    for seed in range(3000):
        got[s.sample_question('add_sub', seed)['text_id']] += 1
    by_n = collections.defaultdict(list)
    for tid, n in got.items():
        by_n[int(settings[int(tid.rsplit('/', 1)[1])])].append(n)
    means = {k: sum(v) / len(v) for k, v in by_n.items()}
    # a text with 3 settings is drawn about as often as a text with 2 (before: 1.5x as often)
    assert max(means.values()) / min(means.values()) < 1.2, (means, n_texts)
