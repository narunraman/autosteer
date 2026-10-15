"""Rebuilt STEER elements condorcet_criterion, add_sub and mult_div (generators lost; see
audit/steer/regen/ and src/filler_tools/{voting,arithmetic}_tools.py).

For each element, N seeded samples (default 1,000, env STEER_TEST_N):
- the independent audit solver (audit/steer/solvers/voting.py, arith_prob.py) agrees on every question it can
  verify; for condorcet it must verify every question;
- add_sub / mult_div: the answer recomputed from the numbers shown and the hand label of the template is the
  unique option equal to the marked one (covers the story templates the audit solver cannot parse);
- exactly one correct option, distinct options, no unfilled placeholders, uniform correct position;
- reproducible with the same seed, also across processes with different PYTHONHASHSEED.
Plus: every template in the datasets has an operation label.
"""
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

from autosteer.paths import templates_dir
from autosteer.filler_tools.arithmetic_tools import base_template, keep_add_sub_template
from helpers import get_sampler, sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get('STEER_TEST_N', 1000))
SEED = 20260928
ELEMENTS = ['condorcet_criterion', 'add_sub', 'mult_div']
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
AS_OPS = {'a+b': lambda a, b: a + b, 'a-b': lambda a, b: a - b, 'b-a': lambda a, b: b - a,
          '-a-b': lambda a, b: -a - b}
MD_OPS = {'a*b': lambda a, b: a * b, 'a/b': lambda a, b: a / b, 'b/a': lambda a, b: b / a}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


voting = _load('audit_voting', 'tests/solvers/steer/solvers/voting.py')
arith = _load('audit_arith', 'tests/solvers/steer/solvers/arith_prob.py')


def labels(element):
    return pd.read_csv(os.path.join(templates_dir(), element, 'template_labels.csv'), keep_default_na=False)


def template_questions(element):
    return get_sampler()._load_template(element).questions


_CACHE = {}


def samples(element):
    if element not in _CACHE:
        _CACHE[element] = sample(element, N, seed=SEED)
    return _CACHE[element]


def solver_frame(element, got):
    return pd.DataFrame({
        'question_text': [s.question for s in got],
        'options': [list(s.options) for s in got],
        'tags': [[s.metadata['tags']] for s in got],
        'type': [s.metadata['type'] for s in got],
    })


# --------------------------------------------------------------------------------------------- generic checks
@pytest.mark.parametrize('element', ELEMENTS)
def test_well_formed(element):
    got = samples(element)
    assert len(got) == N
    for s in got:
        assert sum(s.correct) == 1, (s.question, s.options, s.correct)
        assert len(set(s.options)) == len(s.options), s.options
        blob = s.question + ' || ' + ' | '.join(s.options)
        assert not PLACEHOLDER.search(blob), blob
        assert not re.search(r'\b(None|nan|inf)\b', blob), blob


@pytest.mark.parametrize('element', ELEMENTS)
def test_uniform_position(element):
    got = samples(element)
    by_k = collections.defaultdict(collections.Counter)
    for s in got:
        by_k[len(s.options)][s.answer_index] += 1
    for k, pos in by_k.items():
        total = sum(pos.values())
        if total < 100:
            continue
        shares = [pos[i] / total for i in range(k)]
        sd = ((1 / k) * (1 - 1 / k) / total) ** 0.5
        assert all(abs(x - 1 / k) <= 4 * sd for x in shares), (k, shares)


@pytest.mark.parametrize('element', ELEMENTS)
def test_same_seed_same_output(element):
    a = sample(element, 50, seed=7)
    b = sample(element, 50, seed=7)
    assert [(s.question, s.options, s.correct) for s in a] == [(s.question, s.options, s.correct) for s in b]
    c = sample(element, 50, seed=8)
    assert [s.question for s in a] != [s.question for s in c]


def test_reproducible_across_processes():
    def run(hashseed):
        env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
        out = subprocess.run([sys.executable, os.path.join(HERE, '_repro_dump.py'), ROOT, '30', *ELEMENTS],
                             env=env, capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    a, b = run(1), run(2)
    assert a == b


# --------------------------------------------------------------------------------------------- condorcet
def test_condorcet_solver_agrees_on_every_question():
    got = samples('condorcet_criterion')
    res = voting.solve_condorcet(solver_frame('condorcet_criterion', got))
    assert (res.status == 'ok').all(), res.status.value_counts().to_dict()
    assert [int(p) for p in res.pred] == [s.answer_index for s in got]


def test_condorcet_outcome_mix_and_grades():
    got = samples('condorcet_criterion')
    no_winner = sum(s.answer == 'There is no Condorcet winner' for s in got) / len(got)
    assert 0.2 <= no_winner <= 0.3, no_winner
    for s in got:
        n_cand = len(s.options) - 1
        assert s.options.count('There is no Condorcet winner') == 1
        assert (s.metadata['difficulty_level'] == 2) == (n_cand == 3), (s.metadata, s.options)
        assert n_cand in (3, 4, 5)
        assert len(re.findall(r'voters? voted for', s.question)) == int(s.metadata['type'])
    # the Condorcet winner is not always the plurality leader (plurality does not solve the element)
    ballots = [voting.parse_ballots(s.question) for s in got]
    differs = 0
    for s, b in zip(got, ballots):
        if s.answer == 'There is no Condorcet winner':
            continue
        first = collections.Counter()
        for n, r in b:
            first[r[0]] += n
        if not s.answer.endswith(' ' + first.most_common(1)[0][0]):
            differs += 1
    assert differs > 0.2 * N


# --------------------------------------------------------------------------------------------- arithmetic
def test_every_template_has_an_operation_label():
    for element, ops in (('add_sub', AS_OPS), ('mult_div', MD_OPS)):
        lab = labels(element)
        keep = lab[lab.status == 'keep']
        assert keep.op.isin(list(ops)).all(), keep[~keep.op.isin(list(ops))]
        assert (lab[lab.status == 'drop'].drop_reason.str.len() > 0).all()
        texts = {base_template(element, t) for t in template_questions(element).question_text}
        # add_sub: stories whose negative amounts conflict with the question are dropped (review, Oct 2026)
        expected = {t for t in keep.template if element != 'add_sub' or keep_add_sub_template(t)}
        assert texts == expected


def _value(s: str) -> float:
    return float(s.replace('$', '').replace(',', ''))


def _template_of(question: str, params: dict) -> str:
    t = question
    for k in ('a', 'b'):
        t = t.replace(str(params[k]), '{' + k + '}', 1)
    return t


@pytest.mark.parametrize('element', ['add_sub', 'mult_div'])
def test_answer_matches_label(element):
    ops = AS_OPS if element == 'add_sub' else MD_OPS
    lab = labels(element).set_index('template')
    for s in samples(element):
        t = base_template(element, _template_of(s.question, s.params))
        assert t in lab.index, s.question
        op = lab.loc[t, 'op']
        x = round(ops[op](_value(str(s.params['a'])), _value(str(s.params['b']))), 2)
        hits = [i for i, o in enumerate(s.options) if abs(_value(o) - x) <= 0.0051]
        assert hits == [s.answer_index], (s.question, op, x, s.options, s.correct)


@pytest.mark.parametrize('element', ['add_sub', 'mult_div'])
def test_audit_solver_agrees(element):
    got = samples(element)
    solve = arith.solve_add_sub if element == 'add_sub' else arith.solve_mult_div
    res = solve(solver_frame(element, got))
    status = res.status.str.split(':').str[0]
    assert status.isin(['ok', 'unverified']).all(), res.status.value_counts().to_dict()
    ok = status == 'ok'
    assert ok.mean() > 0.3, ok.mean()  # explicit arithmetic and the solver's validated story families
    assert [int(p) for p in res.pred[ok]] == [s.answer_index for s, k in zip(got, ok) if k]


def test_all_types_and_grades_sampled():
    expect = {'add_sub': {('integer', 0), ('decimal', 1), ('large', 2)},
              'mult_div': {(t, d) for t in ('positive', 'negative', 'both') for d in (0, 1, 2)},
              'condorcet_criterion': {(t, d) for t in ('3', '4', '5') for d in (2, 3)}}
    for element, want in expect.items():
        got = {(s.metadata['type'], s.metadata['difficulty_level']) for s in samples(element)}
        assert got == want, (element, got ^ want)
        rows = template_questions(element)
        assert set(zip(rows.type, rows.difficulty_level)) == want  # no template dropped as unfillable


def test_arithmetic_signs_and_precision():
    for s in samples('mult_div'):
        a, b = _value(str(s.params['a'])), _value(str(s.params['b']))
        ans = _value(s.answer)
        assert (ans > 0) == ((a > 0) == (b > 0)), (s.question, s.answer)  # negative x negative is positive
        if s.metadata['type'] != 'positive':
            assert (a < 0) or (b < 0)
    for element in ('add_sub', 'mult_div'):
        for s in samples(element):
            for o in s.options:  # every option at most 2 decimals, answer included
                assert re.fullmatch(r'-?\d+(\.\d{1,2})?', o), (s.question, s.options)


def test_add_sub_digit_structure():
    for s in samples('add_sub'):
        a, b = abs(_value(str(s.params['a']))), abs(_value(str(s.params['b'])))
        ty = s.metadata['type']
        assert s.metadata['difficulty_level'] == {'integer': 0, 'decimal': 1, 'large': 2}[ty]
        if ty == 'integer':
            assert a == int(a) and b == int(b) and a < 100 and b < 100
        elif ty == 'decimal':
            assert 1 <= a < 100 and 1 <= b < 100 and (a != int(a) or b != int(b))
        else:
            assert 1000 <= a < 10000 and 1000 <= b < 10000
