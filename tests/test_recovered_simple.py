"""Recovered v1-era elements with fixed sentences: plurality_voting, borda_count (voting_tools.py) and
time_inconsistency, transitivity_risk, independence_risk (axiom_tools.py).

For each element, 1,000 seeded questions (env STEER_TEST_N):
- an independent answer computed here from the question text (vote counts; Borda under three scoring conventions
  for truncated ballots, which must agree; the earlier intertemporal choice; transitivity; the sure-thing
  principle), and exactly the keyed option matches it;
- the audit solver for the element (audit/steer/solvers/) agrees on every question it scores and scores most;
- distinct options, no placeholders, the released metadata mix, key position not degenerate.
"""
import collections
import importlib.util
import os
import re

import pandas as pd
import pytest

from helpers import sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get('STEER_TEST_N', 1000))
ELEMENTS = ['plurality_voting', 'borda_count', 'time_inconsistency', 'transitivity_risk', 'independence_risk']
_CACHE = {}


def _load(name):
    spec = importlib.util.spec_from_file_location(f'audit_{name}', os.path.join(ROOT, f'tests/solvers/steer/solvers/{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def got(element):
    if element not in _CACHE:
        _CACHE[element] = sample(element, N, seed=20261003)
    return _CACHE[element]


BALLOT = re.compile(r'(\d+) voters? voted for ([A-Z](?: > [A-Z])*)')


def ballots(q):
    return [(int(n), b.split(' > ')) for n, b in BALLOT.findall(q)]


def cands_of(options):
    return [o.split()[-1] for o in options]


def borda(prof, cands, conv):
    m = len(cands)
    s = dict.fromkeys(cands, 0.0)
    for n, b in prof:
        for i, c in enumerate(b):
            s[c] += n * ((len(b) - 1 - i) if conv == 'ballot' else (m - 1 - i))
        un = [c for c in cands if c not in b]
        if conv == 'avg':
            for c in un:
                s[c] += n * (len(un) - 1) / 2
    return s


def expected(element, s):
    q, opts = s.question, s.options
    if element == 'plurality_voting':
        cands = cands_of(opts)
        cnt = collections.Counter()
        for n, b in ballots(q):
            cnt[b[0]] += n
        top = max(cnt.values())
        win = [c for c in cands if cnt[c] == top]
        assert len(win) == 1
        return f'Candidate {win[0]}'
    if element == 'borda_count':
        cands = cands_of(opts)
        place = 1 if re.search(r'second|runner-up', q) else 0
        ans = set()
        for conv in ('paper', 'avg', 'ballot'):
            sc = borda(ballots(q), cands, conv)
            order = sorted(cands, key=lambda c: -sc[c])
            assert sum(v == sc[order[place]] for v in sc.values()) == 1
            ans.add(order[place])
        assert len(ans) == 1
        return f'Candidate {ans.pop()}'
    if element == 'time_inconsistency':
        chosen = int(re.search(r'(?:chose|chosen|choice) (?:to receive )?\$?(\d+)', q).group(1))
        small = min(opts, key=lambda o: int(o.split()[0]))
        large = max(opts, key=lambda o: int(o.split()[0]))
        today = chosen == int(small.split()[0])
        assert today or chosen == int(large.split()[0])
        return small if today else large
    if element == 'transitivity_risk':
        x = re.search(r'Preference X: A (\d+)% chance of (.+?) is preferred over a (\d+)% chance of (.+?)\.\n'
                      r'Preference Y: A (\d+)% chance of (.+?) is preferred over a (\d+)% chance of (.+?)\.\n', q)
        assert x and x.group(3) == x.group(5) and x.group(4) == x.group(6)
        return f'A {x.group(1)}% chance of {x.group(2)} over a {x.group(7)}% chance of {x.group(8)}'
    if element == 'independence_risk':
        # the option whose description mentions the preferred item first stated in the question
        return None


@pytest.mark.parametrize('element', ELEMENTS[:4])
def test_independent_answer(element):
    for s in got(element):
        assert s.options.index(expected(element, s)) == s.answer_index, (s.question, s.options, s.answer_index)


def test_independence_risk_key():
    # independence: the keyed option mixes in the preferred item (checked by the audit solver below and, per template,
    # by test_conventions_g2). Since the review (October 2026) the keyed lottery is described first in
    # about half the questions and second in the others (it was always first), and the letters are swapped in about
    # half the questions.
    letters, key_first = collections.Counter(), 0
    for s in got('independence_risk'):
        first_named = re.search(r'Option ([XY])', s.question).group(1)
        key_first += s.answer == f'Option {first_named}'
        letters[s.answer] += 1
    assert min(letters.values()) > 0.4 * N, letters
    assert 0.3 * N < key_first < 0.7 * N, key_first


SOLVER = {'plurality_voting': ('voting', 'solve_plurality'), 'borda_count': ('voting', 'solve_borda'),
          'time_inconsistency': ('axioms', 'solve_time_inconsistency'),
          'transitivity_risk': ('axioms', 'solve_transitivity_risk'),
          'independence_risk': ('expected_utility', 'solve_independence_risk')}


@pytest.mark.parametrize('element', ELEMENTS)
def test_audit_solver(element):
    mod, fn = SOLVER[element]
    f = getattr(_load(mod), fn)
    df = pd.DataFrame({'question_text': [s.question for s in got(element)], 'options': [s.options for s in got(element)],
                       'tags': [str(s.metadata['tags']).split(';') for s in got(element)]})
    res = f(df)
    ok = [(st, p, s.answer_index) for st, p, s in zip(res.status, res.pred, got(element)) if st == 'ok']
    wrong = [x for x in ok if x[1] != x[2]]
    assert not wrong and len(ok) > 0.9 * N, (len(ok), wrong[:3], collections.Counter(res.status))


@pytest.mark.parametrize('element', ELEMENTS)
def test_well_formed(element):
    pos = collections.Counter()
    for s in got(element):
        assert sum(s.correct) == 1 and len(set(s.options)) == len(s.options) >= 2
        assert not re.search(r'\{\w+\}', s.question + ' '.join(s.options)), s.question
        pos[(len(s.options), s.answer_index)] += 1
    k, n = collections.Counter(len(s.options) for s in got(element)).most_common(1)[0]
    shares = [pos[(k, i)] / n for i in range(k)]
    assert min(shares) > 0.5 / k, shares


RELEASED = {  # (field, released values) from the staged STEER data
    'plurality_voting': ('difficulty_level', {1, 2}), 'borda_count': ('difficulty_level', {5, 6}),
    'time_inconsistency': ('type', {'today', 'tomorrow'}), 'transitivity_risk': ('domain', {'food', 'gadgets', 'investments', 'vacations'}),
    'independence_risk': ('domain', {'vacation', 'book', 'meal', 'gadget', 'car', 'investment'}),
}


@pytest.mark.parametrize('element', ELEMENTS)
def test_released_mix(element):
    field, values = RELEASED[element]
    seen = collections.Counter(s.metadata[field] for s in got(element))
    assert set(seen) == values, seen
    assert min(seen.values()) > 0.5 * N / len(values), seen


# ----------------------------------------------------------------------------------------- probability elements
def test_gamblers_fallacy():
    for s in sample('gamblers_fallacy', 300, seed=4):
        assert s.options == ['Yes', 'No'] or s.options == ['No', 'Yes']
        assert s.answer == 'No' and re.search(r'\b(three|six|seven|ten|twelve)\b', s.question), s.question


def test_bayes_rule_independent():
    pos = collections.Counter()
    for s in sample('bayes_rule', N, seed=6):
        PB, PBA, PA = (float(x) for x in re.findall(r'(\d+)%', s.question))
        ans = PBA * PA / PB
        assert PBA * PA / 100 <= min(PA, PB) and PB - PBA * PA / 100 <= 100 - PA
        vals = [float(o.rstrip('%')) for o in s.options]
        hits = [i for i, v in enumerate(vals) if abs(v - ans) <= 0.006]
        assert hits == [s.answer_index], (s.question, s.options)
        assert len(set(s.options)) == 3
        pos[s.answer_index] += 1
    assert min(pos.values()) > 0.25 * N, pos
    res = _load('arith_prob').solve_bayes_rule(pd.DataFrame({'question_text': [s.question for s in sample('bayes_rule', 300, seed=6)],
                                                             'options': [s.options for s in sample('bayes_rule', 300, seed=6)]}))
    assert (res.status == 'ok').mean() > 0.9 and all(p == s.answer_index for p, st, s in
                                                      zip(res.pred, res.status, sample('bayes_rule', 300, seed=6)) if st == 'ok')


def test_complement_rule():
    got_ = sample('complement_rule', N, seed=8)
    lv = collections.Counter()
    for s in got_:
        ps = [float(x) for x in re.findall(r'(?<![\d.])0?\.\d+(?![\d.])', s.question)]
        lv[len(ps)] += 1
        ans = 1 - sum(ps)
        assert [i for i, o in enumerate(s.options) if abs(float(o) - ans) < 0.006] == [s.answer_index], (s.question, s.options)
        assert len(set(s.options)) == 3 and all(0 < float(o) < 1 for o in s.options)
    assert set(lv) == {1, 2} and min(lv.values()) > 0.3 * N, lv
    df = pd.DataFrame({'question_text': [s.question for s in got_], 'options': [s.options for s in got_]})
    res = _load('arith_prob').solve_complement_rule(df)
    ok = [(p, s.answer_index) for p, st, s in zip(res.pred, res.status, got_) if st == 'ok']
    assert len(ok) > 0.8 * N and all(p == k for p, k in ok), (len(ok), collections.Counter(res.status))


@pytest.mark.parametrize('element', ELEMENTS + ['gamblers_fallacy', 'bayes_rule', 'complement_rule',
                                                'compute_expectations'])
def test_same_seed_same_output(element):
    a = sample(element, 40, seed=21)
    b = sample(element, 40, seed=21)
    assert [(s.question, s.options, s.correct) for s in a] == [(s.question, s.options, s.correct) for s in b]
    c = sample(element, 40, seed=22)
    assert [s.question for s in a] != [s.question for s in c]
