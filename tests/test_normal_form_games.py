"""Recovered normal-form elements: interpret_games, dominated_strategies, best_response, iterated_removal,
subgame_nash, bayes_nash (src/filler_tools/normal_form_tools.py, multi_agent_tools.py; templates from
STEER-generation's audit/steer/regen/build_normal_form_templates.py).

For each element, 1,000 seeded questions (env STEER_TEST_N):
- brute force from the table printed in the question (parsed here, no filler code): the cell value, the strictly
  dominant action, the unique best response, the unique strictly dominated action before / after removal, the
  unique Nash equilibrium of the stage game (pure; no mixed one, checked by support enumeration for 2x2 and by
  iterated dominance for 3x3) and of the expected Bayesian game; exactly the keyed option matches;
- the independent solver (tests/solvers/steer/solvers/matrix_games.py) agrees on every question;
- distinct options, no placeholders, the released metadata mix (sizes, grades, types/domains), reproducibility.
"""
import collections
import importlib.util
import itertools
import os
import re

import numpy as np
import pandas as pd
import pytest

from helpers import get_sampler
from autosteer.sampler import SamplerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get('STEER_TEST_N', 1000))
ELEMENTS = ['interpret_games', 'dominated_strategies', 'best_response', 'iterated_removal', 'subgame_nash',
            'bayes_nash']
_spec = importlib.util.spec_from_file_location('audit_mg', os.path.join(ROOT, 'tests/solvers/steer/solvers/matrix_games.py'))
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)
_CACHE = {}


def items(element):
    if element not in _CACHE:
        s = get_sampler()
        s.seed(20261002)
        out = []
        for _ in range(N):
            r = s.sample(SamplerConfig(element_name=element))
            out.append((r.get('sub_questions', [r]), r['metadata']))
        _CACHE[element] = out
    return _CACHE[element]


# ----------------------------------------------------------------------------------------- table parsing
def tables(text):
    """[(row labels, col labels, P (n, m, k))] for every printed table, in order."""
    out, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        cols = re.findall(r'Action [A-Z]', lines[i])
        if len(cols) >= 2 and re.fullmatch(r'\s*(Action [A-Z]\s*)+', lines[i]):
            rows, cells, j = [], [], i + 1
            while j < len(lines) and re.match(r'\s*Action [A-Z]\s+[(\d-]', lines[j]):
                lab, rest = re.match(r'\s*(Action [A-Z])\s+(.*)', lines[j]).groups()
                pairs = re.findall(r'\((-?[\d.]+), (-?[\d.]+)\)', rest)
                cells.append([tuple(map(float, p)) for p in pairs] if pairs else [(float(x),) for x in rest.split()])
                rows.append(lab)
                j += 1
            out.append((rows, cols, np.array(cells)))
            i = j
        else:
            i += 1
    return out


def you(text):
    return re.search(r'You are the (row|column) player', text).group(1)


def other(p):
    return 'column' if p == 'row' else 'row'


def view(rows, cols, P, player):
    U = P[:, :, 0] if player == 'row' or P.shape[2] == 1 else P[:, :, 1]
    return (rows, U) if player == 'row' else (cols, U.T)


def dominant(U):
    return [a for a in range(len(U)) if all((U[a] > U[b]).all() for b in range(len(U)) if b != a)]


def dominated(U):
    return [a for a in range(len(U)) if any((U[b] > U[a]).all() for b in range(len(U)) if b != a)]


def nash_pure(A, B):
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] >= A[:, j].max() - 1e-9 and B[i, j] >= B[i, :].max() - 1e-9]


def has_mixed_2x2(A, B):
    """a fully mixed equilibrium of a 2x2 game exists iff both indifference equations have interior solutions."""
    da, db = A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1], B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1]
    if abs(da) < 1e-12 or abs(db) < 1e-12:
        return False
    q = (A[1, 1] - A[0, 1]) / da   # column player's prob of col 0 making row indifferent
    p = (B[1, 1] - B[1, 0]) / db
    return 0 < q < 1 and 0 < p < 1


def keyed(part):
    return [o['correct'] for o in part['options']].index(True)


def texts(part):
    return [o['text'] for o in part['options']]


# ----------------------------------------------------------------------------------------- per element
def check_interpret(parts, md):
    q = parts[0]['question']
    (rows, cols, P), = tables(q)
    me, them = re.search(r'if you play (Action [A-Z]) and your opponent plays (Action [A-Z])', q).groups()
    r_lab, c_lab = (me, them) if you(q) == 'row' else (them, me)
    owner = you(q) if "What is your payoff" in q else other(you(q))
    U = P[:, :, 0] if P.shape[2] == 1 else P[:, :, 0 if owner == 'row' else 1]
    v = U[rows.index(r_lab), cols.index(c_lab)]
    return [i for i, o in enumerate(texts(parts[0])) if abs(float(o) - v) < 1e-9]


def check_dominated(parts, md):
    q = parts[0]['question']
    (rows, cols, P), = tables(q)
    asks_opponent = bool(re.search(r"your opponent('s)? (play|never play|actions)", q))
    assert asks_opponent == md['type'].startswith('other')
    chooser = other(you(q)) if asks_opponent else you(q)
    labs, U = view(rows, cols, P, chooser)
    if re.search(r'dominated\?|never play\?', q):
        assert md['type'].endswith('_dominated')
        d = dominated(U)
    else:
        assert not md['type'].endswith('_dominated')
        d = dominant(U)
    assert len(d) == 1
    return [i for i, o in enumerate(texts(parts[0])) if o == labs[d[0]]]


def check_best_response(parts, md):
    q = parts[0]['question']
    (rows, cols, P), = tables(q)
    given = re.search(r'(Action [A-Z])\?', q).group(1)
    resp = you(q) if md['domain'] == 'self' else other(you(q))
    labs, U = view(rows, cols, P, resp)
    other_labs = cols if resp == 'row' else rows
    v = U[:, other_labs.index(given)]
    assert (v == v.max()).sum() == 1
    return [i for i, o in enumerate(texts(parts[0])) if o == labs[int(v.argmax())]]


def check_iterated(parts, md):
    q0 = parts[0]['question']
    (rows, cols, P), = tables(q0)
    first = you(q0) if md['domain'] == 'self' else other(you(q0))
    labs1, U1 = view(rows, cols, P, first)
    d1 = dominated(U1)
    assert len(d1) == 1
    keep = [x for x in range(3) if x != d1[0]]
    P2, rows2, cols2 = (P[keep], [rows[x] for x in keep], cols) if first == 'row' else (P[:, keep], rows, [cols[x] for x in keep])
    t1 = tables(parts[1]['question'])
    if t1:
        assert t1[0][0] == rows2 and t1[0][1] == cols2 and np.allclose(t1[0][2], P2)
    labs2, U2 = view(rows2, cols2, P2, other(first))
    d2 = dominated(U2)
    assert len(d2) == 1
    h0 = [i for i, o in enumerate(texts(parts[0])) if o == labs1[d1[0]]]
    h1 = [i for i, o in enumerate(texts(parts[1])) if o == labs2[d2[0]]]
    return h0 if h0 == [keyed(parts[0])] and h1 == [keyed(parts[1])] else [None]


def check_subgame(parts, md):
    q = parts[0]['question']
    (rows, cols, P), = tables(q)
    A, B = P[:, :, 0], P[:, :, 1]
    ne = nash_pure(A, B)
    assert len(ne) == 1
    if A.shape[0] == 2:
        assert not has_mixed_2x2(A, B)
    else:  # unique equilibrium: iterated strict dominance must leave one cell
        r, c = list(range(3)), list(range(3))
        for _ in range(6):
            r = [i for i in r if not any((A[k, c] > A[i, c]).all() for k in r if k != i)]
            c = [j for j in c if not any((B[r, k] > B[r, j]).all() for k in c if k != j)]
        assert (r, c) == ([ne[0][0]], [ne[0][1]])
    n = {'two': 2, 'three': 3, 'four': 4}[re.search(r'repeats for (\w+) rounds', q).group(1)]
    i, j = ne[0]
    if md['domain'] == 'both':   # your action first
        target = ([rows[i], cols[j]] if you(q) == 'row' else [cols[j], rows[i]]) * n
    else:
        p = you(q) if md['domain'] == 'self' else other(you(q))
        target = [rows[i] if p == 'row' else cols[j]] * n
    for o in texts(parts[0]):    # plain words, one clause per round
        assert re.fullmatch(r'Round 1: .+(; round \d: .+)*', o) and "'" not in o and '(' not in o, o
        assert o.count('ound ') == n, o
    return [k for k, o in enumerate(texts(parts[0])) if re.findall(r'Action [A-Z]', o) == target]


def typed_bne(q):
    """Brute-force pure BNE of the typed game printed in q: [(x, (s1, s2))], plus the parsed game."""
    p1 = int(re.findall(r'(\d+)%', q)[0]) / 100
    ts = tables(q)
    assert len(ts) == 2 and q.index('Type 1:') < q.index('Type 2:')
    U = [t[2][:, :, 0] for t in ts]
    V = [t[2][:, :, 1] for t in ts]
    prior = [p1, 1 - p1]
    eq = []
    for x in range(2):
        for s in itertools.product(range(2), repeat=2):
            mine = [sum(prior[t] * U[t][a, s[t]] for t in range(2)) for a in range(2)]
            br_me = mine[x] > mine[1 - x]
            br_opp = all(V[t][x, s[t]] > V[t][x, 1 - s[t]] for t in range(2))
            if br_me and br_opp:
                eq.append((x, s))
    return eq, U, V, ts


def single_ne(U, V):
    ne = [(i, j) for i in range(2) for j in range(2) if U[i, j] > U[1 - i, j] and V[i, j] > V[i, 1 - j]]
    return ne[0] if len(ne) == 1 else None


def typed_answer(parts, md):
    q = parts[0]['question']
    eq, U, V, ts = typed_bne(q)
    assert len(eq) == 1, eq
    x, s = eq[0]
    singles = [single_ne(U[t], V[t]) for t in range(2)]
    if md['type'] == 'typed_own':
        label = ts[0][0][x]
        matters = all(ne is None or ne[0] != x for ne in singles)
    else:
        t = int(re.search(r'Type (\d)', q.rsplit('\n', 1)[-1]).group(1)) - 1
        label = ts[0][1][s[t]]
        matters = singles[t] is None or singles[t][1] != s[t]
    return label, matters


def check_bayes(parts, md):
    q = parts[0]['question']
    if md['type'].startswith('typed_'):
        label, _ = typed_answer(parts, md)
        return [k for k, o in enumerate(texts(parts[0])) if o == label]
    probs = [float(p) for p in re.findall(r'probability of state \d+ is (\d+(?:\.\d+)?)', q, re.I)]
    ts = tables(q)
    assert len(ts) == len(probs) == int(md['type'][0]) and abs(sum(probs) - 1) < 1e-9
    E = sum(p * t[2] for p, t in zip(probs, ts))
    ne = nash_pure(E[:, :, 0], E[:, :, 1])
    assert len(ne) == 1
    rows, cols = ts[0][0], ts[0][1]
    return [k for k, o in enumerate(texts(parts[0])) if re.findall(r'Action [A-Z]', o) == [rows[ne[0][0]], cols[ne[0][1]]]]


CHECKS = {'interpret_games': check_interpret, 'dominated_strategies': check_dominated,
          'best_response': check_best_response, 'iterated_removal': check_iterated, 'subgame_nash': check_subgame,
          'bayes_nash': check_bayes}
MIX = {  # (field, released values)
    'interpret_games': ('difficulty_level', {0, 1}), 'dominated_strategies': ('difficulty_level', {0, 1}),
    'best_response': ('difficulty_level', {7, 8}), 'iterated_removal': ('difficulty_level', {5, 6, 7}),
    'subgame_nash': ('difficulty_level', {10, 11, 12, 13}), 'bayes_nash': ('difficulty_level', {12, 13}),
}


def test_bayes_nash_prior_matters():
    by_type = collections.defaultdict(list)
    for parts, md in items('bayes_nash'):
        by_type[md['type']].append(typed_answer(parts, md)[1])
    assert set(by_type) == {'typed_own', 'typed_opponent'}, by_type.keys()
    for t, v in by_type.items():
        assert sum(v) >= 0.5 * len(v), (t, sum(v), len(v))


def test_dominated_strategies_both_kinds():
    kinds = collections.Counter(md['type'].endswith('_dominated') for _, md in items('dominated_strategies'))
    assert 0.4 * N < kinds[True] < 0.6 * N, kinds


@pytest.mark.slow
@pytest.mark.parametrize('element', ELEMENTS)
def test_brute_force_key(element):
    for parts, md in items(element):
        assert CHECKS[element](parts, md) == [keyed(parts[0])], (parts, md)


@pytest.mark.parametrize('element', ELEMENTS)
def test_audit_solver_agrees(element):
    rows = []
    for b, (parts, md) in enumerate(items(element)):
        if str(md['type']).endswith('_dominated') or str(md['type']).startswith('typed_'):
            continue  # the independent solvers cover only the released question kinds
        for k, p in enumerate(parts):
            rows.append(dict(base_id=str(b), part=k, question_text=p['question'], options=texts(p),
                             tags=str(md['tags']).split(';'), keyed=keyed(p)))
    if not rows:
        pytest.skip('no question of a kind the independent solver covers (e.g. bayes_nash typed games)')
    df = pd.DataFrame(rows)
    res = mg.SOLVERS[element](df)
    bad = [(st, pr, k) for st, pr, k in zip(res.status, res.pred, df.keyed) if st != 'ok' or pr != k]
    assert not bad, (len(bad), bad[:5], collections.Counter(res.status))


@pytest.mark.parametrize('element', ELEMENTS)
def test_well_formed_and_mix(element):
    field, values = MIX[element]
    seen = collections.Counter()
    for parts, md in items(element):
        seen[md[field]] += 1
        for p in parts:
            t = texts(p)
            assert len(t) >= 2 and len(set(t)) == len(t) and sum(o['correct'] for o in p['options']) == 1
            assert not re.search(r'\{\w+\}', p['question'] + ' '.join(t)), p
    assert set(seen) == values, seen
    assert min(seen.values()) > 0.6 * N / len(values), seen


def test_reproducible():
    s = get_sampler()
    out = []
    for _ in range(2):
        s.seed(11)
        out.append([s.sample(SamplerConfig(element_name=e)) for e in ELEMENTS for _ in range(5)])
    assert out[0] == out[1]
