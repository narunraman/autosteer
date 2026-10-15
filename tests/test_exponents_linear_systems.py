"""exponents and linear_system_eq (STEER-ME Foundations), recomputed from the question text.

exponents: "base^exponent" is parsed and raised with Python's float power; exactly one option is within half a
hundredth of it, the marked one; options at least 5% apart; whole and fractional exponents occur.
linear_system_eq: every equation "ax + by (+ cz) = d" is parsed; numpy solves the system and the solution is
whole-numbered; the marked option is that solution and satisfies every equation exactly; every other option
violates at least one equation; two- and three-equation systems occur. Sampling is reproducible from the seed.
"""
import collections
import re

import numpy as np

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

POW = re.compile(r'(\d+(?:\.\d+)?)\^(\d+(?:\.\d+)?)')
EQ = re.compile(r'^(-?[\dxyz +-]+?) = (-?\d+)$', re.M)
TERM = re.compile(r'([+-])?\s*(\d*)([xyz])')
ASSIGN = re.compile(r'([xyz]) = (-?\d+)')


def test_exponents():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question('exponents', 111000 + seed), 'test')
        base, e = (float(x) for x in POW.search(row['question_text']).groups())
        want = base ** e
        vals = [float(o) for o in row['options']]
        assert [i for i, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-9 * want] == [row['correct_index']], \
            (row['question_text'], row['options'], want)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.05 * max(vals[i], vals[j]) - 0.011
        seen['integer' if e == int(e) else 'real'] += 1
        assert seen and row['type'] == ('integer' if e == int(e) else 'real_value')
    assert seen['integer'] and seen['real']


def parse_system(text):
    rows, rhs = [], []
    for lhs, d in EQ.findall(text):
        coef = {'x': 0, 'y': 0, 'z': 0}
        for sign, mag, var in TERM.findall(lhs):
            coef[var] = (-1 if sign == '-' else 1) * (int(mag) if mag else 1)
        rows.append(coef)
        rhs.append(int(d))
    names = [v for v in 'xyz' if any(r[v] for r in rows)]
    return np.array([[r[v] for v in names] for r in rows], dtype=float), np.array(rhs, dtype=float), names


def test_linear_systems():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question('linear_system_eq', 112000 + seed), 'test')
        M, b, names = parse_system(row['question_text'])
        n = 2 if row['type'] == 'two_equation' else 3
        assert M.shape == (n, n) and len(names) == n, row['question_text']
        sol = np.linalg.solve(M, b)
        assert np.allclose(sol, np.round(sol), atol=1e-9)
        for i, o in enumerate(row['options']):
            v = dict(ASSIGN.findall(o))
            assert sorted(v) == names
            x = np.array([int(v[k]) for k in names], dtype=float)
            ok = np.allclose(M @ x, b)
            assert ok == (i == row['correct_index']), (row['question_text'], row['options'])
            if ok:
                assert np.allclose(x, sol)
        seen[n] += 1
    assert seen[2] and seen[3]


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for e in ('exponents', 'linear_system_eq'):
        for seed in (0, 7, 123456789):
            q = a.sample_question(e, seed)
            assert b.sample_question(e, seed) == q
            assert a.sample_question(e, seed, template_id=q['template_id']) == q
