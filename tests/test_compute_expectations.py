"""compute_expectations (recovered 2026-10; src/filler_tools/expectation_tools.py).

1,000 seeded questions: the expected value recomputed here from the question text with the family formula
(coin / die faces / N-sided die / spinner / bag draws / days), exactly one option equal to it and keyed; the independent
solver (tests/solvers/steer/solvers/arith_prob.py) agrees wherever it scores; distinct options; domains as released.
"""
import collections
import importlib.util
import os
import re

import pandas as pd

from helpers import sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get('STEER_TEST_N', 1000))
_C = []


def got():
    if not _C:
        _C.extend(sample('compute_expectations', N, seed=20261004))
    return _C


FAIR = (' Assume the coin is fair.', ' Assume a fair six-sided die with faces numbered 1 to 6.')


def ev(q):
    for note in FAIR:
        q = q.replace(note, '')
    n = [int(x) for x in re.findall(r'\d+', q.replace('six-sided', ''))]
    if re.search(r'\b(bag|basket)\b', q):
        return sum(n) / n[0]
    if '%' in q:
        assert n[0] + n[1] == 100
        return n[0] * n[2] / 100
    m = re.search(r'divided into (\d+) equal sections numbered (\d+) to (\d+)', q)
    if m:
        N_, a, b = map(int, m.groups())
        assert b - a + 1 == N_
        return (a + b) / 2
    m = re.search(r'fair (\d+)-sided die', q)
    if m:
        return (int(m.group(1)) + 1) / 2
    fm = re.search(r'\b\d(?:, \d)+\b', q)
    if fm:
        faces = fm.group(0).split(', ')
        assert len(set(faces)) == len(faces) and all(1 <= int(f) <= 6 for f in faces)
        k = len(faces)
        A, B = (int(x) for x in re.findall(r'\d+', q[:fm.start()] + q[fm.end():]))
        return (k * A - (6 - k) * B) / 6
    A, B = n
    return (A - B) / 2


def test_answer_recomputed():
    for s in got():
        v = ev(s.question)
        vals = [float(re.match(r'-?[\d.]+', o).group(0)) for o in s.options]
        assert [i for i, x in enumerate(vals) if abs(x - v) <= 0.006] == [s.answer_index], (s.question, s.options, v)
        assert len(set(s.options)) == 4


def test_audit_solver_and_mix():
    spec = importlib.util.spec_from_file_location('audit_ap', os.path.join(ROOT, 'tests/solvers/steer/solvers/arith_prob.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.solve_compute_expectations(pd.DataFrame({'question_text': [s.question for s in got()],
                                                       'options': [s.options for s in got()]}))
    ok = [(p, s.answer_index) for p, st, s in zip(res.pred, res.status, got()) if st == 'ok']
    assert ok and all(p == k for p, k in ok) and len(ok) > 0.8 * N, (len(ok), collections.Counter(res.status))
    doms = collections.Counter(s.metadata['domain'] for s in got())
    assert set(doms) == {'coin_dice', 'dice', 'bag', 'weather'} and min(doms.values()) > 0.15 * N, doms


def test_fair_and_in_range():
    """Coin and die questions state that the coin / die is fair (and the die's faces); every option lies in the range
    of the outcome, where any expectation lies (review, October 2026: '11.0 days' over 5 days, a spinner on 2
    to 4 with option 1.0)."""
    for s in got():
        q = s.question
        if re.search(r'\b(coin|die|dice)\b', q) and not re.search(r'\d+-sided die', q.replace('six-sided', '')):
            assert q.endswith(FAIR), q
        vals = [float(re.match(r'-?[\d.]+', o).group(0)) for o in s.options]
        assert all(len(o.split()[0].split('.')[1]) == 2 for o in s.options), s.options
        m = re.search(r'numbered (\d+) to (\d+)', q)
        if 'spinner' in q:
            assert all(int(m.group(1)) <= v <= int(m.group(2)) for v in vals), (q, s.options)
        m = re.search(r'over (\d+) days', q)
        if m:
            assert all(0 <= v <= int(m.group(1)) for v in vals), (q, s.options)
        m = re.search(r'fair (\d+)-sided die', q)
        if m:
            assert all(1 <= v <= int(m.group(1)) for v in vals), (q, s.options)
