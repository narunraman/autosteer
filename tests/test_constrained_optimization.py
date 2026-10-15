"""constrained_optimization (STEER 1.2.c): the optimal value recomputed with scipy from the question text.

Linear problems are word problems (October 2026): the per-unit profits or costs, the per-unit uses or contents and
the amounts available or required are read from the story's sentences, and the LP is built from them (max: the two
resources as "<=" constraints; min: the two requirements as ">=" constraints; x, y >= 0). Every number in the story
must be used. Quadratic problems are stated bare; from the released rows alone: the objective (linear "7x + 9y" or quadratic "-2x^2 - y^2 + 6x + 12y") and the
constraint list ("8x + 5y <= 59, 9x + 9y <= 14, x >= 0 and y >= 0") are parsed; the final question says maximum or
minimum. Linear problems are solved with linprog (objective negated for a maximum), quadratic ones with trust-constr from
several starting points. Checks: the question asks for the maximum exactly for type max; exactly one option is within
rounding of the optimal value and it is the marked one; options are at least 5% apart; linear and quadratic problems
of both types occur; sampling is reproducible from the seed. Also the fixed class: linprog's solution of a maximisation
is the best vertex (the old class returned the minimum).
"""
import collections
import re
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

TERM = re.compile(r'([+-])?\s*(\d*)(x\^2|y\^2|x|y)(?![\^\w])')
CONS = re.compile(r'(-?\d*x(?: [+-] \d*y)?|-?\d*y) (<=|>=) (\d+)')
ASK = re.compile(r'What is the (maximum|minimum)\b[^?]*\?\s*$')


def coeffs(expr):
    c = collections.defaultdict(float)
    for sign, mag, var in TERM.findall(expr):
        c[var] += (-1 if sign == '-' else 1) * (int(mag) if mag else 1)
    return c


def parse(text):
    cons_part = re.search(r'((?:(?:\d*x(?: [+-] \d*y)?|\d*y) (?:<=|>=) \d+, )+x >= 0 and y >= 0)', text).group(1)
    obj_part = text[:text.index(cons_part)]
    obj = re.findall(r'(-?\d*x(?:\^2)?(?: [+-] \d*[xy](?:\^2)?)+)', obj_part)[-1]
    cons = [(coeffs(lhs), op, int(rhs)) for lhs, op, rhs in CONS.findall(cons_part)]
    cons = [c for c in cons if not (c[2] == 0 and c[1] == '>=')]
    return coeffs(obj), cons


def solve(obj, cons, maximize):
    sign = -1 if maximize else 1
    quad = obj['x^2'] or obj['y^2']
    A, b = [], []
    for c, op, rhs in cons:
        row = [c['x'], c['y']]
        if op == '>=':
            row, rhs = [-v for v in row], -rhs
        A.append(row)
        b.append(rhs)
    if not quad:
        res = linprog(sign * np.array([obj['x'], obj['y']]), A_ub=A, b_ub=b, bounds=[(0, None)] * 2, method='highs')
        assert res.success
        return sign * res.fun

    def f(v):
        x, y = v
        return obj['x^2'] * x * x + obj['y^2'] * y * y + obj['x'] * x + obj['y'] * y
    lc = LinearConstraint(np.array(A, dtype=float), -np.inf, np.array(b, dtype=float))
    best = None
    for x0 in ([1.0, 1.0], [10.0, 10.0], [30.0, 1.0], [1.0, 30.0]):
        res = minimize(lambda v: sign * f(v), x0, method='trust-constr', bounds=Bounds([0, 0], [np.inf, np.inf]),
                       constraints=[lc], options={'gtol': 1e-12, 'xtol': 1e-12, 'maxiter': 5000})
        viol = max(0.0, float(np.max(np.array(A) @ res.x - np.array(b))), float(-res.x.min()))
        if viol < 1e-7 and (best is None or sign * f(res.x) < sign * best):
            best = f(res.x)
    return best


MAX_USE = re.compile(r'brings a profit of \$(\d+) and uses (\d+) [\w-]+ of [^;.]+? and (\d+) [\w-]+ of ')
MAX_AVAIL = re.compile(r'(\d+) [\w-]+ of [^.]+? and (\d+) [\w-]+ of [^.]+? are available\.')
MIN_USE = re.compile(r'costs \$(\d+) and (?:provides|yields|prints|produces) (\d+) [^;.]+? and (\d+) ')
MIN_NEED = re.compile(r'at least (\d+) [^.]+? and at least (\d+) ')


def parse_story(text, maximize):
    use = (MAX_USE if maximize else MIN_USE).findall(text)
    need = (MAX_AVAIL if maximize else MIN_NEED).findall(text)
    assert len(use) == 2 and len(need) == 1, text
    (c1, a11, a21), (c2, a12, a22) = [tuple(int(v) for v in u) for u in use]
    b1, b2 = (int(v) for v in need[0])
    # every number of the story is one of these nine (the problem is fully determined by the story)
    nums = sorted(int(v) for v in re.findall(r'(?<![\w.])\d+(?![\w.])', text.replace('100 ml', '')))
    assert nums == sorted([c1, c2, a11, a12, a21, a22, b1, b2]), (text, nums)
    obj = collections.defaultdict(float, {'x': c1, 'y': c2})
    op = '<=' if maximize else '>='
    cons = [(collections.defaultdict(float, {'x': a11, 'y': a12}), op, b1),
            (collections.defaultdict(float, {'x': a21, 'y': a22}), op, b2)]
    return obj, cons


def test_optimal_values():
    warnings.filterwarnings('ignore', message='delta_grad == 0.0')
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(300):
        (row,) = build_rows(s.sample_question('constrained_optimization', 131000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        maximize = re.search(r'(maximum|minimum)', text).group(1) == 'maximum'
        assert maximize == (row['type'] == 'max'), text
        if 'linear' in row['tags'] or row['difficulty'] == 4:
            obj, cons = parse_story(text, maximize)
            assert '$' in text and all(o.startswith('$') for o in opts)
        else:
            assert ASK.search(text) and '$' not in text
            obj, cons = parse(text)
        want = solve(obj, cons, maximize)
        vals = [float(o.lstrip('$')) for o in opts]
        hits = [i for i, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-6 * abs(want)]
        assert hits == [k], (text, opts, want)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.05 * max(abs(vals[i]), abs(vals[j])) - 0.011, opts
        seen[(row['type'], 'quadratic' if obj['x^2'] else 'linear')] += 1
        seen[row['domain']] += 1
    assert {(t, f) for t in ('max', 'min') for f in ('linear', 'quadratic')} <= set(seen), seen
    assert len([k for k in seen if isinstance(k, str)]) >= 8, seen


def test_fixed_class_maximises():
    from autosteer.filler_tools.foundations_tools import ConstrainedOptimizationProblem
    rng = np.random.default_rng(3)
    for maximize in (True, False):
        for _ in range(50):
            p = ConstrainedOptimizationProblem(rng, maximize=maximize, problem_type='LP')
            x, v = p.find_optimal()
            vals = [float(p.value(vt)) for vt in p.vertices()]
            assert abs(v - (max(vals) if maximize else min(vals))) < 1e-7
            assert abs(v - float(p.v_opt)) < 1e-7


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question('constrained_optimization', seed)
        assert b.sample_question('constrained_optimization', seed) == q
        assert a.sample_question('constrained_optimization', seed, template_id=q['template_id']) == q
