"""factor_demand: duality of profit maximization and expenditure minimization, recomputed by numerical optimisation.

From the released rows alone: part 0 states Q = A * L^a * K^b, the output price, the wage and the rental rate; the
key (hours of labor, hours of capital, or total cost) is checked against the profit maximum found by scipy
(BFGS on -profit). Part 1 states the same technology and input prices and an output level; the key is checked
against the cost minimum of producing exactly that output found by scipy (bounded 1-D search along the isoquant).
Checks: a + b <= 0.85 (decreasing returns); exactly one option matches the recomputed value and it is the marked
one; the part-1 key is the part-0 key (same text); four options, pairwise at least 10% apart; the shown output is the
part-0 optimum's output; every type, domain and perspective occurs; sampling is reproducible from the seed.
"""
import collections
import math
import re

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'factor_demand'
PF = re.compile(r'Q = (\d+) \* L\^([\d.]+) \* K\^([\d.]+), where')
PRICE = re.compile(r'each [^$]*? for \$(\d+),')
WAGE = re.compile(r'pays? \$(\d+) per hour of [^$]+? and \$(\d+) per hour of')
OUTPUT = re.compile(r'produce exactly ([\d,.]+) ')


def value(o: str) -> float:
    return float(o.replace('$', '').replace(',', ''))


def tech(text):
    m = PF.search(text)
    A, a, b = int(m.group(1)), float(m.group(2)), float(m.group(3))
    w, r = (int(x) for x in WAGE.search(text).groups())
    return A, a, b, w, r


def numeric_profit_max(A, a, b, p, w, r):
    """argmax_{L,K} p A L^a K^b - w L - r K by BFGS in (log L, log K)."""
    def neg(x):
        L, K = np.exp(x)
        q = A * L ** a * K ** b
        return -(p * q - w * L - r * K), -np.array([p * a * q - w * L, p * b * q - r * K])
    res = minimize(neg, x0=np.log([10.0, 10.0]), jac=True, method='BFGS', options={'gtol': 1e-10, 'maxiter': 10000})
    L, K = np.exp(res.x)
    return L, K, A * L ** a * K ** b


def numeric_cost_min(A, a, b, w, r, q):
    """argmin w L + r K subject to A L^a K^b = q: K = (q / (A L^a))^(1/b), searched over log L."""
    def cost(x):
        L = math.exp(x)
        return w * L + r * (q / (A * L ** a)) ** (1 / b)
    res = minimize_scalar(cost, bounds=(-10, 15), method='bounded', options={'xatol': 1e-12})
    L = math.exp(res.x)
    return L, (q / (A * L ** a)) ** (1 / b)


def matches(options, x, tol):
    return [i for i, o in enumerate(options) if abs(value(o) - x) <= tol]


def test_templates():
    ts = ct.templates(E)
    assert len(ts) == 14 * 5 * 3
    for t in ts:
        assert len(t.parts) == 2
        assert '{price}' in t.parts[0] and '{price}' not in t.parts[1]
        assert '{output}' in t.parts[1] and '{output}' not in t.parts[0]


def test_keys_by_numerical_optimisation():
    s = ElementSampler()
    types, domains, persp = collections.Counter(), set(), set()
    for seed in range(600):
        q = s.sample_question(E, 31000 + seed)
        rows = build_rows(q, 'test')
        assert [r['grading'] for r in rows] == ['answer', 'answer']
        t0, t1 = rows[0]['question_text'], rows[1]['question_text']
        A, a, b, w, r = tech(t0)
        assert tech(t1) == (A, a, b, w, r)
        assert a + b <= 0.85 + 1e-9 and a > 0 and b > 0
        p = int(PRICE.search(t0).group(1))
        Q = value(OUTPUT.search(t1).group(1))
        qtype = rows[0]['type']
        L0, K0, Q0 = numeric_profit_max(A, a, b, p, w, r)
        assert abs(Q - Q0) <= 0.006, (Q, Q0)
        L1, K1 = numeric_cost_min(A, a, b, w, r, Q)
        if qtype == 'cost':
            x0, x1, tol = w * L0 + r * K0, w * L1 + r * K1, 0.51
        else:
            x0, x1, tol = (L0, L1, 0.006) if qtype == 'labor' else (K0, K1, 0.006)
        for row, x in ((rows[0], x0), (rows[1], x1)):
            opts = row['options']
            assert len(opts) == 4 and len(set(opts)) == 4
            vals = [value(o) for o in opts]
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.01, opts
            assert matches(opts, x, tol) == [row['correct_index']], (row['question_text'], opts, x)
        k0 = rows[0]['options'][rows[0]['correct_index']]
        k1 = rows[1]['options'][rows[1]['correct_index']]
        assert k0 == k1, (k0, k1)
        types[qtype] += 1
        domains.add(rows[0]['domain'])
        persp.add(rows[0]['perspective'])
    assert set(types) == {'labor', 'capital', 'cost'}
    assert domains == {'consumer_goods', 'entertainment', 'education', 'environmental_policy', 'sports',
                       'technology', 'medical'}
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_closed_forms_match_numerical():
    from autosteer.filler_tools.factor_demand_tools import cost_min, profit_max
    for A, a, b, p, w, r in [(3, 0.4, 0.35, 20, 12, 7), (8, 0.25, 0.45, 9, 6, 15), (2, 0.55, 0.15, 33, 21, 4)]:
        L, K, Q = profit_max(A, a, b, p, w, r)
        Ln, Kn, Qn = numeric_profit_max(A, a, b, p, w, r)
        assert math.isclose(L, Ln, rel_tol=1e-6) and math.isclose(K, Kn, rel_tol=1e-6)
        Lc, Kc = cost_min(A, a, b, w, r, Q)
        assert math.isclose(Lc, L, rel_tol=1e-9) and math.isclose(Kc, K, rel_tol=1e-9)


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
