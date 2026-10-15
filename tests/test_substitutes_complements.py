"""substitutes_complements: the sign of the cross-price effect recomputed from the text, by finite differences.

From the released rows alone:
- linear_demand: x = A - B px +/- C py + D m is parsed; dx/dpy is a finite difference of that function at a few
  random points; the key is Substitutes when it is positive, Complements when negative.
- utility: U = alpha x + beta y - (g/2) x^2 - (h/2) y^2 -/+ t xy + z is parsed; demand at given prices is the
  maximiser of U - px x - py y (scipy, unconstrained), and dx/dpy a finite difference of that numerical demand
  around the stated prices. Both demands are positive at the stated prices.
- elasticity: the cross-price elasticity (dx/dpy) py / x at the stated point, by finite differences of the parsed
  demand, rounded to 2 decimals, is exactly one option, the marked one; options at least 10% apart.
The independent option is never the key; every type, both keys, every domain and perspective occur; sampling is
reproducible from the seed.
"""
import collections
import re

import numpy as np
from scipy.optimize import minimize

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'substitutes_complements'
C = r'(\d+(?:\.\d+)?)?'
DEMAND = re.compile(rf'x = (\d+) - {C}px ([+-]) {C}py \+ {C}m,')
UTIL = re.compile(rf'U = (\d+)x \+ (\d+)y - {C}x\^2 - {C}y\^2 ([+-]) {C}xy \+ z,')
PRICES = re.compile(r'The prices are \$(\d+) for .+? and \$(\d+) for')
POINT = re.compile(r'At px = (\d+), py = (\d+) and m = (\d+),')


def f(x):
    return 1.0 if x is None else float(x)


def demand_fn(text):
    A, B, sign, Cc, D = DEMAND.search(text).groups()
    c = f(Cc) * (1 if sign == '+' else -1)
    return lambda px, py, m: int(A) - f(B) * px + c * py + f(D) * m


def ql_demand(text, px, py):
    al, be, g2, h2, sign, t = UTIL.search(text).groups()
    al, be, g, h = float(al), float(be), 2 * f(g2), 2 * f(h2)
    t = f(t) * (1 if sign == '-' else -1)      # U = ... - t xy

    def neg(v):
        x, y = v
        return -(al * x + be * y - g / 2 * x * x - h / 2 * y * y - t * x * y - px * x - py * y)
    res = minimize(neg, x0=[1.0, 1.0], method='BFGS', options={'gtol': 1e-10})
    return res.x


def test_templates():
    assert len(ct.templates(E)) == 8 * 5 * 3


def test_keys():
    s = ElementSampler()
    rng = np.random.default_rng(0)
    seen = collections.Counter()
    domains, persp = set(), set()
    for seed in range(900):
        (row,) = build_rows(s.sample_question(E, 91000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        assert len(set(opts)) == len(opts)
        if row['type'] in ('linear_demand', 'utility'):
            assert sorted(opts) == sorted(['Substitutes', 'Complements', 'Neither: the two goods are independent'])
            if row['type'] == 'linear_demand':
                x = demand_fn(text)
                pts = rng.uniform(1, 50, size=(3, 3))
                slopes = [(x(a, b + 1e-3, c) - x(a, b - 1e-3, c)) / 2e-3 for a, b, c in pts]
            else:
                px, py = (int(z) for z in PRICES.search(text).groups())
                q = ql_demand(text, px, py)
                assert (q > 0).all(), (text, q)
                slopes = [(ql_demand(text, px, py + 1e-3)[0] - ql_demand(text, px, py - 1e-3)[0]) / 2e-3]
            assert all(abs(sl) > 1e-6 for sl in slopes) and len({sl > 0 for sl in slopes}) == 1
            want = 'Substitutes' if slopes[0] > 0 else 'Complements'
            assert opts[k] == want, (text, opts[k])
            seen[(row['type'], want)] += 1
        else:
            x = demand_fn(text)
            px, py, m = (int(z) for z in POINT.search(text).groups())
            el = (x(px, py + 1e-4, m) - x(px, py - 1e-4, m)) / 2e-4 * py / x(px, py, m)
            vals = [float(o) for o in opts]
            assert [i for i, v in enumerate(vals) if abs(v - el) <= 0.0051] == [k], (text, opts, el)
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= 0.1 * max(abs(vals[i]), abs(vals[j])) - 0.011, opts
            seen[('elasticity', vals[k] > 0)] += 1
        domains.add(row['domain'])
        persp.add(row['perspective'])
    assert {('linear_demand', 'Substitutes'), ('linear_demand', 'Complements'), ('utility', 'Substitutes'),
            ('utility', 'Complements'), ('elasticity', True), ('elasticity', False)} <= set(seen), seen
    assert len(domains) == 8
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
