"""engel_curves, income_elasticity, shephards_lemma: keys recomputed from the question text.

engel_curves: the utility (x^a * y^b or a ln(x) + b ln(y)) and prices are parsed; demand at an income is the maximum
of utility on the budget line found by scipy (bounded 1-D search). The Engel curve m = c x (or c y) has c = m / x(m),
checked at two incomes (it is a straight line through the origin); for income_change the key is x(m + D) - x(m).
income_elasticity: direct: the percentage changes from the original values are computed from the four stated
numbers; quantity: elasticity x income change; income: quantity change / elasticity.
shephards_lemma: C(w, r, Q) is parsed and differentiated numerically (central difference) in w (labor) or r
(capital) at the stated prices and output.
For each: exactly one option is within rounding of the recomputed value and it is the marked one; options at least 10%
apart; every type occurs; sampling is reproducible from the seed.
"""
import collections
import math
import re

from scipy.optimize import minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

POWER = re.compile(r'U\(x, y\) = x\^([\d.]+) \* y\^([\d.]+),')
LOG = re.compile(r'U\(x, y\) = ([\d.]+) ln\(x\) \+ ([\d.]+) ln\(y\),')
PRICES = re.compile(r'is \$(\d+) per [^,]+? and the price of .+? is \$(\d+) per')
DELTA = re.compile(r'rises by \$(\d+) per')
COST = re.compile(r'C\(w, r, Q\) = ([\d.]+) \* w\^([\d.]+) \* r\^([\d.]+) \* Q(?:\^([\d.]+))?\.')
WRQ = re.compile(r'w = \$(\d+) and r = \$(\d+) and produces? Q = (\d+) ')


def apart(vals, rel=0.1, slack=0.011):
    for i in range(len(vals)):
        for j in range(i):
            assert abs(vals[i] - vals[j]) >= rel * max(abs(vals[i]), abs(vals[j])) - slack, vals


def demand(u, px, py, m):
    res = minimize_scalar(lambda x: -u(x, (m - px * x) / py), bounds=(1e-9, m / px - 1e-9), method='bounded',
                          options={'xatol': 1e-12})
    return res.x, (m - px * res.x) / py


def test_engel_curves():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question('engel_curves', 121000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        if POWER.search(text):
            a, b = (float(z) for z in POWER.search(text).groups())
            def u(x, y): return a * math.log(x) + b * math.log(y)   # same ordering as x^a y^b
            seen['power'] += 1
        else:
            a, b = (float(z) for z in LOG.search(text).groups())
            def u(x, y): return a * math.log(x) + b * math.log(y)
            seen['log'] += 1
        px, py = (int(z) for z in PRICES.search(text).groups())
        if row['type'] == 'income_change':
            D = int(DELTA.search(text).group(1))
            want = demand(u, px, py, 1000 + D)[0] - demand(u, px, py, 1000)[0]
            vals = [float(o) for o in opts]
        else:
            i = 0 if row['type'] == 'curve_x' else 1
            c1 = 500 / demand(u, px, py, 500)[i]
            c2 = 2000 / demand(u, px, py, 2000)[i]
            assert math.isclose(c1, c2, rel_tol=1e-6)
            want = c1
            var = 'x' if i == 0 else 'y'
            vals = [float(re.fullmatch(rf'm = ([\d.]+){var}', o).group(1)) for o in opts]
        assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-6 * want] == [k], (text, opts, want)
        apart(vals)
        seen[row['type']] += 1
    assert seen['curve_x'] and seen['curve_y'] and seen['income_change'] and seen['power'] and seen['log']


IE_DIRECT = re.compile(r'income changed from \$([\d,]+) to \$([\d,]+) a .+? from (\d+) to (\d+) ')
IE_E = re.compile(r'income elasticity of demand for .+? is (-?[\d.]+)\.')
IE_PCT = re.compile(r'(?:rises by|changed by) (-?[\d.]+)%')


def test_income_elasticity():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question('income_elasticity', 122000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        if row['type'] == 'direct':
            m1, m2, q1, q2 = (float(z.replace(',', '')) for z in IE_DIRECT.search(text).groups())
            want = ((q2 - q1) / q1) / ((m2 - m1) / m1)
            vals = [float(o) for o in opts]
            seen['inferior' if want < 0 else 'normal'] += 1
        else:
            E = float(IE_E.search(text).group(1))
            p = float(IE_PCT.search(text).group(1))
            want = E * p if row['type'] == 'quantity' else p / E
            assert all(o.endswith('%') for o in opts)
            vals = [float(o.rstrip('%')) for o in opts]
        assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051] == [k], (text, opts, want)
        apart(vals)
        seen[row['type']] += 1
    assert seen['direct'] and seen['quantity'] and seen['income'] and seen['inferior'] and seen['normal']


def test_shephards_lemma():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question('shephards_lemma', 123000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        kc, al, be, g = COST.search(text).groups()
        kc, al, be, g = float(kc), float(al), float(be), float(g) if g else 1.0
        assert math.isclose(al + be, 1.0)
        w, r, Q = (int(z) for z in WRQ.search(text).groups())

        def C(w, r):
            return kc * w ** al * r ** be * Q ** g
        h = 1e-5
        want = ((C(w + h, r) - C(w - h, r)) if row['type'] == 'labor' else (C(w, r + h) - C(w, r - h))) / (2 * h)
        vals = [float(o) for o in opts]
        assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-7 * want] == [k], (text, opts, want)
        apart(vals)
        seen[row['type']] += 1
    assert seen['labor'] and seen['capital']


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for e in ('engel_curves', 'income_elasticity', 'shephards_lemma'):
        for seed in (0, 7, 123456789):
            q = a.sample_question(e, seed)
            assert b.sample_question(e, seed) == q
            assert a.sample_question(e, seed, template_id=q['template_id']) == q
