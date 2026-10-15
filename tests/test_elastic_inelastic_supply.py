"""elastic_inelastic_supply: equilibrium price changes recomputed from the question text.

Perfectly elastic supply: the exponents and the wage and rental-rate changes are parsed; the long-run price is the
minimum unit cost, found by scipy (bounded search over ln(K/L) of the cost (w + r t) / f(1, t) of one unit with the
parsed constant-returns technology), before and after. Fixed supply: the demand Q = α - βP (before and after a shift)
and Q̄ are parsed and the price is where demand equals Q̄ (brentq); a change in the sellers' cost does not enter that
condition (the test checks that the cost stays below the price). Exactly one option within rounding and it is the
marked one; numeric options at least 10% apart; every type, domain and perspective occurs; sampling is reproducible.
"""
import collections
import math
import re

from scipy.optimize import brentq, minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'elastic_inelastic_supply'
TECH = re.compile(r'Q = A \* L\^([\d.]+) \* K\^([\d.]+),')
WAGE = re.compile(r'The wage of .+? (rises by|falls by|stays the same)(?: (\d+)%)? and the rental rate of .+? '
                  r'(rises by|falls by|stays the same)(?: (\d+)%)?\.')
DEMAND = re.compile(r'Demand from .+? is Q = ([\d,]+) - (\d+)P,')
QBAR = re.compile(r'has ([\d,]+) (?:seats|hotel rooms|hectares|berths|places)')
COST = re.compile(r'rises from \$([\d.]+) to \$([\d.]+),')
SHIFT = re.compile(r'demand becomes Q = ([\d,]+) - (\d+)P\.')
X = re.compile(r'hourly wage of workers rises by (\d+)%')


def unit_cost(a, b, w, r):
    res = minimize_scalar(lambda z: (w + r * math.exp(z)) / math.exp(z) ** b, bounds=(-30, 30), method='bounded',
                          options={'xatol': 1e-12})
    return res.fun


def change(m, i):
    if m.group(i) == 'stays the same':
        return 0.0
    return (1 if m.group(i) == 'rises by' else -1) * float(m.group(i + 1)) / 100


def clearing(alpha, beta, qbar):
    return brentq(lambda p: alpha - beta * p - qbar, 0, alpha / beta)


def n(s):
    return float(s.replace(',', ''))


def test_elastic_inelastic_supply():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(500):
        (row,) = build_rows(s.sample_question(E, 201000 + seed), 'test')
        text, opts, k, t = row['question_text'], row['options'], row['correct_index'], row['type']
        if t in ('elastic_cost_shock', 'compare'):
            a, b = (float(z) for z in TECH.search(text).groups())
            assert abs(a + b - 1) < 1e-9
            if t == 'elastic_cost_shock':
                m = WAGE.search(text)
                dw, dr = change(m, 1), change(m, 3)
            else:
                dw, dr = float(X.search(text).group(1)) / 100, 0.0
            elastic = 100 * (unit_cost(a, b, 20 * (1 + dw), 10 * (1 + dr)) / unit_cost(a, b, 20, 10) - 1)
        if t != 'elastic_cost_shock':
            alpha, beta = (n(z) for z in DEMAND.search(text).groups())
            qbar = n(QBAR.search(text).group(1))
            p0 = clearing(alpha, beta, qbar)
        if t == 'compare':
            m = re.fullmatch(r'Market 1: (-?[\d.]+)%; market 2: (-?[\d.]+)%', opts[k])
            assert abs(float(m.group(1))) < 1e-9 and abs(float(m.group(2)) - elastic) <= 0.0051, (text, opts)
            assert len(set(opts)) == 4
        else:
            if t == 'elastic_cost_shock':
                want = elastic
            elif t == 'inelastic_cost_shock':
                c0, c1 = (float(z) for z in COST.search(text).groups())
                assert c0 < c1 < p0
                want = 0.0
            else:
                alpha2, beta2 = (n(z) for z in SHIFT.search(text).groups())
                want = 100 * (clearing(alpha2, beta2, qbar) / p0 - 1)
            vals = [float(o.rstrip('%')) for o in opts]
            assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-7 * abs(want)] == [k], \
                (text, opts, want)
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= 0.1 * max(abs(vals[i]), abs(vals[j])) - 0.011
        seen[t] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[x] for x in ('elastic_cost_shock', 'inelastic_cost_shock', 'inelastic_demand_shift', 'compare',
                                 'first_person', 'second_person', 'man', 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
