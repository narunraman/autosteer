"""fixed_factor_aggregation: the competitive equilibrium recomputed from the question text.

The supply K̄, each firm's production function A_i * K^α_i and output price p_i are parsed. A firm's demand at a
rental rate r is the K where its value of marginal product (a central finite difference of p_i A_i K^α_i) equals r
(scipy brentq on a log scale); the equilibrium rate is the r at which the two demands add up to K̄ (brentq). The key is
firm 1's demand or the rate. Exactly one option within rounding and it is the marked one; options at least 10% apart;
every type, domain and perspective occurs; sampling is reproducible.
"""
import collections
import math
import re

from scipy.optimize import brentq

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'fixed_factor_aggregation'
KBAR = re.compile(r'owns? all ([\d,]+) ')
FIRMS = re.compile(r'firm 1 produces (\d+) \* K\^([\d.]+) .+? at \$(\d+) per .+? firm 2 produces (\d+) \* K\^([\d.]+) '
                   r'.+? at \$(\d+) per ')


def root(f):
    """A root of a monotone f, bracketed by widening [-1, 1] step by step."""
    lo, hi = -1.0, 1.0
    while f(lo) * f(hi) > 0:
        lo, hi = lo - 1, hi + 1
    return brentq(f, lo, hi, xtol=1e-14)


def demand(A, a, p, r):
    def vmp(K):
        e = 1e-6 * K
        return p * (A * (K + e) ** a - A * (K - e) ** a) / (2 * e)
    return math.exp(root(lambda z: math.log(vmp(math.exp(z))) - math.log(r)))


def test_fixed_factor_aggregation():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(450):
        (row,) = build_rows(s.sample_question(E, 181000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        kbar = float(KBAR.search(text).group(1).replace(',', ''))
        A1, a1, p1, A2, a2, p2 = (float(z) for z in FIRMS.search(text).groups())
        z = root(lambda z: math.log(demand(A1, a1, p1, math.exp(z)) + demand(A2, a2, p2, math.exp(z))) -
                 math.log(kbar))
        rate = math.exp(z)
        want = rate if row['type'] == 'rental_rate' else demand(A1, a1, p1, rate)
        vals = [float(o.lstrip('$').replace(',', '')) for o in opts]
        assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-6 * want] == [k], (text, opts, want)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.011
        seen[row['type']] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('equal_exponents', 'different_exponents', 'rental_rate', 'first_person',
                                 'second_person', 'man', 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
