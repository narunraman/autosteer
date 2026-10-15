"""elasticity_of_substitution: keys recomputed from the question text.

from_function and ratio_response: a, b and ρ are parsed; the cost-minimizing capital per hour of labor at input prices
(w, r) is where the ratio of marginal products (central finite differences of the parsed function) equals w/r, found by
scipy's brentq over ln(K/L) (the tangency condition of cost minimization). σ = Δln(K/L) / Δln(w/r) between w/r = 1 and 2.5; the
response is the percentage change in K/L when the stated wage rises by x% at the stated rental rate. from_ratios: the
four prices and two ratios are parsed and the key is Δln(K/L) / Δln(w/r). Exactly one option within rounding and it is
the marked one; options at least 10% apart; every type, domain and perspective occurs; sampling is reproducible.
"""
import collections
import math
import re

from scipy.optimize import brentq

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'elasticity_of_substitution'
CES = re.compile(r'Q = \(a \* L\^ρ \+ b \* K\^ρ\)\^\(1/ρ\) with a = ([\d.]+), b = ([\d.]+) and ρ = (-?[\d.]+),')
RATIOS = re.compile(r'wage was \$(\d+) and the rental rate \$(\d+) per hour, .+? used ([\d.]+) hours .+? wage changed '
                    r'to \$(\d+) and the rental rate to \$(\d+) per hour, it uses ([\d.]+) hours')
RESP = re.compile(r'pays? \$(\d+) per hour of .+? and \$(\d+) per hour of .+? wage rises by (\d+)%')


def kl(a, b, rho, w, r):
    def f(L, K):
        return (a * L ** rho + b * K ** rho) ** (1 / rho)

    def gap(z):                      # ln(MPL / MPK) - ln(w / r) at L = 1, K = e^z (monotone in z)
        t, e = math.exp(z), 1e-6
        mpl = (f(1 + e, t) - f(1 - e, t)) / (2 * e)
        mpk = (f(1, t * (1 + e)) - f(1, t * (1 - e))) / (2 * e * t)
        return math.log(mpl / mpk) - math.log(w / r)
    lo, hi = -0.5, 0.5
    while gap(lo) * gap(hi) > 0:     # widen the bracket from the middle (extremes underflow)
        lo, hi = lo - 0.5, hi + 0.5
    return math.exp(brentq(gap, lo, hi, xtol=1e-13))


def test_elasticity_of_substitution():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(450):
        (row,) = build_rows(s.sample_question(E, 161000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        if row['type'] == 'from_ratios':
            w1, r1, k1, w2, r2, k2 = (float(z) for z in RATIOS.search(text).groups())
            want = math.log(k2 / k1) / math.log((w2 / r2) / (w1 / r1))
        else:
            a, b, rho = (float(z) for z in CES.search(text).groups())
            if row['type'] == 'from_function':
                want = math.log(kl(a, b, rho, 2.5, 1) / kl(a, b, rho, 1, 1)) / math.log(2.5)
            else:
                w, r, x = (float(z) for z in RESP.search(text).groups())
                want = 100 * (kl(a, b, rho, w * (1 + x / 100), r) / kl(a, b, rho, w, r) - 1)
        vals = [float(o) for o in opts]
        assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-5 * abs(want)] == [k], (text, opts, want)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.1 * max(abs(vals[i]), abs(vals[j])) - 0.011
        seen[row['type']] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('from_function', 'from_ratios', 'ratio_response', 'first_person', 'second_person',
                                 'man', 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
