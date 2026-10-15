"""factor_shares_equilibrium: factor shares recomputed from the question text.

The production function (A * L^a * K^b or aL + bK), the output price and the amounts L and K are parsed. The marginal
products are central finite differences of the parsed function; w = p MPL, r = p MPK, and the key is wL / (wL + rK)
(labor) or rK / (wL + rK) (capital), in percent. Exactly one option within rounding and it is the marked one; options
at least 10% apart; every type, factor, domain and perspective occurs, and so do increasing and decreasing returns;
sampling is reproducible.
"""
import collections
import re

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'factor_shares_equilibrium'
CD = re.compile(r'Q = ([\d.]+) \* L\^([\d.]+) \* K\^([\d.]+),')
LIN = re.compile(r'Q = (\d+)L \+ (\d+)K,')
NUMS = re.compile(r'for \$(\d+) and uses? (\d+) hours of .+? and (\d+) hours of ')
WHICH = re.compile(r'goes to (.+?)\? Give')
LAB = re.compile(r'L is the number of hours of (.+?) and K')


def test_factor_shares_equilibrium():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(450):
        (row,) = build_rows(s.sample_question(E, 191000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        m = CD.search(text)
        if m:
            A, a, b = (float(z) for z in m.groups())
            f = lambda L, K: A * L ** a * K ** b    # noqa: E731
            seen['crs' if abs(a + b - 1) < 1e-9 else 'irs' if a + b > 1 else 'drs'] += 1
        else:
            a, b = (float(z) for z in LIN.search(text).groups())
            f = lambda L, K: a * L + b * K          # noqa: E731
        p, L, K = (float(z) for z in NUMS.search(text).groups())
        e = 1e-6
        w = p * (f(L + e, K) - f(L - e, K)) / (2 * e)
        r = p * (f(L, K + e) - f(L, K - e)) / (2 * e)
        labor = WHICH.search(text).group(1) == LAB.search(text).group(1)
        want = 100 * (w * L if labor else r * K) / (w * L + r * K)
        vals = [float(o.rstrip('%')) for o in opts]
        assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0052] == [k], (text, opts, want)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.011
        seen[row['type']] += 1
        seen['labor' if labor else 'capital'] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('cobb_douglas_crs', 'cobb_douglas', 'linear', 'labor', 'capital', 'irs', 'drs',
                                 'first_person', 'second_person', 'man', 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
