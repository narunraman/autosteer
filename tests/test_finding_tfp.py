"""finding_tfp: total factor productivity recomputed from the numbers in the text.

From the released rows alone: the production function (A * L^a * K^b, A * (aL + bK) or A * min(aL, bK)) and the
inputs are parsed. With output given, A = output / f(L, K) evaluated from the parsed formula with A = 1. With the
output price p and the wage w given, the labor shown is the profit-maximising one, so p A MPL = w; MPL (with A = 1)
is a central finite difference of the parsed f in L, and A = w / (p MPL). Checks: exactly one option is within
half a hundredth of A and it is the marked one; options at least 10% apart; every type, domain and perspective
occurs; sampling is reproducible from the seed.
"""
import collections
import re

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'finding_tfp'
CD = re.compile(r'Q = A \* L\^([\d.]+) \* K\^([\d.]+),')
LIN = re.compile(r'Q = A \* \(([\d.]+)L \+ ([\d.]+)K\),')
LEO = re.compile(r'Q = A \* min\(([\d.]+)L, ([\d.]+)K\),')
USED = re.compile(r'used (\d+) hours of .+? and (\d+) hours of .+? and produced ([\d.]+) ')
PRICES = re.compile(r'for \$(\d+) and pays? \$([\d.]+) per hour of .+?\. With (\d+) hours of .+? the profit-maximizing '
                    r'amount of .+? is (\d+) hours')


def technology(text):
    for pat, f in ((CD, lambda a, b: lambda L, K: L ** a * K ** b), (LIN, lambda a, b: lambda L, K: a * L + b * K),
                   (LEO, lambda a, b: lambda L, K: min(a * L, b * K))):
        m = pat.search(text)
        if m:
            return f(float(m.group(1)), float(m.group(2)))
    raise AssertionError(text)


def test_templates():
    assert len(ct.templates(E)) == 14 * 5 * 4


def test_tfp_recomputed():
    s = ElementSampler()
    seen, domains, persp = collections.Counter(), set(), set()
    for seed in range(800):
        (row,) = build_rows(s.sample_question(E, 101000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        f = technology(text)
        if row['type'] == 'input_prices':
            p, w, K, L = PRICES.search(text).groups()
            p, w, K, L = int(p), float(w), int(K), int(L)
            h = 1e-5
            mpl = (f(L + h, K) - f(L - h, K)) / (2 * h)
            A = w / (p * mpl)
        else:
            L, K, Q = USED.search(text).groups()
            A = float(Q) / f(int(L), int(K))
        vals = [float(o) for o in opts]
        assert len(set(opts)) == 4
        assert [i for i, v in enumerate(vals) if abs(v - A) <= 0.0051] == [k], (text, opts, A)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.011, opts
        seen[row['type']] += 1
        domains.add(row['domain'])
        persp.add(row['perspective'])
    assert set(seen) == {'cobb_douglas', 'linear', 'leontief', 'input_prices'}
    assert len(domains) == 7
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
