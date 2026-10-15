"""mrs_labor_supply: keys recomputed from the question text.

The utility U(C, l) (C^a * l^b, aC + bl or a ln(C) + b * l^0.5), wage, price, hours, consumption and leisure are
parsed. Checks that the stated bundle is on the budget line (C = w h / p, l = 16 - h). MRS types: MU_l and MU_C are
central finite differences of the parsed utility at the bundle, the key MU_l / MU_C. optimal_hours: the best number
of hours is found by scipy (bounded search of U(w h / p, 16 - h) over h); the key is keep (|h - h*| < 0.01), work more
or work fewer. Exactly one option matches and it is the marked one; numeric options at least 10% apart; every type,
answer, domain and perspective occurs; sampling is reproducible.
"""
import collections
import math
import re

from scipy.optimize import minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'mrs_labor_supply'
NUMS = re.compile(r'earns? \$(\d+) per hour of work, .+? costs \$(\d+) per unit\. .+? U\(C, l\) = (.+?), where C .+? '
                  r'currently works? ([\d.]+) hours a day, so .+? consumes? ([\d.]+) units and ha(?:ve|s) ([\d.]+) hours')
CD = re.compile(r'C\^([\d.]+) \* l\^([\d.]+)$')
LIN = re.compile(r'([\d.]+)C \+ ([\d.]+)l$')
NL = re.compile(r'([\d.]+) ln\(C\) \+ ([\d.]+) \* l\^0\.5$')


def utility(f):
    m = CD.match(f)
    if m:
        a, b = (float(z) for z in m.groups())
        return lambda C, l: C ** a * l ** b
    m = LIN.match(f)
    if m:
        a, b = (float(z) for z in m.groups())
        return lambda C, l: a * C + b * l
    a, b = (float(z) for z in NL.match(f).groups())
    return lambda C, l: a * math.log(C) + b * l ** 0.5


def test_mrs_labor_supply():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question(E, 141000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        w, p, f, h, C, l = NUMS.search(text).groups()
        w, p, h, C, l = float(w), float(p), float(h), float(C), float(l)
        assert math.isclose(C, w * h / p, abs_tol=0.006) and math.isclose(l, 16 - h)
        u = utility(f)
        if row['type'] == 'optimal_hours':
            res = minimize_scalar(lambda x: -u(w * x / p, 16 - x), bounds=(1e-6, 16 - 1e-6), method='bounded',
                                  options={'xatol': 1e-10})
            want = ('Keep the current hours' if abs(res.x - h) < 0.01 else
                    'Work more hours' if res.x > h else 'Work fewer hours')
            assert len(opts) == 3 and opts[k].startswith(want), (text, opts, want, res.x)
            seen[want] += 1
        else:
            e = 1e-6
            mu_l = (u(C, l + e) - u(C, l - e)) / (2 * e)
            mu_c = (u(C + e, l) - u(C - e, l)) / (2 * e)
            want = mu_l / mu_c
            vals = [float(o) for o in opts]
            assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-6 * want] == [k], (text, opts, want)
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.011
        seen[row['type']] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('cobb_douglas', 'linear', 'non_linear', 'optimal_hours', 'Keep the current hours',
                                 'Work more hours', 'Work fewer hours', 'first_person', 'second_person', 'man',
                                 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
