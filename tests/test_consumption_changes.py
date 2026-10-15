"""consumption_changes: the expenditure share of x recomputed from the question text.

The utility (CES with a, b, ρ, or Cobb-Douglas x^a * y^b), prices, income and the price rise are parsed. The
utility-maximising bundle is found by scipy (bounded 1-D search over the share of income spent on x) at the old and
the new price of x; the shares of income spent on x are compared (direction: rises / falls / stays the same, with a
tolerance of 1e-6) or the new share in percent is the key. Exactly one option matches and it is the marked one;
numeric options at least 10% apart; every type, domain, perspective and direction occurs; sampling is reproducible.
"""
import collections
import math
import re

from scipy.optimize import minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'consumption_changes'
CES = re.compile(r'U\(x, y\) = \(a \* x\^ρ \+ b \* y\^ρ\)\^\(1/ρ\) with a = ([\d.]+), b = ([\d.]+) and ρ = (-?[\d.]+),')
CD = re.compile(r'U\(x, y\) = x\^([\d.]+) \* y\^([\d.]+),')
NUMS = re.compile(r'income of \$([\d,]+) a .+? The price of .+? is \$(\d+) per .+? and the price of .+? is \$(\d+) per '
                  r'.+? now rises by (\d+)%')


def utility(text):
    m = CES.search(text)
    if m:
        a, b, rho = (float(z) for z in m.groups())
        return lambda x, y: (a * x ** rho + b * y ** rho) ** (1 / rho)
    a, b = (float(z) for z in CD.search(text).groups())
    return lambda x, y: a * math.log(x) + b * math.log(y)


def share(u, px, py, m):
    res = minimize_scalar(lambda t: -u(t * m / px, (1 - t) * m / py), bounds=(1e-9, 1 - 1e-9), method='bounded',
                          options={'xatol': 1e-12})
    return res.x


def test_consumption_changes():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(500):
        (row,) = build_rows(s.sample_question(E, 131000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        u = utility(text)
        m, px, py, rise = NUMS.search(text).groups()
        m, px, py, rise = float(m.replace(',', '')), float(px), float(py), float(rise)
        s0, s1 = share(u, px, py, m), share(u, px * (1 + rise / 100), py, m)
        if row['type'] == 'direction':
            want = 'It stays the same' if abs(s1 - s0) < 1e-6 else ('It rises' if s1 > s0 else 'It falls')
            assert sorted(opts) == ['It falls', 'It rises', 'It stays the same']
            assert opts[k] == want, (text, opts, want)
            seen[want] += 1
        else:
            vals = [float(o.rstrip('%')) for o in opts]
            want = 100 * s1
            assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0052] == [k], (text, opts, want)
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.011
        seen[row['type']] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('direction', 'new_share', 'It rises', 'It falls', 'It stays the same', 'first_person',
                                 'second_person', 'man', 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
