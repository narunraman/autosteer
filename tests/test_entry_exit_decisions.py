"""entry_exit_decisions: keys recomputed from the question text.

The cost function TC(q) (F + c q + d q^2, or F + a q - b q^2 + e q^3) and the price are parsed. entry: the most profitable
output is found by scipy (bounded search of P q - TC(q)); open iff that profit is positive (staying out earns 0).
break_even_price: the minimum of TC(q) / q found numerically. shutdown: keep producing iff the best P q - VC(q) over
q > 0 is positive (the fixed cost is paid either way). shutdown_price: the minimum of VC(q) / q found numerically.
Checks also that the planned output in entry questions is a trap (AC above the price when opening is right, MC below
the price when staying out is right). Exactly one option matches and it is the marked one; numeric options at least
10% apart; every type, answer, domain and perspective occurs; sampling is reproducible.
"""
import collections
import re

import numpy as np

from scipy.optimize import minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'entry_exit_decisions'
QUAD = re.compile(r'TC\(q\) = (\d+) \+ ([\d.]+)q \+ ([\d.]+)q\^2 dollars')
CUBIC = re.compile(r'TC\(q\) = (\d+) \+ ([\d.]+)q - ([\d.]+)q\^2 \+ ([\d.]+)q\^3 dollars')
PRICE = re.compile(r'price (?:is|has fallen to) \$([\d.]+) per')
Q0 = re.compile(r'foresees producing (\d+) ')


def argmin(f, hi):
    """Global minimum on (0, hi]: a grid, then a bounded search around the best grid point."""
    grid = np.linspace(hi / 20000, hi, 20000)
    i = int(np.argmin([f(q) for q in grid]))
    lo, up = grid[max(i - 1, 0)] if i else 1e-9, grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(f, bounds=(lo, up), method='bounded', options={'xatol': 1e-12})
    return res.x, res.fun


def test_entry_exit_decisions():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question(E, 171000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        t = row['type']
        m = QUAD.search(text)
        if m:
            F, c, d = (float(z) for z in m.groups())
            vc = lambda q: c * q + d * q * q          # noqa: E731
        else:
            F, a, b, e = (float(z) for z in CUBIC.search(text).groups())
            vc = lambda q: a * q - b * q * q + e * q ** 3   # noqa: E731
        hi = 2000
        if t in ('entry', 'shutdown'):
            P = float(PRICE.search(text).group(1))
            if t == 'entry':
                profit = -argmin(lambda q: -(P * q - F - vc(q)), hi)[1]
                want = 'Yes' if profit > 0 else 'No'
                q0 = float(Q0.search(text).group(1))
                if want == 'Yes':
                    assert (F + vc(q0)) / q0 > P
                else:
                    assert (vc(q0 + 1e-6) - vc(q0 - 1e-6)) / 2e-6 < P
            else:
                op = -argmin(lambda q: -(P * q - vc(q)), hi)[1]
                want = 'Keep' if op > 0 else 'Shut'
                assert P < argmin(lambda q: (F + vc(q)) / q, hi)[1] or want == 'Shut'
            assert len(opts) == 2 and opts[k].startswith(want), (text, opts, want)
            seen[t + want] += 1
        else:
            fixed = F if t == 'break_even_price' else 0
            want = argmin(lambda q: (fixed + vc(q)) / q, hi)[1]
            vals = [float(o.lstrip('$').replace(',', '')) for o in opts]
            assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0051 + 1e-7 * want] == [k], (text, opts, want)
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= 0.1 * max(vals[i], vals[j]) - 0.011
        seen[t] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[x] for x in ('entry', 'break_even_price', 'shutdown', 'shutdown_price', 'entryYes', 'entryNo',
                                 'shutdownKeep', 'shutdownShut', 'first_person', 'second_person', 'man', 'woman',
                                 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
