"""optimize_continuous: the key recomputed by numeric optimisation of the function parsed from the question.

Independent of the generator: the polynomial ("f(x) = 2x^3 - 9x^2 + 12x + 5", "P(q) = ...") and the interval are
parsed from the text, the box volume V(x) = x (s - 2x)^2 on [0, s/2] and the pen area A(x) = x (L - 2x) on
[0, L/2] are built from the stated sheet side and fence length. Each is optimised on a dense grid of the interval
(20,001 points) refined with scipy's bounded minimize_scalar around the best grid point; the keyed option must be
the optimiser (or the optimal value, when asked) within 1e-6 relative, and no other option may be. Also every type
and both goals occur, and seeded reproducibility.
"""
import collections
import re

import numpy as np
from scipy.optimize import minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'optimize_continuous'
TERM = re.compile(r'([+-]?)\s*(\d*)([xq])?(?:\^(\d))?')


def parse_poly(expr):
    coefs = collections.defaultdict(int)
    expr = expr.replace(' ', '')
    for sign, num, var, power in TERM.findall(expr):
        if not (num or var):
            continue
        c = int(num) if num else 1
        p = int(power) if power else (1 if var else 0)
        coefs[p] += -c if sign == '-' else c
    return lambda x: sum(c * x ** p for p, c in coefs.items())


def optimise(f, lo, hi, goal):
    sgn = -1 if goal == 'max' else 1
    xs = np.linspace(lo, hi, 20001)
    vals = sgn * np.array([f(x) for x in xs])
    i = int(np.argmin(vals))
    a, b = xs[max(0, i - 1)], xs[min(len(xs) - 1, i + 1)]
    res = minimize_scalar(lambda x: sgn * f(x), bounds=(a, b), method='bounded', options={'xatol': 1e-10})
    x = res.x if sgn * f(res.x) <= vals[i] else xs[i]
    return x, f(x)


def test_keys_numeric():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question(E, 77000 + seed), 'test')
        text = row['question_text']
        opts = [float(o) for o in row['options']]
        key = opts[row['correct_index']]
        t = row['type']
        if t in ('quadratic', 'cubic'):
            m = re.search(r'(?:f\(x\)|P\(q\)) = ([-+ \dxq^]+?)(?: be|,|\. | is)', text)
            f = parse_poly(m.group(1))
            lo, hi = (int(x) for x in re.search(r'(-?\d+) (?:<= x <=|and) (-?\d+)|\[(-?\d+), (-?\d+)\]', text)
                      .groups() if x is not None)
            goal = 'max' if re.search(r'maxim|largest', text) else 'min'
            x, _ = optimise(f, lo, hi, goal)
            target = x
            seen[(t, goal)] += 1
        elif t == 'box':
            side = int(re.search(r'sheet of cardboard (\d+) cm', text).group(1))
            x, v = optimise(lambda x: x * (side - 2 * x) ** 2, 0, side / 2, 'max')
            target = x if 'side length x' in text else v
            seen[(t, 'x' if 'side length x' in text else 'value')] += 1
        else:
            L = int(re.search(r'has (\d+) metres|have (\d+) metres', text).group(1) or
                    re.search(r'have (\d+) metres', text).group(1))
            x, v = optimise(lambda x: x * (L - 2 * x), 0, L / 2, 'max')
            target = x if 'How long' in text else v
            seen[(t, 'x' if 'How long' in text else 'value')] += 1
        close = [o for o in opts if abs(o - target) <= 1e-6 * max(1, abs(target))]
        assert close == [key], (text, row['options'], target)
    for k in [('quadratic', 'max'), ('quadratic', 'min'), ('cubic', 'max'), ('cubic', 'min'), ('box', 'x'),
              ('box', 'value'), ('fence', 'x'), ('fence', 'value')]:
        assert seen[k], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
