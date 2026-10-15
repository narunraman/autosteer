"""mrs_tangency: MRS and tangency recomputed numerically from the question text.

From the released rows alone: U(x, y) = x^a * y^b, the income and both prices are parsed. The utility maximum on the
budget line is found by scipy (bounded 1-D search), and marginal utilities by central finite differences of U.
- mrs_at_optimum: the key is MU_x / MU_y at the numerical optimum, rounded to 2 decimals; exactly one option matches.
- tangency_bundle: every option costs the stated income (within rounding); the key is the option within 0.006 of
  the numerical optimum, and at it MU_x / MU_y equals px / py (within 1%).
- adjust: the stated bundle costs the income; moving a little along the budget line towards more x raises utility
  exactly when the key says "spend more on x", towards less x when it says "spend less on x", and neither when the
  key keeps the bundle (then it is the numerical optimum).
Every type, all three adjust answers, every domain and perspective occur; sampling is reproducible from the seed.
"""
import collections
import math
import re

from scipy.optimize import minimize_scalar

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'mrs_tangency'
UF = re.compile(r'U\(x, y\) = x\^([\d.]+) \* y\^([\d.]+),')
INCOME = re.compile(r'spends? \$([\d,]+) a ')
PRICES = re.compile(r'is \$(\d+) per [^;]+? and the price of [^$]+? is \$(\d+) per')
NOW = re.compile(r'currently buys? ([\d.]+) .+? and ([\d.]+) ')
BUNDLE = re.compile(r'^x = ([\d.]+), y = ([\d.]+)$')


def parse(text):
    a, b = (float(x) for x in UF.search(text).groups())
    m = int(INCOME.search(text).group(1).replace(',', ''))
    px, py = (int(x) for x in PRICES.search(text).groups())
    return a, b, m, px, py


def optimum(a, b, m, px, py):
    res = minimize_scalar(lambda x: -(x ** a * ((m - px * x) / py) ** b), bounds=(1e-9, m / px - 1e-9),
                          method='bounded', options={'xatol': 1e-12})
    return res.x, (m - px * res.x) / py


def mrs_fd(a, b, x, y, h=1e-6):
    def U(x, y):
        return x ** a * y ** b
    mux = (U(x + h, y) - U(x - h, y)) / (2 * h)
    muy = (U(x, y + h) - U(x, y - h)) / (2 * h)
    return mux / muy


def test_templates():
    assert len(ct.templates(E)) == 10 * 5 * 3


def test_keys_recomputed():
    s = ElementSampler()
    seen = collections.Counter()
    domains, persp = set(), set()
    for seed in range(900):
        (row,) = build_rows(s.sample_question(E, 81000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        a, b, m, px, py = parse(text)
        xo, yo = optimum(a, b, m, px, py)
        assert len(set(opts)) == len(opts)
        if row['type'] == 'mrs_at_optimum':
            g = mrs_fd(a, b, xo, yo)
            vals = [float(o) for o in opts]
            assert [i for i, v in enumerate(vals) if abs(v - g) <= 0.0051 + 1e-6] == [k], (text, opts, g)
            seen['mrs'] += 1
        elif row['type'] == 'tangency_bundle':
            bundles = [tuple(float(z) for z in BUNDLE.match(o).groups()) for o in opts]
            for x, y in bundles:
                assert abs(px * x + py * y - m) <= 0.005 * (px + py) + 1e-9, (text, opts)
            hits = [i for i, (x, y) in enumerate(bundles) if abs(x - xo) <= 0.006 and abs(y - yo) <= 0.006]
            assert hits == [k], (text, opts, xo, yo)
            assert math.isclose(mrs_fd(a, b, *bundles[k]), px / py, rel_tol=0.01)
            seen['bundle'] += 1
        else:
            x, y = (float(z) for z in NOW.search(text).groups())
            assert abs(px * x + py * y - m) <= 1e-6
            eps = 1e-3
            up = (x + eps) ** a * (y - eps * px / py) ** b
            down = (x - eps) ** a * (y + eps * px / py) ** b
            here = x ** a * y ** b
            key = opts[k]
            if key.startswith('Keep'):
                assert abs(x - xo) <= 0.006 and abs(y - yo) <= 0.006
                assert up <= here + 1e-9 and down <= here + 1e-9
            elif key.startswith('Spend more'):
                assert up > here > down, (text, key)
            else:
                assert key.startswith('Spend less') and down > here > up, (text, key)
            seen[key.split()[0] + key.split()[1]] += 1
        domains.add(row['domain'])
        persp.add(row['perspective'])
    assert seen['mrs'] and seen['bundle'] and seen['Keepthis'] and seen['Spendmore'] and seen['Spendless'], seen
    assert len(domains) == 8
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
