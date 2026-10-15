"""hicksian_duality: Marshallian and Hicksian demand recomputed by numerical optimisation from the question text.

From the released rows alone: both parts state U(x, y) = x^a * y^b and the two prices; part 0 states the income,
part 1 a utility level. Part 0's key (units of x, units of y, or the bundle) is checked against the maximum of U on
the budget line found by scipy (bounded 1-D search over the amount of x); part 1's against the cheapest bundle on the
indifference curve of the stated utility (bounded 1-D search over x, with y solved from U = u). Checks: the stated
utility is the part-0 maximum rounded to 2 decimals; exactly one option matches the recomputed value and it is the
marked one; the part-1 key is the part-0 key; four distinct options, at least 10% apart (bundles: in one good); every
type, domain and perspective occurs; sampling is reproducible from the seed.
"""
import collections
import math
import re

from scipy.optimize import minimize_scalar

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'hicksian_duality'
UF = re.compile(r'U\(x, y\) = x\^([\d.]+) \* y\^([\d.]+),')
PRICES = re.compile(r'is \$(\d+) per [^;]+? and the price of [^$]+? is \$(\d+) per')
INCOME = re.compile(r'\$([\d,]+) a \w+ to spend')
UTIL = re.compile(r'utility level of ([\d.]+) per')
BUNDLE = re.compile(r'^x = ([\d.]+), y = ([\d.]+)$')


def numeric_marshall(a, b, px, py, m):
    res = minimize_scalar(lambda x: -(a * math.log(x) + b * math.log((m - px * x) / py)),
                          bounds=(1e-9, m / px - 1e-9), method='bounded', options={'xatol': 1e-12})
    x = res.x
    return x, (m - px * x) / py


def numeric_hicks(a, b, px, py, u):
    def y_of(x):
        return (u / x ** a) ** (1 / b)
    res = minimize_scalar(lambda lx: px * math.exp(lx) + py * y_of(math.exp(lx)), bounds=(-12, 12),
                          method='bounded', options={'xatol': 1e-13})
    x = math.exp(res.x)
    return x, y_of(x)


def parse_opt(o):
    m = BUNDLE.match(o)
    return (float(m.group(1)), float(m.group(2))) if m else float(o)


def test_templates():
    ts = ct.templates(E)
    assert len(ts) == 10 * 5 * 3
    for t in ts:
        assert '{income}' in t.parts[0] and '{income}' not in t.parts[1]
        assert '{utility}' in t.parts[1] and '{utility}' not in t.parts[0]


def test_keys_by_numerical_optimisation():
    s = ElementSampler()
    types, domains, persp = collections.Counter(), set(), set()
    for seed in range(600):
        rows = build_rows(s.sample_question(E, 52000 + seed), 'test')
        assert [r['grading'] for r in rows] == ['answer', 'answer']
        t0, t1 = rows[0]['question_text'], rows[1]['question_text']
        a, b = (float(x) for x in UF.search(t0).groups())
        assert UF.search(t1).groups() == UF.search(t0).groups()
        px, py = (int(x) for x in PRICES.search(t0).groups())
        assert PRICES.search(t1).groups() == PRICES.search(t0).groups()
        m = int(INCOME.search(t0).group(1).replace(',', ''))
        u = float(UTIL.search(t1).group(1))
        x0, y0 = numeric_marshall(a, b, px, py, m)
        assert abs(u - x0 ** a * y0 ** b) <= 0.0051, (u, x0 ** a * y0 ** b)
        x1, y1 = numeric_hicks(a, b, px, py, u)
        qtype = rows[0]['type']
        for row, (x, y) in ((rows[0], (x0, y0)), (rows[1], (x1, y1))):
            opts = [parse_opt(o) for o in row['options']]
            assert len(set(row['options'])) == 4
            if qtype == 'bundle':
                hits = [i for i, o in enumerate(opts) if abs(o[0] - x) <= 0.006 and abs(o[1] - y) <= 0.006]
                for i in range(4):
                    for j in range(i):
                        assert any(abs(opts[i][k] - opts[j][k]) >= 0.1 * max(opts[i][k], opts[j][k]) - 0.01
                                   for k in (0, 1)), row['options']
            else:
                v = x if qtype == 'good_x' else y
                hits = [i for i, o in enumerate(opts) if abs(o - v) <= 0.006]
                for i in range(4):
                    for j in range(i):
                        assert abs(opts[i] - opts[j]) >= 0.1 * max(opts[i], opts[j]) - 0.01, row['options']
            assert hits == [row['correct_index']], (row['question_text'], row['options'], x, y)
        assert rows[0]['options'][rows[0]['correct_index']] == rows[1]['options'][rows[1]['correct_index']]
        types[qtype] += 1
        domains.add(rows[0]['domain'])
        persp.add(rows[0]['perspective'])
    assert set(types) == {'good_x', 'good_y', 'bundle'}
    assert domains == {'consumer_goods', 'entertainment', 'education', 'sports', 'technology', 'medical',
                       'environmental_policy', 'finance'}
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_closed_forms_match_numerical():
    from autosteer.filler_tools.hicksian_duality_tools import hicksian, marshallian
    for a, b, px, py, m in [(0.3, 0.5, 4, 9, 300), (0.75, 0.4, 13, 2, 560), (0.2, 0.8, 7, 7.5, 120)]:
        x, y = marshallian(a, b, px, py, m)
        xn, yn = numeric_marshall(a, b, px, py, m)
        assert math.isclose(x, xn, rel_tol=1e-6) and math.isclose(y, yn, rel_tol=1e-6)
        xh, yh = hicksian(a, b, px, py, x ** a * y ** b)
        assert math.isclose(xh, x, rel_tol=1e-9) and math.isclose(yh, y, rel_tol=1e-9)
        xhn, yhn = numeric_hicks(a, b, px, py, x ** a * y ** b)
        assert math.isclose(xhn, x, rel_tol=1e-6) and math.isclose(yhn, y, rel_tol=1e-6)


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
