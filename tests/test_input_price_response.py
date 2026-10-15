"""input_price_response: the rebuilt Cobb-Douglas type (code templates of the hidden element input_price_cobb; it
replaced comp_stat_cobb's templates in the quality audit) and the rebuilt Leontief type (input_price_leontief) as two
types of one element.

Checks: the merged template set is the two sources' corrected template sets, in order (cobb_douglas first, then
leontief), with only `type` and the question ids changed, for the test and the few-shot pools; a question sampled with
a template id and seed is the question the source element gives for the corresponding template and the same seed;
Cobb-Douglas, independently of the filler: L*(w) maximises p L^a K^0.5 - w L numerically, dL*/dw by central
differences at several wages; exactly the keyed option matches.

Leontief, independently of the filler: a, b, p and r are read from the text; for several wages w the profit
p min(L/a, K/b)^0.5 - w L - r K is maximised numerically over both L and K (a grid over K / L around b / a and a
bounded scalar search over L), dL*/dw is taken by central differences, and every option "dL*/dw = c/(x w + y)^k" is
evaluated at the same wages: exactly the keyed option matches at every wage. Sampling is reproducible from the seed.
"""
import pytest
import re

import numpy as np
from scipy.optimize import minimize_scalar

from helpers import sample
from autosteer.sampler import ElementSampler


E = 'input_price_response'
SOURCES = (('cobb_douglas', 'input_price_cobb'), ('leontief', 'input_price_leontief'))


def test_template_set_is_the_two_sources():
    s = ElementSampler()
    merged = s._load_template(E)
    for attr in ('questions', 'example_questions'):
        m = getattr(merged, attr)
        offset = 0
        for qtype, src in SOURCES:
            q = getattr(s._load_template(src), attr)
            part = m[m['type'] == qtype].sort_values(['question_id', 'sub_id'])
            if q is None or len(q) == 0:  # code templates have no few-shot pool
                assert len(part) == 0
                continue
            assert part['question_text'].tolist() == q.sort_values(['question_id', 'sub_id'])['question_text'].tolist()
            assert sorted(part['question_id'].unique()) == list(range(offset, offset + q['question_id'].nunique()))
            offset += q['question_id'].nunique()


def test_questions_unchanged():
    s = ElementSampler()
    ids = {qtype: sorted(int(i) for i in s._load_template(src).questions['question_id'].unique())
           for qtype, src in SOURCES}
    n_cobb = len(ids['cobb_douglas'])
    for k, seed in enumerate(range(0, 400, 7)):
        qtype, src = SOURCES[k % 2]
        j = k * 37 % len(ids[qtype])
        new_id = j if qtype == 'cobb_douglas' else n_cobb + j
        a = s.sample_question(E, seed, template_id=f'{E}/questions/{new_id}')
        b = s.sample_question(src, seed, template_id=f'{src}/questions/{ids[qtype][j]}')
        assert a['parts'] == b['parts'] and a['metadata']['type'] == qtype


COBB_OPT = re.compile(r'dL\*/dw = (-?[\d,]+)/w\^(\d)')


def cobb_labor(a, K, p, w):
    """argmax_L p L^a K^0.5 - w L, numerically."""
    res = minimize_scalar(lambda L: -(p * L ** a * K ** 0.5 - w * L), bounds=(1e-9, 1e9), method='bounded',
                          options={'xatol': 1e-12})
    return res.x


def test_cobb_key_by_optimisation():
    """Cobb-Douglas type (input_price_cobb), independently of the filler: a, K, p read from the text, L*(w) by
    numerical maximisation, dL*/dw by central differences at several wages; exactly the key matches. Options are
    whole numbers over a power of w (no coefficient rounded to 0, no undefined variable)."""
    for smp in sample(E, 150, seed=78, type='cobb_douglas'):
        a = {'0.5': 0.5, '(2/3)': 2 / 3, '0.75': 0.75}[re.search(r'Q = L\^(\S+) K\^0\.5', smp.question).group(1)]
        K = float(re.search(r'K is fixed at (\d+) hours', smp.question).group(1))
        p = float(re.search(r'for \$(\d+), pay', smp.question).group(1))
        assert len(smp.options) == 4 and len(set(smp.options)) == 4
        match = []
        for o in smp.options:
            g = COBB_OPT.fullmatch(o)
            assert g, o
            c, k = float(g.group(1).replace(',', '')), int(g.group(2))
            assert abs(c) >= 1
            ok = True
            for w in (2.0, 5.0, 11.0):
                h = 1e-4 * w
                d = (cobb_labor(a, K, p, w + h) - cobb_labor(a, K, p, w - h)) / (2 * h)
                ok &= abs(c / w ** k - d) <= 2e-3 * abs(d)
            match.append(ok)
        assert sum(match) == 1 and match.index(True) == smp.answer_index, (smp.question, smp.options, match)


OPT = re.compile(r'dL\*/dw = (-?[\d.]+)/\((\d+)w(?: \+ (\d+))?\)\^(\d)')


def best_labor(a, b, p, w, r):
    """Profit-maximising L, searching over L and over the capital-labor ratio (not assuming K = b L / a)."""
    def profit(L, ratio):
        K = ratio * L
        return p * min(L / a, K / b) ** 0.5 - w * L - r * K

    best = None
    for ratio in np.linspace(0.5, 2.0, 61) * (b / a):
        res = minimize_scalar(lambda L: -profit(L, ratio), bounds=(1e-9, 1e7), method='bounded',
                              options={'xatol': 1e-10})
        if best is None or -res.fun > best[0]:
            best = (-res.fun, res.x, ratio)
    assert abs(best[2] - b / a) < 1e-9  # the best ratio on the grid is exactly b / a (no idle input)
    # refine on the ratio b / a with a tight search
    res = minimize_scalar(lambda L: -profit(L, b / a), bounds=(1e-9, 1e7), method='bounded', options={'xatol': 1e-12})
    return res.x


@pytest.mark.slow
def test_leontief_key_by_optimisation():
    for smp in sample(E, 150, seed=77, type='leontief'):
        m = re.search(r'Q = \(min\(L/(\d+), K/(\d+)\)\)\^0\.5', smp.question)
        a, b = int(m.group(1)), int(m.group(2))
        p = float(re.search(r'for \$(\d+(?:\.\d+)?), pay', smp.question).group(1))
        r = float(re.search(r'and \$(\d+(?:\.\d+)?) per hour of', smp.question).group(1))
        assert len(smp.options) == 4 and len(set(smp.options)) == 4
        match = []
        for o in smp.options:
            g = OPT.fullmatch(o)
            assert g, o
            c, x, y, k = float(g.group(1)), int(g.group(2)), int(g.group(3) or 0), int(g.group(4))
            ok = True
            for w in (2.0, 5.0, 11.0):
                h = 1e-3 * w
                d = (best_labor(a, b, p, w + h, r) - best_labor(a, b, p, w - h, r)) / (2 * h)
                ok &= abs(c / (x * w + y) ** k - d) <= 2e-3 * abs(d)
            match.append(ok)
        assert sum(match) == 1 and match.index(True) == smp.answer_index, (smp.question, smp.options, match)


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
