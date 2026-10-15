"""Fixes from the review of October 2026: markets and welfare elements.

Each test checks the whole class of the finding, not just the reviewed question.
"""
import re

import numpy as np
import pytest

from helpers import sample
from autosteer.sampler import ElementSampler
import conventions as tf
from autosteer.filler_tools import tax_incidence_tools as ti


@pytest.fixture(scope='module')
def sampler():
    return ElementSampler()


def texts(sampler, element):
    return sampler._load_template(element).questions


def _floats(s):
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', s.replace(',', ''))]


# Conventions the keys rely on, stated in every template (inverse curves, divisibility, monopoly pricing and the
# competitive benchmark, the exact monopoly price, the rounded equilibrium price).
@pytest.mark.parametrize('element,needles', [
    ('consumer_surplus', [tf.CONSUMER_SURPLUS_CONVENTION]),
    ('deadweight_loss', [tf.DWL_CONVENTION]),
    ('efficient_surplus', [tf.EFFICIENT_SURPLUS_CONVENTION]),
    ('intertemporal_consumption_smoothing', [tf.INTERTEMPORAL_CONVENTION]),
    ('state_based_consumption', [tf.STATE_BASED_CONVENTION]),
    ('welfare_theorem_1', [tf.WT1_PREMISE]),
    ('welfare_theorem_2', [tf.WT2_PREMISE]),
])
def test_conventions_stated(sampler, element, needles):
    q = texts(sampler, element)
    assert len(q) > 100
    for t in q['question_text']:
        for n in needles:
            assert n in t, t
        # the convention comes before the final question, which still ends the text
        assert t.rstrip().endswith(('?', '.', ':')), t


def test_deadweight_loss_price_exact_and_efficient_price_rounded(sampler):
    for t, typ in zip(*[texts(sampler, 'deadweight_loss')[c] for c in ('question_text', 'type')]):
        assert (tf.DWL_PRICE_EXACT in t) == ('{m_price}' in t)
    q = texts(sampler, 'efficient_surplus')
    for t, typ in zip(q['question_text'], q['type']):
        assert (tf.EFFICIENT_SURPLUS_PRICE_ROUNDED in t) == ('{price}' in t)


def test_efficient_surplus_key_is_exact_intersection():
    for s in sample('efficient_surplus', 40, seed=11, type='intersect_given'):
        (ad, bd), (as_, bs) = _floats(s.params['d_curve']), _floats(s.params['s_curve'])
        q = (bd - bs) / (as_ - ad)
        assert float(s.answer) == pytest.approx(0.5 * (bd - bs) * q, abs=0.006)


def test_find_eq_price_options_between_intercepts():
    # every price option is one at which the curves could cross: above supply's and below demand's price at Q = 0
    for typ in ('linear_price', 'quadratic_price'):
        for s in sample('find_eq_price', 60, seed=3, type=typ):
            lo, hi = _floats(s.params['s_func'])[-1], _floats(s.params['d_func'])[-1]
            vals = [float(o) for o in s.options]
            assert all(lo < v < hi for v in vals), (s.params, vals)
            assert len(set(vals)) == 4


def test_intertemporal_options_differ_in_shape():
    # the old fourth option was the key times 1.1; now every option has its own growth path
    for s in sample('intertemporal_consumption_smoothing', 60, seed=5):
        paths = [_floats(o) for o in s.options]
        ratios = [p[-1] / p[0] for p in paths]
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(ratios[i] - ratios[j]) > 1e-3, s.options
        # and every option exhausts the same lifetime wealth (none is ruled out by the budget alone)
        r = float(s.params['interest_rate'])
        pvs = [sum(c / (1 + r) ** t for t, c in enumerate(p)) for p in paths]
        assert max(pvs) - min(pvs) < 0.01 * max(pvs)


def test_state_based_consistent_story():
    q = texts(ElementSampler(), 'state_based_consumption')
    assert not q['question_text'].str.contains('equally likely', case=False).any()
    for s in sample('state_based_consumption', 60, seed=8, type='linear'):
        p = s.params
        assert float(p['pB_x']) > float(p['pA_x']) and float(p['pB_y']) > float(p['pA_y']), p


def test_tax_incidence_no_half_cent_keys():
    r = np.random.default_rng(0)
    for k in range(400):
        mk = ti.MARKETS[k % len(ti.MARKETS)]
        qtype = ti.TYPES[k % 3]
        x = ti.draw(r, mk, qtype)
        for payer in ('sellers', 'buyers'):
            pb, ps, q = ti.equilibrium(x['a'], x['b'], x['c'], x['d'], x['t'], payer)
            v = {'buyer_price': pb * 100, 'seller_price': ps * 100, 'quantity': q}[qtype]
            assert abs(v % 1 - 0.5) > 1e-6, (x, qtype)


def test_welfare_theorems_well_posed(sampler):
    for el in ('welfare_theorem_1', 'welfare_theorem_2'):
        q = texts(sampler, el)
        for t in q['question_text']:
            assert not tf._WT_EXTERNALITY.search(t) and not tf._WT_NO_GOODS.search(t), t
    q = texts(sampler, 'welfare_theorem_2')
    for t, typ in zip(q['question_text'], q['type']):
        assert t.endswith(tf.WT2_WITHOUT if typ == 'without_transfers' else tf.WT2_WITH), t
    for el, yes_type in (('welfare_theorem_1', 'pareto_efficient'), ('welfare_theorem_2', 'with_transfers')):
        for s in sample(el, 20, seed=2):
            assert s.answer == ('yes' if s.type == yes_type else 'no')


# Re-review round 2

def test_find_eq_price_quantity_options_below_choke():
    # every quantity option is one at which the demand price is still positive
    for typ in ('linear_quantity', 'quadratic_quantity'):
        for s in sample('find_eq_price', 60, seed=4, type=typ):
            d = _floats(re.sub(r'\^\d+', '', s.params['d_func']))   # the coefficients, not the exponent of Q^2
            vals = [float(o) for o in s.options]
            assert all(v > 0 and np.polyval(d, v) > 0 for v in vals), (s.params, vals)
            assert len(set(vals)) == 4


def test_non_distortionary_identify_options_same_form():
    from autosteer.release.rows import build_rows
    s = ElementSampler()
    n = 0
    for seed in range(300):
        (row,) = build_rows(s.sample_question('non_distortionary_taxes', 5000 + seed), 'test')
        if row['type'] != 'identify':
            continue
        n += 1
        opts = row['options']
        assert all(o.startswith('A tax of ') for o in opts), opts
        lens = [len(o) for o in opts]
        assert max(lens) < 1.75 * min(lens), opts
        # the key does not echo the stem
        assert 'whatever' not in opts[row['correct_index']] and 'nobody' not in row['question_text'].split('Assume')[0]
    assert n > 50


def test_wt1_final_question_is_about_pareto_improvements(sampler):
    q = texts(sampler, 'welfare_theorem_1')
    for t, typ in zip(q['question_text'], q['type']):
        if typ == 'further_trades':
            assert t.endswith(tf.WT1_FURTHER), t
    assert 'divisible' in tf.WT1_PREMISE


def test_state_based_options_well_formed():
    # every option is a pair of dollar amounts that spends the wealth (not 2-character fragments)
    for s in sample('state_based_consumption', 60, seed=9):
        W = float(s.params['initial_wealth'])
        for o in s.options:
            m = re.fullmatch(r'\((\d+\.\d+), (\d+\.\d+)\)', o)
            assert m, o
            assert abs(float(m.group(1)) + float(m.group(2)) - W) <= 0.02 or s.type == 'cobb_douglas', (o, W)
