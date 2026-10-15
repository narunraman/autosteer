"""Fixes from the review of October 2026, consumer-theory elements (one test per class of problem).

- structural give-aways: the key must not be identifiable from the form of the option set (a coefficient repeated
  only by the key, the only option repeating the stem's numbers, the only option occurring in both parts, always the
  largest number, the only option on the key's scale, the only option of its sign);
- ill-posed wording: which good is x / y, money spent vs quantities, the answer format, rounding and approximation
  conventions, demand shifts in law-of-demand stories.
"""
import collections
import itertools
import re

import pytest

from helpers import sample
from autosteer.sampler import ElementSampler
from autosteer.sampler.consumer_template_fixes import (
    AGG_DEMAND_CONVENTION, AGG_OFFER_CONVENTION, DERIVING_DEMAND_PRICES, HICKSIAN_ROUNDING, LAW_OF_DEMAND_CONVENTION,
    MARGINAL_UTILITY_FORMAT, MRS_FORMAT, names_good)
from autosteer.filler_tools.micro_utility_tools import no_feature_majority


def _numbers(text):
    return [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', text)]


# ----------------------------------------------------------------------------------------------------------------
# deriving_demand: no "c p_x" twin of the key; every coefficient a natural mistake; x and y named; no money spent
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('typ', ['good_1', 'good_2'])
def test_deriving_demand_option_grid(typ):
    var = 'x' if typ == 'good_1' else 'y'
    for s in sample('deriving_demand', 40, seed=11, type=typ):
        m = re.fullmatch(r'U = x\^([\d.]+) \* y\^([\d.]+)', s.params['utility_func'])
        a, b = float(m.group(1)), float(m.group(2))
        own, other = (a, b) if typ == 'good_1' else (b, a)
        budget = float(s.params['budget'])
        coefs = []
        for o in s.options:
            om = re.fullmatch(rf'([\d.]+)/p_{var}', o)
            assert om, (o, s.options)          # one form for every option: no "226.66p_x" twin of the key
            coefs.append(float(om.group(1)))
        want = sorted([own / (a + b) * budget, other / (a + b) * budget, own * budget, other * budget])
        assert all(abs(c - w) <= 0.011 for c, w in zip(sorted(coefs), want)), (coefs, want)
        q = s.question
        assert names_good(q, 'x') and names_good(q, 'y'), q
        assert DERIVING_DEMAND_PRICES in q
        assert not re.search(r'\b[xy] (?:represents|denotes|is) (?:the )?(?:amount of money|amount spent|spending|'
                             r'expenditure)', q), q


# ----------------------------------------------------------------------------------------------------------------
# marginal_utility / mrs_utility: distractors built from the stem's exponents (no number matching), format stated
# ----------------------------------------------------------------------------------------------------------------
def test_marginal_utility_distractors_use_the_stems_exponents():
    for s in sample('marginal_utility', 40, seed=12):
        m = re.search(r'x\^(\d+(?:\.\d+)?)y\^(\d+(?:\.\d+)?)', s.question)
        a, b = float(m.group(1)), float(m.group(2))
        allowed = {a, b, round(a - 1, 2), round(b - 1, 2)}
        for o in s.options:
            assert set(_numbers(o)) <= allowed, (o, allowed)
        # what each option shows: coefficients written, which good's exponent is lowered in each entry, the first
        # coefficient; counting these over the options does not single out the key
        def shown(o):
            ents = re.findall(r'(\d+(?:\.\d+)?)?x\^(-?[\d.]+)y\^(-?[\d.]+)', o)
            low = lambda e: 'x' if float(e[1]) < 0 else 'y' if float(e[2]) < 0 else None
            return (bool(ents[0][0]), low(ents[0]), low(ents[1]), ents[0][0] or None)
        got = [shown(o) for o in s.options]
        assert len(set(s.options)) == 4
        key = got[s.answer_index]
        assert no_feature_majority(key, [g for i, g in enumerate(got) if i != s.answer_index]), s.options
        assert s.question.endswith(MARGINAL_UTILITY_FORMAT)


def test_no_feature_majority():
    key = (True, 'own', False)
    assert not no_feature_majority(key, [(True, 'own', True), (True, 'other', False), (False, 'own', False)])
    assert no_feature_majority(key, [(True, 'other', True), (False, 'own', True), (False, 'other', False)])


def test_mrs_utility_option_grid():
    for s in sample('mrs_utility', 40, seed=13):
        m = re.search(r'x\^(\d+(?:\.\d+)?)y\^(\d+(?:\.\d+)?)', s.question)
        a, b = m.group(1), m.group(2)
        want = {f'(({c1}{v1})/({c2}{v2}), ({c2}{v2})/({c1}{v1}))'
                for (c1, c2), (v1, v2) in itertools.product(((a, b), (b, a)), (('y', 'x'), ('x', 'y')))}
        assert set(s.options) == want, (s.options, want)
        assert s.answer == f'(({a}y)/({b}x), ({b}x)/({a}y))'
        assert s.question.endswith(MRS_FORMAT)


# ----------------------------------------------------------------------------------------------------------------
# hicksian_duality: both parts show the same four options
# ----------------------------------------------------------------------------------------------------------------
def test_hicksian_duality_same_options_in_both_parts():
    s = ElementSampler()
    for seed in range(30):
        q = s.sample_question('hicksian_duality', 9100 + seed)
        p0, p1 = q['parts']
        assert sorted(p0['options']) == sorted(p1['options']), (p0['options'], p1['options'])
        k0 = p0['options'][p0['correct'].index(True)]
        k1 = p1['options'][p1['correct'].index(True)]
        assert k0 == k1


# ----------------------------------------------------------------------------------------------------------------
# aggregation: the key is not always the largest; function distractors on the key's scale (2 x 2 grid)
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('element,typ', [('aggregation_of_consumer_demand', 'two_markets'),
                                         ('aggregation_of_offer_curve', 'numerical')])
def test_aggregation_numeric_key_rank(element, typ):
    ranks = collections.Counter()
    for s in sample(element, 80, seed=14, type=typ):
        vals = [float(o) for o in s.options]
        assert len(set(vals)) == 4
        ranks[sorted(vals).index(vals[s.answer_index])] += 1
    assert set(ranks) == {0, 1, 2, 3}, ranks
    assert max(ranks.values()) <= 0.45 * sum(ranks.values()), ranks


@pytest.mark.parametrize('element,typ,sign', [('aggregation_of_offer_curve', 'offer_curve', '+')])
def test_aggregation_function_grid(element, typ, sign):
    for s in sample(element, 40, seed=15, type=typ):
        parts = [re.fullmatch(rf'([\d.]+) \{sign} ([\d.]+)P', o) for o in s.options]
        assert all(parts), s.options
        A = collections.Counter(p.group(1) for p in parts)
        B = collections.Counter(p.group(2) for p in parts)
        assert sorted(A.values()) == [2, 2] and sorted(B.values()) == [2, 2], s.options
        # key: the group-size-weighted sums of the shown curves
        na, nb = int(s.params['num_a']), int(s.params['num_b'])
        a, ca = _numbers(s.params['offer_curve_a'])
        b, cb = _numbers(s.params['offer_curve_b'])
        k = parts[s.answer_index]
        assert abs(float(k.group(1)) - (na * a + nb * b)) <= 0.011
        assert abs(float(k.group(2)) - (na * ca + nb * cb)) <= 0.011
        assert AGG_OFFER_CONVENTION in s.question


def test_aggregation_consumer_demand_per_member_convention():
    for s in sample('aggregation_of_consumer_demand', 20, seed=16):
        assert AGG_DEMAND_CONVENTION in s.question
        na, nb, p = int(s.params['num_a']), int(s.params['num_b']), float(s.params['price'])
        a, ca = _numbers(s.params['demand_function_a'].replace(' - ', ' '))
        b, cb = _numbers(s.params['demand_function_b'].replace(' - ', ' '))
        assert abs(float(s.answer) - (na * (a - ca * p) + nb * (b - cb * p))) <= 0.011


# ----------------------------------------------------------------------------------------------------------------
# law_of_demand: movement along an unchanged demand curve, the question asks for the new value
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('typ', ['price_increase', 'price_decrease', 'quantity_increase', 'quantity_decrease'])
def test_law_of_demand_well_posed(typ):
    for s in sample('law_of_demand', 60, seed=17, type=typ):
        q = s.question
        assert LAW_OF_DEMAND_CONVENTION in q
        assert not re.search(r'popularity|trend|\brush\b|perform|availab', q, re.I), q
        # round 2: no yields / rates as prices, no quantities sold or used
        assert not re.search(r'\byield|interest rate|\breturns?\b|\bsold\b|\bsales\b|\bwe use\b', q, re.I), q
        last = q.rsplit(LAW_OF_DEMAND_CONVENTION, 1)[1]
        if typ.startswith('quantity'):
            assert re.search(r'\bprice\b', last, re.I), q
            assert not re.search(r'\bcause\b(?!d)|reason|happen|explanation', last, re.I), q
            assert not re.search(r'\b(?:increase|decrease|rise|drop|fall) in (?:the )?demand\b|\bdemand '
                                 r'(?:increased|decreased|rose|fell)', q, re.I), q
            # the direction of the new price is not stated
            assert not re.search(r'price (?:has )?(?:risen|rose|increased|dropped|fallen|fell|decreased)', q, re.I), q
            vals = sorted(float(o) for o in s.options)
            assert all(b - a >= 0.005 * b for a, b in zip(vals, vals[1:])), s.options   # no 923.55 / 924.14
        else:
            assert not re.search(r'change in (?:the )?quantity demanded', q, re.I), q


# ----------------------------------------------------------------------------------------------------------------
# deriving_hicksian_demand: functional-form options differ in the price exponents too; rounding stated
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('level', [1, 2])
def test_hicksian_functional_options_grid(level):
    for s in sample('deriving_hicksian_demand', 30, seed=18, difficulty_level=1 if level == 1 else 2):
        if 'p_' not in s.options[0]:
            continue
        coefs = collections.Counter(o.split('(', 1)[0] for o in s.options)
        exps = collections.Counter(o.split('(', 1)[1] for o in s.options)
        assert sorted(coefs.values()) == [2, 2] and sorted(exps.values()) == [2, 2], s.options
        e1, e2 = ([float(v) for v in re.findall(r'\^\((-?[\d.]+)\)', e)] for e in exps)
        assert max(abs(a - b) for a, b in zip(e1, e2)) >= 0.05, s.options   # not -0.13 vs -0.14
        assert HICKSIAN_ROUNDING in s.question
        assert not re.search(r'\bearn|\bwage|spending on|expenditure on', s.question, re.I), s.question


# ----------------------------------------------------------------------------------------------------------------
# diminishing_marginal_utility: the question asks about the marginal utility (a derivative) of the varied good
# ----------------------------------------------------------------------------------------------------------------
def test_dmu_asks_for_marginal_utility():
    for s in sample('diminishing_marginal_utility', 40, seed=19):
        var = 'y' if s.type.startswith('x_fixed') else 'x'
        assert f'how does the marginal utility of {var} (the partial derivative' in s.question, s.question
        assert names_good(s.question, 'x') and names_good(s.question, 'y')


# ----------------------------------------------------------------------------------------------------------------
# elasticity_of_substitution: the key is never the only option of its sign
# ----------------------------------------------------------------------------------------------------------------
def test_elasticity_of_substitution_sign_not_a_giveaway():
    for s in sample('elasticity_of_substitution', 80, seed=20):
        vals = [float(o) for o in s.options]
        key = vals[s.answer_index]
        assert any((v > 0) == (key > 0) for i, v in enumerate(vals) if i != s.answer_index), s.options


# ----------------------------------------------------------------------------------------------------------------
# conventions stated: Engel coefficients rounded, income-elasticity approximation, rounded tangency bundles,
# interior optimum for quasi-linear utility, price elasticity quantities are quantities demanded
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('element,typ,phrase', [
    ('engel_curves', 'curve_x', 'Coefficients are rounded to two decimals.'),
    ('engel_curves', 'curve_y', 'Coefficients are rounded to two decimals.'),
    ('income_elasticity', 'quantity', 'Use the approximation percentage change in quantity demanded'),
    ('income_elasticity', 'income', 'Use the approximation percentage change in quantity demanded'),
    ('mrs_tangency', 'tangency_bundle', 'spends the whole $'),
    ('substitutes_complements', 'utility', 'x, y and z are all positive at the optimum'),
])
def test_conventions_stated(element, typ, phrase):
    for s in sample(element, 10, seed=21, type=typ):
        assert phrase in s.question, s.question
        assert 'costs exactly' not in s.question or typ != 'tangency_bundle'


def test_price_elasticity_quantity_demanded():
    for s in sample('price_elasticity', 60, seed=22):
        assert not re.search(r'distribut|produc(?:e|ed|es|tion)\b|in stock', s.question, re.I), s.question


def test_price_elasticity_no_single_sign_pair():
    """Round 2: the key was the negative member of the only +/- pair. Now 0 or 2 pairs, and both kinds occur."""
    kinds = collections.Counter()
    for s in sample('price_elasticity', 80, seed=23):
        vals = [float(o) for o in s.options]
        pairs = [(a, b) for a, b in itertools.combinations(vals, 2) if a == -b]
        assert len(pairs) in (0, 2), s.options
        kinds[len(pairs)] += 1
        if not pairs:
            key = vals[s.answer_index]
            assert all((v > 0) == (key > 0) for v in vals), s.options
    assert kinds[0] and kinds[2], kinds


@pytest.mark.parametrize('typ', ['direct', 'quantity', 'income'])
def test_income_elasticity_no_single_sign_pair(typ):
    """Round 3: -key was the only +/- pair. Now 0 or 2 pairs, and both kinds occur."""
    kinds = collections.Counter()
    for s in sample('income_elasticity', 60, seed=24, type=typ):
        vals = [float(o.rstrip('%')) for o in s.options]
        pairs = [(a, b) for a, b in itertools.combinations(vals, 2) if a == -b]
        assert len(pairs) in (0, 2), s.options
        kinds[len(pairs)] += 1
    assert kinds[0] and kinds[2], kinds


def _half_up(d):
    from decimal import Decimal, ROUND_HALF_UP
    return d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


@pytest.mark.parametrize('typ', ['quantity', 'income'])
def test_income_elasticity_key_is_half_up_of_shown_inputs(typ):
    """1.31 x 24.5% = 32.095% was printed +32.09%; keys on a rounding tie are redrawn, so the key is the half-up
    (and the half-even) rounding of the exact value computed from the shown inputs."""
    from decimal import Decimal, getcontext
    getcontext().prec = 40
    for s in sample('income_elasticity', 150, seed=25, type=typ):
        E, k = Decimal(s.params['elasticity']), Decimal(s.params['pct'])
        exact = E * k if typ == 'quantity' else (k / E if k > 0 else -(-k / E))
        assert Decimal(s.answer.rstrip('%')) == _half_up(exact), (s.params, s.answer)


def test_price_elasticity_key_is_half_up_of_shown_inputs():
    from decimal import Decimal, getcontext
    getcontext().prec = 40
    for s in sample('price_elasticity', 150, seed=26):
        p = {k: Decimal(str(s.params[k])) for k in ('start_price', 'end_price', 'start_quantity', 'end_quantity')}
        exact = ((p['end_quantity'] - p['start_quantity']) / p['start_quantity']) / \
                ((p['end_price'] - p['start_price']) / p['start_price'])
        assert Decimal(s.answer) == _half_up(exact), (s.params, s.answer)


def test_near_tie():
    from autosteer.filler_tools.utilities.rounding import near_tie
    assert near_tie(1.31 * 24.5) and near_tie(-0.125) and not near_tie(32.094) and not near_tie(0.12)
