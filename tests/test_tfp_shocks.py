"""tfp_shocks: long-run competitive price = unit cost of a CRS Cobb-Douglas firm.

P = (1/A) (w/a)^a (r/(1-a))^(1-a), so (exact, not the first-order approximations -x and a*x)
  productivity +-x%:  %dP = 1/(1 +- x) - 1      (= -x/(1+x) for an increase)
  wage +-x%:          %dP = (1 +- x)^a - 1      (a = labor share)
Recomputed from the displayed shock and exponent; tolerance 0.006 (points). Every question states the
competitive long-run assumption. The capital-stock types were dropped (decision of 3 October 2026).
"""
import re

import pytest

from helpers import sample, num, matching, get_sampler
from econ_solvers import unit_cost_price_change_productivity, unit_cost_price_change_wage

TYPES = ['increase_productivity', 'decrease_productivity', 'increase_labor', 'decrease_labor']


def test_only_productivity_and_wage_types():
    assert sorted(get_sampler()._manifest['elements']['tfp_shocks']['types']) == sorted(TYPES)


@pytest.mark.parametrize('typ', TYPES)
def test_tfp_shocks(typ):
    for s in sample('tfp_shocks', 100, seed=11, type=typ):
        p = s.params
        sign = 1 if typ.startswith('increase') else -1
        a = float(re.search(r'L\^([\d.]+)', p['prod_fn']).group(1))
        b = float(re.search(r'K\^([\d.]+)', p['prod_fn']).group(1))
        assert abs(a + b - 1) < 1e-9  # constant returns
        assert p['prod_fn'] in s.question
        assert 'in the long run the price of output equals the unit cost of production' in s.question, s.question
        if typ.endswith('productivity'):
            shock = p[f'productivity_{typ.split("_")[0]}']
            expected = unit_cost_price_change_productivity(sign * num(shock))
        else:
            shock = p[f'labor_price_{typ.split("_")[0]}']
            expected = unit_cost_price_change_wage(sign * num(shock), a)
        assert str(shock) in s.question
        assert all(o.endswith('%') for o in s.options)
        assert len({num(o) for o in s.options}) == len(s.options) == 4, s.options
        assert matching(s.options, expected) == [s.answer_index], (s.question, s.options, expected)
        # direction: productivity up / wage down -> price falls
        falls = (typ == 'increase_productivity') or (typ == 'decrease_labor')
        assert (num(s.answer) < 0) == falls
