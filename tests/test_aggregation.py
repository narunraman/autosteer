"""aggregation_of_consumer_demand and aggregate_demand.

aggregation_of_consumer_demand: total demand = n_a D_a(P) + n_b D_b(P) with
each segment's demand required to be positive at the asked price (the
filler redraws otherwise; no truncation). Tolerance 0.006 (exact inputs).

aggregate_demand has no committed template, so its filler is called
directly: options must be a single (not double-wrapped) list per part.
"""
import re

import pytest

from helpers import sample, matching
from autosteer.filler_tools import function_registry


def linear_demand(s):
    m = re.fullmatch(r'([\d.]+) - ([\d.]+)P', s)
    return float(m.group(1)), float(m.group(2))


def test_aggregation_of_consumer_demand():
    for s in sample('aggregation_of_consumer_demand', 80, seed=6):
        p = s.params
        a, ca = linear_demand(p['demand_function_a'])
        b, cb = linear_demand(p['demand_function_b'])
        price = float(p['price'])
        da, db = a - ca * price, b - cb * price
        assert da > 0 and db > 0, p
        assert int(p['num_consumers']) == int(p['num_a']) + int(p['num_b'])
        total = int(p['num_a']) * da + int(p['num_b']) * db
        assert matching(s.options, total) == [s.answer_index], (s.question, s.options, total)


@pytest.mark.parametrize('typ', ['simple_labor', 'simple_capital', 'homo_labor', 'homo_capital'])
def test_aggregate_demand_not_double_wrapped(typ):
    fn = function_registry.get('aggregate_demand')
    for _ in range(5):
        out = fn({'type': typ, 'domain': 'consumer_goods', 'difficulty_level': 0, 'tags': ''})
        opts, ans = out['options_list'], out['correct_answer']
        assert len(opts) == 1 and len(ans) == 1
        assert all(not isinstance(o, list) for o in opts[0]), opts
        assert sum(bool(x) for x in ans[0]) == 1


def test_aggregation_of_labor_demand():
    """Total labor demand of n_a firms with D_a(P) = a - c_a P and n_b with D_b(P) = b - c_b P is
    (n_a a + n_b b) - (n_a c_a + n_b c_b) P; the key is that function (both coefficients to 2 decimals)."""
    func = re.compile(r'^(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)P$')
    for s in sample('aggregation_of_labor_demand', 200, seed=7):
        p = s.params
        a, ca = linear_demand(p['demand_function_a'])
        b, cb = linear_demand(p['demand_function_b'])
        na, nb = int(p['num_a']), int(p['num_b'])
        assert int(p['num_firms']) == na + nb
        for k in ('demand_function_a', 'demand_function_b', 'num_a', 'num_b'):
            assert str(p[k]) in s.question, (k, s.question)
        intercept, slope = na * a + nb * b, na * ca + nb * cb
        assert len(set(s.options)) == len(s.options) == 4, s.options
        hits = []
        for i, o in enumerate(s.options):
            m = func.match(o)
            assert m, o
            if abs(float(m.group(1)) - intercept) < 0.006 and abs(float(m.group(2)) - slope) < 0.006:
                hits.append(i)
        assert hits == [s.answer_index], (s.question, s.options, intercept, slope)
