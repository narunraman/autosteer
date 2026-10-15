"""law_of_demand: exactly one option is on the law-consistent side of the start value.

price up -> only one option below the start quantity; price down -> only
one option above it; quantity up -> only one option below the start price;
quantity down -> only one above. No option equals the start value, and the
start and end values always differ. Values are compared exactly (they are
shown as-is).
"""
import pytest

from helpers import sample, num

TYPES = ['price_increase', 'price_decrease', 'quantity_increase', 'quantity_decrease']


@pytest.mark.parametrize('typ', TYPES)
def test_law_of_demand(typ):
    for s in sample('law_of_demand', 60, seed=3, type=typ):
        p = s.params
        p0, q0 = float(p['start_price']), float(p['start_quantity'])
        vals = [num(o) for o in s.options]
        if typ.startswith('price'):
            p1 = float(p['end_price'])
            assert p1 != p0
            assert (p1 > p0) == (typ == 'price_increase')
            consistent = [i for i, v in enumerate(vals) if (v < q0 if p1 > p0 else v > q0)]
            assert q0 not in vals
        else:
            q1 = float(p['end_quantity'])
            assert q1 != q0
            assert (q1 > q0) == (typ == 'quantity_increase')
            consistent = [i for i, v in enumerate(vals) if (v < p0 if q1 > q0 else v > p0)]
            assert p0 not in vals
        assert consistent == [s.answer_index], (s.question, s.options)
