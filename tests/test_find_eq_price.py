"""find_eq_price: solve inverse demand = inverse supply; equilibrium must be positive.

Tolerance 0.006: the filler solves with the displayed (rounded) coefficients.
"""
import pytest

from helpers import sample, matching
from econ_solvers import parse_terms, quadratic_roots


def coeffs(expr):
    out = {0: 0.0, 1: 0.0, 2: 0.0}
    for c, pw in parse_terms(expr, 'Q'):
        out[int(pw.get('Q', 0))] += c
    return out


@pytest.mark.parametrize('typ', ['linear_price', 'linear_quantity', 'quadratic_price', 'quadratic_quantity'])
def test_find_eq_price(typ):
    for s in sample('find_eq_price', 60, seed=5, type=typ):
        d, su = coeffs(s.params['d_func']), coeffs(s.params['s_func'])
        a, b, c = (d[2] - su[2], d[1] - su[1], d[0] - su[0])
        if typ.startswith('linear'):
            assert a == 0
            qs = [-c / b]
        else:
            qs = [q for q in quadratic_roots(a, b, c) if q > 0]
        assert len(qs) == 1, (s.params, qs)
        q = qs[0]
        p = d[2] * q * q + d[1] * q + d[0]
        assert q > 0 and p > 0, (s.params, q, p)
        expected = p if typ.endswith('price') else q
        assert matching(s.options, expected) == [s.answer_index], (s.question, s.options, expected)
