"""supply_elasticity: e = Q'(P) P / Q(P) for the displayed supply Q(P).

The derivative is taken numerically (central difference) as an independent
check. The type label must match the displayed family. Tolerance 0.006
(exact displayed inputs).
"""
import re

import pytest

from helpers import sample, matching
from econ_solvers import parse_terms, eval_terms


@pytest.mark.parametrize('typ', ['Linear_supply_given', 'Quadratic_supply_given'])
def test_supply_elasticity(typ):
    got = sample('supply_elasticity', 80, seed=9, type=typ)
    for s in got:
        f = s.params['s_func']
        assert f.startswith('Q = '), f
        terms = parse_terms(f[4:], 'P')
        has_sq = any(pw.get('P', 0) == 2 for _, pw in terms)
        assert has_sq == (typ == 'Quadratic_supply_given'), (typ, f)
        assert 'marginal cost' not in s.question.lower()
        price = float(s.params['price'])
        h = 1e-6
        q = eval_terms(terms, {'P': price})
        dq = (eval_terms(terms, {'P': price + h}) - eval_terms(terms, {'P': price - h})) / (2 * h)
        e = dq * price / q
        assert matching(s.options, e, abs_=0.006) == [s.answer_index], (s.question, s.options, e)
