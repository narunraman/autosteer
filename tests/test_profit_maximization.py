"""profit_maximization: short-run optimum with capital fixed; must be finite.

Closed forms in solvers.py, cross-checked numerically (profit is lower 1%
either side of L*). Profit = revenue - labor cost (no capital price is shown).
Tolerance 0.006 on L* and 1e-6 relative on profit (exact displayed inputs).
"""
import re

import pytest

from helpers import sample, matching
from econ_solvers import cd_short_run_profit_max, leontief_short_run_profit_max


@pytest.mark.parametrize('typ', ['compute_labor', 'compute_profit'])
def test_profit_maximization(typ):
    for s in sample('profit_maximization', 80, seed=8, type=typ):
        p = s.params
        price, w, k = float(p['price']), float(p['cost']), float(p['capital'])
        f = p['p_func']
        cd = re.fullmatch(r'L\^([\d.]+)K\^([\d.]+)', f)
        if cd:
            a, b = float(cd.group(1)), float(cd.group(2))
            assert 0 < a < 1, f  # finite optimum
            L, profit = cd_short_run_profit_max(price, a, b, k, w)
            F = lambda x: x ** a * k ** b
        else:
            m = re.fullmatch(r'min\(L/([\d.]+),K/([\d.]+)\)\^([\d.]+)', f)
            assert m, f
            cl, ck, v = (float(g) for g in m.groups())
            L, profit = leontief_short_run_profit_max(price, cl, ck, v, k, w)
            F = lambda x: min(x / cl, k / ck) ** v
        pi = lambda x: price * F(x) - w * x
        assert pi(L) >= pi(L * 0.99) - 1e-9 and pi(L) >= pi(L * 1.01) - 1e-9
        assert abs(L) < 1e7 and abs(profit) < 1e9
        expected = L if typ == 'compute_labor' else profit
        assert matching(s.options, expected, rel=1e-6) == [s.answer_index], (s.question, s.options, expected)
