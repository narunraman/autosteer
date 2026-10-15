"""diminishing_marginal_product: recompute MP at start/end from the shown function.

Tolerance: the function, start, end and fixed values are shown exactly, and
the filler computes from those same values, so options must match the
recomputed MP to 0.011 (2-decimal rounding).
"""
import re

import pytest

from helpers import sample
from econ_solvers import parse_terms, diff_terms, eval_terms

TYPES = [f'{fam}_{fix}' for fam in ('linear', 'cobb_douglas', 'quadratic')
         for fix in ('capital_fixed', 'labor_fixed')]
OPT = re.compile(r'marginal product of (L|K) (increases from|decreases from|remains constant at) '
                 r'(-?[\d.]+)(?: to (-?[\d.]+))?')


def close(a, b):
    return abs(a - b) <= 0.011 + 1e-6 * abs(b)


@pytest.mark.parametrize('typ', TYPES)
def test_dmp(typ):
    for s in sample('diminishing_marginal_product', 40, seed=1, type=typ):
        p = s.params
        f = p['p_func']
        terms = parse_terms(f, 'LK')
        var = 'L' if 'capital_fixed' in typ else 'K'
        other = 'K' if var == 'L' else 'L'
        mp = diff_terms(terms, var)
        mp0 = eval_terms(mp, {var: p['start'], other: p['fixed']})
        mp1 = eval_terms(mp, {var: p['end'], other: p['fixed']})

        if typ.startswith('quadratic'):
            # full function string: all five terms are shown
            assert {'L^2', 'K^2', 'LK'} <= set(re.findall(r'L\^2|K\^2|LK', f)), f
        if typ.startswith('linear'):
            assert mp0 == mp1
        else:
            # generated functions must have truly diminishing, positive MP
            assert mp1 < mp0 and mp1 > 0, (f, mp0, mp1)
            if typ.startswith('cobb_douglas'):
                exp = dict(terms[0][1])[var]
                assert 0 < exp < 1

        matches = []
        for i, o in enumerate(s.options):
            m = OPT.search(o)
            assert m, o
            assert m.group(1) == var
            kind, a, b = m.group(2), float(m.group(3)), m.group(4)
            if not close(a, mp0):
                continue
            if kind == 'remains constant at' and mp0 == mp1:
                matches.append(i)
            elif kind == 'decreases from' and mp1 < mp0 and close(float(b), mp1):
                matches.append(i)
            elif kind == 'increases from' and mp1 > mp0 and close(float(b), mp1):
                matches.append(i)
        assert matches == [s.answer_index], (s.question, s.options, mp0, mp1)
