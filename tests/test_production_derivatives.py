"""marginal_production and output_elasticity of a production function F(L, K).

marginal_production (Cobb-Douglas F = L^a K^b): MP_L = a L^(a-1) K^b, MP_K = b L^a K^(b-1). The options are
expressions c L^x K^y; the key must be the one whose (c, x, y) is the derivative asked for, recomputed from the
exponents shown in the question.

output_elasticity, with capital fixed at the stated K0:
  Cobb-Douglas F = L^a K^b:  elasticity of output w.r.t. L = a (a number; no other option equal to it);
  linear F = c L + d K:      elasticity = c L / (c L + d K0), an expression in L; the key is that expression
                             with the constant d K0 recomputed (to 2 decimals) from the shown d and K0 > 0.
"""
import re

import pytest

from helpers import sample, num

NUMBER = r'-?\d+(?:\.\d+)?'
CD = re.compile(rf'^L\^({NUMBER})K\^({NUMBER})$')
TERM = re.compile(rf'^({NUMBER})L\^({NUMBER})K\^({NUMBER})$')
LINEAR = re.compile(rf'^({NUMBER})L \+ ({NUMBER})K$')
RATIO = re.compile(rf'^({NUMBER})L/\(({NUMBER})L\+({NUMBER})\)$')


@pytest.mark.parametrize('typ', ['marginal_labor', 'marginal_capital'])
def test_marginal_production(typ):
    for s in sample('marginal_production', 200, seed=41, type=typ):
        m = CD.match(s.params['p_func'])
        assert m and s.params['p_func'] in s.question, s.params
        a, b = float(m.group(1)), float(m.group(2))
        want = (a, a - 1, b) if typ == 'marginal_labor' else (b, a, b - 1)
        terms = []
        for o in s.options:
            t = TERM.match(o)
            assert t, o
            terms.append(tuple(float(x) for x in t.groups()))
        assert len(set(terms)) == len(terms) == 4, s.options
        hits = [i for i, t in enumerate(terms) if all(abs(x - y) < 0.006 for x, y in zip(t, want))]
        assert hits == [s.answer_index], (s.question, s.options, want)


def test_output_elasticity_cobb_douglas():
    for s in sample('output_elasticity', 200, seed=42, type='Cobb_Douglas'):
        m = CD.match(s.params['p_function'])
        assert m and s.params['p_function'] in s.question, s.params
        a = float(m.group(1))
        vals = [num(o) for o in s.options]
        assert len(set(vals)) == len(vals) == 4, s.options
        hits = [i for i, v in enumerate(vals) if abs(v - a) < 0.006]
        assert hits == [s.answer_index], (s.question, s.options, a)


def test_output_elasticity_linear():
    for s in sample('output_elasticity', 200, seed=43, type='linear'):
        m = LINEAR.match(s.params['p_function'])
        assert m and s.params['p_function'] in s.question, s.params
        c, d = float(m.group(1)), float(m.group(2))
        k0 = float(s.params['fixed_val'].split('=')[1])
        assert s.params['fixed_val'] in s.question and k0 > 0, s.params
        assert len(set(s.options)) == len(s.options) == 4, s.options
        const = round(d * k0, 2)
        hits = []
        for i, o in enumerate(s.options):
            r = RATIO.match(o)
            if r and abs(float(r.group(1)) - c) < 1e-9 and abs(float(r.group(2)) - c) < 1e-9 \
                    and abs(float(r.group(3)) - const) < 0.006:
                hits.append(i)
        assert hits == [s.answer_index], (s.question, s.options, c, const)
        # the constant 1 is the elasticity only when K0 = 0, which is excluded above
