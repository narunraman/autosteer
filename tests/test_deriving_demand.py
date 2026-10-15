"""deriving_demand: Cobb-Douglas Marshallian demand, coefficient x income / price.

For U = x^a y^b and income m: x = a/(a+b) m / p_x, y = b/(a+b) m / p_y.
good_1 asks for x, good_2 for y. Tolerance 0.011 on the coefficient (the
shown a, b, m are exact).
"""
import re

import pytest

from helpers import sample
from econ_solvers import cobb_douglas_marshallian
from conventions import y_good_name

OPT = re.compile(r'^(-?[\d.]+)/p_([xy])$')


@pytest.mark.parametrize('typ', ['good_1', 'good_2'])
def test_deriving_demand(typ):
    for s in sample('deriving_demand', 60, seed=4, type=typ):
        m = re.fullmatch(r'U = x\^([\d.]+) \* y\^([\d.]+)', s.params['utility_func'])
        a, b = float(m.group(1)), float(m.group(2))
        good = 1 if typ == 'good_1' else 2
        var = 'x' if good == 1 else 'y'
        c = cobb_douglas_marshallian(a, b, float(s.params['budget']), good)
        hits = []
        for i, o in enumerate(s.options):
            om = OPT.match(o)
            if om and om.group(2) == var and abs(float(om.group(1)) - c) <= 0.011:
                hits.append(i)
        assert hits == [s.answer_index], (s.question, s.options, c)
        if good == 2:
            # the question must ask for the good called y
            y = y_good_name(s.question)
            assert y and y.lower() in s.question.rsplit('demand function', 1)[1].lower(), s.question
