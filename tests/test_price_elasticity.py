"""price_elasticity: wording direction matches the numbers; value = %dQ / %dP.

Elasticity uses the starting point as base (the filler's definition), from
the displayed 2-decimal prices and quantities; tolerance 0.006.
"""
import re

import pytest

from helpers import sample, matching
from econ_solvers import elasticity_arc_simple

UP = re.compile(r'\b(increase[sd]?|increasing|rise[sn]?|rising|rose|grow[sn]?|grew|jump(s|ed)?|climb(s|ed)?|surge[sd]?|up|soar(s|ed)?|boost(s|ed)?|expand(s|ed)?)\b')
DOWN = re.compile(r'\b(decrease[sd]?|decreasing|drop(s|ped)?|fall(s|en)?|fell|decline[sd]?|reduce[sd]?|down|shrink(s)?|shrank|dip(s|ped)?|plummet(s|ed)?|slump(s|ed)?)\b')


def quantity_words(question, q0):
    # the clause right before "from <start quantity>"
    i = question.find(f'from {q0} to')
    head = question[max(0, i - 45):i].lower()
    head = re.split(r'[,.;]', head)[-1]
    return bool(UP.search(head)), bool(DOWN.search(head))


@pytest.mark.parametrize('typ', ['price_increase', 'price_decrease'])
def test_price_elasticity(typ):
    for s in sample('price_elasticity', 80, seed=7, type=typ):
        p = s.params
        p0, p1, q0, q1 = (float(p[k]) for k in ('start_price', 'end_price', 'start_quantity', 'end_quantity'))
        assert (p1 > p0) == (typ == 'price_increase')
        # wording: the verb before "from <q0>" must match the quantity change
        up, down = quantity_words(s.question, p['start_quantity'])
        if q1 > q0:
            assert not (down and not up), s.question
        else:
            assert not (up and not down), s.question
        e = elasticity_arc_simple(p0, p1, q0, q1)
        assert matching(s.options, e) == [s.answer_index], (s.question, s.options, e)
