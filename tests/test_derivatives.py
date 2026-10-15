"""derivatives: the marked option equals d/dx of the shown polynomial.

Covers both the test templates and the few-shot example templates: the
published example rows showed a hard-coded equation while the options were
drawn for a different random polynomial. Coefficients compared to 0.006.
"""
import pytest

from helpers import sample
from econ_solvers import poly_x


def derivative(p):
    return {k - 1: k * c for k, c in p.items() if k > 0}


def same_poly(p, q):
    keys = set(p) | set(q)
    return all(abs(p.get(k, 0.0) - q.get(k, 0.0)) <= 0.006 for k in keys)


@pytest.mark.parametrize('pool', ['questions', 'examples'])
@pytest.mark.parametrize('typ', ['linear', 'quadratic', 'cubic'])
def test_derivatives(typ, pool):
    for s in sample('derivatives', 40, seed=10, type=typ, pool=pool):
        eq = s.params['equation']
        assert eq in s.question
        want = derivative(poly_x(eq))
        hits = [i for i, o in enumerate(s.options) if same_poly(poly_x(o), want)]
        assert hits == [s.answer_index], (s.question, s.options, want)
