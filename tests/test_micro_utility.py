"""micro_utility (STEER 1.2.a, Optimize over a Discrete Set): pick the highest-utility bundle from a short list.

Recomputed from the utility function shown in the question (Cobb-Douglas x^a * y^b, linear a x + b y or
min(a x, b y)) and the bundles in the options ("x = 4, y = 7"): the key is the unique bundle with the highest
utility, at least 5% above the runner-up; every other option is another listed bundle; the bundles are distinct;
the best bundle does not have at least as much of both goods as every other one; templates that speak of two
bundles have two, the others four; every question says which good is x and which is y.
"""
import re

from helpers import sample
from autosteer.filler_tools.micro_utility_tools import TWO_BUNDLES
from autosteer.sampler.template_fixes import _names_x_and_y

BUNDLE = re.compile(r'^x = (\d+), y = (\d+)$')
NUM = r'(\d+(?:\.\d+)?)'
COEF = r'(\d*)'  # a coefficient of 1 is not written
FAMILIES = [
    (re.compile(rf'x\^{NUM} \* y\^{NUM}'), lambda a, b, x, y: x ** a * y ** b),
    (re.compile(rf'min\({COEF}x, {COEF}y\)'), lambda a, b, x, y: min(a * x, b * y)),
    (re.compile(rf'(?<![\d.]){COEF}x \+ {COEF}y'), lambda a, b, x, y: a * x + b * y),
]


def utility_of(question):
    for k, (pat, fn) in enumerate(FAMILIES):
        found = pat.findall(question)
        if found:
            assert len(set(found)) == 1, found
            assert '1x' not in question and '1y' not in question
            a, b = (float(v) if v else 1.0 for v in found[0])
            return k, lambda x, y: fn(a, b, x, y)
    raise AssertionError(f'no utility function in {question!r}')


def test_micro_utility():
    families = set()
    for s in sample('micro_utility', 400, seed=51):
        family, u = utility_of(s.question)
        families.add(family)
        assert _names_x_and_y(s.question), s.question
        bundles = []
        for o in s.options:
            m = BUNDLE.match(o)
            assert m, o
            bundles.append((int(m.group(1)), int(m.group(2))))
        assert len(set(bundles)) == len(bundles), s.options
        two = TWO_BUNDLES.search(s.question)
        assert len(bundles) == (2 if two else 4), (s.question, s.options)
        utils = [u(x, y) for x, y in bundles]
        best = max(range(len(bundles)), key=lambda i: utils[i])
        assert best == s.answer_index, (s.question, s.options, utils)
        assert all(utils[best] >= 1.05 * v for i, v in enumerate(utils) if i != best), (s.options, utils)
        bx, by = bundles[best]
        assert not all(bx >= x and by >= y for x, y in bundles), s.options
    assert families == {0, 1, 2}
