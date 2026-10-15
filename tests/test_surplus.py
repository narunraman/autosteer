"""consumer_surplus, producer_surplus, efficient_surplus: triangle areas under linear curves.

Curves are shown as inverse functions P = aQ + b. Recomputed from the displayed coefficients and price:
  consumer surplus at price p (demand a < 0):  Q = (p - b) / a,  CS = (b - p) Q / 2
  producer surplus at price p (supply a > 0):  Q = (p - b) / a,  PS = (p - b) Q / 2
  total surplus at the competitive equilibrium: Q* = (b_d - b_s) / (a_s - a_d),  TS = (b_d - b_s) Q* / 2
(for intersect_given the stated price must be the equilibrium price to 2 decimals). Tolerance 0.006: the fillers
compute from the displayed (rounded) values. Options are distinct numbers, exactly one matches.
"""
import re

import pytest

from helpers import sample, matching, num

CURVE = re.compile(r'^(-?\d+(?:\.\d+)?)Q \+ (\d+(?:\.\d+)?)$')


def curve(text):
    m = CURVE.match(text)
    assert m, text
    return float(m.group(1)), float(m.group(2))


def distinct_numbers(options):
    vals = [num(o) for o in options]
    assert len(set(options)) == len(options) == 4 and len(set(vals)) == 4, options


def test_consumer_surplus():
    for s in sample('consumer_surplus', 200, seed=31):
        a, b = curve(s.params['d_function'])
        p = float(s.params['price'])
        assert s.params['d_function'] in s.question and a < 0 and 0 <= p <= b
        cs = (b - p) * ((p - b) / a) / 2
        distinct_numbers(s.options)
        assert matching(s.options, cs) == [s.answer_index], (s.question, s.options, cs)


def test_producer_surplus():
    for s in sample('producer_surplus', 200, seed=32):
        a, b = curve(s.params['s_curve'])
        p = float(s.params['price'])
        assert s.params['s_curve'] in s.question and a > 0 and p >= b
        ps = (p - b) * ((p - b) / a) / 2
        distinct_numbers(s.options)
        assert matching(s.options, ps) == [s.answer_index], (s.question, s.options, ps)


@pytest.mark.parametrize('typ', ['compute intersect', 'intersect_given', 'maximize_surplus'])
def test_efficient_surplus(typ):
    for s in sample('efficient_surplus', 120, seed=33, type=typ):
        ad, bd = curve(s.params['d_curve'])
        as_, bs = curve(s.params['s_curve'])
        assert s.params['d_curve'] in s.question and s.params['s_curve'] in s.question
        assert ad < 0 < as_ and bd > bs
        q = (bd - bs) / (as_ - ad)
        if typ == 'intersect_given':
            assert abs(float(s.params['price']) - (ad * q + bd)) <= 0.005 + 1e-9, s.params
        ts = (bd - bs) * q / 2
        distinct_numbers(s.options)
        assert matching(s.options, ts) == [s.answer_index], (s.question, s.options, ts)
