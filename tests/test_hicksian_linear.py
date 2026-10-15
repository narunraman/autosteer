"""deriving_hicksian_demand, linear-utility branch (three goods, numeric prices).

Independent corner-solution check: with U = a1 x1 + a2 x2 + a3 x3 and prices p, reaching utility u at least cost
means buying only the good with the highest a_i / p_i, so the compensated demand for the asked good k is u / a_k
if k is that good and 0 otherwise. Computed here from the numbers shown in the question. Also: the key is the
only option equal to that value, options are distinct and well separated, 0 is always an option, and the key is
0 in roughly half the questions (the asked good is the one bought about half the time).
"""
import re

from helpers import sample

N = 1500


def linear_samples():
    out = []
    for s in sample('deriving_hicksian_demand', N, seed=31, difficulty_level=1):
        if 'price_3' in s.params and str(s.params['utility_func']).startswith('f(x) ='):
            out.append(s)
    return out


_CACHE = []


def got():
    if not _CACHE:
        _CACHE.extend(linear_samples())
    return _CACHE


def test_branch_is_sampled():
    assert len(got()) > 100


def test_key_is_corner_solution():
    for s in got():
        a = [float(x) for x in re.findall(r'([\d.]+) \* x[123]', s.params['utility_func'])]
        p = [float(s.params[f'price_{i}']) for i in (1, 2, 3)]
        u = float(s.params['utility_level'])
        assert len(a) == 3
        b = max(range(3), key=lambda i: a[i] / p[i])
        k = int(s.metadata['type'].split('_')[1]) - 1
        expect = round(u / a[k], 2) if k == b else 0.0
        hits = [i for i, o in enumerate(s.options) if abs(float(o) - expect) <= 0.011]
        assert hits == [s.answer_index], (s.question, s.options, expect)


def test_options_well_formed():
    zero_key = 0
    for s in got():
        vals = [float(o) for o in s.options]
        assert len(vals) == 4 and sum(s.correct) == 1
        assert len(set(s.options)) == 4
        assert 0.0 in vals, s.options
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(vals[i] - vals[j]) > max(0.015, 0.03 * max(abs(vals[i]), abs(vals[j]))) - 1e-9, s.options
        zero_key += float(s.answer) == 0.0
    share = zero_key / len(got())
    assert 0.35 <= share <= 0.65, share
