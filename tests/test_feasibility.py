"""feasibility: repeated prisoner's dilemmas, keys recomputed from the question text (design of October 2026).

The story is read from its own payoff sentence ("If both X, each earns $R per period; if both Y, each earns $P ...;
if one Y' while the other X', the one that Y' earns $T and the other earns $S ..."); the test checks that the
phrases name the same two actions and that, by these labels, the game is a genuine prisoner's dilemma: T > R > P > S
and 2R > T + S (the property tests/test_multi_agent_regen.py::test_story_is_pd checked on the pipeline templates).

folk: the option pairs are read and classified with exact arithmetic: feasible iff the point lies in one of the
    triangles spanned by three of the four stage payoff pairs (Caratheodory), individually rational iff both payoffs
    exceed P. Exactly the keyed option is both; the distractors include an infeasible and a not individually rational
    one; no discount factor is stated.
discount: the discount factor is read; each option is turned back into its periodic plan; the plan is checked by
    simulating discounted sums over 1,200 periods (floats, independent of the generator's closed form): at each
    period of the cycle, each player's continuation value must be at least its best one-period deviation followed
    by mutual defection forever. Exactly the keyed plan passes, and every option has the same verdict at the stated
    factor +- 0.02. Every keyable plan is the key somewhere.
"""
import collections
import re
from fractions import Fraction

import pytest

from helpers import sample

pytestmark = pytest.mark.slow

N = 400
PAY = re.compile(r'If both (.+?), each ([a-z ]+?) \$([\d,]+) per (\w+); if both (.+?), each \2 \$([\d,]+) per \4; '
                 r'if one (.+?) while the other (.+?), the one that (.+?) \2 \$([\d,]+) and the other \2 \$([\d,]+) '
                 r'per \4\.')


def usd(s):
    return int(s.replace(',', ''))


def _norm(phrase):
    out = []
    for w in phrase.split():
        w = {'its': 'their', 'itself': 'themselves', 'prices': 'price', 'quotas': 'quota'}.get(w, w)
        if w.endswith('es') and w[:-2] in ('overfish', 'launch'):
            w = w[:-2]
        elif w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        out.append(w)
    return ' '.join(out)


def read_game(q):
    m = PAY.search(q)
    assert m, q
    c_pl, _, r, period, d_pl, p, d_sg, c_sg, d_sg2, t, s = m.groups()
    assert d_sg == d_sg2
    assert _norm(d_sg) == _norm(d_pl) and _norm(c_sg) == _norm(c_pl), (c_pl, c_sg, d_pl, d_sg)
    g = {'R': usd(r), 'P': usd(p), 'T': usd(t), 'S': usd(s)}
    return g, period, (c_pl, c_sg, d_pl, d_sg)


@pytest.fixture(scope='module')
def got():
    return sample('feasibility', N, seed=8080)


def test_story_is_pd(got):
    for s in got:
        g, _, _ = read_game(s.question)
        assert g['T'] > g['R'] > g['P'] > g['S'] and 2 * g['R'] > g['T'] + g['S'], (g, s.question)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def in_triangle(p, a, b, c):
    d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def feasible(g, x, y):
    pts = [(g['R'], g['R']), (g['S'], g['T']), (g['T'], g['S']), (g['P'], g['P'])]
    return any(in_triangle((x, y), *[pts[i] for i in range(4) if i != k]) for k in range(4))


def test_folk_keys(got):
    kinds = collections.Counter()
    n = 0
    for s in got:
        if s.type != 'folk':
            continue
        n += 1
        assert 'discount' not in s.question.lower()
        g, _, _ = read_game(s.question)
        good = []
        for o in s.options:
            m = re.fullmatch(r'\w+ A: \$([\d,]+), \w+ B: \$([\d,]+)', o)
            assert m, o
            x, y = usd(m.group(1)), usd(m.group(2))
            f, ir = feasible(g, x, y), x > g['P'] and y > g['P']
            good.append(f and ir)
            kinds['infeasible' if not f and ir else 'not_ir' if f and not ir else 'key' if f else 'neither'] += 1
            if not f:
                assert max(x, y) < g['T'], o     # infeasible only jointly (re-review)
        assert good == list(s.correct), (s.question, s.options, s.correct)
    assert n > 100 and kinds['infeasible'] > n and kinds['not_ir'] > n, kinds


def path_of(option, A, B, acts):
    c_pl, c_sg, d_pl, d_sg = acts
    C, D = 'C', 'D'
    if option.startswith('They take turns'):
        assert f'{A} {d_sg} while {B} {c_sg}' in option
        return [(D, C), (C, D)]
    if option.startswith('A three-'):
        assert f'first both {c_pl}; then {A} {d_sg} while {B} {c_sg}; then {B} {d_sg} while {A} {c_sg}' in option
        return [(C, C), (D, C), (C, D)]
    if option.startswith(f'{B} {c_sg} in every'):
        assert f'while {A} {d_sg} in the first' in option and f', {c_sg} in the next' in option
        return [(D, C), (C, C)]
    if option == f'Both {c_pl} in every ' + option.split(' in every ')[-1]:
        return [(C, C)]
    if re.match(rf'Both {re.escape(c_pl)} for two \w+, then both {re.escape(d_pl)} for one', option):
        return [(C, C), (C, C), (D, D)]
    if re.match(rf'Both {re.escape(c_pl)} in the first \w+, both {re.escape(d_pl)} in the next', option):
        return [(C, C), (D, D)]
    raise AssertionError(option)


def sim_ok(g, path, d, horizon=1200):
    u = {('C', 'C'): (g['R'], g['R']), ('C', 'D'): (g['S'], g['T']), ('D', 'C'): (g['T'], g['S']),
         ('D', 'D'): (g['P'], g['P'])}
    n = len(path)
    for t in range(n):
        for i in (0, 1):
            follow = (1 - d) * sum(d ** k * u[path[(t + k) % n]][i] for k in range(horizon))
            mine, theirs = path[t][i], path[t][1 - i]
            alt = 'D' if mine == 'C' else 'C'
            prof = (alt, theirs) if i == 0 else (theirs, alt)
            deviate = (1 - d) * u[prof][i] + d * g['P']
            if follow < deviate - 1e-9:
                return False
    return True


def test_discount_keys(got):
    keys = collections.Counter()
    n = 0
    for s in got:
        if s.type != 'discount':
            continue
        n += 1
        g, period, acts = read_game(s.question)
        d = float(re.search(r'(?:factor of|discount factor) (0\.\d\d)', s.question).group(1))
        A, B = re.search(r'(\w+ A) and (\w+ B)', s.question).groups()
        verdicts = []
        for o in s.options:
            path = path_of(o, A, B, acts)
            v = sim_ok(g, path, d)
            assert sim_ok(g, path, d - 0.02) == v == sim_ok(g, path, d + 0.02), (o, g, d)
            verdicts.append(v)
        assert verdicts == list(s.correct), (s.question, s.options, verdicts)
        keys[s.answer.split(':')[0][:20]] += 1
    assert n > 100 and len(keys) >= 5, keys


def test_threshold_monotone():
    """The generator's bisection assumes sustainability is monotone in the discount factor: check on a grid."""
    import random
    from autosteer.filler_tools.feasibility_tools import PATHS, sustainable
    from autosteer.filler_tools.multi_agent_tools import sample_pd
    random.seed(5)
    for _ in range(60):
        g = sample_pd(10, 99, integer=True)
        for path in PATHS.values():
            v = [sustainable(g, path, Fraction(k, 100)) for k in range(1, 100)]
            assert v == sorted(v), (g, path)
