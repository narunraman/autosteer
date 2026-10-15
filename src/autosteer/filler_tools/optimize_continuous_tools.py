"""optimize_continuous (STEER 1.2.b, Optimize a Continuous Function): hand-written templates.

Design of 6 October 2026 (no earlier generator or data). Types:
    quadratic (difficulty 1): f(x) = a x^2 + b x + c on a closed interval [lo, hi], whole coefficients and vertex;
        asked for the x that maximises or minimises f. Half the draws have the vertex as the answer, half an
        endpoint (the vertex outside the interval, or the parabola opening the wrong way).
    cubic (difficulty 2): f(x) = a x^3 + b x^2 + c x + d with f'(x) = 3a (x - r1)(x - r2), r1 < r2 whole numbers
        inside [lo, hi]; asked for the maximiser or minimiser on the interval (a critical point or an endpoint; the
        best value is at least 1 ahead of the others). Also as a firm's profit P(q) over a capacity range (maximum).
    box (difficulty 3): squares of side x cut from the corners of a square sheet of side s (a multiple of 6) and the
        sides folded up: V(x) = x (s - 2x)^2 on [0, s/2], maximised at x = s/6. Asked for x* or the largest volume.
    fence (difficulty 2): a rectangular pen against a straight wall or river, L units of fencing for the other three
        sides (L a multiple of 4): A(x) = x (L - 2x), the sides from the wall x* = L/4, largest area L^2/8. Asked for
        x* or the largest area.
Distractors: for "which x" questions, the other critical points and the endpoints, then other whole numbers of the
interval (every option is a point of the interval; cubic intervals have length >= 3); for the box and the fence,
other feasible cuts / sides (s/4, s/3, s/12, x +- 1; L/3, 2L/5, L/5, L/6, L/8) and, for the largest volume / area, the
values of other cuts / sides and slips that overstate it (box: x (s - x)^2, x s^2; fence: 3L^2/16, L^2/4).
(Before the review of October 2026 f(x*) was offered as an x, and an area as a length.) Options are whole
numbers, distinct. Perspectives: anon for plain functions; anon, first_person, second_person, man, woman for the
word problems.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import voice

function_registry = FillRegister()

ELEMENT = 'optimize_continuous'
WORD_PERSPECTIVES = ('anon', 'first_person', 'second_person', 'man', 'woman')


def poly_text(coefs, var='x') -> str:
    """[a3, a2, a1, a0] (highest power first) -> '2x^3 - 9x^2 + 12x + 5'."""
    deg = len(coefs) - 1
    out = []
    for i, c in enumerate(coefs):
        p = deg - i
        if c == 0:
            continue
        mag = abs(c)
        body = (f'{mag}' if p == 0 else ('' if mag == 1 else f'{mag}') + (var if p == 1 else f'{var}^{p}'))
        sign = '-' if c < 0 else '+'
        out.append((sign, body))
    s = ('-' if out[0][0] == '-' else '') + out[0][1]
    for sign, body in out[1:]:
        s += f' {sign} {body}'
    return s


def evalp(coefs, x):
    v = 0
    for c in coefs:
        v = v * x + c
    return v


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for goal in ('maximum', 'minimum'):
        verb = 'maximizes' if goal == 'maximum' else 'minimizes'
        for qtype, lvl in (('quadratic', 1), ('cubic', 2)):
            for k, phr in enumerate([
                    "Let f(x) = {f} be a {kind} function on the interval {lo} <= x <= {hi}. At what value of x "
                    "does f attain its {goal} on this interval?",
                    "Consider the {kind} function f(x) = {f}, defined for {lo} <= x <= {hi}. Which x in this "
                    "interval {verb} f(x)?",
                    "On the closed interval [{lo}, {hi}], find the point x at which the {kind} function f(x) = {f} is "
                    "{goalpast}."]):
                text = phr.replace('{kind}', qtype).replace('{goal}', goal).replace('{verb}', verb).replace(
                    '{goalpast}', 'largest' if goal == 'maximum' else 'smallest')
                out.append(ct.CodeTemplate(parts=(text,), type=qtype, domain='math', tags='anon', difficulty_level=lvl,
                                           spec={'type': qtype, 'goal': goal}))
    for k, persp in enumerate(WORD_PERSPECTIVES):
        v = voice(persp, k, 'the firm', 'its')
        firm = {'anon': 'A firm', 'first_person': 'I run a firm, and my', 'second_person': 'You run a firm, and your',
                'man': f"{v['subj']} runs a firm, and his", 'woman': f"{v['subj']} runs a firm, and her"}[persp]
        lead = (f"{firm} weekly profit" if persp != 'anon' else "A firm's weekly profit")
        text = (lead + ", in thousands of dollars, from producing q hundred units is P(q) = {f}. Capacity allows "
                "between {lo} and {hi} hundred units. How many hundred units should " + v['q'] +
                " produce to maximize profit?")
        out.append(ct.CodeTemplate(parts=(text,), type='cubic', domain='business', tags=persp, difficulty_level=2,
                                   spec={'type': 'cubic', 'goal': 'maximum'}))
        maker = {'anon': 'A workshop makes', 'first_person': 'I make', 'second_person': 'You make',
                 'man': f"{v['subj']} makes", 'woman': f"{v['subj']} makes"}[persp]
        for ask in ('x', 'value'):
            q = (f"What side length x, in cm, of the cut squares gives the box the largest volume?" if ask == 'x' else
                 "What is the largest possible volume of the box, in cubic cm?")
            text = (maker + " an open-top box from a square sheet of cardboard {s} cm on each side by cutting a square "
                    "of side x cm from each corner and folding up the sides. " + q)
            out.append(ct.CodeTemplate(parts=(text,), type='box', domain='manufacturing', tags=persp,
                                       difficulty_level=3, spec={'type': 'box', 'ask': ask}))
            fencer = {'anon': 'A farmer has', 'first_person': 'I have', 'second_person': 'You have',
                      'man': f"{v['subj']} has", 'woman': f"{v['subj']} has"}[persp]
            q = ("How long, in metres, should each of the two sides perpendicular to the river be to enclose the "
                 "largest area?" if ask == 'x' else "What is the largest area, in square metres, that can be enclosed?")
            text = (fencer + " {L} metres of fencing to make a rectangular pen along a straight river; the side along "
                    "the river needs no fence. " + q)
            out.append(ct.CodeTemplate(parts=(text,), type='fence', domain='agriculture', tags=persp,
                                       difficulty_level=2, spec={'type': 'fence', 'ask': ask}))
    return out


def _opts(r, key, pool, fill=()):
    """key and three distinct distractors: from `pool` first (in random order), then from `fill`."""
    out = [key]
    for group in (pool, fill):
        for i in r.permutation(len(group)):
            x = group[int(i)]
            if x not in out:
                out.append(x)
            if len(out) == 4:
                return out
    return None


def _in_interval(r, x_star, cands, lo, hi):
    """Options for 'which x': the key, the other candidates (endpoints, critical points) and, if those are fewer than
    three, other whole numbers of [lo, hi]. Every option is a point of the interval, so none can be ruled out without
    comparing values (review, October 2026: f(x*) and out-of-interval vertices were offered as x)."""
    pool = [x for x in cands if lo <= x <= hi and x != x_star]
    return _opts(r, x_star, pool, [x for x in range(lo, hi + 1)])


def _quadratic(r, goal):
    while True:
        a = int(r.choice([-3, -2, -1, 1, 2, 3]))
        h, k0 = int(r.integers(-6, 7)), int(r.integers(-20, 21))
        vertex_answer = r.random() < 0.5
        lo = int(r.integers(-8, 5))
        hi = lo + int(r.integers(3, 9))
        opens_right_way = (a < 0) == (goal == 'maximum')
        inside = lo < h < hi
        if vertex_answer != (opens_right_way and inside):
            continue
        coefs = [a, -2 * a * h, a * h * h + k0]
        cand = {x: evalp(coefs, x) for x in (lo, hi) + ((h,) if inside else ())}
        best = (max if goal == 'maximum' else min)(cand.values())
        winners = [x for x, v in cand.items() if v == best]
        if len(winners) != 1:
            continue
        x_star = winners[0]
        if sorted(cand.values(), reverse=goal == 'maximum')[1] == best:
            continue
        o = _in_interval(r, x_star, (lo, hi, h), lo, hi)
        if o is not None:
            return {'f': poly_text(coefs), 'lo': lo, 'hi': hi}, o


def _cubic(r, goal, profit: bool):
    while True:
        a = int(r.choice([-2, -1, 1, 2])) if not profit else -int(r.choice([1, 2]))
        r1, r2 = sorted(int(x) for x in r.choice(range(-5, 9) if not profit else range(1, 12), size=2,
                                                 replace=False))
        if a % 2 and (r1 + r2) % 2:
            continue
        b, c = -3 * a * (r1 + r2) // 2, 3 * a * r1 * r2
        d = int(r.integers(-20, 21)) if not profit else int(r.integers(-40, -4))
        coefs = [a, b, c, d]
        lo = r1 - int(r.integers(0, 4))
        hi = r2 + int(r.integers(0, 4))
        if profit:
            lo = max(0, lo)
        if lo > r1 or hi < r2 or hi - lo < 3:
            continue
        pts = sorted({lo, r1, r2, hi})
        vals = {x: evalp(coefs, x) for x in pts}
        best = (max if goal == 'maximum' else min)(vals.values())
        winners = [x for x, v in vals.items() if v == best]
        if len(winners) != 1:
            continue
        x_star = winners[0]
        others = sorted(vals.values(), reverse=goal == 'maximum')
        if abs(others[1] - best) < 1:
            continue
        o = _in_interval(r, x_star, (lo, hi, r1, r2), lo, hi)
        if o is not None:
            return {'f': poly_text(coefs, 'q' if profit else 'x'), 'lo': lo, 'hi': hi}, o


@function_registry.register(ELEMENT)
def fill_optimize_continuous(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    t = spec['type']
    if t == 'quadratic':
        out, opts = _quadratic(r, spec['goal'])
    elif t == 'cubic':
        out, opts = _cubic(r, spec['goal'], metadata['domain'] == 'business')
    elif t == 'box':
        s = 6 * int(r.integers(2, 11))
        x = s // 6
        vol = x * (s - 2 * x) ** 2
        if spec['ask'] == 'x':
            # cuts strictly between 0 and s/2 (a possible box): s/4 and s/3 (the fence's and a cube's thirds), and
            # neighbouring whole numbers
            pool = [s // 4, s // 3, s // 12, x - 1, x + 1, 2 * x]
            out, key = {'s': s}, x
            pool = [p for p in pool if 0 < p < s / 2]
        else:
            # volumes of other cuts (s/4, s/3) and two slips that overstate it: subtracting x once, x(s - x)^2, and
            # not subtracting it, x s^2 (so the key is not always the largest option)
            pool = [(s // 4) * (s - 2 * (s // 4)) ** 2, (s // 3) * (s - 2 * (s // 3)) ** 2, x * (s - x) ** 2,
                    x * s * s]
            out, key = {'s': s}, vol
        opts = _opts(r, key, [p for p in pool if p > 0])
    else:
        L = 4 * int(r.integers(10, 101))
        if spec['ask'] == 'x':  # sides strictly between 0 and L/2
            out, key, pool = {'L': L}, L // 4, [L // 3, 2 * L // 5, L // 5, L // 6, L // 8]
        else:  # areas of other side lengths (L/3, the square L/4 x L/4) and two that overstate it
            out, key, pool = {'L': L}, L * L // 8, [L * L // 9, L * L // 16, 3 * L * L // 16, L * L // 4]
        opts = _opts(r, key, pool)
    out['options_list'] = [[str(o) for o in opts]]
    out['correct_answer'] = [[1, 0, 0, 0]]
    return out


ct.register(ELEMENT, _templates())
