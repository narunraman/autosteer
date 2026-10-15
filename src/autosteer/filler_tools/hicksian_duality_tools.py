"""hicksian_duality (STEER-ME, Consumption Decisions, Deriving Demand): Duality of Hicksian Demand.

Decision of 5 October 2026; templates written here, built like factor_demand. Each question has two parts about the
same consumer with Cobb-Douglas utility U(x, y) = x^a y^b over two divisible goods with prices px, py.

Part 0, utility maximisation (Marshallian demand) with income m:

    x* = a / (a + b) * m / px,   y* = b / (a + b) * m / py,   u* = U(x*, y*).

Part 1, expenditure minimisation (Hicksian demand): reach utility u (the part-0 maximum u*, stated rounded to 2
decimals) at the lowest cost. The tangency condition a y / (b x) = px / py gives y = (b px / (a py)) x and

    x^h(u) = (u (a py / (b px))^b)^(1 / (a + b)),   y^h(u) = (b px / (a py)) x^h(u).

Duality: at u = u* the cheapest bundle is the utility-maximising one (and costs m), so the part-1 key equals the
part-0 key. Part 1 is computed from the shown, rounded u*; draws where the two keys differ at the shown precision
are redrawn.

Types (what both parts ask): good_x (units of x), good_y (units of y), bundle (the pair "x = .., y = ..").

Distractors are the asked quantity under a natural mistake:
  utility maximisation: the exponent itself as the budget share (a m / px, without dividing by a + b); the prices
      swapped; the exponents swapped; the budget split equally; the other good's amount (good types).
  expenditure minimisation: the prices swapped in the tangency condition; the tangency condition inverted
      (MRS = b x / (a y)); the 1 / (a + b) power left out; the utility target spent as if it were income
      (a / (a + b) u / px); the other good's amount (good types).
Both parts show the same four options (review, October 2026: with separate distractors for each part,
the key was the only option that occurred in both parts). Three distractors are chosen at random from the two lists
together, at least one from each, at least 10% apart from the key and from each other (for bundles: in at least one
of the two goods); otherwise the parameters are redrawn. The sampler shuffles each part's options on its own.

Perspectives (tags): first_person, second_person, man, woman, anon ("A student", "A household", ...).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, stable, voice

function_registry = FillRegister()

ELEMENT = 'hicksian_duality'
TYPES = ('good_x', 'good_y', 'bundle')

# domain, anonymous consumer, period, good x (name, unit, counted plural), good y
STORIES = [
    dict(domain='consumer_goods', anon='a shopper', per='month', gx='coffee', ux='bag', uxs='bags of coffee',
         gy='tea', uy='box', uys='boxes of tea'),
    dict(domain='consumer_goods', anon='a household', per='week', gx='fresh fruit', ux='kilogram',
         uxs='kilograms of fresh fruit', gy='bread', uy='loaf', uys='loaves of bread'),
    dict(domain='entertainment', anon='a film fan', per='month', gx='cinema tickets', ux='ticket',
         uxs='cinema tickets', gy='streaming rentals', uy='rental', uys='streaming rentals'),
    dict(domain='entertainment', anon='a music lover', per='year', gx='concert tickets', ux='ticket',
         uxs='concert tickets', gy='vinyl records', uy='record', uys='vinyl records'),
    dict(domain='education', anon='a student', per='term', gx='textbooks', ux='book', uxs='textbooks',
         gy='tutoring', uy='hour', uys='hours of tutoring'),
    dict(domain='sports', anon='a runner', per='year', gx='race entries', ux='entry', uxs='race entries',
         gy='running shoes', uy='pair', uys='pairs of running shoes'),
    dict(domain='technology', anon='a gamer', per='year', gx='video games', ux='game', uxs='video games',
         gy='cloud storage', uy='terabyte-month', uys='terabyte-months of cloud storage'),
    dict(domain='medical', anon='a retiree', per='month', gx='physiotherapy', ux='session',
         uxs='physiotherapy sessions', gy='massage', uy='session', uys='massage sessions'),
    dict(domain='environmental_policy', anon='a commuter', per='month', gx='train travel', ux='ticket',
         uxs='train tickets', gy='bike sharing', uy='ride', uys='bike-share rides'),
    dict(domain='finance', anon='a young saver', per='year', gx='financial planning', ux='session',
         uxs='financial-planning sessions', gy='investing courses', uy='course', uys='investing courses'),
]


def _intro(tag: str, k: int, st: dict, with_income: bool) -> tuple[str, dict]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}", None)
    if tag == 'anon':
        v['Subj'] = st['anon'][0].upper() + st['anon'][1:]
        v['poss'], v['Poss'] = 'their', 'Their'
    if with_income:
        first = f"{v['Subj']} {v['have']} ${{income}} a {st['per']} to spend on {st['gx']} (good x) and {st['gy']} (good y)."
    else:
        first = f"{v['Subj']} buy{v['s']} {st['gx']} (good x) and {st['gy']} (good y)."
    text = (f"{first} {v['Poss']} utility per {st['per']} is U(x, y) = {{ufunc}}, where x is the number of "
            f"{st['uxs']} and y the number of {st['uys']}. The price of {st['gx']} is ${{px}} per {st['ux']} and the "
            f"price of {st['gy']} is ${{py}} per {st['uy']}; both can be bought in any fractional amount.")
    return text, v


def _ask(qtype: str, st: dict, v: dict) -> str:
    if qtype == 'good_x':
        return f"how many {st['uxs']} should {v['q']} buy per {st['per']}?"
    if qtype == 'good_y':
        return f"how many {st['uys']} should {v['q']} buy per {st['per']}?"
    return f"which bundle should {v['q']} buy per {st['per']}?"


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str, str]:
    t0, v = _intro(tag, k, st, True)
    t1, _ = _intro(tag, k, st, False)
    q = _ask(qtype, st, v)
    p0 = f"{t0} To maximize {v['poss']} utility, {q}"
    p1 = (f"{t1} {v['Subj']} want{v['s']} to reach a utility level of {{utility}} per {st['per']} at the lowest "
          f"possible cost. To do so, {q}")
    return p0, p1


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=2 if qtype == 'bundle' else 1,
                                           spec={'story': k, 'type': qtype}))
    return out


# ------------------------------------------------------------------------------------------------------------------
def marshallian(a, b, px, py, m):
    return a / (a + b) * m / px, b / (a + b) * m / py


def hicksian(a, b, px, py, u):
    x = (u * (a * py / (b * px)) ** b) ** (1 / (a + b))
    return x, (b * px / (a * py)) * x


def utility(a, b, x, y):
    return x ** a * y ** b


def ufunc(a: float, b: float) -> str:
    return f'x^{a:g} * y^{b:g}'


def bundle_text(x: float, y: float) -> str:
    return f'x = {num(x)}, y = {num(y)}'


def _pick_bundles(r, key, cands, n=3):
    def apart(p, q):
        return any(abs(p[i] - q[i]) >= 0.1 * max(abs(p[i]), abs(q[i])) for i in (0, 1))

    order = [int(i) for i in r.permutation(len(cands))]
    chosen = []
    for i in order:
        c = cands[i]
        if all(c[j] > 0 for j in (0, 1)) and all(apart(c, z) for z in [key] + chosen):
            chosen.append(c)
            if len(chosen) == n:
                return chosen
    return None


def draw(r, qtype: str) -> dict:
    while True:
        a = round(0.05 * int(r.integers(4, 17)), 2)    # 0.20 .. 0.80
        b = round(0.05 * int(r.integers(4, 17)), 2)
        if a == b or not 0.6 <= a + b <= 1.4:
            continue
        px, py = int(r.integers(2, 31)), int(r.integers(2, 31))
        if px == py:
            continue
        m = 10 * int(r.integers(10, 81))               # 100 .. 800
        x0, y0 = marshallian(a, b, px, py, m)
        if not (2 <= x0 <= 300 and 2 <= y0 <= 300):
            continue
        u0 = utility(a, b, x0, y0)
        us = round(u0, 2)
        if us < 1 or not stable(num(us)):
            continue
        x1, y1 = hicksian(a, b, px, py, us)
        if num(x1) != num(x0) or num(y1) != num(y0):
            continue
        s = a + b
        # part 0 mistakes (bundles)
        m0 = [(a * m / px, b * m / py), marshallian(a, b, py, px, m), marshallian(b, a, px, py, m),
              (m / (2 * px), m / (2 * py))]
        # part 1 mistakes (bundles)
        xs = (us * (a * px / (b * py)) ** b) ** (1 / s)
        xi = (us * (b * py / (a * px)) ** b) ** (1 / s)
        xn = us * (a * py / (b * px)) ** b
        m1 = [(xs, (b * py / (a * px)) * xs), (xi, (a * px / (b * py)) * xi), (xn, (b * px / (a * py)) * xn),
              (a / s * us / px, b / s * us / py)]
        if qtype == 'bundle':
            d = _pick_bundles(r, (x0, y0), m0 + m1)
            if d is None or not (set(d) & set(m0) and set(d) & set(m1)):
                continue
            o0 = o1 = [bundle_text(*z) for z in [(x0, y0)] + d]
        else:
            i = 0 if qtype == 'good_x' else 1
            k0 = (x0, y0)[i]
            c0 = [z[i] for z in m0] + [(x0, y0)[1 - i]]
            c1 = [z[i] for z in m1] + [(x1, y1)[1 - i]]
            d = pick_apart(r, k0, c0 + c1, lo=0.01, shown=num)
            if d is None or not (set(d) & set(c0) and set(d) & set(c1)):
                continue
            o0 = o1 = [num(z) for z in [k0] + d]
        if len(set(o0)) != 4 or len(set(o1)) != 4 or o0[0] != o1[0]:
            continue
        return dict(a=a, b=b, px=px, py=py, m=m, u=us, opts=[o0, o1])


@function_registry.register(ELEMENT)
def fill_hicksian_duality(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    d = draw(filler_utils.rng, spec['type'])
    return {
        'ufunc': ufunc(d['a'], d['b']),
        'px': str(d['px']), 'py': str(d['py']), 'income': f"{d['m']:,}", 'utility': num(d['u']),
        'options_list': d['opts'],
        'correct_answer': [[1, 0, 0, 0], [1, 0, 0, 0]],
    }


ct.register(ELEMENT, _templates())
