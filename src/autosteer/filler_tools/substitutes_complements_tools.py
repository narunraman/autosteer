"""substitutes_complements (STEER-ME, Consumption Decisions, Properties of Utility Functions): Substitutes and
Complements ("the ability to distinguish between substitutes and complements in consumption decisions").

Decision of 5 October 2026; templates written here. Gross substitutes / complements are read from the sign of the
cross-price derivative dx/dpy: positive means substitutes, negative complements. Cobb-Douglas is avoided (its
cross-price effect is zero). The two goods have neutral names ("Snack X" and "Snack Y") so that the answer has to
come from the numbers, not from what one knows about real goods.

Types:
  linear_demand: an estimated demand x = A - B px + C py + D m (B, D > 0, C != 0). Options Substitutes /
      Complements / Neither (independent); key the sign of C. The independent option is never the key; a reader of
      the own-price sign alone always says "complements".
  utility: quasi-linear utility U = alpha x + beta y - (g/2) x^2 - (h/2) y^2 - t x y + z over the two goods and
      money z, with g h > t^2 and prices at which both demands are positive. The first-order conditions give
      demand with dx/dpy = t / (g h - t^2), so the goods are substitutes when t > 0 and complements when t < 0.
      Same options as linear_demand.
  elasticity: the linear demand at stated prices and income; the key is the cross-price elasticity
      (dx/dpy) py / x = C py / x (2 decimals). Distractors: the sign flipped; the own-price elasticity -B px / x; the
      slope C itself; the ratio inverted, C x / py. Three at least 10% apart from the key and from each other, within
      [-100, 100]; the key is at least 0.1 in absolute value.

Perspectives (tags): first_person, second_person, man, woman, anon (a market researcher or a consumer).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'substitutes_complements'
TYPES = ('linear_demand', 'utility', 'elasticity')
SUBS, COMPS, NEITHER = 'Substitutes', 'Complements', 'Neither: the two goods are independent'

# domain, seller, product kind (X and Y), period
STORIES = [
    dict(domain='consumer_goods', seller='a vending-machine company', kind='Snack', per='week'),
    dict(domain='entertainment', seller='a streaming service', kind='Add-on', per='month'),
    dict(domain='technology', seller='an app store', kind='App', per='week'),
    dict(domain='sports', seller='a sports centre', kind='Class', per='month'),
    dict(domain='education', seller='an online learning platform', kind='Course',
         per='month'),
    dict(domain='medical', seller='a pharmacy chain', kind='Remedy', per='week'),
    dict(domain='environmental_policy', seller='a city transit agency', kind='Pass', per='month'),
    dict(domain='finance', seller='a bank', kind='Plan', per='month'),
]


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    X, Y = f"{st['kind']} X", f"{st['kind']} Y"
    if qtype == 'utility':
        v = voice(tag, k, 'a consumer')
        first = 'A consumer' if tag == 'anon' else v['Subj']
        poss = 'The consumer\'s' if tag == 'anon' else v['Poss']
        text = (f"{first} buy{v['s']} two products from {st['seller']}, {X} and {Y}, and spend{v['s']} the rest of "
                f"{'their' if tag == 'anon' else v['poss']} money on other goods. {poss} utility is U = {{ufunc}}, "
                f"where x is the number of units of {X}, y the number of units of {Y} and z the dollars left for "
                f"other goods. The prices are ${{px}} for {X} and ${{py}} for {Y}. Quantities are continuous, and income is "
                f"large enough that x, y and z are all positive at the optimum, before and after a small price "
                f"change. Judging by how the quantity of one product demanded responds to the price of the other, "
                f"are {X} and {Y} substitutes or complements for this consumer?")
        return (text,)
    v = voice(tag, k, 'a market researcher')
    first = 'A market researcher' if tag == 'anon' else v['Subj']
    text = (f"{first} {'works' if tag in ('man', 'woman', 'anon') else 'work'} for {st['seller']}, which sells "
            f"{X} and {Y}. The estimated demand for {X} is x = {{demand}}, where x is the quantity of {X} "
            f"sold per {st['per']}, px and py are the prices of {X} and {Y} in dollars, and m is the "
            f"average customer income in thousands of dollars.")
    if qtype == 'linear_demand':
        text += f" Based on this demand, are {X} and {Y} substitutes or complements?"
    else:
        text += (f" At px = {{px}}, py = {{py}} and m = {{income}}, what is the cross-price elasticity of demand for "
                 f"{X} with respect to the price of {Y}? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=2 if qtype != 'linear_demand' else 1,
                                           spec={'story': k, 'type': qtype}))
    return out


# ------------------------------------------------------------------------------------------------------------------
def term(c: float, var: str) -> str:
    """'3.5px', or 'px' for a coefficient of 1."""
    return var if num(c) == '1' else f'{num(c)}{var}'


def demand_text(A, B, C, D) -> str:
    cross = f'+ {term(C, "py")}' if C > 0 else f'- {term(-C, "py")}'
    return f'{A} - {term(B, "px")} {cross} + {term(D, "m")}'


def quasilinear_demand(alpha, beta, g, h, t, px, py):
    """Interior demand from alpha - g x - t y = px, beta - h y - t x = py."""
    det = g * h - t * t
    x = (h * (alpha - px) - t * (beta - py)) / det
    y = (g * (beta - py) - t * (alpha - px)) / det
    return x, y


def ufunc_text(alpha, beta, g, h, t) -> str:
    cross = f'- {term(t, "xy")}' if t > 0 else f'+ {term(-t, "xy")}'
    return f'{alpha}x + {beta}y - {term(g / 2, "x^2")} - {term(h / 2, "y^2")} {cross} + z'


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        subs = bool(r.integers(0, 2))
        if qtype == 'utility':
            g, h = int(r.integers(1, 6)) * 2, int(r.integers(1, 6)) * 2    # shown halves are integers
            t = round(float(r.uniform(0.15, 0.8)) * (g * h) ** 0.5, 1) * (1 if subs else -1)
            if t == 0 or g * h <= t * t:
                continue
            alpha, beta = int(r.integers(30, 121)), int(r.integers(30, 121))
            px, py = int(r.integers(2, alpha // 2)), int(r.integers(2, beta // 2))
            x, y = quasilinear_demand(alpha, beta, g, h, t, px, py)
            if x < 1 or y < 1:
                continue
            opts = [SUBS if subs else COMPS, COMPS if subs else SUBS, NEITHER]
            return dict(ufunc=ufunc_text(alpha, beta, g, h, t), px=str(px), py=str(py),
                        options_list=[opts], correct_answer=[[1, 0, 0]])
        B = round(float(r.uniform(0.5, 12)) * 2) / 2
        C = round(float(r.uniform(0.2, 0.9)) * B * 2) / 2 * (1 if subs else -1)
        D = round(float(r.uniform(0.1, 3)) * 2) / 2
        if B <= 0 or C == 0 or D <= 0:
            continue
        A = int(r.integers(50, 801))
        out = dict(demand=demand_text(A, B, C, D))
        if qtype == 'linear_demand':
            opts = [SUBS if subs else COMPS, COMPS if subs else SUBS, NEITHER]
            out.update(options_list=[opts], correct_answer=[[1, 0, 0]])
            return out
        px, py, m = int(r.integers(2, 41)), int(r.integers(2, 41)), int(r.integers(20, 121))
        x = A - B * px + C * py + D * m
        if x < 10:
            continue
        key = C * py / x
        cands = [-key, -B * px / x, C, C * x / py]
        if abs(key) < 0.1:
            continue
        dist = pick_apart(r, key, cands, shown=num, lo=-100, hi=100)
        if dist is None:
            continue
        opts = [num(z) for z in [key] + dist]
        if len(set(opts)) != 4:
            continue
        out.update(px=str(px), py=str(py), income=str(m), options_list=[opts], correct_answer=[[1, 0, 0, 0]])
        return out
    raise RuntimeError(f'substitutes_complements: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_substitutes_complements(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
