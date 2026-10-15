"""engel_curves (STEER-ME, Consumption Decisions, Comparative Statics of Demand): Engel Curves ("the ability to
calculate the Engel curve for a good given a utility function and a budget constraint").

Decision of 5 October 2026; templates written here. The old filler (`gen_engel_curves` in demand_static_tools.py, no
templates) is replaced: its option formatting mixed a list with a list of lists, its y branch used the price of x, and
its quadratic type was unverified (dropped). Its Cobb-Douglas formula is kept: with U = x^a y^b (or the same
preferences written a ln x + b ln y) and prices px, py, demand is x = a/(a+b) m/px, so the Engel curve of x, income as
a function of the quantity bought, is

    m = (1 + b/a) px x      (and for y: m = (1 + a/b) py y).

Types (the consumers of hicksian_duality):
  curve_x: which equation is the Engel curve of good x. Options "m = 26.67x". Distractors: the exponent ratio
      inverted ((1 + a/b) px), the other good's price ((1 + b/a) py), all income spent on x (px), the exponent a as
      the budget share (px / a).
  curve_y: the same for good y (key (1 + a/b) py).
  income_change: income rises by a stated amount D with prices fixed; by how many units does the purchase of x rise?
      Key a/(a+b) D/px (the slope of the Engel curve times D). Distractors: a D/px, the other price a/(a+b) D/py,
      the other good's share b/(a+b) D/px, all extra income on x (D/px).
Half of the templates write the utility as x^a * y^b, half as a ln(x) + b ln(y) (same preferences). Distractors at least
10% apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .hicksian_duality_tools import STORIES
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'engel_curves'
TYPES = ('curve_x', 'curve_y', 'income_change')


def _parts(st: dict, tag: str, k: int, qtype: str, form: str) -> tuple[str]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}")
    first = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    poss = 'their' if tag == 'anon' else v['poss']
    text = (f"{first} buy{v['s']} {st['gx']} (good x) and {st['gy']} (good y) and spend{v['s']} all of {poss} income "
            f"m on them. {'Their' if tag == 'anon' else v['Poss']} utility is U(x, y) = {{ufunc}}, where x is the "
            f"number of {st['uxs']} and y the number of {st['uys']} per {st['per']}. The price of {st['gx']} is "
            f"${{px}} per {st['ux']} and the price of {st['gy']} is ${{py}} per {st['uy']}, and both can be bought in "
            f"any fractional amount.")
    if qtype == 'curve_x':
        text += (f" Prices stay fixed. Coefficients are rounded to two decimals. Which equation is the Engel curve for {st['gx']}, giving the income m per "
                 f"{st['per']} at which {v['subj']} buy{v['s']} x {st['uxs']}?")
    elif qtype == 'curve_y':
        text += (f" Prices stay fixed. Coefficients are rounded to two decimals. Which equation is the Engel curve for {st['gy']}, giving the income m per "
                 f"{st['per']} at which {v['subj']} buy{v['s']} y {st['uys']}?")
    else:
        text += (f" If {poss} income rises by ${{delta}} per {st['per']} and prices stay fixed, by how many "
                 f"{st['uxs']} per {st['per']} does {poss} purchase of {st['gx']} rise? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                form = 'power' if (k + PERSPECTIVES.index(tag)) % 2 == 0 else 'log'
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype, form), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=1 if qtype == 'income_change' else 2,
                                           spec={'story': k, 'type': qtype, 'form': form}))
    return out


def ufunc(a: float, b: float, form: str) -> str:
    return f'x^{a:g} * y^{b:g}' if form == 'power' else f'{a:g} ln(x) + {b:g} ln(y)'


def draw(r, qtype: str, form: str) -> dict:
    for _ in range(100000):
        a = round(0.05 * int(r.integers(4, 17)), 2)
        b = round(0.05 * int(r.integers(4, 17)), 2)
        if a == b:
            continue
        px, py = int(r.integers(2, 41)), int(r.integers(2, 41))
        if px == py:
            continue
        out = {'ufunc': ufunc(a, b, form), 'px': str(px), 'py': str(py)}
        if qtype in ('curve_x', 'curve_y'):
            if qtype == 'curve_x':
                key = (1 + b / a) * px
                cands = [(1 + a / b) * px, (1 + b / a) * py, px, px / a]
                var = 'x'
            else:
                key = (1 + a / b) * py
                cands = [(1 + b / a) * py, (1 + a / b) * px, py, py / b]
                var = 'y'
            dist = pick_apart(r, key, cands, lo=0.01, shown=num)
            if dist is None:
                continue
            out.update(options_list=[[f'm = {num(c)}{var}' for c in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
            return out
        D = 10 * int(r.integers(5, 61))
        key = a / (a + b) * D / px
        cands = [a * D / px, a / (a + b) * D / py, b / (a + b) * D / px, D / px]
        if key < 0.5:
            continue
        dist = pick_apart(r, key, cands, lo=0.01, shown=num)
        if dist is None:
            continue
        out.update(delta=str(D), options_list=[[num(c) for c in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
        return out
    raise RuntimeError(f'engel_curves: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_engel_curves(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'], spec['form'])


ct.register(ELEMENT, _templates())
