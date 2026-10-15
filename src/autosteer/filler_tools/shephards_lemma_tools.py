"""shephards_lemma (STEER-ME, Production Decisions, Comparative Statics with Production): Shephard's Lemma ("the
ability to calculate factor demands given a cost function using the derivatives with respect to prices").

Decision of 5 October 2026; templates written here. The old fillers in production_functions_tools.py (no templates)
are replaced: the Cobb-Douglas level-0 type is kept in substance (conditional factor demand = derivative of the cost
function of a Cobb-Douglas technology), with the cost function now actually shown (the old text showed a meaningless
"E = p_L * L^h + p_K * K^h"); the linear type (whose "conditional demands" rose in their own price) and level 1
(a hard-coded wrong demand formula) are dropped.

The firms are those of factor_demand. The cost function of a Cobb-Douglas technology Q = A L^a K^b is
C(w, r, Q) = k w^al r^(1-al) Q^g with al = a / (a + b) and g = 1 / (a + b); the question shows it with k (2 decimals),
al and g as numbers, and the wage w, rental rate r and output Q. By Shephard's lemma the cost-minimising inputs are

    L(w, r, Q) = dC/dw = k al w^(al-1) r^(1-al) Q^g,   K(w, r, Q) = dC/dr = k (1-al) w^al r^(-al) Q^g.

Types: labor (hours of labor), capital (hours of capital); answers to 2 decimals.
Distractors are the same input computed with a common mistake: the exponent not brought down (k w^(al-1) ...
without the factor al, which is cost divided by the input price, as if all spending went on that input); the exponent
not reduced (k al w^al ..., which is al C); the other input's derivative; the derivative with respect to output
(marginal cost). Three at
least 10% apart from the key and from each other, within a factor 10 of the key.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .factor_demand_tools import SCENARIOS, _voice
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart

function_registry = FillRegister()

ELEMENT = 'shephards_lemma'
TYPES = ('labor', 'capital')
GAMMAS = (0.8, 1, 1.25, 1.5, 2)


def _parts(sc: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = _voice(tag, sc, k)
    s = v['s']
    text = (f"{v['intro']} The {sc['firm']} produces at the lowest possible cost. Its weekly cost, in dollars, of "
            f"producing Q {sc['units']} when an hour of {sc['lab']} costs w dollars and an hour of {sc['cap']} costs "
            f"r dollars is C(w, r, Q) = {{cfunc}}. {v['subj']} pay{s} w = ${{wage}} and r = ${{rent}} and "
            f"produce{s} Q = {{output}} {sc['units']} a week.")
    what = sc['lab'] if qtype == 'labor' else sc['cap']
    text += f" How many hours of {what} does the {sc['firm']} use per week? Round to two decimals."
    return (text.replace(' a online ', ' an online ').replace(' a electronics ', ' an electronics '),)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(sc, tag, k, qtype), type=qtype, domain=sc['domain'],
                                           tags=tag, difficulty_level=0, spec={'scenario': k, 'type': qtype}))
    return out


def cost(kc, al, g, w, r, Q):
    return kc * w ** al * r ** (1 - al) * Q ** g


def demands(kc, al, g, w, r, Q):
    """(dC/dw, dC/dr, dC/dQ)."""
    c = cost(kc, al, g, w, r, Q)
    return al * c / w, (1 - al) * c / r, g * c / Q


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        kc = round(float(r.uniform(0.5, 5)), 2)
        al = round(0.05 * int(r.integers(4, 17)), 2)
        be = round(1 - al, 2)
        g = GAMMAS[int(r.integers(0, len(GAMMAS)))]
        w, rr = int(r.integers(5, 61)), int(r.integers(3, 61))
        Q = int(r.integers(10, 301))
        if w == rr:
            continue
        L, K, mc = demands(kc, al, g, w, rr, Q)
        C = cost(kc, al, g, w, rr, Q)
        if not (1 <= L <= 5000 and 1 <= K <= 5000):
            continue
        if qtype == 'labor':
            key = L
            cands = [C / w, al * C, K, mc]
        else:
            key = K
            cands = [C / rr, (1 - al) * C, L, mc]
        dist = pick_apart(r, key, cands, lo=key / 10, hi=key * 10, shown=num)
        if dist is None:
            continue
        cf = f'{kc:g} * w^{al:g} * r^{be:g}' + ('' if g == 1 else f' * Q^{g:g}')
        if g == 1:
            cf += ' * Q'
        return {'cfunc': cf, 'wage': str(w), 'rent': str(rr), 'output': str(Q),
                'options_list': [[num(z) for z in [key] + dist]], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError(f'shephards_lemma: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_shephards_lemma(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
