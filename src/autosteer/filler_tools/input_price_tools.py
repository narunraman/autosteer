"""input_price_response (STEER-ME, Production Decisions, Comparative Statics with Production): how a firm's optimal
labor responds to the wage. The paper's element "Input Price Elasticity" ("the ability to calculate how the optimal
input bundle changes with changes in input prices for a given production function").

Decision of 5 October 2026: comp_stat_cobb and comp_stat_leonteif are types of one element (the 2024 report cards
already remapped both to input_price_elasticity). This module registers the merged element as a composite of their
template sets (code_templates.register_composite): type `cobb_douglas` = comp_stat_cobb's templates and filler, type
`leontief` = comp_stat_leonteif's. Questions and keys are unchanged. Both ask for the derivative dL*/dw of the
profit-maximising labor, not for an elasticity, so the element is named for what it asks and the paper name is kept
as an alias (elements.json `paper_name`); the old ids are listed in `aliases`. The source elements stay registered
(their fillers are used here) but are no longer listed on the site or exported.

Leontief type rebuilt (October 2026 cleanup). The comp_stat_leonteif templates gave min(L/a, K/b)^0.5 and a price but
never the amount or price of capital; its key -p^2/(2a) w^-3 is right only if capital is free, and with a capital
price r or a fixed capital stock the answer is different (at a fixed stock binding at the kink, labor does not respond
to the wage at all). They also wrote the options in a "C" the question never defines and gave coefficients down to
1e-05. Type `leontief` now comes from the hidden code-template element `input_price_leontief` below: a firm (the
scenarios of factor_demand) with Q = (min(L/a, K/b))^0.5, a, b whole numbers 2-5, sells at p and rents capital at r,
both inputs freely chosen, competitive markets. It never holds idle input, so L = a z, K = b z for z bundles, profit
p z^0.5 - (a w + b r) z, and

    z* = p^2 / (4 (a w + b r)^2),   L*(w) = a p^2 / (4 (a w + b r)^2),   dL*/dw = -a^2 p^2 / (2 (a w + b r)^3).

The options are expressions in w: the key, and three of: capital's cost left out (-a^2 p^2 / (2 (a w)^3), the old
key's reading), the bundle z* differentiated instead of L* (coefficient a p^2 / 2), the sign flipped, and the power 2
instead of 3. comp_stat_leonteif stays registered and unchanged (its digests), but is not used.

Cobb-Douglas type rebuilt the same way (quality audit, October 2026). comp_stat_cobb drew capital and price down to
0.01 and exponents rounded to two decimals, so its options read "-0.0C^-2.72" against "-0.01C^-2.72" (coefficients
rounded to zero or nearly equal), in an undefined "C". Type `cobb_douglas` now comes from the hidden code-template
element `input_price_cobb`: the same firms, short run, Q = L^a K^0.5 with capital fixed at a square number of hours K
(so K^0.5 is whole) and a in {0.5, 2/3, 0.75}; output price p, wage $w. With m = a p K^0.5 (whole by choice of p) and
n = 1 / (1 - a) in {2, 3, 4}:

    L*(w) = (m / w)^n,   dL*/dw = -n m^n / w^(n + 1).

Distractors (three of four): sign flipped; the factor n left out; power n instead of n + 1; capital left out of the
marginal product (m = a p). comp_stat_cobb stays registered and unchanged, but is not used.
"""
from __future__ import annotations

from . import code_templates as ct
from .factor_demand_tools import MEN, PERSPECTIVES, SCENARIOS, WOMEN
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'input_price_response'
LEONTIEF = 'input_price_leontief'
COBB = 'input_price_cobb'
PARTS = {'cobb_douglas': COBB, 'leontief': LEONTIEF}
# the sources' own type values, passed on to their fillers
_SOURCE_TYPES = {'cobb_douglas': 'cobb_douglas', 'leontief': 'leontief'}


def _voice(tag: str, sc: dict, k: int) -> dict:
    firm = sc['firm']
    if tag == 'first_person':
        return dict(intro=f"I run a {firm} {sc['what']}.", subj='I', s='', q='my')
    if tag == 'second_person':
        return dict(intro=f"You run a {firm} {sc['what']}.", subj='You', s='', q='your')
    if tag in ('man', 'woman'):
        name = (MEN if tag == 'man' else WOMEN)[k % 7]
        return dict(intro=f"{name} runs a {firm} {sc['what']}.", subj=name, s='s', q='his' if tag == 'man' else 'her')
    return dict(intro=f"Consider a {firm} {sc['what']}.", subj=f'The {firm}', s='s', q='its')


STEMS = [
    'Let L*(w) be the profit-maximizing number of hours of {lab} per week when the wage is $w per hour. Which '
    'expression is the derivative dL*/dw?',
    'How does the profit-maximizing amount of {lab} respond to the wage? Writing L*(w) for {q} profit-maximizing '
    'hours of {lab} per week at a wage of $w per hour, which expression gives dL*/dw?',
]


def _leontief_templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            v = _voice(tag, sc, k)
            body = (f"{v['intro']} Its weekly output, in {sc['units']}, is Q = {{pfunc}}, where L is the number of "
                    f"hours of {sc['lab']} and K the number of hours of {sc['cap']} used per week. {v['subj']} "
                    f"sell{v['s']} each {sc['unit']} for ${{price}}, pay{v['s']} a wage of $w per hour of {sc['lab']} "
                    f"and ${{rent}} per hour of {sc['cap']}, and choose{v['s']} both inputs freely to maximize "
                    f"profit; all markets are competitive.")
            for j, stem in enumerate(STEMS):
                out.append(ct.CodeTemplate(parts=(body + ' ' + stem.format(lab=sc['lab'], q=v['q']),),
                                           type='leontief', domain=sc['domain'], tags=tag, difficulty_level=2,
                                           spec={'scenario': k, 'stem': j}))
    return out


def _cobb_templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            v = _voice(tag, sc, k)
            body = (f"{v['intro']} Its weekly output, in {sc['units']}, is Q = {{pfunc}}, where L is the number of "
                    f"hours of {sc['lab']} and K the number of hours of {sc['cap']} used per week. In the short run "
                    f"K is fixed at {{capital}} hours per week. {v['subj']} sell{v['s']} each {sc['unit']} for "
                    f"${{price}}, pay{v['s']} a wage of $w per hour of {sc['lab']}, and choose{v['s']} L to maximize "
                    f"profit; all markets are competitive.")
            for j, stem in enumerate(STEMS):
                out.append(ct.CodeTemplate(parts=(body + ' ' + stem.format(lab=sc['lab'], q=v['q']),),
                                           type='cobb_douglas', domain=sc['domain'], tags=tag, difficulty_level=2,
                                           spec={'scenario': k, 'stem': j}))
    return out


# labor exponent a (as written) -> n = 1 / (1 - a), the step of p that makes a p whole, and the largest m = a p K^0.5
_COBB_A = {'0.5': (2, 2, 60), '(2/3)': (3, 3, 25), '0.75': (4, 4, 12)}


def cobb_expr(coef: int, power: int) -> str:
    """'dL*/dw = -coef/w^power' (coef with thousands separators)."""
    sign = '-' if coef < 0 else ''
    return f'dL*/dw = {sign}{abs(coef):,}/w^{power}'


@function_registry.register(COBB)
def fill_input_price_cobb(metadata: dict) -> dict:
    r = filler_utils.rng
    a_txt = list(_COBB_A)[int(r.integers(0, 3))]
    n, step, m_max = _COBB_A[a_txt]
    a_num, a_den = {'0.5': (1, 2), '(2/3)': (2, 3), '0.75': (3, 4)}[a_txt]
    while True:
        root = int(r.integers(2, 7))                   # K = root^2 hours
        p = step * int(r.integers(1, 40 // step + 1))  # a p whole
        ap = a_num * p // a_den
        m = ap * root
        if 3 <= m <= m_max and p >= 4:
            break
    key = cobb_expr(-n * m ** n, n + 1)
    wrong = [cobb_expr(n * m ** n, n + 1),           # sign flipped
             cobb_expr(-(m ** n), n + 1),            # the factor n left out
             cobb_expr(-n * m ** n, n),              # power n instead of n + 1
             cobb_expr(-n * ap ** n, n + 1)]         # capital left out of the marginal product
    pick = [wrong[int(i)] for i in r.choice(len(wrong), size=3, replace=False)]
    options = [key] + pick
    return {'pfunc': f'L^{a_txt} K^0.5', 'price': str(p), 'capital': str(root * root),
            'options_list': [options], 'correct_answer': [[o == key for o in options]]}


def num(x: float) -> str:
    return f'{x:.2f}'.rstrip('0').rstrip('.')


def leontief_expr(coef: float, a: int, c: int, power: int) -> str:
    """'dL*/dw = -coef/(a w + c)^power' (c = 0: the capital term left out)."""
    sign = '-' if coef < 0 else ''
    den = f'{a}w + {c}' if c else f'{a}w'
    return f'dL*/dw = {sign}{num(abs(coef))}/({den})^{power}'


@function_registry.register(LEONTIEF)
def fill_input_price_leontief(metadata: dict) -> dict:
    r = filler_utils.rng
    a, b = (int(x) for x in r.integers(2, 6, size=2))
    p = int(r.integers(10, 41))
    rent = int(r.integers(2, 13))
    c = b * rent
    coef = a * a * p * p / 2
    key = leontief_expr(-coef, a, c, 3)
    wrong = [leontief_expr(-coef, a, 0, 3),          # capital's cost left out
             leontief_expr(-coef / a, a, c, 3),      # the bundle z* differentiated instead of L* = a z*
             leontief_expr(coef, a, c, 3),           # sign flipped
             leontief_expr(-coef, a, c, 2)]          # power 2
    pick = [wrong[int(i)] for i in r.choice(len(wrong), size=3, replace=False)]
    options = [key] + pick
    return {'pfunc': f'(min(L/{a}, K/{b}))^0.5', 'price': str(p), 'rent': str(rent),
            'options_list': [options], 'correct_answer': [[o == key for o in options]]}


def _fill_leontief_part(metadata: dict) -> dict:
    return fill_input_price_leontief(metadata)


_FILLERS = {'cobb_douglas': fill_input_price_cobb, 'leontief': _fill_leontief_part}


@function_registry.register(ELEMENT)
def fill_input_price_response(metadata: dict) -> dict:
    qtype = metadata['type']
    return _FILLERS[qtype](dict(metadata, type=_SOURCE_TYPES[qtype]))


ct.register(LEONTIEF, _leontief_templates())
ct.register(COBB, _cobb_templates())
ct.register_composite(ELEMENT, PARTS)
