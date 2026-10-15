"""factor_shares_equilibrium (STEER-ME, Non-Strategic Decisions in Multi-Agent Environments, Prices in Static
Market Equilibrium): Factor Shares in Equilibrium (the paper's definition, paraphrased: the share of income paid to each factor of production in
competitive equilibrium).

Decision of 7 October 2026; templates written here. The old filler (`gen_factor_shares_equilibrium_question` in
production_functions_tools.py, no templates) is replaced. Shares are shares of total factor payments wL + rK (triage
D7), stated in every question, so they are well defined with or without constant returns to scale.

The firms are those of factor_demand. Each input is paid the value of its marginal product (w = p MPL, r = p MPK); the
firm uses stated amounts L and K. Types, each asking for labor's or capital's share (in percent, 2 decimals):
  cobb_douglas_crs: Q = A L^a K^(1-a); labor's share a. Distractors: the other factor's share, the share of hours
      L / (L + K), 50%, a / (1 + a).
  cobb_douglas: Q = A L^a K^b with a + b ≠ 1 (0.5 to 0.85 or 1.15 to 1.4); labor's share a / (a + b). Distractors:
      the exponent itself (the share of revenue, not of factor payments), the other factor's share, the other
      exponent, the share of hours.
  linear: Q = aL + bK; w = p a, r = p b, labor's share aL / (aL + bK). Distractors: a / (a + b) (the amounts left out),
      the share of hours, the other factor's share, the share with the coefficients swapped.
Options at least 10% (and 1 point) apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .factor_demand_tools import SCENARIOS, _voice
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart

function_registry = FillRegister()

ELEMENT = 'factor_shares_equilibrium'
TYPES = ('cobb_douglas_crs', 'cobb_douglas', 'linear')
FACTORS = ('labor', 'capital')


def _parts(sc: dict, tag: str, k: int, qtype: str, factor: str) -> tuple[str]:
    v = _voice(tag, sc, k)
    s = v['s']
    tech = {'cobb_douglas_crs': 'a Cobb-Douglas technology with constant returns to scale',
            'cobb_douglas': 'a Cobb-Douglas technology', 'linear': 'a linear technology'}[qtype]
    which = sc['lab'] if factor == 'labor' else sc['cap']
    text = (f"{v['intro']} The {sc['firm']} has {tech}: its weekly output, in {sc['units']}, is Q = {{pfunc}}, where L "
            f"is the number of hours of {sc['lab']} and K the number of hours of {sc['cap']} used per week. "
            f"{v['subj']} sell{s} each {sc['unit']} for ${{price}} and use{s} {{labor}} hours of {sc['lab']} and "
            f"{{capital}} hours of {sc['cap']} a week. The input markets are competitive, and each input is paid the "
            f"value of its marginal product: the wage w per hour of {sc['lab']} equals the price times the marginal "
            f"product of labor, and the rental rate r per hour of {sc['cap']} equals the price times the marginal "
            f"product of capital. What share of the total payments to the two inputs, wL + rK, goes to {which}? Give "
            f"the percentage, rounded to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                for factor in FACTORS:
                    out.append(ct.CodeTemplate(parts=_parts(sc, tag, k, qtype, factor), type=qtype,
                                               domain=sc['domain'], tags=tag,
                                               difficulty_level={'cobb_douglas_crs': 0, 'cobb_douglas': 1,
                                                                 'linear': 2}[qtype],
                                               spec={'firm': k, 'type': qtype, 'factor': factor}))
    return out


def draw(r, qtype: str, factor: str) -> dict:
    for _ in range(100000):
        L, K = int(r.integers(10, 201)), int(r.integers(10, 201))
        price = int(r.integers(2, 61))
        hours = L / (L + K)
        if qtype == 'cobb_douglas_crs':
            a = round(0.05 * int(r.integers(4, 17)), 2)
            b = round(1 - a, 2)
            A = round(0.1 * int(r.integers(5, 51)), 1)
            pfunc = f'{A:g} * L^{a:g} * K^{b:g}'
            lab = a
            cands = {'labor': [b, hours, 0.5, a / (1 + a)], 'capital': [a, 1 - hours, 0.5, b / (1 + b)]}[factor]
        elif qtype == 'cobb_douglas':
            a, b = (round(0.05 * int(z), 2) for z in r.integers(4, 17, 2))
            if not (0.5 <= a + b <= 0.85 or 1.15 <= a + b <= 1.4) or a == b:
                continue
            A = round(0.1 * int(r.integers(5, 51)), 1)
            pfunc = f'{A:g} * L^{a:g} * K^{b:g}'
            lab = a / (a + b)
            cands = {'labor': [a, b / (a + b), b, hours], 'capital': [b, a / (a + b), a, 1 - hours]}[factor]
        else:
            a, b = int(r.integers(1, 21)), int(r.integers(1, 21))
            if a == b:
                continue
            pfunc = f'{a}L + {b}K'
            lab = a * L / (a * L + b * K)
            swapped = b * L / (b * L + a * K)
            cands = {'labor': [a / (a + b), hours, 1 - lab, swapped],
                     'capital': [b / (a + b), 1 - hours, lab, 1 - swapped]}[factor]
        key = 100 * (lab if factor == 'labor' else 1 - lab)
        if not 3 <= key <= 97:
            continue
        dist = pick_apart(r, key, [100 * c for c in cands], abs_gap=1.0, lo=0.5, hi=99.5, shown=num)
        if dist is None:
            continue
        return {'pfunc': pfunc, 'price': str(price), 'labor': str(L), 'capital': str(K),
                'options_list': [[f'{num(z)}%' for z in [key] + dist]], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError(f'factor_shares_equilibrium: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_factor_shares_equilibrium(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'], spec['factor'])


ct.register(ELEMENT, _templates())
