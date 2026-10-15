"""elasticity_of_substitution (STEER-ME, Production Decisions, Properties of Production Functions): Elasticity of
Substitution (the paper's definition, paraphrased: how easily one input can be substituted for another while keeping output constant).

Decision of 7 October 2026; templates written here. The old filler (`gen_elasticity_of_substitution` in
production_functions_tools.py, no templates) returned 1/(1 - exponent) for a Cobb-Douglas technology, whose
elasticity of substitution is 1; it is replaced. A CES technology Q = (a L^ρ + b K^ρ)^(1/ρ) has elasticity of
substitution σ = 1/(1 - ρ): the cost-minimizing K/L = ((b/a) (w/r))^σ, so σ = Δln(K/L) / Δln(w/r).

The firms are those of factor_demand. Types:
  from_function: the CES function is shown; the key is σ = 1/(1 - ρ). Distractors: ρ, 1/ρ, 1 - ρ, 1 (the
      Cobb-Douglas answer), 1/(1 + ρ).
  from_ratios: the technology is said to have a constant elasticity of substitution (the function is not shown); the
      cost-minimizing capital per hour of labor is given at two pairs of input prices (wage and rental rate both
      change). Key Δln(K/L) / Δln(w/r) from the shown numbers. Distractors: the ratio of simple percentage changes
      measured from the first values, its inverse, the wage change alone in the denominator, the sign flipped
      (labor per hour of capital), the inverse of the key.
  ratio_response: the CES function is shown (σ at most 2.5) and the wage rises by x% (20% to 100%) with the rental rate unchanged; the
      key is the percentage change in the cost-minimizing K/L, 100 ((1 + x)^σ - 1). Distractors: the first-order
      approximation σ x, the exponent ρ or 1 - ρ in place of σ, the change in L/K (100 ((1 + x)^(-σ) - 1)), x itself.
Numeric options have 2 decimals and are at least 10% apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

import math

from . import code_templates as ct
from .factor_demand_tools import SCENARIOS, _voice
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart

function_registry = FillRegister()

ELEMENT = 'elasticity_of_substitution'
TYPES = ('from_function', 'from_ratios', 'ratio_response')
RHOS = (-4, -3, -2, -1.5, -1, -0.5, -0.25, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8)


def _parts(sc: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = _voice(tag, sc, k)
    s = v['s']
    func = (f"Its weekly output, in {sc['units']}, is Q = (a * L^ρ + b * K^ρ)^(1/ρ) with {{params}}, where L is the "
            f"number of hours of {sc['lab']} and K the number of hours of {sc['cap']} used per week.")
    if qtype == 'from_function':
        text = (f"{v['intro']} {func} What is the elasticity of substitution between {sc['lab']} and {sc['cap']} "
                f"in this technology? Round to two decimals.")
    elif qtype == 'from_ratios':
        text = (f"{v['intro']} The {sc['firm']} produces with {sc['lab']} (L, hours per week) and {sc['cap']} "
                f"(K, hours per week), and its technology has a constant elasticity of substitution between the two "
                f"inputs. {v['subj']} always choose{s} the input mix that minimizes the cost of the "
                f"{sc['firm']}'s output. When the wage was ${{w1}} and the rental rate ${{r1}} per hour, the cost-minimizing mix used "
                f"{{k1}} hours of {sc['cap']} per hour of {sc['lab']}. After the wage changed to ${{w2}} and the "
                f"rental rate to ${{r2}} per hour, it uses {{k2}} hours of {sc['cap']} per hour of {sc['lab']}. What "
                f"is the elasticity of substitution between {sc['lab']} and {sc['cap']}? Round to two decimals.")
    else:
        text = (f"{v['intro']} {func} {v['subj']} pay{s} ${{wage}} per hour of {sc['lab']} and ${{rent}} per hour of "
                f"{sc['cap']} and choose{s} the input mix that minimizes cost. If the wage rises by {{x}}% and the "
                f"rental rate stays the same, by what percentage does the cost-minimizing number of hours of "
                f"{sc['cap']} per hour of {sc['lab']} (K/L) change? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(sc, tag, k, qtype), type=qtype, domain=sc['domain'],
                                           tags=tag, difficulty_level=1 if qtype == 'from_function' else 2,
                                           spec={'firm': k, 'type': qtype}))
    return out


def _params(r):
    a = round(0.05 * int(r.integers(4, 17)), 2)
    b = round(1 - a, 2)
    rho = float(r.choice(RHOS))
    return a, b, rho, f'a = {a:g}, b = {b:g} and ρ = {rho:g}'


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        a, b, rho, params = _params(r)
        sigma = 1 / (1 - rho)
        if qtype == 'ratio_response' and sigma > 2.5:
            continue
        if qtype == 'from_function':
            key = sigma
            # 1 - sigma = rho / (rho - 1): sigma and its complement confused
            cands = [rho, 1 / rho, 1 - rho, 1.0, 1 / (1 + rho) if rho != -1 else None, 1 - sigma]
            out = {'params': params}
        elif qtype == 'ratio_response':
            x = 10 * int(r.integers(2, 11))
            key = 100 * ((1 + x / 100) ** sigma - 1)
            cands = [sigma * x, 100 * ((1 + x / 100) ** rho - 1), 100 * ((1 + x / 100) ** (1 - rho) - 1),
                     100 * ((1 + x / 100) ** (-sigma) - 1), float(x)]
            out = {'params': params, 'x': str(x), 'wage': str(int(r.integers(12, 61))),
                   'rent': str(int(r.integers(5, 41)))}
        else:
            w1, r1 = int(r.integers(10, 41)), int(r.integers(5, 31))
            w2, r2 = int(r.integers(10, 61)), int(r.integers(5, 31))
            R1, R2 = w1 / r1, w2 / r2
            if abs(math.log(R2 / R1)) < 0.4 or w1 == w2 or r1 == r2:
                continue
            k1 = round((b / a * R1) ** sigma, 2)
            k2 = round((b / a * R2) ** sigma, 2)
            if min(k1, k2) < 0.2 or max(k1, k2) > 50 or k1 == k2:
                continue
            key = math.log(k2 / k1) / math.log(R2 / R1)
            if abs(key - sigma) > 0.02 * sigma:
                continue                         # the shown (rounded) ratios must still pin down sigma
            simple = (k2 / k1 - 1) / (R2 / R1 - 1)
            cands = [simple, 1 / simple, math.log(k2 / k1) / math.log(w2 / w1), -key, 1 / key]
            out = {'w1': str(w1), 'r1': str(r1), 'w2': str(w2), 'r2': str(r2), 'k1': num(k1), 'k2': num(k2)}
        dist = pick_apart(r, key, cands, shown=num)
        if dist is None or abs(key) < 0.05:
            continue
        if all((z > 0) != (key > 0) for z in dist):
            continue                             # the key alone of its sign gives it away (review)
        out.update(options_list=[[num(z) for z in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
        return out
    raise RuntimeError(f'elasticity_of_substitution: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_elasticity_of_substitution(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
