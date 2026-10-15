"""finding_tfp (STEER-ME, Production Decisions, Comparative Statics with Production): Total Factor Productivity
("the ability to calculate total factor productivity given a production function and input prices").

Decision of 5 October 2026; templates written here. The old filler (`gen_find_tfp` in production_functions_tools.py,
no templates) recovered A = Y / f(L, K) with unshown random options; it is replaced by this module. This element is
separate from `tfp_shocks` (the paper's 4.4 Comparative Statics with Total Factor Production Shocks, which asks how a
productivity change moves the competitive price): here the productivity level itself is computed.

The firms are those of factor_demand. Types:
  cobb_douglas: Q = A L^a K^b with observed L, K and output Q; A = Q / (L^a K^b).
  linear: Q = A (a L + b K); A = Q / (a L + b K).
  leontief: Q = A min(a L, b K); A = Q / min(a L, b K).
  input_prices: Q = A L^a K^b, the output price p and the wage w are given, and the firm's labor is the profit-
      maximising choice given its capital (p MPL = w); A = w / (p a L^(a-1) K^b). Output is not given.
A is drawn with 2 decimals in [0.5, 5]; Q is shown with 2 decimals; the key is A recomputed from the shown numbers.

Distractors are A computed with a natural mistake:
  cobb_douglas: exponents swapped (Q / (L^b K^a)); output per hour of labor (Q / L); the exponents ignored
      (Q / (L K)); the inputs summed (Q / (L^a + K^b)).
  linear: the weights ignored (Q / (L + K)); the weights swapped (Q / (b L + a K)); output per hour of labor.
  leontief: the larger term (Q / max(a L, b K)); the weights ignored (Q / min(L, K)); the sum (Q / (a L + b K)).
  input_prices: the exponent a left out of the marginal product (the average product, w / (p L^(a-1) K^b)); the
      price left out (w / (a L^(a-1) K^b)); L^a in place of L^(a-1) (w / (p a L^a K^b)); the wage set equal to the
      value of output (w / (p L^a K^b)). The wage is shown to the cent and never ends in 0 cents.
Three are chosen at random among those within a factor 15 of the key (and at least 0.05), at least 10% apart
from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .factor_demand_tools import SCENARIOS, _voice
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, stable

function_registry = FillRegister()

ELEMENT = 'finding_tfp'
TYPES = ('cobb_douglas', 'linear', 'leontief', 'input_prices')
FORMS = {'cobb_douglas': 'A * L^{a} * K^{b}', 'linear': 'A * ({a}L + {b}K)', 'leontief': 'A * min({a}L, {b}K)'}


def _parts(sc: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = _voice(tag, sc, k)
    s = v['s']
    tech = {'cobb_douglas': 'Cobb-Douglas', 'input_prices': 'Cobb-Douglas', 'linear': 'linear',
            'leontief': 'fixed-proportions (Leontief)'}[qtype]
    text = (f"{v['intro']} The {sc['firm']} has a {tech} technology: its weekly output, in {sc['units']}, is "
            f"Q = {{pfunc}}, where L is the "
            f"number of hours of {sc['lab']} and K the number of hours of {sc['cap']} used per week, and A is total "
            f"factor productivity.")
    if qtype == 'input_prices':
        text += (f" {v['subj']} sell{s} each {sc['unit']} for ${{price}} and pay{s} ${{wage}} per hour of {sc['lab']}. "
                 f"With {{capital}} hours of {sc['cap']} a week, the profit-maximizing amount of {sc['lab']} is "
                 f"{{labor}} hours a week, and that is what {v['q']} use{s}.")
    else:
        text += (f" Last week {v['subj'] if tag != 'anon' else 'the ' + sc['firm']} used {{labor}} hours of "
                 f"{sc['lab']} and {{capital}} hours of {sc['cap']} and produced {{output}} {sc['units']}.")
    text += " What is the value of A? Round to two decimals."
    return (text.replace(' a online ', ' an online ').replace(' a electronics ', ' an electronics '),)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(sc, tag, k, qtype), type=qtype, domain=sc['domain'],
                                           tags=tag, difficulty_level=2 if qtype == 'input_prices' else 1,
                                           spec={'scenario': k, 'type': qtype}))
    return out


def base(qtype, a, b, L, K):
    if qtype in ('cobb_douglas', 'input_prices'):
        return L ** a * K ** b
    if qtype == 'linear':
        return a * L + b * K
    return min(a * L, b * K)


def _mistakes(qtype, a, b, L, K, Q=None, p=None, w=None):
    if qtype == 'cobb_douglas':
        return [Q / (L ** b * K ** a), Q / L, Q / (L * K), Q / (L ** a + K ** b)]
    if qtype == 'linear':
        return [Q / (L + K), Q / (b * L + a * K), Q / L]
    if qtype == 'leontief':
        return [Q / max(a * L, b * K), Q / min(L, K), Q / (a * L + b * K)]
    mpl = L ** (a - 1) * K ** b
    return [w / (p * mpl), w / (a * mpl), w / (p * a * L ** a * K ** b), w / (p * L ** a * K ** b)]


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        A = round(float(r.uniform(0.5, 5)), 2)
        if qtype in ('cobb_douglas', 'input_prices'):
            a = round(0.05 * int(r.integers(4, 15)), 2)
            b = round(0.05 * int(r.integers(4, 15)), 2)
        else:
            a = round(0.5 * int(r.integers(1, 9)), 1)
            b = round(0.5 * int(r.integers(1, 9)), 1)
        if a == b:
            continue
        L, K = int(r.integers(10, 201)), int(r.integers(10, 201))
        pf = FORMS['cobb_douglas' if qtype == 'input_prices' else qtype].format(a=f'{a:g}', b=f'{b:g}')
        out = {'pfunc': pf, 'labor': str(L), 'capital': str(K)}
        if qtype == 'input_prices':
            p = int(r.integers(2, 41))
            w = round(p * A * a * L ** (a - 1) * K ** b, 2)
            if not 5 <= w <= 150:
                continue
            key = w / (p * a * L ** (a - 1) * K ** b)
            cands = _mistakes(qtype, a, b, L, K, p=p, w=w)
            out.update(price=str(p), wage=f'{w:.2f}')
            if not stable(out['wage']):
                continue
        else:
            Q = round(A * base(qtype, a, b, L, K), 2)
            key = Q / base(qtype, a, b, L, K)
            cands = _mistakes(qtype, a, b, L, K, Q=Q)
            out.update(output=num(Q))
        if num(key) != num(A):
            continue
        dist = pick_apart(r, key, cands, lo=max(0.05, key / 15), hi=key * 15, shown=num)
        if dist is None:
            continue
        out.update(options_list=[[num(z) for z in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
        return out
    raise RuntimeError(f'finding_tfp: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_finding_tfp(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
