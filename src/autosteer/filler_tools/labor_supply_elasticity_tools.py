"""labor_supply_elasticity (STEER-ME, Factor Markets): using a wage elasticity of labor supply to translate a percentage
change in the wage into a percentage change in labor supplied, or the reverse.

Review, October 2026: the AI-pipeline templates gave changes without units ("wages change by 19.58"), sometimes
stated a product's price elasticity of supply (handcrafted chairs) instead of a wage elasticity of labor supply, did not
say that the elasticity is to be used as a linear approximation along an unchanged supply curve, and keyed decreases of
more than 100%. Templates are now written here:

  quantity_supplied: the wage changes by x% (2-20%, either sign); the answer is e x %, with the linear approximation
      stated. Distractors: x / e, -e x, x, -x / e, e x / 2 (a random number of them below the key).
  wage: the labor supplied is to change by y% (2-15%, either sign); the answer is y / e %, |y / e| <= 40%.
      Distractors: e y, -y / e, y, -e y, y / (2e).
The elasticity e is positive, 0.1-1.5 (0.4-1.5 for the wage type). Answers are percentages rounded to two decimals.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.rank_pick import pick_ranked
from .utilities.story_utils import PERSPECTIVES, num, voice

function_registry = FillRegister()

ELEMENT = 'labor_supply_elasticity'
TYPES = ('quantity_supplied', 'wage')

STORIES = [
    dict(domain='medical', group='nurses', place='a hospital network'),
    dict(domain='education', group='part-time tutors', place='a city'),
    dict(domain='technology', group='freelance programmers', place='an online platform'),
    dict(domain='consumer_goods', group='furniture makers', place='a manufacturing region'),
    dict(domain='entertainment', group='session musicians', place='a music city'),
    dict(domain='sports', group='referees', place='an amateur league'),
    dict(domain='finance', group='call-centre staff', place='a bank'),
    dict(domain='environmental_policy', group='tree planters', place='a reforestation programme'),
]

RULE = ('Treat this as a movement along an unchanged labor supply curve and use the elasticity as a linear '
        'approximation: the percentage change in the quantity of labor supplied equals the elasticity times the '
        'percentage change in the wage.')


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, 'the labor economist')
    first = 'A labor economist' if tag == 'anon' else v['Subj']
    head = (f"{first} estimate{v['s']} that the wage elasticity of labor supply of {st['group']} working for "
            f"{st['place']} is {{elasticity}}: hours of work offered respond to the hourly wage they are paid. ")
    if qtype == 'quantity_supplied':
        ask = (f"The hourly wage {{direction}} by {{wage_change}}%, and nothing else that affects labor supply changes. "
               f"{RULE} By what percentage does the quantity of labor supplied change? Give the percentage rounded to "
               f"two decimals, negative for a decrease.")
    else:
        ask = (f"The goal is for the quantity of labor supplied to {{direction}} by {{labor_change}}%, with nothing "
               f"else that affects labor supply changing. {RULE} By what percentage must the hourly wage change? Give "
               f"the percentage rounded to two decimals, negative for a decrease.")
    return (head + ask,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=0 if qtype == 'quantity_supplied' else 1,
                                           spec={'story': k, 'type': qtype}))
    return out


def _draw(r, qtype):
    fmt = lambda x: f'{num(x)}%'
    for _ in range(100000):
        sign = 1 if r.random() < 0.5 else -1
        if qtype == 'quantity_supplied':
            e = 0.05 * int(r.integers(2, 31))                 # 0.1 .. 1.5
            x = sign * 0.5 * int(r.integers(4, 41))           # 2% .. 20%
            key = e * x
            cands = [x / e, -e * x, x, -x / e, e * x / 2]
            params = {'wage_change': num(abs(x)), 'direction': 'rises' if sign > 0 else 'falls'}
        else:
            e = 0.05 * int(r.integers(8, 31))                 # 0.4 .. 1.5
            y = sign * 0.5 * int(r.integers(4, 31))           # 2% .. 15%
            key = y / e
            cands = [e * y, -y / e, y, -e * y, y / (2 * e)]
            params = {'labor_change': num(abs(y)), 'direction': 'rise' if sign > 0 else 'fall'}
        dist = pick_ranked(r, key, cands, rel=0.1, abs_gap=0.1, shown=fmt)
        if dist is None or abs(key) < 0.2:
            continue
        params.update(elasticity=num(e), options_list=[[fmt(z) for z in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
        return params
    raise RuntimeError(f'{ELEMENT}: no {qtype} draw')


@function_registry.register(ELEMENT)
def fill_labor_supply_elasticity(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return _draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
