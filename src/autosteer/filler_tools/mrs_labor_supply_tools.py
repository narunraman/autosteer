"""mrs_labor_supply (STEER-ME, Consumption Decisions, Labor Supply): Marginal Rate of Substitution in Labor Supply
(the paper's definition, paraphrased: the marginal rate of substitution between consumption and leisure).

Decision of 7 October 2026; templates written here. The old filler (`gen_mrs_labor_supply_values` in
labor_supply_tools.py, no templates) is replaced: its Cobb-Douglas key was MU_C / MU_l under a question that did not
say which ratio it meant, it printed debug output, its Leontief key was the string "Undefined" (type dropped) and it
drew a wage and a price it never used. Here the direction is stated in every question: the marginal rate of
substitution of leisure for consumption, MRS = MU_l / MU_C, the units of consumption the worker would give up for one
more hour of leisure (mrs_utility likewise states which ratio it asks for).

A worker has 16 hours a day to divide between work h and leisure l = 16 - h, earns a wage w per hour, has no other
income and spends all earnings on a consumption good that costs p per unit, so C = w h / p. The current bundle is on
this budget line. Types:
  cobb_douglas: U = C^a * l^b; MRS = b C / (a l).
  linear: U = a C + b l; MRS = b / a.
  non_linear: U = a ln(C) + b * l^0.5; MRS = b C / (2 a l^0.5).
      MRS distractors: the inverse MU_C / MU_l; the exponents (weights) swapped; the real wage w / p (the MRS equals it
      only at the optimum, and the current bundle is not optimal); for Cobb-Douglas the exponent ratio b / a; for
      the linear utility b * C / (a * l); for the non-linear one the factor 1/2 left out, and the ln derivative left
      out (b / (2 a l^0.5)). Three at least 10% apart from the key and from each other.
  optimal_hours: Cobb-Douglas utility (a + b = 1); is the current number of hours the best? Options: work more
      hours, work fewer hours, keep the current hours. Key by comparing MRS = MU_l / MU_C with w / p: fewer hours if
      the MRS is larger, more if smaller; keep (a third of the questions) when the hours are the optimum h* = 16 a.
      Bundles where comparing with the inverted ratio p / w gives another answer are preferred.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'mrs_labor_supply'
TYPES = ('cobb_douglas', 'linear', 'non_linear', 'optimal_hours')
T = 16
MORE, FEWER, KEEP = ('Work more hours a day (take less leisure)', 'Work fewer hours a day (take more leisure)',
                     'Keep the current hours: no other number of hours gives higher utility')

WORKERS = [
    dict(domain='consumer_goods', job='barista at a coffee shop', anon='a barista'),
    dict(domain='entertainment', job='stagehand at a theater', anon='a stagehand'),
    dict(domain='education', job='private tutor', anon='a private tutor'),
    dict(domain='sports', job='fitness instructor at a gym', anon='a fitness instructor'),
    dict(domain='technology', job='freelance web developer', anon='a freelance web developer'),
    dict(domain='medical', job='home-care aide paid by the hour', anon='a home-care aide'),
    dict(domain='environmental_policy', job='tree planter on a reforestation project', anon='a tree planter'),
    dict(domain='finance', job='bank teller paid by the hour', anon='a bank teller'),
]


def _parts(wk: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, f"the {wk['anon'].split(' ', 1)[1]}")
    first = wk['anon'][0].upper() + wk['anon'][1:] if tag == 'anon' else v['Subj']
    poss, Poss = ('their', 'Their') if tag == 'anon' else (v['poss'], v['Poss'])
    intro = f"{first} work{v['s']} as a {wk['job']}." if tag != 'anon' else f"{first} is paid by the hour."
    fam = {'cobb_douglas': 'a Cobb-Douglas utility function', 'optimal_hours': 'a Cobb-Douglas utility function',
           'linear': 'a linear utility function', 'non_linear': 'a non-linear utility function'}[qtype]
    text = (f"{intro} {v['Subj']} earn{v['s']} ${{wage}} per hour of work, {v['have']} no other income and spend{v['s']} "
            f"all earnings on a consumption good that costs ${{price}} per unit. {v['Subj']} {v['have']} {T} hours a "
            f"day to divide between work and leisure. {Poss} utility is U(C, l) = {{ufunc}}, where C is the number of "
            f"units of the consumption good and l the number of hours of leisure per day ({fam}). {v['Subj']} currently "
            f"work{v['s']} {{hours}} hours a day, so {v['subj']} consume{v['s']} {{cons}} units and "
            f"{v['have']} {{leisure}} hours of leisure.")
    if qtype == 'optimal_hours':
        text += (f" Comparing {poss} marginal rate of substitution between leisure and consumption with the real "
                 f"wage, what should {v['subj']} do to raise {poss} utility?")
    else:
        text += (f" At this bundle, what is {poss} marginal rate of substitution of leisure for consumption, "
                 f"MRS = MU_l / MU_C, the number of units of consumption {v['subj']} would give up for one more hour "
                 f"of leisure? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, wk in enumerate(WORKERS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                lvl = {'linear': 0, 'cobb_douglas': 1, 'non_linear': 2, 'optimal_hours': 2}[qtype]
                out.append(ct.CodeTemplate(parts=_parts(wk, tag, k, qtype), type=qtype, domain=wk['domain'],
                                           tags=tag, difficulty_level=lvl, spec={'worker': k, 'type': qtype}))
    return out


def _wage_price(r):
    p = int(r.choice([1, 2, 5]))
    w = int(r.integers(8, 61))
    return w, p


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        w, p = _wage_price(r)
        if qtype == 'optimal_hours':
            a = round(0.05 * int(r.integers(5, 16)), 2)      # 0.25 .. 0.75
            b = round(1 - a, 2)
            hstar = round(T * a, 2)
            keep = r.random() < 1 / 3
            if keep:
                h = hstar
            else:
                h = int(r.integers(2, 15))
                if abs(h - hstar) < 0.1 * hstar:
                    continue
            l, C = T - h, w * h / p
            g = b * C / (a * l)
            if keep:
                ans = KEEP
            else:
                ans = FEWER if g > w / p else MORE
                wrong = FEWER if g > p / w else MORE
                if wrong == ans and r.random() < 0.8:
                    continue
            opts = [ans] + [o for o in (MORE, FEWER, KEEP) if o != ans]
            ufunc = f'C^{a:g} * l^{b:g}'
        else:
            a = round(0.05 * int(r.integers(4, 17)), 2) if qtype != 'non_linear' else float(r.integers(2, 10))
            b = round(0.05 * int(r.integers(4, 17)), 2) if qtype != 'non_linear' else float(r.integers(2, 10))
            if a == b:
                continue
            h = int(r.integers(2, 15))
            l, C = T - h, w * h / p
            if qtype == 'cobb_douglas':
                ufunc = f'C^{a:g} * l^{b:g}'
                key = b * C / (a * l)
                cands = [a * l / (b * C), a * C / (b * l), b / a, w / p]
            elif qtype == 'linear':
                ufunc = f'{a:g}C + {b:g}l'
                key = b / a
                cands = [a / b, w / p, b * C / (a * l), b]
            else:
                ufunc = f'{a:g} ln(C) + {b:g} * l^0.5'
                key = b * C / (2 * a * l ** 0.5)
                cands = [2 * a * l ** 0.5 / (b * C), b * C / (a * l ** 0.5), b / (2 * a * l ** 0.5), w / p]
            if abs(key - w / p) < 0.1 * max(key, w / p) or key < 0.05:
                continue                         # the current bundle must be clearly off the optimum
            dist = pick_apart(r, key, cands, lo=0.01, shown=num)
            if dist is None:
                continue
            opts = [num(z) for z in [key] + dist]
        out = {'wage': str(w), 'price': str(p), 'ufunc': ufunc, 'hours': num(h), 'leisure': num(l),
               'cons': num(C), 'options_list': [opts], 'correct_answer': [[1] + [0] * (len(opts) - 1)]}
        if abs(round(C, 2) - C) > 1e-9:
            continue
        return out
    raise RuntimeError(f'mrs_labor_supply: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_mrs_labor_supply(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
