"""labor_supply_distortions (STEER-ME, Factor Market Distortions): how a labor income tax changes hours worked, labor
supplied and the labor-market equilibrium.

Review, October 2026: the AI-pipeline templates used a supply curve LS = a + bT that rises with the tax and
ignores the wage, called a per-hour amount T a "tax rate" while subtracting it from the wage, keyed post-tax levels
where they asked for changes, and keyed the first-order elasticity approximation without saying so (or what the
initial tax was, or that the answer is a fraction). Templates are now written here, and every question states its
model and units:

  hours_worked: a worker who pays no income tax faces a new tax of t% on labor income; the gross wage is unchanged and
      the wage elasticity of labor supply (with respect to the after-tax wage) is e. With the linear approximation
      %dH = e x %d(after-tax wage), the answer is -e t, in percent. Distractors: the sign flipped, t / e, the after-tax
      wage as the base (-e t / (1 - t)), the tax rate itself (either sign), -(1 - e) t, -e (100 - t),
      -e t (1 - t).
  labor_supply: hours supplied per week LS = a + b(W - T), with W the hourly wage employers pay (held fixed) and T a
      tax per hour worked in dollars; T rises from T0 to T1, and the answer is the change -b (T1 - T0). Distractors: the
      sign flipped, -b T1, -b T0, -(T1 - T0), twice and half the change.
  equilibrium_wage_quantity: supply LS = a + b(W - T) and demand LD = c - dW, with W the gross hourly wage employers
      pay and T the tax per hour paid by workers; the answer is the equilibrium (W, hours) =
      ((c - a + bT) / (b + d), c - dW). Distractors: the no-tax equilibrium, the net wage W - T with the same hours,
      the tax added on the wrong side, the whole tax passed on to the wage, and supply read without the tax.
Answers rounded to two decimals; the shown parameters are exact. A random number of numeric distractors lies below
the key.

Perspectives (tags): first_person, second_person, man, woman, anon (hours_worked); anon and second_person for the
market types.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.rank_pick import pick_ranked
from .utilities.story_utils import PERSPECTIVES, num, voice

function_registry = FillRegister()

ELEMENT = 'labor_supply_distortions'
TYPES = ('hours_worked', 'labor_supply', 'equilibrium_wage_quantity')

STORIES = [
    dict(domain='medical', job='nurse', group='nurses', place='a regional hospital network'),
    dict(domain='education', job='teacher', group='substitute teachers', place='a large school district'),
    dict(domain='technology', job='software tester', group='freelance software testers', place='a tech hub'),
    dict(domain='consumer_goods', job='warehouse worker', group='warehouse workers', place='a logistics park'),
    dict(domain='entertainment', job='stage technician', group='stage technicians', place='a theatre district'),
    dict(domain='sports', job='fitness instructor', group='fitness instructors', place='a city'),
    dict(domain='finance', job='bank teller', group='bank tellers', place='a mid-sized city'),
    dict(domain='environmental_policy', job='solar-panel installer', group='solar-panel installers',
         place='a sunny state'),
]

PCT_RULE = ('Use the elasticity as a linear approximation: the percentage change in hours equals the elasticity times '
            'the percentage change in the after-tax wage.')


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    if qtype == 'hours_worked':
        v = voice(tag, k, f"the {st['job']}")
        lead = (f"A {st['job']} earns" if tag == 'anon' else f"{v['Subj']} work{v['s']} as a {st['job']} and earn{v['s']}")
        text = (f"{lead} a gross wage of ${{wage}} an hour and currently pay{v['s']} no income tax. A labor income tax "
                f"of {{tax_rate}}% is introduced, and the gross wage stays at ${{wage}} an hour. {v['Poss']} wage "
                f"elasticity of labor supply, with respect to the after-tax wage, is {{elasticity}}. {PCT_RULE} By what "
                f"percentage do {v['poss']} hours of work change? Give the percentage rounded to two decimals, "
                f"negative for a decrease.")
        return (text,)
    lead = 'You are studying' if tag == 'second_person' else 'A labor economist is studying'
    supply = (f"the hours of labor that {st['group']} supply per week are LS = {{supply_a}} + {{supply_b}}(W - T), where "
              f"W is the hourly wage employers pay, in dollars, and T is a tax per hour worked, in dollars, that the "
              f"workers pay")
    if qtype == 'labor_supply':
        text = (f"{lead} the market for {st['group']} in {st['place']}. In this market {supply}. Employers keep paying "
                f"W = ${{wage}}. The tax rises from ${{tax_start}} to ${{tax_end}} per hour worked. By how many hours "
                f"per week does the labor supplied change? Give the change rounded to two decimals, negative for a "
                f"decrease.")
    else:
        text = (f"{lead} the market for {st['group']} in {st['place']}. In this market {supply}, and the hours "
                f"employers demand per week are LD = {{demand_c}} - {{demand_d}}W. The tax is ${{tax}} per hour "
                f"worked. In equilibrium, what hourly wage W do employers pay, and how many hours per week are "
                f"worked? Round both to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for qtype in TYPES:
            tags = PERSPECTIVES if qtype == 'hours_worked' else ('anon', 'second_person')
            for tag in tags:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level={'equilibrium_wage_quantity': 2}.get(qtype, 1),
                                           spec={'story': k, 'type': qtype}))
    return out


def _numeric(r, key, cands, fmt):
    dist = pick_ranked(r, key, cands, rel=0.1, abs_gap=0.1, shown=fmt)
    if dist is None:
        return None
    return [[fmt(x) for x in [key] + dist]]


def _hours_worked(r):
    for _ in range(100000):
        wage = round(float(r.uniform(12, 60)), 2)
        t = int(r.integers(5, 41))
        e = 0.05 * int(r.integers(2, 19))              # 0.1 .. 0.9
        key = -e * t
        cands = [e * t, -t / e, -e * t / (1 - t / 100), -t, -(1 - e) * t, -e * (100 - t), -e * t * (1 - t / 100), t]
        opts = _numeric(r, key, cands, lambda x: f'{num(x)}%')
        if opts:
            return {'wage': f'{wage:.2f}', 'tax_rate': str(t), 'elasticity': num(e), 'options_list': opts}
    raise RuntimeError('hours_worked')


def _labor_supply(r):
    for _ in range(100000):
        a, b = int(r.integers(500, 3001)), int(r.integers(10, 81))
        wage = round(float(r.uniform(15, 60)), 2)
        t0 = round(float(r.uniform(0.5, 4)), 2)
        t1 = round(t0 + float(r.uniform(0.5, 4)), 2)
        d = t1 - t0
        key = -b * d
        cands = [b * d, -b * t1, -b * t0, -d, -2 * b * d, -b * d / 2]
        opts = _numeric(r, key, cands, num)
        if opts:
            return {'supply_a': str(a), 'supply_b': str(b), 'wage': f'{wage:.2f}', 'tax_start': f'{t0:.2f}',
                    'tax_end': f'{t1:.2f}', 'options_list': opts}
    raise RuntimeError('labor_supply')


def equilibrium(a, b, c, d, T):
    W = (c - a + b * T) / (b + d)
    return W, c - d * W


def _equilibrium(r):
    fmt = lambda p: f'W = ${num(p[0])}, {num(p[1])} hours'
    for _ in range(100000):
        a, b, d = int(r.integers(100, 1001)), int(r.integers(5, 41)), int(r.integers(5, 41))
        w0 = int(r.integers(15, 61))
        c = a + (b + d) * w0
        T = round(float(r.uniform(1, 8)), 2)
        W, Q = equilibrium(a, b, c, d, T)
        Wm = (c - a - b * T) / (b + d)
        cands = [(w0, c - d * w0), (W - T, Q), (Wm, c - d * Wm), (w0 + T, c - d * (w0 + T)), (W, a + b * W)]
        chosen = [cands[int(i)] for i in r.permutation(len(cands))[:3]]
        allp = [(W, Q)] + chosen
        if any(abs(p[0] - x[0]) < 0.1 and abs(p[1] - x[1]) < 1 for i, p in enumerate(allp) for x in allp[i + 1:]):
            continue
        texts = [fmt(p) for p in [(W, Q)] + chosen]
        if len(set(texts)) < 4 or Q <= 0:
            continue
        return {'supply_a': str(a), 'supply_b': str(b), 'demand_c': str(c), 'demand_d': str(d), 'tax': f'{T:.2f}',
                'options_list': [texts]}
    raise RuntimeError('equilibrium')


@function_registry.register(ELEMENT)
def fill_labor_supply_distortions(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    out = {'hours_worked': _hours_worked, 'labor_supply': _labor_supply,
           'equilibrium_wage_quantity': _equilibrium}[spec['type']](filler_utils.rng)
    out['correct_answer'] = [[1, 0, 0, 0]]
    return out


ct.register(ELEMENT, _templates())
