"""capital_market_distortions (STEER-ME, Factor Market Distortions): how a tax on the return to capital compounds over
several years to lower the capital stock and output.

Review, October 2026: the AI-pipeline templates did not state the model the key used (a fall in the
"accumulation rate" read as the stock shrinking; profit or patent income asked for as a capital stock; taxes on
"capital gains" keyed as a levy on the whole stock, K0 [(1 + g)(1 - t)]^n; output keyed where the stock was asked).
Templates are now written here and every question states the model:

The capital is worth K0 (thousands of dollars). Each year it earns a pre-tax return of g of its value; the return is
taxed at t and the whole after-tax return is reinvested (no depreciation, nothing else added or taken out), so
    K_n = K0 (1 + g (1 - t))^n.
Types:
  capital_stock: K_n.
  capital_shortfall: K0 (1 + g)^n - K_n, how much lower the capital is than without the tax.
  output: Y = A L^alpha K_n^beta (beta 0.5-0.8) with the number of workers L fixed (K in thousands of dollars).
Answers rounded to two decimals; every shown parameter is exact (integers, or percentages with at most one decimal).
Distractors are plausible mistakes (no tax; the tax levied on the whole stock, K0 [(1 + g)(1 - t)]^n; simple instead of
compound growth; one year too few or too many; the tax rate subtracted from the return; the tax rate taken as the share
kept; the tax taken once at the end on the whole value or on the whole gain, only one year's return taxed), three of them at least 3% apart (2% for capital_stock, 1.5% for output), with a
random number below the key. For capital_stock and output only mistakes that leave the capital above K0 are used (the
capital must grow; re-review: an option below K0, or output below F(L, K0), is ruled out without computing).

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.rank_pick import pick_ranked
from .utilities.story_utils import PERSPECTIVES, num, voice

function_registry = FillRegister()

ELEMENT = 'capital_market_distortions'
TYPES = ('capital_stock', 'capital_shortfall', 'output')

STORIES = [
    dict(domain='consumer_goods', anon='a furniture maker', business='a furniture workshop',
         capital='woodworking machinery', output='chairs'),
    dict(domain='technology', anon='a software entrepreneur', business='a data-hosting company',
         capital='server equipment', output='terabytes of hosted data'),
    dict(domain='medical', anon='a physician', business='a diagnostic clinic',
         capital='imaging equipment', output='scans'),
    dict(domain='entertainment', anon='a music producer', business='a recording studio',
         capital='sound equipment', output='hours of finished recordings'),
    dict(domain='environmental_policy', anon='an engineer', business='a recycling plant',
         capital='sorting and baling machines', output='tonnes of recycled material'),
    dict(domain='sports', anon='a former athlete', business='a fitness centre',
         capital='training equipment', output='member visits'),
    dict(domain='education', anon='a teacher', business='a private tutoring school',
         capital='classroom technology', output='tutoring hours'),
    dict(domain='finance', anon='a banker', business='a payments company',
         capital='payment-processing hardware', output='thousand card payments'),
]


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}", 'their' if tag == 'anon' else None)
    first = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    setup = (f"{first} run{v['s']} {st['business']}. {v['Poss']} {st['capital']} is worth ${{initial_stock}} "
             f"thousand today. Each year the capital earns a pre-tax return of {{growth_rate}}% of its value, the "
             f"government taxes that return at {{capital_tax}}%, and the whole after-tax return is reinvested in more "
             f"{st['capital']}. The capital does not depreciate, and nothing else is added to it or taken out of it.")
    if qtype == 'capital_stock':
        ask = (f"What will {v['poss']} capital be worth after {{num_years}} years? Give the value in thousands of "
               f"dollars, rounded to two decimals.")
    elif qtype == 'capital_shortfall':
        ask = (f"Compared with no tax on the return (and everything else the same), how much less will {v['poss']} "
               f"capital be worth after {{num_years}} years? Give the difference in thousands of dollars, rounded to "
               f"two decimals.")
    else:
        ask = (f"The {st['business'].split(' ', 1)[1]} employs {{labor}} workers, a number that does not change, and "
               f"produces Y = {{prod_fn}} {st['output']} a year, where L is the number of workers and K is the value of "
               f"the capital in thousands of dollars. How many {st['output']} a year will it produce with the capital "
               f"it has after {{num_years}} years? Round to two decimals.")
    return (setup + ' ' + ask,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level={'capital_stock': 0}.get(qtype, 1),
                                           spec={'story': k, 'type': qtype}))
    return out


def stock(K0, g, t, n):
    """Capital after n years, all after-tax returns reinvested (g, t as fractions)."""
    return K0 * (1 + g * (1 - t)) ** n


def stock_mistakes(K0, g, t, n):
    no_tax = K0 * (1 + g) ** n
    return [no_tax,                                   # tax ignored
            K0 * ((1 + g) * (1 - t)) ** n,            # tax levied on the whole stock every year
            K0 * (1 + n * g * (1 - t)),               # simple growth
            stock(K0, g, t, n - 1), stock(K0, g, t, n + 1),
            K0 * (1 + g - t) ** n if g < t + 1 else None,   # tax rate subtracted from the return
            K0 * (1 + g * t) ** n,                    # the tax rate taken as the share kept
            no_tax * (1 - t),                         # whole value taxed once at the end
            no_tax - t * (no_tax - K0),               # whole gain taxed once at the end
            no_tax - t * g * K0,                      # only the first year's return taxed
            no_tax - t * g * K0 * (1 + g) ** (n - 1)]  # only the last year's return taxed


def _draw(r, qtype):
    for _ in range(100000):
        K0 = int(r.integers(100, 1000))
        g = 0.5 * int(r.integers(8, 25))              # 4% .. 12%
        t = int(r.integers(15, 46))                   # 15% .. 45%
        n = int(r.integers(3, 11))
        G, T = g / 100, t / 100
        rel = 0.03
        key_k = stock(K0, G, T, n)
        params = {'initial_stock': str(K0), 'growth_rate': num(g, 1), 'capital_tax': str(t), 'num_years': str(n)}
        fmt = lambda x: f'${num(x)} thousand'
        if qtype == 'capital_stock':
            # with a positive after-tax return the capital grows: a value at or below K0 is ruled out at sight
            key, rel = key_k, 0.02
            cands = [m for m in stock_mistakes(K0, G, T, n) if m is not None and m > 1.01 * K0]
        elif qtype == 'capital_shortfall':
            no_tax = K0 * (1 + G) ** n
            key = no_tax - key_k
            cands = [no_tax - m for m in stock_mistakes(K0, G, T, n)[1:] if m is not None] + [
                K0 * n * G * T,                                         # simple growth
                no_tax - K0 * (1 + G * (1 - T)) ** (n - 1),             # mixed years
                key_k]                                                  # the taxed stock itself
        else:
            A = int(r.integers(2, 10))
            L = int(r.integers(10, 81))
            alpha = 0.05 * int(r.integers(4, 13))     # 0.2 .. 0.6
            beta = 0.05 * int(r.integers(10, 17))     # 0.5 .. 0.8 (output responds visibly to the capital)
            if alpha + beta > 1.0001:
                continue
            F = lambda K: A * L ** alpha * K ** beta
            key = F(key_k)
            cands = [F(m) for m in stock_mistakes(K0, G, T, n) if m is not None and m > 1.01 * K0]
            params.update(labor=str(L), prod_fn=f'{A}L^{num(alpha)}K^{num(beta)}')
            fmt, rel = num, 0.015
        if key <= 0:
            continue
        dist = pick_ranked(r, key, [c for c in cands if c is not None and c > 0],
                           rel=rel, abs_gap=0.05, shown=fmt)
        if dist is None:
            continue
        params.update(options_list=[[fmt(x) for x in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
        return params
    raise RuntimeError(f'{ELEMENT}: no {qtype} draw')


@function_registry.register(ELEMENT)
def fill_capital_market_distortions(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return _draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
