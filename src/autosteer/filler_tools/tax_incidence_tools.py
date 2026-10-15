"""tax_incidence (STEER-ME, Evaluating Equilibria and Externalities, Welfare Analysis of Market Equilibrium):
Irrelevance of Tax Incidence.

Decision of 5 October 2026; templates written here. Each question has two parts about the same competitive market
with linear demand and supply

    Q_d = a - b P,   Q_s = c + d P   (a, b, d > 0; c may be negative),

and a per-unit tax of t dollars. Part 0: sellers must pay the tax to the government for every unit they sell.
Part 1: buyers must pay it for every unit they buy. Let P_b be what buyers pay per unit in total (the market price
plus any tax they pay themselves) and P_s what sellers keep (the market price minus any tax they pay). Either way
Q_d(P_b) = Q_s(P_s) with P_b - P_s = t, so

    P_b = P0 + d t / (b + d),   P_s = P0 - b t / (b + d),   Q = Q0 - b d t / (b + d),

where (P0, Q0) is the untaxed equilibrium. Who is legally liable does not matter: the two parts have the same key.

Types (what both parts ask): buyer_price (P_b, to the cent), seller_price (P_s, to the cent), quantity (Q, to the
nearest whole unit).

The parameters are drawn from the untaxed equilibrium: P0, Q0 integers, slopes b = e_d Q0 / P0 and d = e_s Q0 / P0
rounded to whole numbers (point elasticities e_d, e_s with e_d / e_s in [1/3, 3], so each side bears at least a
quarter of the tax), a = Q0 + b P0, c = Q0 - d P0, and t a multiple of $0.25 between 10% and 40% of P0 (not ending in 50 cents, which the sampler would show as
"$2.5"). Draws with a price option whose cents end in 0 are redrawn for the same reason, and so are draws whose
exact key is a half cent (prices) or half unit (quantity), where half-up and half-even rounding differ.

Distractors are the asked quantity under a common mistake: no tax (the untaxed equilibrium); the whole tax borne
by one side (the statutory payer, or the other side); the tax split 50/50; the incidence shares swapped (the more
elastic side bearing more); the other side's price (P_s for P_b and vice versa); for the quantity, demand at P0 + t,
supply at P0 - t, demand at P0 + t/2 and the tax counted twice. Three are chosen, at least 2% (prices; 1.5% and 2
units for quantities) apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon (an analyst studying the market, or "Consider").
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, dollars_amount, money, pick_apart, stable, voice

function_registry = FillRegister()

ELEMENT = 'tax_incidence'
TYPES = ('buyer_price', 'seller_price', 'quantity')

# domain, market, unit, units, period, P0 range, Q0 range
MARKETS = [
    dict(domain='consumer_goods', market='the market for bottled olive oil in a region', unit='bottle',
         units='bottles', per='week', p0=(8, 30), q0=(2000, 9000)),
    dict(domain='entertainment', market='the market for concert tickets in a city', unit='ticket', units='tickets',
         per='month', p0=(30, 120), q0=(500, 5000)),
    dict(domain='education', market='the market for private tutoring in a town', unit='hour of tutoring',
         units='hours of tutoring', per='week', p0=(20, 70), q0=(300, 3000)),
    dict(domain='environmental_policy', market='the market for firewood in a mountain county', unit='cord',
         units='cords', per='month', p0=(150, 400), q0=(2000, 12000)),
    dict(domain='sports', market='the market for gym memberships in a suburb', unit='membership',
         units='memberships', per='month', p0=(25, 90), q0=(400, 4000)),
    dict(domain='technology', market='the market for phone chargers in a country', unit='charger', units='chargers',
         per='week', p0=(10, 40), q0=(5000, 40000)),
    dict(domain='medical', market='the market for flu shots at private clinics in a city', unit='flu shot',
         units='flu shots', per='month', p0=(15, 60), q0=(1000, 8000)),
    dict(domain='politics', market='the market for cigarettes in a province', unit='pack', units='packs',
         per='week', p0=(6, 20), q0=(10000, 60000)),
]


def _intro(tag: str, k: int, mk: dict) -> str:
    if tag == 'anon':
        return f"Consider {mk['market']}."
    v = voice(tag, k, '')
    return f"{v['Subj']} {v['be']} an analyst studying {mk['market']}."


def _parts(mk: dict, tag: str, k: int, qtype: str) -> tuple[str, str]:
    unit, units, per = mk['unit'], mk['units'], mk['per']
    setup = (f"{_intro(tag, k, mk)} The quantity demanded per {per} is Q_d = {{demand}} and the quantity supplied "
             f"per {per} is Q_s = {{supply}}, where P is the price in dollars per {unit} and Q is the number of "
             f"{units}. The market is competitive.")
    who = [f"sellers must pay to the government for every {unit} they sell",
           f"buyers must pay to the government for every {unit} they buy"]
    if qtype == 'buyer_price':
        ask = (f"Once the market adjusts to the tax, what is the total amount buyers pay per {unit}, including any "
               f"tax they pay to the government themselves? Give the amount to the nearest cent.")
    elif qtype == 'seller_price':
        ask = (f"Once the market adjusts to the tax, how much do sellers keep per {unit}, after any tax they pay to "
               f"the government themselves? Give the amount to the nearest cent.")
    else:
        ask = (f"Once the market adjusts to the tax, how many {units} are bought and sold per {per}? Round to the "
               f"nearest whole number.")
    return tuple(f"{setup} The government introduces a tax of ${{tax}} per {unit}, which {w}. {ask}" for w in who)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, mk in enumerate(MARKETS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(mk, tag, k, qtype), type=qtype, domain=mk['domain'],
                                           tags=tag, difficulty_level=1, spec={'market': k, 'type': qtype}))
    return out


def equilibrium(a, b, c, d, t, payer: str):
    """(P_b, P_s, Q) with the tax collected from `payer` ('sellers' or 'buyers'), solved on the shifted curve."""
    if payer == 'sellers':
        # supply in terms of the market price P_b: sellers keep P_b - t
        pb = (a - c + d * t) / (b + d)
        ps = pb - t
    else:
        # demand in terms of the market price P_s: buyers pay P_s + t
        ps = (a - c - b * t) / (b + d)
        pb = ps + t
    return pb, ps, a - b * pb


def linear(intercept: int, slope: int, sign: str) -> str:
    """'1200 - 30P' / '300 + 25P' / '25P - 300' ('P' for a slope of 1)."""
    p = 'P' if slope == 1 else f'{slope:,}P'
    if sign == '-':
        return f'{intercept:,} - {p}'
    if intercept >= 0:
        return f'{intercept:,} + {p}'
    return f'{p} - {-intercept:,}'


def draw(r, mk: dict, qtype: str) -> dict:
    while True:
        P0 = int(r.integers(mk['p0'][0], mk['p0'][1] + 1))
        Q0 = int(r.integers(mk['q0'][0] // 10, mk['q0'][1] // 10 + 1)) * 10
        ed = float(r.uniform(0.3, 2.0))
        es = float(r.uniform(0.3, 2.0))
        if not 1 / 3 <= ed / es <= 3:
            continue
        b, d = round(ed * Q0 / P0), round(es * Q0 / P0)
        if b < 1 or d < 1 or b == d:
            continue
        a, c = Q0 + b * P0, Q0 - d * P0
        t = round(float(r.uniform(0.1, 0.4)) * P0 * 4) / 4
        if t < 0.25 or not stable(dollars_amount(t)):
            continue
        pb0, ps0, q0 = equilibrium(a, b, c, d, t, 'sellers')
        pb1, ps1, q1 = equilibrium(a, b, c, d, t, 'buyers')
        if q0 <= 0 or ps0 <= 0:
            continue
        sh_b, sh_s = d / (b + d), b / (b + d)
        if qtype == 'buyer_price':
            keys = (pb0, pb1)
            cands = [P0, P0 + t, P0 + t / 2, P0 + sh_s * t, ps0]
            fmt, opt = money, dict(rel=0.02, abs_gap=0.05, lo=0.01)
        elif qtype == 'seller_price':
            keys = (ps0, ps1)
            cands = [P0, P0 - t, P0 - t / 2, P0 - sh_b * t, pb0]
            fmt, opt = money, dict(rel=0.02, abs_gap=0.05, lo=0.01)
        else:
            keys = (q0, q1)
            cands = [Q0, Q0 - b * t, Q0 - d * t, Q0 - b * t / 2, Q0 - 2 * b * d * t / (b + d)]
            fmt, opt = (lambda x: f'{round(x):,}'), dict(rel=0.015, abs_gap=2, lo=1)
        if fmt(keys[0]) != fmt(keys[1]):
            continue
        # review: an exact half cent ($300.025) or half unit has no agreed rounding (half-up vs half-even)
        scale = 1 if qtype == 'quantity' else 100
        if any(abs((k * scale) % 1 - 0.5) < 1e-6 for k in keys):
            continue
        opts = []
        for key in keys:
            dist = pick_apart(r, key, cands, shown=fmt, **opt)
            if dist is None:
                break
            opts.append([fmt(x) for x in [key] + dist])
        if len(opts) < 2 or not all(stable(o) for o in opts[0] + opts[1]):
            continue
        return dict(a=a, b=b, c=c, d=d, t=t, opts=opts)


@function_registry.register(ELEMENT)
def fill_tax_incidence(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    x = draw(filler_utils.rng, MARKETS[spec['market']], spec['type'])
    return {
        'demand': linear(x['a'], x['b'], '-'),
        'supply': linear(x['c'], x['d'], '+'),
        'tax': dollars_amount(x['t']),
        'options_list': x['opts'],
        'correct_answer': [[1, 0, 0, 0], [1, 0, 0, 0]],
    }


ct.register(ELEMENT, _templates())
