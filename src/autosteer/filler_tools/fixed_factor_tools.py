"""fixed_factor_aggregation (STEER-ME, Non-Strategic Decisions in Multi-Agent Environments, Factor Market
Aggregation): Aggregation of Fixed Factor Supply (the paper's definition, paraphrased: aggregating the demand of several firms for a factor whose total
supply is fixed).

Decision of 7 October 2026; templates written here. The old filler (`gen_fixed_factor_aggregation` in
production_functions_tools.py, no templates) split a fixed K between firms with Y = K^α by equalising α/K, but the
marginal product is α K^(α-1); that is right only in the log case. It is replaced.

A fixed supply K̄ of a factor (land, scanner time, studio time, ...) is rented in a competitive market to two firms
that take the rental rate as given. Firm i produces Y_i = A_i K_i^(α_i) and sells its output at a given price p_i, so
it rents until its value of marginal product p_i A_i α_i K_i^(α_i - 1) equals the rental rate; the whole supply is
rented (the rate adjusts until the two demands add up to K̄).

Types:
  equal_exponents: α_1 = α_2 = α; the key is firm 1's amount K̄ z_1 / (z_1 + z_2), z_i = (p_i A_i)^(1/(1-α)).
      Distractors: shares proportional to p_i A_i (the exponent ignored), an equal split, the prices ignored
      (A_i^(1/(1-α))), firm 2's amount.
  different_exponents: α_1 ≠ α_2; the key is firm 1's amount from p_1 A_1 α_1 K_1^(α_1-1) = p_2 A_2 α_2 (K̄ - K_1)^(α_2-1)
      (brentq). Distractors: the old rule α_1 / K_1 = α_2 / K_2 (K_1 = K̄ α_1 / (α_1 + α_2)), an equal split, shares
      proportional to p_i A_i, equal values of average product, firm 2's amount.
  rental_rate: equal exponents; the key is the market rental rate, the common value of marginal product. Distractors:
      firm 1's or firm 2's value of marginal product at an equal split, the value of average product (α left out),
      firm 1's value of marginal product if it rented the whole supply.
Options at least 10% apart from the key and from each other; amounts to 2 decimals, rates in dollars to the cent.

Perspectives (tags): first_person, second_person, man, woman, anon (the owner of the fixed supply).
"""
from __future__ import annotations

from scipy.optimize import brentq

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'fixed_factor_aggregation'
TYPES = ('equal_exponents', 'different_exponents', 'rental_rate')

STORIES = [
    dict(domain='environmental_policy', owner='a landowner', what='hectares of irrigated farmland in a valley',
         unit='hectare', per='season',
         f1=dict(name='a vineyard', out='tonnes of grapes', unit='tonne'),
         f2=dict(name='a wheat farm', out='tonnes of wheat', unit='tonne')),
    dict(domain='technology', owner='a data-centre operator', what='server-hours of computing capacity a day',
         unit='server-hour', per='day',
         f1=dict(name='an animation studio', out='rendered scenes', unit='scene'),
         f2=dict(name='a weather-forecasting firm', out='forecast runs', unit='run')),
    dict(domain='consumer_goods', owner='a market-hall owner', what='square meters of stall space in a market hall',
         unit='square meter', per='month',
         f1=dict(name='a bakery', out='loaves of bread', unit='loaf'),
         f2=dict(name='a cheese shop', out='kilograms of cheese', unit='kilogram')),
    dict(domain='entertainment', owner='a studio owner', what='hours of time on the only sound stage in town',
         unit='hour', per='month',
         f1=dict(name='a film producer', out='minutes of finished footage', unit='minute'),
         f2=dict(name='an advertising producer', out='finished commercials', unit='commercial')),
    dict(domain='medical', owner='an imaging centre', what='hours of MRI-scanner time a week', unit='hour',
         per='week',
         f1=dict(name='a sports-injury clinic', out='diagnostic scans', unit='scan'),
         f2=dict(name='a research laboratory', out='study scans', unit='scan')),
    dict(domain='sports', owner='a rink operator', what='hours of ice time a week at the town rink', unit='hour',
         per='week',
         f1=dict(name='a hockey school', out='player-lessons', unit='player-lesson'),
         f2=dict(name='a figure-skating club', out='skater-classes', unit='skater-class')),
]


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, st['owner'])
    f1, f2 = st['f1'], st['f2']
    the1 = 'the ' + f1['name'].split(' ', 1)[1]
    owner = st['owner'][0].upper() + st['owner'][1:] if tag == 'anon' else v['Subj']
    text = (f"{owner} own{v['s']} all {{kbar}} {st['what']} available and rent{v['s']} them out in a competitive "
            f"market to two firms, {f1['name']} (firm 1) and {f2['name']} (firm 2). Each firm takes the rental rate "
            f"per {st['unit']} as given and rents the amount that maximizes its profit, and the rate adjusts until "
            f"the whole supply is rented. With K {st['unit']}s, firm 1 produces {{f1}} {f1['out']} per {st['per']}, "
            f"which it sells at ${{p1}} per {f1['unit']}, and firm 2 produces {{f2}} {f2['out']} per {st['per']}, "
            f"which it sells at ${{p2}} per {f2['unit']}; the firms have no other costs that depend on K.")
    text += (" The two production functions have different exponents on K." if qtype == 'different_exponents' else
             " The two production functions have the same exponent on K.")
    if qtype == 'rental_rate':
        text += (f" What is the equilibrium rental rate per {st['unit']} per {st['per']}? Round to the cent.")
    else:
        text += (f" In equilibrium, how many {st['unit']}s does {the1} (firm 1) rent? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=1 if qtype == 'equal_exponents' else 2,
                                           spec={'story': k, 'type': qtype}))
    return out


def _root(f, kbar):
    lo, hi = 1e-6 * kbar, kbar * (1 - 1e-6)
    if f(lo) * f(hi) > 0:
        return None                              # the split is within a millionth of a corner: not used
    return brentq(f, lo, hi, xtol=1e-12)


def split(kbar, v1, a1, v2, a2):
    """Firm 1's amount where v1 a1 K^(a1-1) = v2 a2 (kbar - K)^(a2-1); v_i = p_i A_i."""
    return _root(lambda K: v1 * a1 * K ** (a1 - 1) - v2 * a2 * (kbar - K) ** (a2 - 1), kbar)


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        kbar = 10 * int(r.integers(10, 121))
        A1, A2 = int(r.integers(2, 31)), int(r.integers(2, 31))
        p1, p2 = int(r.integers(2, 41)), int(r.integers(2, 41))
        v1, v2 = p1 * A1, p2 * A2
        if qtype == 'different_exponents':
            a1, a2 = (round(0.05 * int(z), 2) for z in r.integers(4, 17, 2))
            if abs(a1 - a2) < 0.1:
                continue
        else:
            a1 = a2 = round(0.05 * int(r.integers(4, 17)), 2)
        if v1 == v2 or A1 == A2:
            continue
        K1 = split(kbar, v1, a1, v2, a2)
        if K1 is None or not 0.1 * kbar <= K1 <= 0.9 * kbar:
            continue
        if qtype == 'equal_exponents':
            e = 1 / (1 - a1)
            key = K1
            cands = [kbar * v1 / (v1 + v2), kbar / 2, kbar * A1 ** e / (A1 ** e + A2 ** e), kbar - K1]
        elif qtype == 'different_exponents':
            key = K1
            ap = _root(lambda K: v1 * K ** (a1 - 1) - v2 * (kbar - K) ** (a2 - 1), kbar)
            cands = [kbar * a1 / (a1 + a2), kbar / 2, kbar * v1 / (v1 + v2), ap, kbar - K1]
        else:
            key = v1 * a1 * K1 ** (a1 - 1)
            h = kbar / 2
            cands = [v1 * a1 * h ** (a1 - 1), v2 * a2 * h ** (a2 - 1), v1 * K1 ** (a1 - 1), v1 * a1 * kbar ** (a1 - 1)]
        if key < 0.5:
            continue
        dist = pick_apart(r, key, cands, lo=0.01, shown=num)
        if dist is None:
            continue
        fmt = (lambda z: f'${num(z)}') if qtype == 'rental_rate' else num
        return {'kbar': f'{kbar:,}', 'f1': f'{A1} * K^{a1:g}', 'f2': f'{A2} * K^{a2:g}', 'p1': str(p1),
                'p2': str(p2), 'options_list': [[fmt(z) for z in [key] + dist]], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError(f'fixed_factor_aggregation: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_fixed_factor_aggregation(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
