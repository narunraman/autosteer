"""elastic_inelastic_supply (STEER-ME, Non-Strategic Decisions in Multi-Agent Environments, Comparative Statics of
Equilibrium Prices): Comparative Statics with Inelastic or Perfectly Elastic Supply (the paper's definition, paraphrased: how the equilibrium price
responds to cost and demand shocks when supply is perfectly inelastic or perfectly elastic).

Decision of 7 October 2026; templates written here. The old filler (`gen_comparative_statics_price_shock` in
production_functions_tools.py, no templates; data name comparative_stats_supply_price_shock, kept as an alias)
returned the percentage change in unit cost after an input-price shock, which is only the perfectly elastic case. It
is replaced. The element is kept distinct from tfp_shocks (one input-price or productivity shock in a single
Cobb-Douglas firm): here the supply regime is the point, both input prices can move, and the fixed-supply case is
included.

Perfectly elastic supply: a constant-returns industry Q = A L^a K^(1-a) with free entry and input prices that do not
depend on the industry's size; the long-run price equals unit cost, proportional to w^a r^(1-a).
Perfectly inelastic supply: a fixed quantity Q̄ that is offered whatever the price (costs already below the price),
with linear demand Q = α - βP; the price is (α - Q̄) / β.

Types:
  elastic_cost_shock: the wage changes by x% and the rental rate by y% (one of them may be 0); key the percentage price
      change 100 ((1 + x)^a (1 + y)^(1-a) - 1). Distractors: the first-order a x + (1-a) y, the full wage change x,
      0% (the fixed-supply answer), x + y, the wage effect alone.
  inelastic_cost_shock: Q̄ fixed and the sellers' cost per unit rises from c0 to c1 (still below the price); key 0%.
      Distractors: the cost's percentage change, the cost increase as a percentage of the price, half of that.
  inelastic_demand_shift: Q̄ fixed and demand shifts out (α to α'); key 100 (α' - α) / (α - Q̄). Distractors: the
      percentage change of α, 0% (the elastic-supply answer), the shift as a percentage of Q̄, half the key.
  compare: the same wage rise x% in a fixed-supply market (market 1) and a constant-returns free-entry industry
      (market 2); options pair the two price changes. Key (0%, the elastic change). Distractors: swapped, both the
      elastic change, (0%, x%) (full wage pass-through).
Percentages to 2 decimals; numeric options at least 10% (and 0.5 points) apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon (an analyst studying the market).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'elastic_inelastic_supply'
TYPES = ('elastic_cost_shock', 'inelastic_cost_shock', 'inelastic_demand_shift', 'compare')

ELASTIC = [
    dict(domain='consumer_goods', good='plain cotton T-shirts', unit='T-shirt', lab='sewing labor',
         cap='sewing-machine time'),
    dict(domain='technology', good='basic USB cables', unit='cable', lab='assembly labor', cap='extruder time'),
    dict(domain='medical', good='disposable face masks', unit='box', lab='operator labor', cap='mask-machine time'),
    dict(domain='entertainment', good='printed event posters', unit='poster', lab='print-shop labor',
         cap='press time'),
    dict(domain='environmental_policy', good='recycled paper', unit='ream', lab='mill labor', cap='pulper time'),
    dict(domain='sports', good='tennis balls', unit='can', lab='factory labor', cap='moulding-press time'),
]
INELASTIC = [
    dict(domain='entertainment', good='tickets to a sold-out stadium concert', unit='ticket',
         fixed='the stadium has {qbar} seats, all of which are sold', cost='staffing and cleaning per seat',
         buyers='fans'),
    dict(domain='consumer_goods', good='hotel rooms in an island town during its festival week', unit='room',
         fixed='the town has {qbar} hotel rooms, all of which are let for the week',
         cost='cleaning and laundry per room', buyers='visitors'),
    dict(domain='environmental_policy', good='irrigated farmland in a valley, rented for one season', unit='hectare',
         fixed='the valley has {qbar} hectares, all of which are rented out',
         cost='maintaining the irrigation channels per hectare', buyers='farmers'),
    dict(domain='sports', good='mooring berths in the only marina on a lake, rented for the summer', unit='berth',
         fixed='the marina has {qbar} berths, all of which are rented out', cost='maintenance per berth',
         buyers='boat owners'),
    dict(domain='education', good='places at a summer school with a fixed size', unit='place',
         fixed='the school has {qbar} places, all of which are filled', cost='teaching materials per place',
         buyers='families'),
]


def _intro(tag, k, good):
    v = voice(tag, k, 'an analyst')
    if tag == 'anon':
        return f'Consider the market for {good}.'
    return f"{v['Subj']} {v['be']} studying the market for {good}."


def _fixed_text(st):
    return (f"The quantity is fixed: {st['fixed']} whatever the price, because "
            f"no more can be added and every {st['unit']} is worth offering at any price above its running cost. "
            f"Demand from {st['buyers']} is Q = {{alpha}} - {{beta}}P, where P is the price per {st['unit']} in "
            f"dollars.")


def _elastic_text(st, wage_word='wage'):
    return (f"Every producer uses the constant-returns technology Q = A * L^{{a}} * K^{{b}}, where L is hours of "
            f"{st['lab']} and K hours of {st['cap']}; firms can freely enter and leave the industry, and the "
            f"industry is too small to affect the {wage_word} or the rental rate, so in the long run supply is "
            f"perfectly elastic at the minimum unit cost.")


def _parts(tag: str, k: int, qtype: str) -> tuple[str]:
    if qtype == 'elastic_cost_shock':
        st = ELASTIC[k]
        text = (f"{_intro(tag, k, st['good'])} {_elastic_text(st)} The wage of {st['lab']} {{wchange}} and the "
                f"rental rate of {st['cap']} {{rchange}}. By what percentage does the long-run equilibrium price of "
                f"{st['good']} change? Round to two decimals (negative for a fall).")
    elif qtype == 'compare':
        e, f = ELASTIC[k], INELASTIC[k]
        text = (f"{_intro(tag, k, f['good'] + ' (market 1) and the market for ' + e['good'] + ' (market 2)')} "
                f"In market 1, {_fixed_text(f)[0].lower() + _fixed_text(f)[1:]} In market 2, "
                f"{_elastic_text(e)[0].lower() + _elastic_text(e)[1:]} In both markets the hourly wage of workers "
                f"rises by {{x}}% (in market 1 this raises the cost of {f['cost']}, which stays well below the price), and "
                f"nothing else changes. By what percentage does the equilibrium price change in each market? Round "
                f"to two decimals.")
    else:
        st = INELASTIC[k]
        text = f"{_intro(tag, k, st['good'])} {_fixed_text(st)}"
        if qtype == 'inelastic_cost_shock':
            text += (f" The cost of {st['cost']} rises from ${{c0}} to ${{c1}}, still well below the price, and "
                     f"nothing else changes. By what percentage does the equilibrium price per {st['unit']} change? "
                     f"Round to two decimals.")
        else:
            text += (f" Demand rises: at every price, {st['buyers']} now want {{shift}} more {st['unit']}s, so demand "
                     f"becomes Q = {{alpha2}} - {{beta}}P. By what percentage does the equilibrium price per "
                     f"{st['unit']} change? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for qtype in TYPES:
        n = len(ELASTIC) if qtype == 'elastic_cost_shock' else len(INELASTIC)
        for k in range(n):
            dom = (ELASTIC if qtype == 'elastic_cost_shock' else INELASTIC)[k]['domain']
            for tag in PERSPECTIVES:
                out.append(ct.CodeTemplate(parts=_parts(tag, k, qtype), type=qtype, domain=dom, tags=tag,
                                           difficulty_level={'inelastic_cost_shock': 1, 'compare': 1,
                                                             'elastic_cost_shock': 2,
                                                             'inelastic_demand_shift': 2}[qtype],
                                           spec={'k': k, 'type': qtype}))
    return out


def pct(x: float) -> str:
    return f'{num(x)}%'


def _change(z: int) -> str:
    if z == 0:
        return 'stays the same'
    return f'rises by {z}%' if z > 0 else f'falls by {-z}%'


def _demand(r):
    beta = int(r.integers(1, 21))
    p0 = int(r.integers(20, 201))
    qbar = 10 * int(r.integers(10, 201))
    return beta, p0, qbar, qbar + beta * p0


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        if qtype in ('elastic_cost_shock', 'compare'):
            a = round(0.05 * int(r.integers(4, 17)), 2)
            b = round(1 - a, 2)
        if qtype == 'elastic_cost_shock':
            x = int(r.integers(-30, 61))
            y = int(r.choice([0, int(r.integers(-30, 61))]))
            if x == 0 or x == y:
                continue
            key = 100 * ((1 + x / 100) ** a * (1 + y / 100) ** b - 1)
            cands = [a * x + b * y, float(x), 0.0, float(x + y), 100 * ((1 + x / 100) ** a - 1)]
            dist = pick_apart(r, key, cands, abs_gap=0.5, shown=num)
            if dist is None:
                continue
            return {'a': f'{a:g}', 'b': f'{b:g}', 'wchange': _change(x), 'rchange': _change(y),
                    'options_list': [[pct(z) for z in [key] + dist]], 'correct_answer': [[1, 0, 0, 0]]}
        beta, p0, qbar, alpha = _demand(r)
        out = {'qbar': f'{qbar:,}', 'alpha': f'{alpha:,}', 'beta': str(beta)}
        if qtype == 'inelastic_cost_shock':
            c0 = int(r.integers(2, max(3, p0 // 3)))
            x = 5 * int(r.integers(2, 13))
            c1 = c0 * (1 + x / 100)
            if c1 >= 0.5 * p0 or abs(round(c1, 2) - c1) > 1e-9:
                continue
            key = 0.0
            cands = [float(x), 100 * (c1 - c0) / p0, 50 * (c1 - c0) / p0]
            out.update(c0=num(c0), c1=num(c1))
        elif qtype == 'inelastic_demand_shift':
            shift = 10 * int(r.integers(2, 41))
            key = 100 * shift / (alpha - qbar)
            cands = [100 * shift / alpha, 0.0, 100 * shift / qbar, key / 2]
            out.update(shift=f'{shift:,}', alpha2=f'{alpha + shift:,}')
        else:
            x = int(r.integers(5, 41))
            e = 100 * ((1 + x / 100) ** a - 1)
            if abs(e - x) < 0.1 * x:
                continue
            pair = lambda m1, m2: f'Market 1: {pct(m1)}; market 2: {pct(m2)}'   # noqa: E731
            opts = [pair(0, e), pair(e, 0), pair(e, e), pair(0, x)]
            out.update(x=str(x), a=f'{a:g}', b=f'{b:g}', options_list=[opts], correct_answer=[[1, 0, 0, 0]])
            return out
        dist = pick_apart(r, key, cands, abs_gap=0.5, shown=num)
        if dist is None:
            continue
        out.update(options_list=[[pct(z) for z in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
        return out
    raise RuntimeError(f'elastic_inelastic_supply: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_elastic_inelastic_supply(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())
