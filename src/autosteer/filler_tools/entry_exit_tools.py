"""entry_exit_decisions (STEER-ME, Production Decisions, Dynamic Production Decisions): Entry and Exit Decisions
(the paper's definition, paraphrased: whether a firm should enter or exit a market, in the long run, or shut down in the short run).

Decision of 7 October 2026; templates written here. The old filler (`gen_entry_exit_values` in dynamic_profit_tools.py,
no templates) showed a decorative production function, decided from an unrelated random cost polynomial via fsolve
(P = MC, which could return a negative output) and had two options; it is replaced.

A competitive (price-taking) firm; the firms are those of factor_demand. Two cost structures:
  long run (entry): TC(q) = F + c q + d q^2 per week, where F is avoidable by not entering. Minimum average cost
      c + 2 sqrt(d F) at q = sqrt(F / d). Enter iff P > min AC.
  short run (shut-down): TC(q) = F + a q - b q^2 + e q^3, F already committed (sunk for the season). Average variable
      cost a - b q + e q^2 is lowest at q = b / (2e), min AVC = a - b^2 / (4e). Keep producing iff P > min AVC.
Types:
  entry: Yes (open) / No (stay out). The planned output q0 stated in the story is a trap: in "enter" questions
      AC(q0) > P (comparing the price with average cost at an arbitrary output says stay out); in "stay out" questions
      MC(q0) < P (comparing the price with marginal cost says enter). Prices at least 3% from min AC.
  break_even_price: the lowest price at which entering does not lose money, c + 2 sqrt(d F). Distractors: c (the fixed
      cost ignored), c + sqrt(d F) (the factor 2 dropped), AC at the planned output, MC at the planned output.
  shutdown: keep producing / shut down temporarily. In "keep" questions P < min AC (the firm loses money either way,
      but less by producing); in "shut down" questions P is above the minimum of marginal cost (P = MC has a
      solution). Prices at least 3% from min AVC.
  shutdown_price: the price below which the firm shuts down, min AVC. Distractors: the minimum of marginal cost
      a - b^2 / (3e), a (AVC as q goes to 0), the minimum of average total cost, AVC at the output where MC is
      lowest (a - 2 b^2 / (9e)), the derivative slip a - b^2 / (2e), and average total cost at the output where AVC
      is lowest (min AVC + F / q).
Numeric options in dollars to the cent, at least 10% apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

import math

import numpy as np

from . import code_templates as ct
from .factor_demand_tools import SCENARIOS
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, stable, voice

function_registry = FillRegister()

ELEMENT = 'entry_exit_decisions'
TYPES = ('entry', 'break_even_price', 'shutdown', 'shutdown_price')
ENTER, STAY_OUT = 'Yes: open it and produce where price equals marginal cost', 'No: stay out of the market'
KEEP, SHUT = ('Keep producing, at the output where price equals marginal cost',
              'Shut down temporarily and produce nothing until the season ends')


def _parts(sc: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, f"the {sc['firm']}'s owner")
    s, firm, unit, units = v['s'], sc['firm'], sc['unit'], sc['units']
    art = 'an' if firm[0] in 'aeiou' else 'a'
    comp = (f"The market is competitive: the {firm} can sell any number of {units} at the market price, which it "
            f"takes as given.")
    if qtype in ('entry', 'break_even_price'):
        subj = 'An entrepreneur' if tag == 'anon' else v['Subj']
        be = 'is' if tag == 'anon' else v['be']
        text = (f"{subj} {be} deciding whether to open {art} {firm} {sc['what']}. If it opens, its weekly cost of "
                f"producing q {units} would be TC(q) = {{tc}} dollars, where the fixed ${{fixed}} (rent and equipment "
                f"leases) is avoided entirely by not opening. {comp} The business plan foresees producing {{q0}} "
                f"{units} a week.")
        if qtype == 'entry':
            text += f" The market price is ${{price}} per {unit}. In the long run, should the {firm} be opened?"
        else:
            text += (f" What is the lowest market price per {unit} at which opening the {firm} would not lose money in "
                     f"the long run? Round to the cent.")
    else:
        subj = f'The owner of {art} {firm}' if tag == 'anon' else v['Subj']
        text = (f"{subj} run{s} {art} {firm} {sc['what']}. Its weekly cost of producing q {units} is TC(q) = {{tc}} "
                f"dollars, where the fixed ${{fixed}} is rent under a lease for the rest of the season, which must be "
                f"paid even if the {firm} produces nothing. {comp}")
        if qtype == 'shutdown':
            text += (f" The market price has fallen to ${{price}} per {unit}. For the rest of the season, what should "
                     f"the {firm} do?")
        else:
            text += (f" Below what market price per {unit} should the {firm} shut down temporarily for the rest of the "
                     f"season? Round to the cent.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                lvl = {'entry': 1, 'shutdown': 1, 'break_even_price': 2, 'shutdown_price': 2}[qtype]
                out.append(ct.CodeTemplate(parts=_parts(sc, tag, k, qtype), type=qtype, domain=sc['domain'],
                                           tags=tag, difficulty_level=lvl, spec={'firm': k, 'type': qtype}))
    return out


def money(x: float) -> str:
    """Dollars rounded to the cent without trailing zeros ($12.3, $12), as the sampler prints them."""
    return f'${num(x)}'


def _long_run(r, qtype):
    for _ in range(100000):
        d = float(r.choice([0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 2.0]))
        F = 50 * int(r.integers(4, 61))
        c = int(r.integers(2, 41))
        qs = math.sqrt(F / d)
        mac = c + 2 * math.sqrt(d * F)
        q0 = int(round(qs * float(r.choice([0.25, 0.33, 0.4, 2.5, 3.0, 4.0]))))
        if q0 < 2:
            continue
        ac0, mc0 = c + d * q0 + F / q0, c + 2 * d * q0
        out = {'tc': f'{F} + {c}q + {d:g}q^2', 'fixed': f'{F:,}', 'q0': str(q0)}
        if qtype == 'entry':
            enter = r.random() < 0.5
            P = round(mac * (1 + float(r.uniform(0.03, 0.2))) if enter else mac * (1 - float(r.uniform(0.03, 0.2))), 2)
            trap = ac0 > P if enter else mc0 < P
            if not trap or P <= c:
                continue
            ans = ENTER if enter else STAY_OUT
            opts = [ans, STAY_OUT if enter else ENTER]
            out['price'] = num(P)
        else:
            key = mac
            dist = pick_apart(r, key, [c, c + math.sqrt(d * F), ac0, mc0], shown=money)
            if dist is None:
                continue
            opts = [money(z) for z in [key] + dist]
        if not all(stable(z) for z in opts + [out.get('price', '')]):
            continue                             # the sampler would print $12.30 as $12.3
        out.update(options_list=[opts], correct_answer=[[1] + [0] * (len(opts) - 1)])
        return out
    raise RuntimeError(f'entry_exit_decisions: no draw for {qtype}')


def min_ac(F, a, b, e):
    """Minimum of (F + a q - b q^2 + e q^3) / q over q > 0: the positive root of 2e q^3 - b q^2 - F = 0."""
    q = max(z.real for z in np.roots([2 * e, -b, 0, -F]) if abs(z.imag) < 1e-9 and z.real > 0)
    return F / q + a - b * q + e * q * q


def _short_run(r, qtype):
    for _ in range(100000):
        e = float(r.choice([0.01, 0.02, 0.03, 0.04, 0.05]))
        qm = int(r.integers(10, 41))
        b = round(2 * e * qm, 2)
        v = int(r.integers(1000, 6001)) / 100
        a = round(v + e * qm * qm, 2)                    # min AVC = a - b^2 / (4e) = v
        F = 50 * int(r.integers(4, 61))
        mavc = a - b * b / (4 * e)
        mmc = a - b * b / (3 * e)
        mac = min_ac(F, a, b, e)
        if mmc <= 1 or mac < 1.1 * mavc or mmc > 0.9 * mavc:
            continue
        out = {'tc': f'{F} + {a:g}q - {b:g}q^2 + {e:g}q^3', 'fixed': f'{F:,}'}
        if qtype == 'shutdown':
            keep = r.random() < 0.5
            P = round(float(r.uniform(mavc * 1.03, mac * 0.98)) if keep else
                      float(r.uniform(max(mmc * 1.01, 0.5), mavc * 0.97)), 2)
            if not (mavc * 1.03 <= P < mac if keep else mmc < P <= mavc * 0.97):
                continue
            ans = KEEP if keep else SHUT
            opts = [ans, SHUT if keep else KEEP]
            out['price'] = num(P)
        else:
            key = mavc
            dist = pick_apart(r, key, [mmc, a, mac, a - 2 * b * b / (9 * e), a - b * b / (2 * e), key + F / qm],
                              lo=0.5, shown=money)
            if dist is None:
                continue
            opts = [money(z) for z in [key] + dist]
        if not all(stable(z) for z in opts + [out.get('price', '')]):
            continue                             # the sampler would print $12.30 as $12.3
        out.update(options_list=[opts], correct_answer=[[1] + [0] * (len(opts) - 1)])
        return out
    raise RuntimeError(f'entry_exit_decisions: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_entry_exit_decisions(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    return _long_run(r, spec['type']) if spec['type'] in ('entry', 'break_even_price') else _short_run(r, spec['type'])


ct.register(ELEMENT, _templates())
