"""non_distortionary_taxes (STEER-ME, Evaluating Equilibria and Externalities, Welfare Analysis of Market
Equilibrium): Identify Non-Distortionary Taxes (the paper's definition, paraphrased: which taxes raise revenue without distorting behavior).

Decision of 7 October 2026; templates written here. The old filler (`gen_lump_sum_tax_efficiency_question` in
production_functions_tools.py, no templates; data name lump_sum_tax_efficiency, kept as an alias) keyed "labor is
unchanged at A/2" under a lump-sum tax with u = ln c + ln(A - l), but with c = w l - T the optimum is (A + T/w)/2 (an
income effect); it is replaced. A tax is non-distortionary when nobody can change what they owe by changing what they
do (a lump-sum tax), or when the taxed quantity cannot respond (a fixed factor); then it creates no deadweight loss.

Types:
  identify: a government's adviser considers four taxes, described in words with amounts. The key is either a
      lump-sum charge (a fixed amount per adult, whatever they earn, buy or do) or a tax on the rent of land whose
      amount is fixed (half each); the other three come from seven distortionary taxes (on wages, a per-unit sales
      tax, interest on savings, an import tariff, overtime hours, an employer payroll tax, a per-room hotel tax).
      The lump-sum and the land tax are never both options.
  zero_dwl: four per-unit taxes, each in a market with stated linear demand and supply (Q_d = a - bP, Q_s = c + dP);
      in one market supply is fixed (Q_s = constant, a good such as parking spaces; the text does not say it is fixed
      except through the supply equation), so the tax there creates no deadweight loss. The four tax amounts are
      drawn so that the key's rank among them is uniform (review, October 2026: the zero-deadweight-loss tax used to be always the largest,
      which gave the key away), the four amounts are distinct; every other tax
      has a deadweight loss of at least $5.
  The identify stem asks which tax creates no deadweight loss (it said "without distorting anyone's choices"; a
  lump-sum tax still has income effects) and states that nobody moves because of the taxes and the amount of land is
  fixed; every option has the form "A tax of <amount> on <base>" (the land tax is on bare land, without buildings).

Perspectives (tags): first_person, second_person, man, woman, anon (an adviser to the government).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, voice

function_registry = FillRegister()

ELEMENT = 'non_distortionary_taxes'
TYPES = ('identify', 'zero_dwl')
LETTERS = 'ABCD'

CONTEXTS = [
    dict(domain='finance', gov='a national government', need='to raise revenue to reduce its debt'),
    dict(domain='environmental_policy', gov='a regional council', need='to pay for flood defences'),
    dict(domain='education', gov='a city government', need='to fund its public schools'),
    dict(domain='medical', gov='a provincial government', need='to fund its hospitals'),
    dict(domain='consumer_goods', gov='the government of a small island', need='to pay for a new harbour'),
    dict(domain='technology', gov='a town council', need='to pay for public broadband'),
]

# Options are short and of one form, "A tax of <amount> on <base>" (re-review: the key used to repeat the stem's
# condition, "the same whatever they earn, buy or do", and was much longer than the distractors).
LUMP_SUM = 'A tax of ${amt} a year on every adult resident'
LAND = 'A tax of {pct}% on the rental value of bare land'
DISTORTING = [
    'A tax of {pct}% on all wage income',
    'A tax of ${t} on every restaurant meal sold',
    'A tax of {pct}% on interest earned on savings',
    'A tax of {pct}% on all imported cars',
    'A tax of ${t} on every hour of overtime worked',
    'A tax of {pct}% on the wages employers pay',
    'A tax of ${t} on every hotel room let per night',
]
GOODS = ['apples (kilograms)', 'bicycle repairs', 'cinema tickets', 'gym memberships', 'printer paper (reams)',
         'taxi rides', 'potted plants', 'phone cases']
FIXED_GOODS = ['parking spaces in the old town', 'riverside allotment plots', 'mooring berths in the harbour',
               'stalls in the historic market hall']


def _intro(tag, k, ctx):
    v = voice(tag, k, 'an economist')
    subj = 'An economist' if tag == 'anon' else v['Subj']
    return (f"{subj} advise{v['s']} {ctx['gov']}, which needs {ctx['need']}.", v)


def _parts(ctx: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    intro, v = _intro(tag, k, ctx)
    if qtype == 'identify':
        text = (f"{intro} The government is considering four taxes, each raising about the same revenue. "
                f"Assume nobody moves into or out of the area because of any of these taxes, and the total amount of "
                f"land is fixed. Which of these taxes creates no deadweight loss?")
    else:
        text = (f"{intro} The government is considering a per-unit tax in one of four competitive markets, where P is "
                f"the price in dollars and quantities are per month: {{markets}}. Comparing the deadweight loss each "
                f"tax would create, which tax creates no deadweight loss?")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, ctx in enumerate(CONTEXTS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(ctx, tag, k, qtype), type=qtype, domain=ctx['domain'],
                                           tags=tag, difficulty_level=1 if qtype == 'identify' else 2,
                                           spec={'context': k, 'type': qtype}))
    return out


def _fill_tax(r, text: str) -> str:
    return text.format(amt=f'{50 * int(r.integers(4, 41)):,}', pct=str(int(r.integers(2, 31))),
                       t=num(0.25 * int(r.integers(2, 21))))


def _identify(r):
    good = _fill_tax(r, LUMP_SUM if r.random() < 0.5 else LAND)
    idx = [int(i) for i in r.choice(len(DISTORTING), 3, replace=False)]
    bad = [_fill_tax(r, DISTORTING[i]) for i in idx]
    return {'options_list': [[good] + bad], 'correct_answer': [[1, 0, 0, 0]]}


def dwl(a, b, c, d, t):
    """Deadweight loss of a per-unit tax t with Q_d = a - bP and Q_s = c + dP (d = 0: fixed supply)."""
    q0 = (a * d + b * c) / (b + d)
    qt = (a * d + b * c - b * d * t) / (b + d)
    return 0.5 * t * (q0 - qt)


def _zero_dwl(r):
    rank = int(r.integers(0, 4))   # rank of the zero-DWL tax among the four (0 = smallest), uniform
    for _ in range(100000):
        markets = []
        for _ in range(3):
            b, d = int(r.integers(1, 16)), int(r.integers(1, 16))
            p0 = int(r.integers(10, 101))
            c = int(r.integers(0, 101))
            a = c + (b + d) * p0                      # equilibrium price p0, quantity c + d p0
            t = num(0.5 * int(r.integers(1, 21)))
            markets.append((a, b, c, d, float(t)))
        b = int(r.integers(1, 16))
        p0 = int(r.integers(10, 101))
        qbar = 10 * int(r.integers(5, 101))
        tf = float(num(0.5 * int(r.integers(1, 21))))
        fixed = (qbar + b * p0, b, qbar, 0, tf)
        taxes = [m[4] for m in markets]
        if len(set(taxes + [tf])) < 4 or sorted(taxes + [tf]).index(tf) != rank:
            continue
        if min(dwl(*m) for m in markets) < 5:
            continue
        if any((m[0] * m[3] + m[1] * m[2] - m[1] * m[3] * m[4]) / (m[1] + m[3]) <= 0 for m in markets):
            continue
        goods = [str(g) for g in r.choice(GOODS, 3, replace=False)]
        fgood = str(r.choice(FIXED_GOODS))
        allm = markets + [fixed]
        names = goods + [fgood]
        order = [int(i) for i in r.permutation(4)]
        parts = []
        for j, i in enumerate(order):
            a, b_, c, d, t = allm[i]
            sup = f'Q_s = {c:,}' if d == 0 else f'Q_s = {c:,} + {d}P'
            parts.append(f'market {LETTERS[j]}, {names[i]}: Q_d = {a:,} - {b_}P and {sup}, tax ${num(t)} per unit')
        key = LETTERS[order.index(3)]
        opts = [f'The tax in market {key}'] + [f'The tax in market {x}' for x in LETTERS if x != key]
        return {'markets': '; '.join(parts), 'options_list': [opts], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError('non_distortionary_taxes: no zero_dwl draw')


@function_registry.register(ELEMENT)
def fill_non_distortionary_taxes(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    return _identify(r) if spec['type'] == 'identify' else _zero_dwl(r)


ct.register(ELEMENT, _templates())
