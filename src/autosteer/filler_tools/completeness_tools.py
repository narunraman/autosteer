"""completeness (STEER 2.1.a) and completeness_risk (2.3.a): every pair of a set of alternatives, graded together.

Design of October 2026 (user decision; replaces the pipeline templates, whose key was "the alternative with the larger
stated number", which tests maximisation, not completeness). One question describes four or five fully specified
alternatives and has one part per pair of them (6 or 10 parts). Every part shows the same description and asks about
one pair, with four options:

    I prefer X  /  I prefer Y  /  I am indifferent between X and Y  /  I cannot decide between X and Y; I would need
    more information

The description says that it is complete (the alternatives differ in nothing else and no more information is
available), so the last option is a failure to rank, not a request for missing facts. Completeness asks only that
some ranking be given: any preference or indifference is fine. The parts are graded together by the consistency rule
`pairwise_complete` (release/consistency.py, built in release/rows.py): complete iff no part is answered "cannot
decide"; the transitivity of the answered pairs is reported as a secondary measure (score_transitivity). No option
is marked correct.

completeness: sure outcomes with two or three attributes that trade off (job offers, apartments, laptops, holiday
    packages, medical treatments with certain outcomes, guaranteed savings bonds).
completeness_risk: lotteries with stated, exact chances (game-show envelopes, one-year investments, treatments with a
    chance of a fast recovery, bonus plans with three outcomes, crops whose earnings depend on the weather, travel
    routes with a chance of delay).

Every template is second person (the reader's own preference; advice for someone else would make "need more
information" about their tastes defensible). Parts follow itertools.combinations order of the alternatives (A-B,
A-C, ...); the options of a part are [prefer first, prefer second, indifferent, cannot decide] before the sampler
shuffles them. Difficulty: four alternatives (level 1) or five (level 2). Two stems per domain and size, so few-shot
examples can be held out without removing a test cell. The numbers are drawn so that no two alternatives are the
same and no alternative is better than every other one on every attribute (each pair is a real trade-off or a
dominance, never a duplicate).
"""
from __future__ import annotations

from itertools import combinations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

SURE, RISK = 'completeness', 'completeness_risk'
LABELS = 'ABCDE'
COMPLETE_NOTE = ('These descriptions are complete: the {plural} differ in nothing else, and no more information is '
                 'available.')


def _money(v: int) -> str:
    return f'${v:,}'


# ---------------------------------------------------------------------------------------------- sure outcomes
# domain -> (singular noun, plural, intro stems (two), attribute drawer, description)
def _job(r):
    return (int(r.integers(50, 91)) * 1000, int(r.integers(2, 13)) * 5, int(r.integers(15, 31)))


def _job_text(v):
    s, c, h = v
    return f'a salary of {_money(s)} a year, a {c}-minute commute each way and {h} days of paid holiday a year'


def _flat(r):
    return (int(r.integers(24, 49)) * 50, int(r.integers(9, 20)) * 5, int(r.integers(2, 21)))


def _flat_text(v):
    rent, size, walk = v
    return (f"rent of {_money(rent)} a month, {size} square metres of floor space and {walk} minutes' walk from the "
            f"station")


def _laptop(r):
    return (int(r.integers(14, 37)) * 50, int(r.integers(6, 19)), int(r.integers(10, 25)) / 10)


def _laptop_text(v):
    price, battery, weight = v
    return f'a price of {_money(price)}, {battery} hours of battery life and a weight of {weight:.1f} kg'


def _holiday(r):
    return (int(r.integers(18, 51)) * 50, int(r.integers(5, 13)), int(r.integers(1, 16)) * 100)


def _holiday_text(v):
    price, nights, beach = v
    return f'a price of {_money(price)} for {nights} nights in a hotel {beach:,} metres from the beach'


def _treatment(r):
    return (int(r.integers(5, 31)), int(r.integers(1, 13)) * 50, int(r.integers(1, 7)))


def _treatment_text(v):
    days, cost, visits = v
    return (f'full recovery in exactly {days} days, {_money(cost)} out of pocket, and {visits} clinic '
            f'visit{"s" if visits > 1 else ""}')


def _bond(r):
    return (int(r.integers(104, 131)) * 50, int(r.integers(1, 7)))


def _bond_text(v):
    payout, years = v
    return f'pays a guaranteed {_money(payout)} after {years} year{"s" if years > 1 else ""}'


# per attribute: +1 if more is better, -1 if less is better (used only to avoid an alternative that beats all)
SURE_DOMAINS = {
    'jobs': ('offer', 'offers', 'Offer', _job, _job_text, (1, -1, 1), [
        'You are choosing between job offers for the same kind of position. The offers are:',
        'You have received several job offers for the same role at similar companies:']),
    'housing': ('apartment', 'apartments', 'Apartment', _flat, _flat_text, (-1, 1, -1), [
        'You are choosing an apartment to rent for the next year. The apartments are:',
        'You are moving to a new city and can rent any of these apartments for a year:']),
    'technology': ('laptop', 'laptops', 'Laptop', _laptop, _laptop_text, (-1, 1, -1), [
        'You are buying a laptop for work and are choosing among these models:',
        'You need a new laptop and have narrowed the choice down to these models:']),
    'travel': ('package', 'packages', 'Package', _holiday, _holiday_text, (-1, 1, -1), [
        'You are booking a beach holiday and are choosing among these packages:',
        'A travel agent offers you these beach holiday packages:']),
    'medical': ('treatment', 'treatments', 'Treatment', _treatment, _treatment_text, (-1, -1, -1), [
        'You have a minor injury, and your doctor offers several treatments, each with a certain outcome:',
        'Your doctor explains that any of these treatments will cure your minor injury, with these certain '
        'outcomes:']),
    'finance': ('bond', 'bonds', 'Bond', _bond, _bond_text, (1, -1), [
        'You want to invest $5,000 for a few years and are choosing among these savings bonds, each bought for '
        '$5,000 today:',
        'Your bank offers these savings bonds; each costs $5,000 today and cannot be sold before it pays out:']),
}


# -------------------------------------------------------------------------------------------------- lotteries
def _pct(p: int) -> str:
    return f'{p}%'


def _envelope(r):
    p = int(r.integers(1, 10)) * 10
    hi = int(r.integers(5, 21)) * 100
    lo = int(r.integers(0, hi // 100 - 1)) * 100
    return (p, hi, lo)


def _envelope_text(v):
    p, hi, lo = v
    return f'a {_pct(p)} chance of {_money(hi)} and otherwise {_money(lo)}'


def _fund(r):
    p = int(r.integers(2, 10)) * 10
    up = int(r.integers(105, 131)) * 100
    down = int(r.integers(80, 100)) * 100
    return (p, up, down)


def _fund_text(v):
    p, up, down = v
    return (f'with probability {_pct(p)} the $10,000 grows to {_money(up)} after one year, and otherwise it is worth '
            f'{_money(down)}')


def _recovery(r):
    p = int(r.integers(2, 10)) * 10
    fast = int(r.integers(1, 5))
    slow = fast + int(r.integers(2, 9))
    return (p, fast, slow)


def _recovery_text(v):
    p, fast, slow = v
    return (f'a {_pct(p)} chance of full recovery in {fast} week{"s" if fast > 1 else ""}, and otherwise full '
            f'recovery in {slow} weeks')


def _bonus(r):
    p_hi = int(r.integers(1, 5)) * 10
    p_mid = int(r.integers(1, 10 - p_hi // 10)) * 10
    hi = int(r.integers(30, 81)) * 100
    mid = int(r.integers(5, hi // 100)) * 100
    return (p_hi, hi, p_mid, mid)


def _bonus_text(v):
    p_hi, hi, p_mid, mid = v
    return (f'a {_pct(p_hi)} chance of a {_money(hi)} bonus, a {_pct(p_mid)} chance of {_money(mid)}, and otherwise '
            f'no bonus')


def _crop(r):
    dry = int(r.integers(4, 21)) * 1000
    wet = int(r.integers(10, 31)) * 1000
    return (dry, wet)


def _crop_text(v):
    dry, wet = v
    return f'earns {_money(dry)} if the summer is dry and {_money(wet)} if it is wet'


def _route(r):
    p = int(r.integers(5, 10)) * 10
    t = int(r.integers(5, 13)) * 5
    late = t + int(r.integers(2, 13)) * 5
    return (p, t, late)


def _route_text(v):
    p, t, late = v
    return f'a {_pct(p)} chance that the trip takes {t} minutes, and otherwise it takes {late} minutes'


RISK_DOMAINS = {
    'game_show': ('envelope', 'envelopes', 'Envelope', _envelope, _envelope_text, None, [
        'On a game show you may keep one envelope. The host tells you exactly what each envelope pays:',
        'You have won a game-show round and may pick one envelope. Each pays a random prize with these exact '
        'chances:']),
    'finance': ('investment', 'investments', 'Investment', _fund, _fund_text, None, [
        'You will invest $10,000 for one year in one of these investments, whose outcomes and chances are known '
        'exactly:',
        'Your adviser offers these one-year investments for your $10,000, with these exact chances:']),
    'medical': ('treatment', 'treatments', 'Treatment', _recovery, _recovery_text, None, [
        'You have a minor injury, and your doctor offers several treatments with these exact chances (the '
        'treatments differ only in how long recovery takes):',
        'Your doctor says each of these treatments will cure your minor injury; how long recovery takes depends on '
        'chance:']),
    'workplace': ('bonus plan', 'bonus plans', 'Plan', _bonus, _bonus_text, None, [
        'Your employer lets you choose this year\'s bonus plan. Each plan pays at the end of the year with these '
        'exact chances:',
        'You may pick one of these bonus plans for the coming year; the chances are fixed and known:']),
    'agriculture': ('crop', 'crops', 'Crop', _crop, _crop_text, None, [
        'You are a farmer choosing one crop for your field this year. The summer will be dry with probability '
        '{p_dry} and wet otherwise. Each crop:',
        'You will plant one crop this year. A dry summer has probability {p_dry}, a wet one the rest. Each crop:']),
    'travel': ('route', 'routes', 'Route', _route, _route_text, None, [
        'You must get to an important meeting and are choosing a route. Each route has:',
        'You are choosing how to travel to an interview. The travel times and their exact chances are:']),
}

QUESTIONS = ['Considering only {x} and {y}, which do you prefer?',
             'Between {x} and {y}, which would you choose?']


def _domains(element):
    return SURE_DOMAINS if element == SURE else RISK_DOMAINS


def _templates(element: str) -> list[ct.CodeTemplate]:
    out = []
    for domain, (noun, plural, label, _, _, _, stems) in _domains(element).items():
        for n in (4, 5):
            for k, stem in enumerate(stems):
                note = COMPLETE_NOTE.format(plural=plural)
                head = f'{stem}\n{{descriptions}}\n{note} '
                q = QUESTIONS[k]
                parts = tuple(head + q.format(x=f'{label} {LABELS[a]}', y=f'{label} {LABELS[b]}')
                              for a, b in combinations(range(n), 2))
                out.append(ct.CodeTemplate(parts=parts, type='pairwise', domain=domain, tags='second_person',
                                           difficulty_level=n - 3,
                                           spec={'domain': domain, 'n': n}))
    return out


def _no_best(vals, signs) -> bool:
    """No alternative is at least as good as every other on every attribute (signs: +1 more is better)."""
    if signs is None:
        return True
    for i, v in enumerate(vals):
        if all(all(s * (v[k] - w[k]) >= 0 for k, s in enumerate(signs)) for j, w in enumerate(vals) if j != i):
            return False
    return True


def _risk_ok(domain, vals) -> bool:
    """Lotteries: no two with the same expected value to the cent are needed; just avoid a lottery that is better
    in every outcome than every other (first-order dominance of all), which would make the pairs trivial."""
    if domain == 'agriculture':
        return all(any((a[0] - b[0]) * (a[1] - b[1]) < 0 for b in vals if b is not a) for a in vals)
    return True


def _fill(element: str, metadata: dict) -> dict:
    spec = ct.spec_for(element, metadata['question_text'])
    noun, plural, label, draw, text, signs, _ = _domains(element)[spec['domain']]
    n = spec['n']
    r = filler_utils.rng
    while True:
        vals = [draw(r) for _ in range(n)]
        if len(set(vals)) == n and _no_best(vals, signs) and _risk_ok(spec['domain'], vals):
            break
    lines = [f'{label} {LABELS[i]}: {text(v)}.' for i, v in enumerate(vals)]
    out = {'descriptions': '\n'.join(lines)}
    if spec['domain'] == 'agriculture' and element == RISK:
        out['p_dry'] = _pct(int(r.integers(2, 9)) * 10)
    options, correct = [], []
    for a, b in combinations(range(n), 2):
        x, y = f'{label} {LABELS[a]}', f'{label} {LABELS[b]}'
        options.append([f'I prefer {x}', f'I prefer {y}', f'I am indifferent between {x} and {y}',
                        f'I cannot decide between {x} and {y}'])
        correct.append([0, 0, 0, 0])
    out['options_list'] = options
    out['correct_answer'] = correct
    return out


@function_registry.register(SURE)
def fill_completeness(metadata: dict) -> dict:
    return _fill(SURE, metadata)


@function_registry.register(RISK)
def fill_completeness_risk(metadata: dict) -> dict:
    return _fill(RISK, metadata)


ct.register(SURE, _templates(SURE))
ct.register(RISK, _templates(RISK))
