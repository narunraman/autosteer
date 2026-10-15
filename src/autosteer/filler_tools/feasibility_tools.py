"""feasibility (STEER 3.4.a, Feasibility in Infinitely Repeated Games): hand-written repeated prisoner's dilemmas.

Design of October 2026 (replaces the pipeline templates, which stated a discount factor that the key
ignored). Six hand-written stories (pricing, fishing, advertising, oil quotas, data sharing, irrigation), each a
symmetric prisoner's dilemma whose payoffs per period are stated by the story's own action words: both cooperate R,
both defect P, the defector T and the cooperator S, with T > R > P > S and 2R > T + S (sample_pd of
multi_agent_tools), in whole thousands of dollars. Two types:

    folk (difficulty 7): no discount factor. Which pair of long-run average payoffs per period is feasible (in the
        convex hull of the four stage payoff pairs) and individually rational (each player gets more than its minmax
        payoff, which in a prisoner's dilemma is the mutual-defection payoff P)? Tag `defined` states both terms;
        tag `named` asks which pair the folk theorem says can be sustained by a subgame-perfect equilibrium when
        both players are very patient (the same set, strictly individually rational). Exactly one option qualifies;
        the distractors are individually rational but infeasible (beyond the payoff frontier, with each payoff at most
        T, so a bound on one player's payoff does not rule them out) or feasible but not individually rational (one
        player below P), at least one of each. Every option is at least 2% of T - S
        from the boundary of the set, on a $250 grid.
    discount (difficulty 8): a stated discount factor delta, grim-trigger punishment (both defect forever after any
        departure), and four plans (periodic paths): both cooperate every period; take turns defecting; cooperate and
        defect in alternate periods; two periods of cooperation then one of defection; a three-period cycle of
        cooperation and turn-taking; one player always cooperates while the other alternates (never the key: when it
        is sustainable, most of the others are too). Exactly one is a
        subgame-perfect equilibrium path at delta, decided exactly (Fractions) by the one-shot deviation principle:
        at every period of the cycle, each player's normalised continuation value is at least (1 - delta) times
        its best one-period deviation payoff plus delta P. The status of every option is the same at delta +- 0.01
        and +- 0.02, so delta is at least 0.02 from every threshold. Tags `named` (says "grim trigger") or
        `described`.
Perspective: anonymous (the two players are named A and B).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import code_templates as ct
from .fill_registry import FillRegister
from .multi_agent_tools import sample_pd
from autosteer.filler_tools.rng import pyrng

function_registry = FillRegister()

ELEMENT = 'feasibility'
C, D = 'C', 'D'

# domain, setting sentence, (A, B), plural noun, singular noun, period, cooperate (plural, singular),
# defect (plural, singular), "the one that ..." (defector), verb for payoffs
STORIES = [
    dict(domain='business', setting='Two petrol stations on the same road, Station A and Station B, set their prices '
         'every month.', A='Station A', B='Station B', who='stations', noun='station', period='month',
         c=('keep their prices high', 'keeps its price high'), d=('cut their prices', 'cuts its price'),
         earn='earns'),
    dict(domain='environment', setting='Two fishing companies, Company A and Company B, share a lake. Every season '
         'each company either keeps to the agreed catch limit or overfishes.', A='Company A', B='Company B',
         who='companies', noun='company', period='season', c=('keep to the catch limit', 'keeps to the catch limit'),
         d=('overfish', 'overfishes'), earn='earns'),
    dict(domain='marketing', setting='Two soft-drink makers, Maker A and Maker B, each decide every year whether to '
         'keep their advertising modest or to launch an advertising blitz.', A='Maker A', B='Maker B',
         who='makers', noun='maker', period='year', c=('keep advertising modest', 'keeps advertising modest'),
         d=('launch an advertising blitz', 'launches an advertising blitz'), earn='earns a profit of'),
    dict(domain='energy', setting='Two oil producers, Producer A and Producer B, have agreed on output quotas. Every '
         'quarter each producer either sticks to its quota or produces extra.', A='Producer A', B='Producer B',
         who='producers', noun='producer', period='quarter', c=('stick to their quotas', 'sticks to its quota'),
         d=('produce extra', 'produces extra'), earn='earns'),
    dict(domain='research', setting='Two research labs, Lab A and Lab B, work on a joint project. Every month each lab '
         'either shares its new data with the other or keeps it to itself; a lab\'s payoff is the funding its '
         'results bring in.', A='Lab A', B='Lab B', who='labs', noun='lab', period='month',
         c=('share their data', 'shares its data'), d=('keep their data to themselves', 'keeps its data to itself'),
         earn='gains funding worth'),
    dict(domain='agriculture', setting='Two farms, Farm A and Farm B, draw water from the same river. Every week each '
         'farm either draws its fair share or draws extra water.', A='Farm A', B='Farm B', who='farms', noun='farm',
         period='week', c=('draw their fair share', 'draws its fair share'), d=('draw extra water', 'draws extra water'),
         earn='earns'),
]


def money(v) -> str:
    return f'${int(round(v)):,}'


def payoff_sentence(st: dict) -> str:
    cp, cs = st['c']
    dp, ds = st['d']
    p, e = st['period'], st['earn']
    return (f'If both {cp}, each {e} {{both_y}} per {p}; if both {dp}, each {e} {{both_x}} per {p}; if one {ds} '
            f'while the other {cs}, the one that {ds} {e} {{x_payoff}} and the other {e} {{y_payoff}} per {p}.')


FOLK_ASK = {
    'defined': ('The two {who} interact in this way every {period}, forever. Which of the following pairs of long-run '
                'average payoffs per {period} is both feasible (it can be obtained as a long-run average of the four '
                'outcomes above, by mixing them over time) and individually rational (each {noun} gets more than it '
                'can guarantee itself, which here is the payoff when both {d_pl})?'),
    'named': ('The two {who} interact in this way every {period}, forever, and both are very patient. According to '
              'the folk theorem, which of the following pairs of long-run average payoffs per {period} can be '
              'sustained by a subgame-perfect equilibrium of this infinitely repeated game?'),
}
DISCOUNT_ASK = {
    'named': ('The two {who} interact in this way every {period}, forever, and each values a payoff one {period} '
              'later at {{discount_factor}} times its value now (discount factor {{discount_factor}}). They agree on '
              'a plan for what each does in every {period}, enforced by grim trigger: if either ever departs from the '
              'plan, both {d_pl} in every {period} after that, forever. Which of the following plans is a '
              'subgame-perfect equilibrium of the repeated game with this punishment, so that neither {noun} can ever '
              'gain by departing from it?'),
    'described': ('The {who} interact in this way every {period}, forever, and discount the future by a factor of '
                  '{{discount_factor}} per {period}. Suppose they agree on a plan for every {period}, and that if '
                  'either one ever departs from it, both will {d_pl} in every later {period}. Exactly one of the '
                  'following plans can be sustained this way as a subgame-perfect equilibrium (no {noun} ever gains '
                  'by departing from it). Which one?'),
}


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        fmt = dict(who=st['who'], period=st['period'], noun=st['noun'], d_pl=st['d'][0])
        head = f"{st['setting']} {payoff_sentence(st)} "
        for tag, ask in FOLK_ASK.items():
            out.append(ct.CodeTemplate(parts=(head + ask.format(**fmt),), type='folk', domain=st['domain'],
                                       tags=f'anon;{tag}', difficulty_level=7, spec={'story': k, 'type': 'folk'}))
        for tag, ask in DISCOUNT_ASK.items():
            out.append(ct.CodeTemplate(parts=(head + ask.format(**fmt),), type='discount', domain=st['domain'],
                                       tags=f'anon;{tag}', difficulty_level=8, spec={'story': k, 'type': 'discount'}))
    return out


# --------------------------------------------------------------------------------------------------------- folk
def _scores(g, x, y):
    """(hull score, individual-rationality score): < 0 inside (minus the distance to the boundary), > 0 outside."""
    from scipy.spatial import ConvexHull
    pts = np.array([(g['R'], g['R']), (g['S'], g['T']), (g['T'], g['S']), (g['P'], g['P'])], dtype=float)
    hull = max(float(eq[0] * x + eq[1] * y + eq[2]) for eq in ConvexHull(pts).equations)
    return hull, max(g['P'] - x, g['P'] - y)


def _folk(g, st):
    span = g['T'] - g['S']
    m = 0.02 * span
    lo, hi = max(0.0, g['S'] - 0.15 * span), g['T'] + 0.15 * span

    def draw(kind):
        for _ in range(3000):
            x = round(pyrng.uniform(lo, hi) / 0.25) * 0.25
            y = x if pyrng.random() < 0.25 else round(pyrng.uniform(lo, hi) / 0.25) * 0.25
            h, ir = _scores(g, x, y)
            if kind == 'key' and h <= -m and ir <= -m:
                return x, y
            if kind == 'infeasible' and h >= m and ir <= -m and max(x, y) <= g['T'] - m:
                return x, y     # beyond the frontier, but no coordinate above the largest stage payoff
            if kind == 'not_ir' and h <= -m and ir >= m:
                return x, y
        return None

    kinds = ['infeasible', 'not_ir', pyrng.choice(['infeasible', 'not_ir'])]
    chosen = [draw('key')]
    for kind in kinds:
        for _ in range(20):
            o = draw(kind)
            if o is not None and o not in chosen:
                chosen.append(o)
                break
    if None in chosen or len(chosen) < 4:
        return None      # a game with too little room for one kind of option: the filler draws another game
    texts = [f"{st['A']}: {money(1000 * x)}, {st['B']}: {money(1000 * y)}" for x, y in chosen]
    return texts, [1, 0, 0, 0]


# ----------------------------------------------------------------------------------------------------- discount
PATHS = {
    'cc': [(C, C)],
    'alt': [(D, C), (C, D)],
    'cc_dd': [(C, C), (D, D)],
    'cc_cc_dd': [(C, C), (C, C), (D, D)],
    'cc_alt': [(C, C), (D, C), (C, D)],
    'b_carries': [(D, C), (C, C)],
}


def stage(g, a1, a2):
    return {(C, C): (g['R'], g['R']), (C, D): (g['S'], g['T']), (D, C): (g['T'], g['S']),
            (D, D): (g['P'], g['P'])}[(a1, a2)]


def sustainable(g, path, delta) -> bool:
    """Exact: the periodic path is a subgame-perfect equilibrium path with grim-trigger reversion to (D, D)."""
    d = Fraction(delta)
    n = len(path)
    for t in range(n):
        for i in (0, 1):
            value = (1 - d) * sum(d ** k * Fraction(stage(g, *path[(t + k) % n])[i]) for k in range(n)) / (1 - d ** n)
            own, other = path[t][i], path[t][1 - i]
            dev = D if own == C else C
            dev_profile = (dev, other) if i == 0 else (other, dev)
            if value < (1 - d) * stage(g, *dev_profile)[i] + d * g['P']:
                return False
    return True


def path_text(name: str, st: dict) -> str:
    A, B, p = st['A'], st['B'], st['period']
    cp, cs = st['c']
    dp, ds = st['d']
    return {
        'cc': f'Both {cp} in every {p}.',
        'alt': (f'They take turns: in the first {p} {A} {ds} while {B} {cs}, in the next {p} {B} {ds} while {A} {cs}, '
                f'and so on, alternating.'),
        'cc_dd': f'Both {cp} in the first {p}, both {dp} in the next, and so on, alternating.',
        'cc_cc_dd': f'Both {cp} for two {p}s, then both {dp} for one {p}, and this three-{p} pattern repeats.',
        'cc_alt': (f'A three-{p} cycle that repeats: first both {cp}; then {A} {ds} while {B} {cs}; then {B} {ds} '
                   f'while {A} {cs}.'),
        'b_carries': (f'{B} {cs} in every {p}, while {A} {ds} in the first {p}, {cs} in the '
                      f'next, and so on, alternating.'),
    }[name]


def robust_status(g, path, delta: Fraction):
    """Sustainability at delta, or None if it differs anywhere within 0.02 of delta."""
    s = sustainable(g, path, delta)
    for e in (-2, -1, 1, 2):
        if sustainable(g, path, delta + Fraction(e, 100)) != s:
            return None
    return s


def threshold(g, path) -> float:
    """Smallest discount factor at which the path is sustainable (bisection on a 1e-4 grid; 1.0 if never below
    0.999). Sustainability is monotone in delta for these paths (checked in tests/test_feasibility.py); the
    question's statuses are checked exactly with robust_status."""
    if not sustainable(g, path, Fraction(999, 1000)):
        return 1.0
    lo, hi = 0, 9990
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sustainable(g, path, Fraction(mid, 10000)):
            hi = mid
        else:
            lo = mid
    return hi / 10000


# b_carries is never the only sustainable plan among four (when it is sustainable, so are most others): distractor only
KEYS = ('cc', 'alt', 'cc_dd', 'cc_cc_dd', 'cc_alt')


def _discount(st):
    key = pyrng.choice(KEYS)
    while True:
        g = sample_pd(10, 99, integer=True)
        th = {n: threshold(g, p) for n, p in PATHS.items()}
        lo = max(0.10, th[key] + 0.03)
        harder = [n for n in PATHS if n != key and th[n] >= lo + 0.03]
        if th[key] >= 0.9 or len(harder) < 3:
            continue
        others = pyrng.sample(harder, 3)
        hi = min(0.95, min(th[o] for o in others) - 0.03)
        if hi < lo:
            continue
        delta = Fraction(round(pyrng.uniform(lo, hi) * 100), 100)
        if robust_status(g, PATHS[key], delta) and all(robust_status(g, PATHS[o], delta) is False for o in others):
            texts = [path_text(n, st) for n in [key] + others]
            return g, delta, texts, [1, 0, 0, 0]


@function_registry.register(ELEMENT)
def fill_feasibility(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    st = STORIES[spec['story']]
    out = {}
    if spec['type'] == 'folk':
        while True:
            g = sample_pd(10, 99, integer=True)
            got = _folk(g, st)
            if got is not None:
                texts, ans = got
                break
    else:
        g, delta, texts, ans = _discount(st)
        out['discount_factor'] = f'{float(delta):.2f}'
    out.update({'both_y': money(1000 * g['R']), 'both_x': money(1000 * g['P']), 'x_payoff': money(1000 * g['T']),
                'y_payoff': money(1000 * g['S']), 'options_list': [texts], 'correct_answer': [ans]})
    return out


ct.register(ELEMENT, _templates())
