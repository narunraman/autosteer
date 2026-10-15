"""factor_demand (STEER-ME 3.3, Deriving Factor Demand): Duality of Profit Maximization and Expenditure Minimization.

Decision of 5 October 2026. The data directory's templates (a firm with a production function in K and L asking
for "how many of each" with no filler) are not used; the templates are written here. Each question has two parts
about the same price-taking firm, with Cobb-Douglas technology

    Q = f(L, K) = A * L^a * K^b,   a + b = s < 1 (decreasing returns, so long-run profit has a finite maximum),

wage w per hour of labor (L), rental rate r per hour of capital (K) and output price p. The firm currently uses
(Lnow, Know), but both inputs can be changed.

Part 0, long-run profit maximisation: max_{L,K} p f(L, K) - w L - r K. The first-order conditions
p a Q / L = w and p b Q / K = r give L = a p Q / w and K = b p Q / r; substituting into Q = A L^a K^b,

    Q* = [A (a p / w)^a (b p / r)^b]^(1 / (1 - s)),   L* = a p Q* / w,   K* = b p Q* / r,
    total cost C* = w L* + r K* = s p Q*.

Part 1, expenditure (cost) minimisation of producing exactly Q* (stated, rounded to 2 decimals as shown): min w L + r K
subject to A L^a K^b = Q. The tangency condition (a/b)(K/L) = w/r gives K = (b w / (a r)) L, and

    L^c(Q) = (Q / A)^(1/s) (a r / (b w))^(b/s),   K^c(Q) = (Q / A)^(1/s) (b w / (a r))^(a/s),
    C(Q) = w L^c + r K^c = (s / a) w L^c.

Duality: at Q = Q* the cost-minimising bundle is the profit-maximising one, so the part-1 key equals the part-0 key.
The parts are computed independently (part 1 from the shown, rounded Q*) and draws whose two keys differ at the
shown precision (2 decimals for hours, whole dollars for costs) are redrawn.

Types (which quantity both parts ask for): labor (hours of labor), capital (hours of capital), cost (part 0: total
cost of labor and capital at the profit maximum; part 1: the minimum cost of producing Q*).

Distractors are the same quantity computed with a natural mistake (each from the full bundle (L, K) of the mistake;
a cost is w L + r K at the true prices):
  part 0: w and r swapped; exponents a and b swapped; the first-order conditions solved with Q held at
      A (a p / w)^a (b p / r)^b (the 1 / (1 - s) power from decreasing returns dropped); the short-run optimum with the
      other input fixed at its current level (labor: L = (a p A Know^b / w)^(1 / (1 - a)), K = Know); the other input
      (labor / capital types); revenue p Q*, profit p Q* - C* and the labor cost w L* alone (cost type).
  part 1: w and r swapped in the tangency condition; the exponent b (a) instead of b/s (a/s) on the price ratio;
      the short-run bundle with the other input fixed at its current level (labor: (Q / (A Know^b))^(1/a)); constant
      returns assumed (the bundle for one unit scaled by Q); the other input (labor / capital); the labor cost alone
      (cost type).
Three are chosen at random among the mistakes of both parts that are within a factor 8 of the key and at least 10%
apart from the key and from each other (relative to the larger), giving four options; otherwise the parameters are
redrawn. Both parts offer the same four values (Review, October 2026: separately drawn distractors left the key
as the only value common to both parts).

Perspectives (tags): first_person, second_person, man, woman, anon. Domains as in the data directory's templates.
"""
from __future__ import annotations

import math

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'factor_demand'

# domain, firm, what it does, unit, units, labor noun, capital noun
SCENARIOS = [
    dict(domain='consumer_goods', firm='bakery', what='that bakes artisan bread', unit='loaf', units='loaves',
         lab='baker labor', cap='oven time'),
    dict(domain='consumer_goods', firm='furniture workshop', what='that builds wooden chairs', unit='chair',
         units='chairs', lab='carpenter labor', cap='CNC machine time'),
    dict(domain='entertainment', firm='animation studio', what='that produces short animated clips for advertisers',
         unit='clip', units='clips', lab='animator labor', cap='render-farm time'),
    dict(domain='entertainment', firm='recording studio', what='that mixes and masters music tracks', unit='track',
         units='tracks', lab='sound-engineer labor', cap='mixing-console time'),
    dict(domain='education', firm='tutoring center', what='that runs exam-preparation sessions', unit='session',
         units='sessions', lab='tutor labor', cap='computer-lab time'),
    dict(domain='education', firm='online course provider', what='that records video lessons', unit='lesson',
         units='lessons', lab='instructor labor', cap='filming-studio time'),
    dict(domain='environmental_policy', firm='recycling plant', what='that turns plastic waste into pellets',
         unit='tonne of pellets', units='tonnes of pellets', lab='sorter labor', cap='shredder time'),
    dict(domain='environmental_policy', firm='tree nursery', what='that grows seedlings for reforestation projects',
         unit='tray of seedlings', units='trays of seedlings', lab='gardener labor', cap='greenhouse time'),
    dict(domain='sports', firm='sporting-goods maker', what='that makes tennis rackets', unit='racket',
         units='rackets', lab='assembly labor', cap='molding-press time'),
    dict(domain='sports', firm='bicycle workshop', what='that services racing bikes', unit='service',
         units='services', lab='mechanic labor', cap='workstand time'),
    dict(domain='technology', firm='electronics assembler', what='that builds laptops', unit='laptop',
         units='laptops', lab='technician labor', cap='assembly-line time'),
    dict(domain='technology', firm='data-labelling company', what='that annotates images for machine learning',
         unit='batch of images', units='batches of images', lab='annotator labor', cap='GPU-server time'),
    dict(domain='medical', firm='diagnostic laboratory', what='that runs blood tests', unit='test', units='tests',
         lab='lab-technician labor', cap='analyser time'),
    dict(domain='medical', firm='compounding pharmacy', what='that prepares custom prescriptions',
         unit='prescription', units='prescriptions', lab='pharmacist labor', cap='compounding-equipment time'),
]

MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel', 'Mateo', 'Arjun']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah', 'Amara', 'Lena']
PERSPECTIVES = ['first_person', 'second_person', 'man', 'woman', 'anon']
TYPES = ('labor', 'capital', 'cost')


def _voice(tag: str, sc: dict, k: int) -> dict:
    """Intro sentence, sentence subject, third-person verb suffix and question subject for a perspective."""
    firm = sc['firm']
    if tag == 'first_person':
        return dict(intro=f"I run a {firm} {sc['what']}.", subj='I', s='', q='I')
    if tag == 'second_person':
        return dict(intro=f"You run a {firm} {sc['what']}.", subj='You', s='', q='you')
    if tag in ('man', 'woman'):
        name = (MEN if tag == 'man' else WOMEN)[k % 7]
        return dict(intro=f"{name} runs a {firm} {sc['what']}.", subj=name, s='s', q=name)
    return dict(intro=f"Consider a {firm} {sc['what']}.", subj=f'The {firm}', s='s', q=f'the {firm}')


def _setup(sc: dict, v: dict, with_price: bool) -> str:
    lab, cap, s = sc['lab'], sc['cap'], v['s']
    text = (f"{v['intro']} The {sc['firm']}'s weekly output, in {sc['units']}, is Q = {{pfunc}}, where L is the "
            f"number of hours of {lab} and K the number of hours of {cap} used per week. ")
    if with_price:
        text += f"{v['subj']} sell{s} each {sc['unit']} for ${{price}}, pay{s} "
    else:
        text += f"{v['subj']} pay{s} "
    text += (f"${{wage}} per hour of {lab} and ${{rent}} per hour of {cap}, and currently use{s} {{lnow}} hours of "
             f"{lab} and {{know}} hours of {cap} a week. Both inputs can be changed freely, and all markets are "
             f"competitive.")
    return text


def _parts(sc: dict, tag: str, k: int, qtype: str) -> tuple[str, str]:
    v = _voice(tag, sc, k)
    s = v['s']
    p0 = _setup(sc, v, with_price=True)
    p1 = _setup(sc, v, with_price=False) + (f" {v['subj']} need{s} to produce exactly {{output}} {sc['units']} per "
                                            f"week at the lowest possible cost.")
    if qtype == 'labor':
        p0 += f" To maximize profit, how many hours of {sc['lab']} should {v['q']} use per week?"
        p1 += f" How many hours of {sc['lab']} should {v['q']} use per week?"
    elif qtype == 'capital':
        p0 += f" To maximize profit, how many hours of {sc['cap']} should {v['q']} use per week?"
        p1 += f" How many hours of {sc['cap']} should {v['q']} use per week?"
    else:
        p0 += (" At the profit-maximizing choice of L and K, what is the total weekly cost of labor and capital, to "
               "the nearest dollar?")
        p1 += " What is the lowest weekly cost of producing that output, to the nearest dollar?"
    return p0, p1


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(sc, tag, k, qtype), type=qtype, domain=sc['domain'],
                                           tags=tag, difficulty_level=2 if qtype == 'cost' else 1,
                                           spec={'scenario': k, 'type': qtype}))
    return out


# --------------------------------------------------------------------------------------------------------------
# Closed forms (see the module docstring)
# --------------------------------------------------------------------------------------------------------------
def profit_max(A, a, b, p, w, r):
    """Long-run profit-maximising (L, K, Q) for Q = A L^a K^b, a + b < 1."""
    s = a + b
    q = (A * (a * p / w) ** a * (b * p / r) ** b) ** (1 / (1 - s))
    return a * p * q / w, b * p * q / r, q


def cost_min(A, a, b, w, r, q):
    """Cost-minimising (L, K) producing exactly q."""
    s = a + b
    return (q / A) ** (1 / s) * (a * r / (b * w)) ** (b / s), (q / A) ** (1 / s) * (b * w / (a * r)) ** (a / s)


def _pick(r, key: float, candidates: list[float], n: int = 3):
    """n candidates within a factor 8 of key, at least 10% apart from key and each other; None if too few."""
    def apart(x, y):
        return abs(x - y) >= 0.1 * max(abs(x), abs(y))

    cands = [c for c in candidates if math.isfinite(c) and key / 8 <= c <= key * 8]
    order = [int(i) for i in r.permutation(len(cands))]
    chosen = []
    for i in order:
        c = cands[i]
        if all(apart(c, x) for x in [key] + chosen):
            chosen.append(c)
            if len(chosen) == n:
                return chosen
    return None


def hours(x: float) -> str:
    return str(round(x, 2))


def dollars(x: float) -> str:
    return f'${round(x):,}'


def duality_draw(r, qtype: str):
    """Parameters, keys and distractors of one question (redrawn until both parts have four good options)."""
    while True:
        a = 0.05 * int(r.integers(4, 12))        # 0.20 .. 0.55
        b = 0.05 * int(r.integers(3, 10))        # 0.15 .. 0.45
        a, b = round(a, 2), round(b, 2)
        s = a + b
        if not 0.5 <= s <= 0.85 + 1e-9:
            continue
        A = int(r.integers(2, 10))
        p = int(r.integers(5, 41))
        # prices that put the optimum near (Lt, Kt), rounded to whole dollars; the optimum is recomputed from them
        Lt, Kt = int(r.integers(20, 201)), int(r.integers(10, 151))
        Yt = A * Lt ** a * Kt ** b
        w, rr = round(a * p * Yt / Lt), round(b * p * Yt / Kt)
        if w < 5 or rr < 3 or w == rr:
            continue
        L0, K0, Q0 = profit_max(A, a, b, p, w, rr)
        if not (5 <= L0 <= 1000 and 5 <= K0 <= 1000):
            continue
        C0 = w * L0 + rr * K0

        def now(x):
            f = float(r.uniform(0.4, 0.75)) if r.integers(0, 2) == 0 else float(r.uniform(1.3, 2.0))
            return max(1, round(x * f))
        Ln, Kn = now(L0), now(K0)

        Qs = round(Q0, 2)  # shown in part 1
        L1, K1 = cost_min(A, a, b, w, rr, Qs)
        C1 = w * L1 + rr * K1
        if round(L1, 2) != round(L0, 2) or round(K1, 2) != round(K0, 2) or round(C1) != round(C0):
            continue

        # part 0 mistakes, as bundles (L, K)
        sw = profit_max(A, a, b, p, rr, w)[:2]
        se = profit_max(A, b, a, p, w, rr)[:2]
        qn = A * (a * p / w) ** a * (b * p / rr) ** b
        nf = (a * p * qn / w, b * p * qn / rr)
        sr_l = ((a * p * A * Kn ** b / w) ** (1 / (1 - a)), Kn)
        sr_k = (Ln, (b * p * A * Ln ** a / rr) ** (1 / (1 - b)))
        # part 1 mistakes
        t_sw = cost_min(A, a, b, rr, w, Qs)
        t_ex = ((Qs / A) ** (1 / s) * (a * rr / (b * w)) ** b, (Qs / A) ** (1 / s) * (b * w / (a * rr)) ** a)
        t_sr_l = ((Qs / (A * Kn ** b)) ** (1 / a), Kn)
        t_sr_k = (Ln, (Qs / (A * Ln ** a)) ** (1 / b))
        u = cost_min(A, a, b, w, rr, 1.0)
        t_crs = (Qs * u[0], Qs * u[1])

        def cost(bundle):
            return w * bundle[0] + rr * bundle[1]

        if qtype == 'labor':
            key0, key1 = L0, L1
            c0 = [sw[0], se[0], nf[0], sr_l[0], K0]
            c1 = [t_sw[0], t_ex[0], t_sr_l[0], t_crs[0], K1]
        elif qtype == 'capital':
            key0, key1 = K0, K1
            c0 = [sw[1], se[1], nf[1], sr_k[1], L0]
            c1 = [t_sw[1], t_ex[1], t_sr_k[1], t_crs[1], L1]
        else:
            key0, key1 = C0, C1
            c0 = [cost(sw), cost(se), cost(nf), cost(sr_l), p * Q0, p * Q0 - C0, w * L0]
            c1 = [cost(t_sw), cost(t_ex), cost(t_sr_l), cost(t_crs), w * L1]
        fmt = dollars if qtype == 'cost' else hours
        # Review: with separate distractors the key was the only value offered in both parts. Both parts
        # now offer the same four values (the keys agree at the shown precision); the distractors are drawn from
        # the mistakes of either part. The sampler shuffles each part's options independently.
        d = _pick(r, key0, c0 + c1)
        if d is None:
            continue
        opts0 = [fmt(x) for x in [key0] + d]
        opts1 = list(opts0)
        if len(set(opts0)) != 4 or fmt(key1) != opts0[0]:
            continue
        return dict(A=A, a=a, b=b, p=p, w=w, r=rr, Lnow=Ln, Know=Kn, output=Qs, opts0=opts0, opts1=opts1)


def pfunc(A: int, a: float, b: float) -> str:
    return f'{A} * L^{a:g} * K^{b:g}'


@function_registry.register(ELEMENT)
def fill_factor_demand(metadata: dict) -> dict:
    qtype = ct.spec_for(ELEMENT, metadata['question_text'])['type']
    d = duality_draw(filler_utils.rng, qtype)
    return {
        'pfunc': pfunc(d['A'], d['a'], d['b']),
        'price': str(d['p']), 'wage': str(d['w']), 'rent': str(d['r']),
        'lnow': str(d['Lnow']), 'know': str(d['Know']), 'output': f"{d['output']:,}",
        'options_list': [d['opts0'], d['opts1']],
        'correct_answer': [[1, 0, 0, 0], [1, 0, 0, 0]],
    }


ct.register(ELEMENT, _templates())
