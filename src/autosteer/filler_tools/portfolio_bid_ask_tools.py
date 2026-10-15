"""optimal_portfolio_choice (STEER-ME, Consumption Decisions, Dynamic Consumption Decisions): Optimal Portfolio
Choice with Bid-Ask Spreads (the paper's definition, paraphrased: the optimal portfolio allocation when trading involves bid-ask spreads).

Decision of 7 October 2026; templates written here. The old filler (`gen_optimal_portfolio_choice` in
dynamic_consumption_tools.py, no templates) is replaced: its spreads (0.5-1.5) were subtracted from returns of
0.05-0.08, so every net return was negative, its two-asset path returned no options or key and its three-asset path
ignored the spreads.

An investor holds a portfolio for one year: a share w of savings in a risky fund, the rest in a risk-free account at
rf. The fund's return has expected value mu and standard deviation sigma; buying it and selling it at the end of the
year costs a round-trip bid-ask spread s of the amount invested, deducted from the fund's return (a sure cost, so the
variance is unchanged). The investor maximizes the mean-variance utility U = E[r] - (A/2) Var(r) (as in
price_of_risk), with 0 <= w <= 1 (no borrowing or short sales).

Types:
  one_asset: the optimal share w* = (mu - s - rf) / (A sigma^2), clipped to [0, 1], as a percentage (2 decimals). In
      about one question in eight the spread eats the whole risk premium (key 0%) and in about one in twelve the
      unclipped share is above 1 (key 100%). Distractors (clipped the same way): the spread ignored, half the spread
      deducted, the spread deducted twice, sigma in place of sigma^2, the factor 1/2 of the utility dropped
      (w / 2), and the last two combined with the spread ignored. Three at least 10% (and 1 point) apart from the key and from each other.
  best_fund: four funds (expected return, standard deviation, spread); the investor puts the risky part in exactly
      one of them. The best is the one with the highest Sharpe ratio net of the spread, (mu - s - rf) / sigma (the
      utility at the optimum is rf + S^2 / (2A) when w* <= 1, which every fund satisfies here). The draw makes the
      fund with the highest gross Sharpe ratio, the one with the highest expected return net of the spread and the
      one with the lowest spread all differ from the key; the key's net Sharpe ratio is at least 10% above the next.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'optimal_portfolio_choice'
TYPES = ('one_asset', 'best_fund')
FUNDS = 'ABCD'

STORIES = [
    dict(domain='finance', anon='a retirement saver', money='savings', fund='a broad stock-market index fund'),
    dict(domain='education', anon='a university endowment committee', money='endowment', its=True,
         fund='an emerging-markets equity fund'),
    dict(domain='technology', anon='a software engineer', money='year-end bonus', fund='a technology-sector stock fund'),
    dict(domain='medical', anon='a doctor', money='savings for a new practice', fund='a health-care sector stock fund'),
    dict(domain='environmental_policy', anon='a climate charity', money='cash reserves', its=True,
         fund='a clean-energy stock fund'),
    dict(domain='entertainment', anon='a musician', money='royalty income', fund='a small-company stock fund'),
    dict(domain='sports', anon='a retired athlete', money='prize money', fund='a real-estate investment trust fund'),
    dict(domain='consumer_goods', anon='a shop owner', money='spare cash', fund='a consumer-staples stock fund'),
]


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}")
    first = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    poss = ('its' if st.get('its') else 'their') if tag == 'anon' else v['poss']
    obj = ('it' if st.get('its') else 'them') if tag == 'anon' else v['obj']
    money = st['money']
    util = (f"{v['subj']} choose{v['s']} the share w of the {money} in the fund to maximize the mean-variance utility "
            f"U = E[r] - (A/2) Var(r) of the portfolio's one-year return r (returns measured as decimals), with "
            f"risk aversion A = {{aversion}}, and w must be between 0 and 1 (no borrowing or short sales).")
    if qtype == 'one_asset':
        text = (f"{first} invest{v['s']} {poss} {money} for one year, splitting the money between a risk-free account "
                f"paying {{rf}}% and {st['fund']}. Over the year the fund's return has an expected value of {{mu}}% "
                f"and a standard deviation of {{sd}}%. Buying the fund and selling it at the end of the year costs a "
                f"round-trip bid-ask spread of {{spread}}% of the amount invested, which is deducted from the fund's "
                f"return. {util[0].upper() + util[1:]} What share of {poss} {money} should {v['subj']} put in the "
                f"fund? Give the percentage, rounded to two decimals.")
    else:
        text = (f"{first} invest{v['s']} {poss} {money} for one year, splitting the money between a risk-free account "
                f"paying {{rf}}% and exactly one of four funds of the same kind as {st['fund']}. For each fund, the "
                f"expected one-year return, its standard deviation and the round-trip bid-ask spread paid to buy the "
                f"fund and sell it at the end of the year (as a percentage of the amount invested, deducted from the "
                f"return) are: {{table}}. Whichever fund {v['subj']} pick{v['s']}, {util} Which fund gives "
                f"{obj} the highest utility?")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=1 if qtype == 'one_asset' else 2,
                                           spec={'story': k, 'type': qtype}))
    return out


def clip(w: float) -> float:
    return min(1.0, max(0.0, w))


def weight(mu, s, rf, sd, A):
    """Optimal share (percent inputs), unclipped."""
    return (mu - s - rf) / 100 / (A * (sd / 100) ** 2)


def _one_asset(r):
    mode = r.random()
    for _ in range(100000):
        A = int(r.integers(2, 9))
        rf = round(0.25 * int(r.integers(4, 17)), 2)          # 1% .. 4%
        sd = int(r.integers(10, 31))
        mu = round(0.1 * int(r.integers(50, 141)), 1)         # 5% .. 14%
        s = round(0.1 * int(r.integers(2, 31)), 1)            # 0.2% .. 3%
        w = weight(mu, s, rf, sd, A)
        if mode < 0.125:
            if not (-0.5 < w <= 0 and mu - rf > 0.5 and weight(mu, 0, rf, sd, A) > 0.15):
                continue
        elif mode < 0.21:
            if not 1.0 < w < 1.6:
                continue
        elif not 0.08 <= w <= 0.95:
            continue
        key = 100 * clip(w)
        gross = weight(mu, 0, rf, sd, A)
        cands = [100 * clip(z) for z in (gross, weight(mu, s / 2, rf, sd, A), weight(mu, 2 * s, rf, sd, A),
                                         (mu - s - rf) / 100 / (A * sd / 100), w / 2, gross / 2,
                                         (mu - rf) / 100 / (A * sd / 100))]
        dist = pick_apart(r, key, cands, abs_gap=1.0, shown=num)
        if dist is None:
            continue
        opts = [f'{num(z)}%' for z in [key] + dist]
        return {'aversion': str(A), 'rf': num(rf), 'mu': num(mu), 'sd': str(sd), 'spread': num(s),
                'options_list': [opts], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError('optimal_portfolio_choice: no one_asset draw')


def _best_fund(r):
    for _ in range(100000):
        A = int(r.integers(2, 9))
        rf = round(0.25 * int(r.integers(4, 17)), 2)
        funds = []
        for _ in range(4):
            sd = int(r.integers(8, 31))
            mu = round(rf + 0.1 * int(r.integers(10, 101)), 2)
            s = round(0.1 * int(r.integers(1, 31)), 1)
            funds.append((mu, sd, s))
        if any(weight(mu, s, rf, sd, A) > 1 for mu, sd, s in funds):
            continue
        net = [(mu - s - rf) / sd for mu, sd, s in funds]
        gross = [(mu - rf) / sd for mu, sd, s in funds]
        best = max(range(4), key=lambda i: net[i])
        rest = sorted(net[i] for i in range(4) if i != best)
        if net[best] <= 0 or net[best] < 1.1 * rest[-1]:
            continue
        g = max(range(4), key=lambda i: gross[i])
        m = max(range(4), key=lambda i: funds[i][0] - funds[i][2])
        lo = min(range(4), key=lambda i: funds[i][2])
        if best in (g, m, lo) or len({round(z, 9) for z in gross}) < 4 or \
                len({round(f[0] - f[2], 9) for f in funds}) < 4 or len({f[2] for f in funds}) < 4:
            continue
        table = '; '.join(f'Fund {FUNDS[i]}: expected return {num(mu)}%, standard deviation {sd}%, spread {num(s)}%'
                          for i, (mu, sd, s) in enumerate(funds))
        opts = [f'Fund {FUNDS[best]}'] + [f'Fund {FUNDS[i]}' for i in range(4) if i != best]
        return {'aversion': str(A), 'rf': num(rf), 'table': table, 'options_list': [opts],
                'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError('optimal_portfolio_choice: no best_fund draw')


@function_registry.register(ELEMENT)
def fill_optimal_portfolio_choice(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return (_one_asset if spec['type'] == 'one_asset' else _best_fund)(filler_utils.rng)


ct.register(ELEMENT, _templates())
