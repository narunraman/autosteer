"""price_of_risk (STEER-ME, Consumption Decisions): a mean-variance investor's optimal share of wealth in a risky asset,
and how that share responds to the asset's price of risk (its Sharpe ratio).

Review, October 2026: the AI-pipeline templates did not give the utility normalization (U = E[r] - (A/2) Var(r)
versus E[r] - A Var(r) halves the answer), did not define the price of risk, gave a market standard deviation for a
single fund, keyed shares above 1 without saying borrowing was allowed, asked for "the optimal allocation" while keying
its change, asked for a percentage while keying a fraction, and the options were always 0.5x, 1x, 1.5x and 2x the key
(the key was always the second smallest). Templates are now written here and state all of it:

The investor splits wealth between a risk-free asset paying rf and one risky fund with expected return mu and standard
deviation sigma (one year), and chooses the share w of wealth in the fund to maximize U = E[r] - (A/2) Var(r), returns
as decimals. Optimum w* = (mu - rf) / (A sigma^2) = S / (A sigma), with S = (mu - rf) / sigma the price of risk.
Every question says whether borrowing at the risk-free rate is allowed (then w can exceed 100%) or not (w in [0, 100%];
short sales are never needed since mu > rf).

Types:
  risk_allocation: the optimal share, in percent. In about a quarter of the questions the unconstrained share is above
      100%: the key is then 100% without borrowing and the unconstrained share with it. Distractors: A in place of A/2
      (half the key), A/2 in the denominator (twice), sigma in place of sigma^2, the risk-free rate ignored, and the
      clipped / unclipped share (all clipped to [0, 100%] without borrowing).
  utility_analysis: S rises from S0 to S1 with sigma unchanged; both optimal shares are in (5%, 95%). ask 'change': the
      change in the optimal share, in percentage points, 100 (S1 - S0) / (A sigma); ask 'level': the new share.
      Distractors: sigma^2 for sigma, twice and half the key, the level for the change (or the change / old share for
      the level), sigma forgotten, and the change in utility (S1^2 - S0^2) / (2A).
Answers rounded to two decimals; distractors at least 10% (and 1 point) from the key and each other, with a random
number below the key, and never both half and twice the key.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.rank_pick import pick_ranked
from .utilities.story_utils import PERSPECTIVES, num, voice

function_registry = FillRegister()

ELEMENT = 'price_of_risk'

STORIES = [
    dict(domain='finance', anon='a retirement saver', money='savings', fund='a broad stock-market index fund'),
    dict(domain='education', anon='a university lecturer', money='research prize money', fund='an emerging-markets equity fund'),
    dict(domain='technology', anon='a software engineer', money='year-end bonus', fund='a technology-sector stock fund'),
    dict(domain='medical', anon='a doctor', money='savings', fund='a health-care sector stock fund'),
    dict(domain='environmental_policy', anon='an environmental consultant', money='savings', fund='a clean-energy stock fund'),
    dict(domain='entertainment', anon='a musician', money='royalty income', fund='a small-company stock fund'),
    dict(domain='sports', anon='a retired athlete', money='prize money', fund='a real-estate investment trust fund'),
    dict(domain='consumer_goods', anon='a shop owner', money='spare cash', fund='a consumer-staples stock fund'),
]

BORROW = {
    True: ("{subj} can borrow at the risk-free rate, so the share in the fund may be above 100% (borrowing to invest "
           "more than {poss} {money})."),
    False: "{subj} cannot borrow or sell short, so the share in the fund must be between 0% and 100%.",
}


def _parts(st: dict, tag: str, k: int, qtype: str, borrow: bool, ask: str) -> tuple[str]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}", 'their' if tag == 'anon' else None)
    first = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    money = st['money']
    util = (f"{v['Subj']} choose{v['s']} the share w of {v['poss']} {money} in the fund to maximize the mean-variance "
            f"utility U = E[r] - (A/2) Var(r) of the portfolio's one-year return r, with returns measured as decimals, "
            f"not percentages, and risk aversion A = {{gamma}}. ")
    rule = BORROW[borrow].format(subj=v['Subj'], poss=v['poss'], money=money)
    if qtype == 'risk_allocation':
        text = (f"{first} invest{v['s']} {v['poss']} {money} for one year, splitting it between a risk-free account "
                f"paying {{risk_free_rate}}% and {st['fund']}, whose one-year return has an expected value of "
                f"{{expected_return}}% and a standard deviation of {{std_dev}}%. {util}{rule} What share of "
                f"{v['poss']} {money} should {v['q']} put in the fund? Give the percentage, rounded to two decimals.")
    else:
        q = (f"By how many percentage points does the optimal share of {v['poss']} {money} in the fund change? Round to "
             f"two decimals, negative for a decrease." if ask == 'change' else
             f"What is the optimal share of {v['poss']} {money} in the fund after the change? Give the percentage, "
             f"rounded to two decimals.")
        text = (f"{first} invest{v['s']} {v['poss']} {money} for one year, splitting it between a risk-free account "
                f"and {st['fund']}. The fund's price of risk is its Sharpe ratio S = (expected return - risk-free "
                f"rate) / standard deviation, with returns as decimals. {util}{rule} The fund's standard deviation is "
                f"{{std_dev}}% and stays the same, while its price of risk rises from {{price_of_risk_start}} to "
                f"{{price_of_risk_end}}. {q}")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for i, tag in enumerate(PERSPECTIVES):
            borrow = (k + i) % 2 == 0
            out.append(ct.CodeTemplate(parts=_parts(st, tag, k, 'risk_allocation', borrow, ''), type='risk_allocation',
                                       domain=st['domain'], tags=tag, difficulty_level=1,
                                       spec={'type': 'risk_allocation', 'borrow': borrow}))
            for ask in ('change', 'level'):
                b2 = (k + i + (ask == 'level')) % 2 == 0
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, 'utility_analysis', b2, ask), type='utility_analysis',
                                           domain=st['domain'], tags=tag, difficulty_level=2,
                                           spec={'type': 'utility_analysis', 'borrow': b2, 'ask': ask}))
    return out


def share(mu, rf, sd, A):
    """Unconstrained optimal share (fraction); mu, rf, sd in percent."""
    return (mu - rf) / 100 / (A * (sd / 100) ** 2)


def _half_and_double(key, dist):
    """Both half and twice the key among the options would make the key the one in the middle."""
    return any(abs(d - key / 2) < 1e-9 for d in dist) and any(abs(d - 2 * key) < 1e-9 for d in dist)


def _allocation(r, borrow):
    lever = r.random() < 0.25
    for _ in range(100000):
        A = int(r.integers(2, 9))
        rf = 0.25 * int(r.integers(4, 17))                 # 1% .. 4%
        mu = 0.1 * int(r.integers(50, 151))                # 5% .. 15%
        sd = int(r.integers(10, 31))
        w = share(mu, rf, sd, A)
        if lever != (1.05 < w < 2.5) or (not lever and not 0.08 <= w <= 0.95):
            continue
        clip = (lambda z: z) if borrow else (lambda z: min(1.0, max(0.0, z)))
        key = 100 * clip(w)
        # mistakes: half / twice the share, sigma not squared, the risk-free rate ignored or added instead of
        # subtracted, the other reading of the borrowing rule, the risk-free share
        cands = [100 * clip(z) for z in (w / 2, 2 * w, (mu - rf) / 100 / (A * sd / 100), mu / 100 / (A * (sd / 100) ** 2),
                                         (mu + rf) / 100 / (A * (sd / 100) ** 2))]
        cands.append(100 * (min(1.0, w) if borrow else w))   # the other reading of the borrowing rule
        if w < 1:
            cands.append(100 * (1 - w))   # the share left in the risk-free asset
        dist = pick_ranked(r, key, cands, rel=0.1, abs_gap=1.0, shown=num)
        if dist is None or _half_and_double(key, dist):
            continue
        return {'gamma': str(A), 'risk_free_rate': num(rf), 'expected_return': num(mu), 'std_dev': str(sd),
                'options_list': [[f'{num(z)}%' for z in [key] + dist]]}
    raise RuntimeError('risk_allocation')


def _analysis(r, ask):
    for _ in range(100000):
        A = int(r.integers(2, 9))
        sd = int(r.integers(10, 31))
        s0 = 0.01 * int(r.integers(10, 51))
        s1 = round(s0 + 0.01 * int(r.integers(10, 41)), 2)
        sig = sd / 100
        w0, w1 = s0 / (A * sig), s1 / (A * sig)
        if not (0.05 < w0 < 0.95 and 0.05 < w1 < 0.95):
            continue
        ds = s1 - s0
        if ask == 'change':
            key = 100 * ds / (A * sig)
            cands = [100 * ds / (A * sig ** 2), 2 * key, key / 2, 100 * w1, 100 * ds / A,
                     100 * (s1 ** 2 - s0 ** 2) / (2 * A)]
        else:
            key = 100 * w1
            cands = [100 * w0, 100 * ds / (A * sig), 100 * s1 / (A * sig ** 2), 2 * key, key / 2, 100 * s1 / A]
        dist = pick_ranked(r, key, cands, rel=0.1, abs_gap=1.0, shown=num)
        if dist is None or _half_and_double(key, dist):
            continue
        unit = ' percentage points' if ask == 'change' else '%'
        return {'gamma': str(A), 'std_dev': str(sd), 'price_of_risk_start': num(s0), 'price_of_risk_end': num(s1),
                'options_list': [[f'{num(z)}{unit}' for z in [key] + dist]]}
    raise RuntimeError('utility_analysis')


@function_registry.register(ELEMENT)
def fill_price_of_risk(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    out = _allocation(r, spec['borrow']) if spec['type'] == 'risk_allocation' else _analysis(r, spec['ask'])
    out['correct_answer'] = [[1, 0, 0, 0]]
    return out


ct.register(ELEMENT, _templates())
