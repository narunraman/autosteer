"""present_value (STEER-ME, Consumption Decisions, Dynamic Consumption Decisions): the paper's "Exponential
Discounting" ("the ability to exponentially discount future rewards or costs").

Decision of 5 October 2026 (triage D1, option b); templates written here. The 2024 data the paper's heatmap reports
under this name were time-consistency questions, the design of STEER's `time_inconsistency`, which elements.json
records as an alias of the paper name. This element asks for a present value.

Types:
  amount: one payment X received t years from now;          PV = X / (1 + r)^t = X d^t
  stream: payments C_1, ..., C_n at the end of years 1..n;   PV = sum_k C_k / (1 + r)^k = sum_k C_k d^k
  perpetuity: C every year forever, the first in one year;   PV = C / r = C d / (1 - d)
Half of the templates state an annual interest rate r (compounded annually) at which the person discounts, the other
half a yearly discount factor d (a dollar a year from now is worth d dollars today); tags carry the perspective,
the spec says which (`rate` or `factor`).

Distractors are the present value under a common mistake:
  amount: hyperbolic / simple-interest discounting X / (1 + r t) (with d: r = 1/d - 1); one period too few
      (t - 1) or too many (t + 1); the rate subtracted instead of divided, X (1 - r)^t (with d: X (1 - t (1 - d)),
      straight-line discounting).
  stream: each payment discounted hyperbolically; the first payment treated as paid today (every exponent one
      too small) or every exponent one too large; all payments discounted as if received in the last year.
  perpetuity: the first payment counted today (C / r + C, or C / (1 - d)); the first payment two years away
      (C / (r (1 + r)), or C d^2 / (1 - d)); only the first payment (C / (1 + r), or C d).
Three are chosen at random among those at least 2% apart from the key and from each other. Shown cents never end
in 0 (the sampler's string rounding would print "$12.3").

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, money, pick_apart, stable, voice

function_registry = FillRegister()

ELEMENT = 'present_value'
TYPES = ('amount', 'stream', 'perpetuity')
YEAR_WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}

# domain, anonymous person, single payment, stream, perpetuity (each "be owed / receive ..." phrase)
STORIES = [
    dict(domain='finance', anon='an investor', amount='a savings bond that pays ${amount} when it matures in {years} years',
         stream='a bond that pays {stream}', perp='a share of a trust that pays ${amount} every year forever'),
    dict(domain='gambling', anon='a lottery winner', amount='a lottery prize of ${amount} that will be paid out in {years} years',
         stream='a lottery prize paid in instalments: {stream}', perp='a lottery prize that pays ${amount} a year in perpetuity'),
    dict(domain='education', anon='a student', amount='a scholarship award of ${amount} that will be paid in {years} years',
         stream='a scholarship that pays {stream}', perp='an endowed prize fund that pays ${amount} every year forever'),
    dict(domain='medical', anon='a patient', amount='an injury settlement of ${amount} that will be paid in {years} years',
         stream='an injury settlement that pays {stream}', perp='an injury settlement that pays ${amount} every year in perpetuity'),
    dict(domain='sports', anon='an athlete', amount='a deferred signing bonus of ${amount} that will be paid in {years} years',
         stream='a contract bonus paid as {stream}', perp='a sponsorship deal that pays ${amount} every year in perpetuity'),
    dict(domain='entertainment', anon='a songwriter', amount='a royalty payment of ${amount} that will arrive in {years} years',
         stream='royalty payments of {stream}', perp='royalties of ${amount} every year forever'),
    dict(domain='technology', anon='a software developer', amount='a licence fee of ${amount} that will be paid in {years} years',
         stream='licence fees of {stream}', perp='a licence that pays ${amount} every year forever'),
    dict(domain='environmental_policy', anon='a landowner', amount='a carbon-offset payment of ${amount} that will be paid in {years} years',
         stream='carbon-offset payments of {stream}', perp='a conservation easement that pays ${amount} every year forever'),
]


def _parts(st: dict, tag: str, k: int, qtype: str, how: str) -> tuple[str]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}")
    first = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    have = v['have']
    if qtype == 'amount':
        claim = st['amount']
    elif qtype == 'stream':
        claim = st['stream']
    else:
        claim = st['perp'] + ', with the first payment one year from now'
    text = f"{first} {have} {claim}."
    if how == 'rate':
        text += (f" {v['Subj']} discount{v['s']} future money exponentially at an interest rate of {{rate}}% per year, "
                 f"compounded annually.")
    else:
        text += (f" {v['Subj']} discount{v['s']} the future exponentially with a discount factor of {{delta}} per "
                 f"year: a dollar received one year later is worth {{delta}} of a dollar received now.")
    what = 'this payment' if qtype == 'amount' else 'these payments'
    text += f" What is the present value of {what} today, to the nearest cent?"
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                for how in ('rate', 'factor'):
                    out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype, how), type=qtype, domain=st['domain'],
                                               tags=tag, difficulty_level=0 if qtype == 'amount' else 1,
                                               spec={'story': k, 'type': qtype, 'how': how}))
    return out


def stream_text(cs: list[int]) -> str:
    items = [f"${c:,} {YEAR_WORDS[i + 1]} year{'s' if i else ''} from now" for i, c in enumerate(cs)]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def pv(cs, d):
    """sum_k C_k d^k, k = 1..n."""
    return sum(c * d ** (i + 1) for i, c in enumerate(cs))


def draw(r, qtype: str, how: str) -> dict:
    for _ in range(100000):
        if how == 'rate':
            rate = int(r.integers(3, 13))                # percent
            d, k = 1 / (1 + rate / 100), rate / 100      # k: per-year impatience of the hyperbolic mistake
        else:
            delta = round(0.01 * int(r.integers(85, 98)), 2)
            d, k = delta, 1 / delta - 1
        out = {'rate': str(rate)} if how == 'rate' else {'delta': f'{delta:g}'}
        if qtype == 'amount':
            X = 50 * int(r.integers(20, 1001))
            t = int(r.integers(2, 13))
            key = X * d ** t
            lin = X * (1 - k) ** t if how == 'rate' else X * (1 - t * (1 - d))
            cands = [X / (1 + k * t), X * d ** (t - 1), X * d ** (t + 1), lin]
            out.update(amount=f'{X:,}', years=str(t))
        elif qtype == 'stream':
            n = int(r.integers(3, 6))
            cs = [50 * int(r.integers(10, 201)) for _ in range(n)]
            key = pv(cs, d)
            cands = [sum(c / (1 + k * (i + 1)) for i, c in enumerate(cs)), pv(cs, d) / d, pv(cs, d) * d,
                     sum(cs) * d ** n]
            out.update(stream=stream_text(cs))
        else:
            C = int(r.integers(200, 10001))  # any whole amount: with C a multiple of 50, C d always ends in 0 cents
            key = C * d / (1 - d)
            cands = [key + C, key * d, C * d]
            out.update(amount=f'{C:,}')
        dist = pick_apart(r, key, cands, rel=0.02, lo=0.01, shown=money)
        if dist is None:
            continue
        opts = [money(x) for x in [key] + dist]
        if not all(stable(o) for o in opts):
            continue
        out.update(options_list=[opts], correct_answer=[[1, 0, 0, 0]])
        return out
    raise RuntimeError(f'present_value: no draw for {qtype}/{how}')


@function_registry.register(ELEMENT)
def fill_present_value(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'], spec['how'])


ct.register(ELEMENT, _templates())
