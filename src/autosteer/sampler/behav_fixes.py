"""Load-time template corrections for behavioural STEER elements (review, October 2026).

Registered into the tables of sampler/template_fixes.py (see the end of that module). Each correction answers a
finding of the review of the released elements:

gamblers_fallacy    GAMBLERS_FALLACY_DROPS: templates where "Yes" is defensible or the question is not about a change in
                    probability (a "different outcome" being more likely than the streak outcome, card games dealt from
                    a shoe or deck, which are not independent trials, questions without a comparison or horizon).
independence_risk   every template is emitted twice; the second copy describes the two lotteries in the other order
                    (the preferred item's lottery second, with its label), so the key is not always the lottery
                    described first.
certainty_effect,   stated_prior: a sentence before the question says that the preferences (reflection_effect: the
reflection_effect   strict preference and the attitude to risk) have not changed since the earlier choice, so that
                    the earlier choice determines what the person should pick.
reflection_effect   loss-frame options say "chance of having to pay $Y" (was "chance of paying $Y", which also reads
                    as a payout).
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------------
# gamblers_fallacy
# ----------------------------------------------------------------------
_OTHER = 'asks whether a different outcome is more likely than the streak outcome, not whether it became more likely'
_DEPENDENT = 'cards dealt from a shoe or deck: successive hands are not independent'
_NO_CHANGE = 'does not ask whether the probability changed (no comparison or no horizon)'
GAMBLERS_FALLACY_DROPS: dict[int, str] = {
    **{i: _OTHER for i in (32, 44, 65, 71, 93, 95, 116, 121, 122, 123, 144)},
    145: 'a total of seven is more likely than six with two dice, so "Yes" is defensible',
    **{i: _DEPENDENT for i in (10, 17, 23, 24, 45, 64, 80, 86, 129, 135, 136)},
    40: 'poker hands won depend on skill, not independent fair trials',
    **{i: _NO_CHANGE for i in (2, 6, 47, 68, 70, 72, 101, 131, 142)},
    46: 'garbled ("the same number three has been drawn for the past few times")',
    128: 'garbled ("is your next bet more likely to be odd?")',
}


GAMBLERS_FALLACY_DROPS.update({
    **{i: _DEPENDENT for i in (0, 49, 84)},   # review round 2: a card game's tie, poker hands dealt from a deck
    **{i: 'lottery parity with no number range: odd can be more likely than even (1-49 has 25 odd numbers)'
       for i in (66, 67, 146)},
})
# review round 3: a slot machine that has hit the jackpot several times in a row is evidence of a loose machine, so
# "is it MORE likely to hit again / does it affect your chances" is not a pure gambler's-fallacy item (the "less
# likely" versions stay)
GAMBLERS_FALLACY_DROPS.update({i: 'jackpot streak: asks whether hitting again is more likely (evidence about the '
                                  'machine)' for i in (7, 21, 39, 69, 78, 83, 94)})
# review round 4: a single die cannot show seven or more ("the die has rolled a seven", template 27)
_ONE_DIE_IMPOSSIBLE = re.compile(r'\b(?:a|the) die\b[^?]*\b(?:seven|eight|nine|ten|eleven|twelve)\b', re.I)
# any card game is dropped unless it is dice ("poker dice"); rounds dealt from a deck or shoe need not be independent
_CARDS = re.compile(r'\bcards?\b|\bdeal\w*|\bdeck\b|\bshoe\b|\bhands?\b|\bpoker\b(?! dice)|blackjack|baccarat|'
                    r'\bflush\b|\bbust', re.I)


def _gamblers_fallacy_filter(row: pd.Series) -> bool:
    text = str(row['question_text'])
    return (int(row['question_id']) not in GAMBLERS_FALLACY_DROPS and not _CARDS.search(text)
            and not _ONE_DIE_IMPOSSIBLE.search(text))


# ----------------------------------------------------------------------
# independence_risk
# ----------------------------------------------------------------------
def swap_lottery_order(text: str) -> Optional[str]:
    """Describe the two lotteries in the other order. Every template reads '... {p}% <preferred item> ... {q}% <c>
    ... {p}% <other item> ... {q}% <c> ...' with the labels Option {first} / Option {second}; the two '{p} ... {q}'
    segments are exchanged and, when the labels sit outside them, so are the labels, so that Option {first} (the key)
    still names the lottery with the preferred item, now described second."""
    ps = [m.start() for m in re.finditer(r'\{p\}', text)]
    qs = [m.start() for m in re.finditer(r'\{q\}', text)]
    if len(ps) != 2 or len(qs) != 2 or not ps[0] < qs[0] < ps[1] < qs[1]:
        return None
    s1, s2 = text[ps[0]:qs[0]], text[ps[1]:qs[1]]
    out = text[:ps[0]] + s2 + text[qs[0]:ps[1]] + s1 + text[qs[1]:]
    inside = [('{first}' in s) or ('{second}' in s) for s in (s1, s2)]
    if any(inside):
        return out if ('{first}' in s1 and '{second}' in s2) else None
    if out.count('{first}') != 1 or out.count('{second}') != 1:
        return None
    return out.replace('{first}', '\0').replace('{second}', '{first}').replace('\0', '{second}')


def _independence_risk_swapped(row: pd.Series) -> Optional[str]:
    return swap_lottery_order(row['question_text'])


# ----------------------------------------------------------------------
# certainty_effect / reflection_effect, stated_prior
# ----------------------------------------------------------------------
_STATED_Q = re.compile(r'Which (.+?) should (?:I|you|he|she|they|[A-Z]\w+|the [\w\- ]+) (?:pick|choose|select|take|go '
                       r'with)\?\s*$')
_CHOOSER = re.compile(r'\band (I|you|he|she|they) chose\b')
_POSS = {'I': 'my', 'you': 'your', 'he': 'his', 'she': 'her', 'they': 'their'}


def stated_prior_question(text: str, element: str) -> Optional[str]:
    """Insert an assumption before the final question (which keeps its 'should'; the template wording avoids cue
    words such as 'consistent', see tests/test_stated_prior.py)."""
    m, c = _STATED_Q.search(text), _CHOOSER.search(text)
    if not m or not c:
        return None
    who = c.group(1)
    poss = _POSS[who]
    have = {'I': 'have', 'you': 'have', 'they': 'have'}.get(who, 'has')
    if element == 'reflection_effect':
        lead = (f'Assume that {poss} earlier choice reflected a strict preference and that {poss} attitude to risk '
                f'has not changed since then.')
    else:
        lead = f'Assume that {poss} preferences {have} not changed since then.'
    return text[:m.start()] + lead + ' ' + text[m.start():]


def _stated_prior_rewrite(element: str):
    def rewrite(row: pd.Series) -> Optional[str]:
        if str(row['type']).strip() != 'stated_prior':
            return row['question_text']
        return stated_prior_question(row['question_text'], element)
    return rewrite


_PAYING = re.compile(r'chance of paying (\$\{\w+\})')


def _reflection_hook(element, q, opts):
    """Loss-frame options (part 1 of two_part, the options of stated_prior): 'chance of having to pay $Y'."""
    if opts is None or not len(opts):
        return q, opts
    types = dict(zip(q['question_id'], q['type'].astype(str).str.strip()))
    loss = [(types.get(qid) == 'stated_prior') or int(sid) == 1 for qid, sid in zip(opts['question_id'], opts['sub_id'])]
    opts = opts.copy()
    opts.loc[loss, 'option_text'] = opts.loc[loss, 'option_text'].astype(str).str.replace(
        _PAYING, r'chance of having to pay \1', regex=True)
    return q, opts


# ----------------------------------------------------------------------
# Registration (read by template_fixes)
# ----------------------------------------------------------------------
FILTERS = {'gamblers_fallacy': _gamblers_fallacy_filter}
REWRITES = {'certainty_effect': _stated_prior_rewrite('certainty_effect'),
            'reflection_effect': _stated_prior_rewrite('reflection_effect')}
VARIANTS = {'independence_risk': [('sure_thing', lambda row: row['question_text']),
                                  ('sure_thing', _independence_risk_swapped)]}
HOOKS = {'reflection_effect': _reflection_hook}
AFTER_HOOKS: dict = {}  # completeness / completeness_risk: code templates since October 2026 (completeness_tools.py)
