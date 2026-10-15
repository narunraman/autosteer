"""Fillers for the v1-era STEER elements time_inconsistency, transitivity_risk and independence_risk (recovered
2026-10; templates: audit/steer/regen/recover_axiom_templates.py). The original fill code is lost; number ranges
follow the released rows.

time_inconsistency  You chose $a today over $b tomorrow (type today) or the reverse (type tomorrow); now $a in t
                    days or $b in t+1 days. A time-consistent (exponential) discounter makes the same choice, so
                    the key is the option matching the earlier choice. a 30-400, b - a 2-20, t 5-200 (as released).
transitivity_risk   X: p% of I1 preferred to q% of I2; Y: q% of I2 preferred to r% of I3. Options: "p% of I1 over
                    r% of I3" (key, by transitivity), the reverse, "Cannot determine preference". p, q, r distinct
                    integers 10-90.
independence_risk   prefers a to b; Option X = p% a + (100-p)% c, Option Y = p% b + (100-p)% c. The option mixing
                    in the preferred item is the key (independence axiom). The letters X / Y are swapped in half
                    the questions so the key is not always "Option X". p 10-90, not 50.
"""
from __future__ import annotations

import re

from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()


def _rng():
    return filler_utils.rng


def _int(a: int, b: int) -> int:
    return int(_rng().integers(a, b + 1))


@function_registry.register('time_inconsistency')
def fill_time_inconsistency(metadata: dict) -> dict:
    a = _int(30, 400)
    b = a + _int(2, 20)
    t = _int(5, 200)
    opts = [f'{a} dollars {t} days from now', f'{b} dollars {t + 1} days from now']
    key = 0 if metadata['type'] == 'today' else 1
    return {'a': a, 'b': b, 't': t, 't1': t + 1, 'options_list': [opts],
            'correct_answer': [[int(i == key) for i in range(2)]]}


ITEMS = re.compile(r'A \{p\}% chance of (.+?) is preferred over a \{q\}% chance of (.+?)\.\n'
                   r'Preference Y: A \{q\}% chance of \2 is preferred over a \{r\}% chance of (.+?)\.\n')


@function_registry.register('transitivity_risk')
def fill_transitivity_risk(metadata: dict) -> dict:
    m = ITEMS.search(str(metadata.get('question_text', '')))
    i1, _, i3 = m.groups() if m else ('item one', 'item two', 'item three')
    while True:
        p, q, r = (_int(10, 90) for _ in range(3))
        if len({p, q, r}) == 3:
            break
    opts = [f'A {p}% chance of {i1} over a {r}% chance of {i3}', f'A {r}% chance of {i3} over a {p}% chance of {i1}',
            'Cannot determine preference']
    return {'p': p, 'q': q, 'r': r, 'options_list': [opts], 'correct_answer': [[1, 0, 0]]}


@function_registry.register('independence_risk')
def fill_independence_risk(metadata: dict) -> dict:
    while True:
        p = _int(10, 90)
        if p != 50:
            break
    first, second = ('X', 'Y') if _rng().random() < 0.5 else ('Y', 'X')
    opts = [f'Option {first}', f'Option {second}']
    return {'p': p, 'q': 100 - p, 'first': first, 'second': second, 'options_list': [opts],
            'correct_answer': [[1, 0]]}
