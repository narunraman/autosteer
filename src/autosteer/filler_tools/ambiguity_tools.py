"""ambiguity_aversion (STEER 2.5.d): Ellsberg three-colour urn, two parts, graded by consistency (ellsberg_seu).

Decision of 3 October 2026: recover the element from the Ellsberg templates. The templates are the v1 GPT-4
templates of utility_gen/ambiguity.pkl, labelled by scripts/recover_ellsberg_templates.py (440 of 737 kept; see
that script) into ellsberg_templates.json: for each template, which colour has the known count, which part-0 bet is
on an unknown colour, and the role of each question slot. Values as in the recovered fill code
(fill_generations/ambiguity.ipynb): the urn holds n balls of the known colour and 2n of the two others (n = 1..100,
total 3n), and every bet pays the same reward of 1, 2, 4 or 7.

The filler returns the options with the known-probability bet first in both parts (part 0: the known colour;
part 1: the two unknown colours together) and "Indifferent between the two options" last; release/rows.py builds the
rule "ellsberg_seu:<a><c>,<b><d>,<i0><i1>" from where they are shown (release/consistency.py). No option is marked
correct. Review, October 2026: the indifference option was added (at the belief p = 1/3 an SEU agent is
indifferent in both parts, so without it the known-probability bet in both parts is a defensible tie-break), and
part 0 ends with a sentence saying that one item is drawn at random, each equally likely, and that its colour cannot
be seen or chosen (some stories "select" or "encounter" an item, which does not make the known probability 1/3).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

TEMPLATES = json.loads(Path(__file__).with_name('ellsberg_templates.json').read_text(encoding='utf-8'))['templates']
REWARDS = (1, 2, 4, 7)
UNIT = re.compile(r'\b(\d+) ((?:gold |golden |silver |treasure |magic )?)(dollar|coin|gemstone|chest|token|piece|jewel|'
                  r'gem|point|credit|star|crystal|potion)\b(?!s)')


def _named(q0: str, roles: list[str]) -> str:
    out = q0.strip()
    for role in roles:
        out = out.replace('{}', '{' + role + '}', 1)
    return out


INDIFFERENT = 'Indifferent between the two options'
RANDOM_DRAW = ('Exactly one of them will be drawn at random, each equally likely, and its colour cannot be seen or '
               'chosen before the draw.')


def _with_draw(q0: str) -> str:
    q0 = q0.rstrip()
    sep = '' if q0.endswith(('.', ':', '?', '!')) else '.'
    if q0.endswith(':'):   # "... Here are your options:" -> put the sentence before it
        head, _, tail = q0.rpartition('. ')
        if head:
            return f'{head}. {RANDOM_DRAW} {tail}'
    return f'{q0}{sep} {RANDOM_DRAW}'


def _templates() -> list[ct.CodeTemplate]:
    return [ct.CodeTemplate(parts=(_with_draw(_named(t['q0'], t['roles'])), t['q1']), type='two_part', domain=t['domain'],
                            tags='second_person', difficulty_level=5, spec={'index': k})
            for k, t in enumerate(TEMPLATES)]


def reward_text(option: str, reward: int) -> str:
    text = option.strip().replace('{}', str(reward), 1)
    if reward != 1:
        text = UNIT.sub(lambda m: f'{m.group(1)} {m.group(2)}{m.group(3)}s', text)
    return text


@function_registry.register('ambiguity_aversion')
def fill_ambiguity_aversion(metadata: dict) -> dict:
    t = TEMPLATES[ct.spec_for('ambiguity_aversion', metadata['question_text'])['index']]
    r = filler_utils.rng
    n = int(r.integers(1, 101))
    reward = REWARDS[int(r.integers(0, len(REWARDS)))]
    # option texts end without a period (the recovered texts mix both; quality audit)
    options = [[reward_text(o, reward).rstrip('.') for o in t['options0']] + [INDIFFERENT],
               [reward_text(o, reward).rstrip('.') for o in t['options1']] + [INDIFFERENT]]
    return {'known': n, 'unknown': 2 * n, 'total': 3 * n,
            'options_list': options, 'correct_answer': [[0, 0, 0], [0, 0, 0]]}


ct.register('ambiguity_aversion', _templates())
