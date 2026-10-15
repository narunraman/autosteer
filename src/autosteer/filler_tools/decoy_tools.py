"""independence (STEER 2.1.b): the decoy effect, two parts, graded by consistency (same_brand).

Decision of 3 October 2026. Part 0 offers brands A and B, each with a price and a quality rating: one is cheaper,
the other better, so neither dominates. Part 1 offers the same A and B (same texts) plus a decoy, brand C, built as
the standard asymmetric decoy: C is worse than the target (A or B, drawn at random) on both attributes (a higher
price and a lower quality rating) but not worse than the other brand on both (it beats it on one attribute).
A rational chooser is not moved by an option it would never pick: the choices are consistent iff the same brand is
chosen in both parts. Margins: B costs 30-200% more than A and rates 10-30 points higher; C is 5-25% dearer and
4-12 points worse than the target.

The scenario sentences are the v1 decoy templates (preferences_gen/decoy.pkl), recovered by
scripts/recover_decoy_templates.py into decoy_templates.json, with "two" in part 0 and "three" in part 1 as
released; the option wording is the released one ("Brand A: price is $200, with quality rating 55"). The filler
returns part 0 as [target, other, no preference] and part 1 as [target, other, decoy, no preference]; release/rows.py
builds the rule "same_brand:<t0><t1>,<o0><o1>,<i0><i1>" from where they are shown. No option is marked correct.
Review, October 2026: "No preference between Brand A and Brand B" was added to both parts (a chooser
indifferent between A and B could break the tie with the decoy without violating independence); no preference in
both parts is consistent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

NO_PREFERENCE = 'No preference between Brand A and Brand B'
TEMPLATES = json.loads(Path(__file__).with_name('decoy_templates.json').read_text(encoding='utf-8'))['templates']


def _count(text: str, word: str) -> str:
    """Fill the count slot, capitalised at the start of a sentence ("knives. two knife sets" -> "knives. Two ...")."""
    text = re.sub(r'(^|[.!?]\s+)\{\}', lambda m: m.group(1) + word.capitalize(), text)
    return text.replace('{}', word)


def _templates() -> list[ct.CodeTemplate]:
    return [ct.CodeTemplate(parts=(_count(t['text'], 'two'), _count(t['text'], 'three')),
                            type='asymmetric_decoy', domain=t['domain'],
                            tags='second_person;context' if t['context'] else 'second_person;no_context',
                            difficulty_level=1, spec={'index': k})
            for k, t in enumerate(TEMPLATES)]


def brand(label: str, price: int, quality: int) -> str:
    return f'Brand {label}: price is ${price:,}, with quality rating {quality}'


def decoy_draw(r):
    """(A, B, C, target) with A = (price, quality) cheaper and worse than B, and C the decoy of the target."""
    while True:
        pa = int(r.integers(20, 501))
        pb = int(round(pa * float(r.uniform(1.3, 3.0))))
        qa = int(r.integers(30, 71))
        qb = qa + int(r.integers(10, 31))
        if qb > 99:
            continue
        target = 'A' if r.integers(0, 2) == 0 else 'B'
        if target == 'A':
            pc = int(round(pa * float(r.uniform(1.05, 1.25))))
            qc = qa - int(r.integers(4, 13))
            ok = pa < pc < pb and qc > 0
        else:
            pc = int(round(pb * float(r.uniform(1.05, 1.25))))
            qc = qb - int(r.integers(4, 13))
            ok = pc > pb and qa < qc < qb
        if ok:
            return (pa, qa), (pb, qb), (pc, qc), target


@function_registry.register('independence')
def fill_independence(metadata: dict) -> dict:
    ct.spec_for('independence', metadata['question_text'])  # a code template of this element
    a, b, c, target = decoy_draw(filler_utils.rng)
    texts = {'A': brand('A', *a), 'B': brand('B', *b)}
    other = 'B' if target == 'A' else 'A'
    part0 = [texts[target], texts[other], NO_PREFERENCE]
    part1 = [texts[target], texts[other], brand('C', *c), NO_PREFERENCE]
    return {'options_list': [part0, part1], 'correct_answer': [[0, 0, 0], [0, 0, 0, 0]]}


ct.register('independence', _templates())
