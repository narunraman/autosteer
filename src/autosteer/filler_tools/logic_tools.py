"""Filler for the STEER element equiv_contrapositive (STEER 1.4.c; generator built 2026-10, cleaned up 2026-10-06).

The ICML items were written one by one by GPT-4, and some of their stored keys were the converse or the inverse. The
first rebuild took the recovered contexts from the data directory (templates/equiv_contrapositive/template_labels.csv)
and copied the key's clauses verbatim from one GPT-written option while the distractors were assembled from other
phrasings, so the key could differ in style from the distractors (tense, "the patient" / "a patient", a name for a
pronoun).

Now the contexts are the normalised bank `contrapositive_bank.json` (package data, written by
scripts/build_contrapositive_bank.py from the recovered contexts; 429 of 560 kept, the script prints why the others
were dropped). Each context has the four clause strings P, Q, not-P, not-Q, with the negations built from P and Q
themselves (same subject and tense). Templates are code templates: intro + "If P, then Q" + outro. Every option is
"If A, then B." with A, B among those four strings, so key and distractors share their wording exactly:

    key        If not-Q, then not-P        (contrapositive: equivalent to P -> Q)
    converse   If Q, then P                (not equivalent)
    inverse    If not-P, then not-Q        (not equivalent)
    p_notq     If P, then not-Q            (not equivalent; contradicts the statement when P holds)

Two distractors (three in 10% of questions) are drawn from the three kinds. A {speaker_name} in the intro is filled
with a first name.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities.filler_utils import rng

function_registry = FillRegister()

ELEMENT = 'equiv_contrapositive'
FOUR_OPTION_RATE = 0.1
NAMES = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Jamie', 'Avery', 'Quinn', 'Robin', 'Sam',
         'Chris', 'Dana', 'Kim', 'Lee', 'Pat', 'Drew', 'Jesse', 'Sasha', 'Noor', 'Ravi', 'Mei', 'Omar', 'Ines']
BANK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contrapositive_bank.json')


@lru_cache(maxsize=None)
def bank() -> list[dict]:
    with open(BANK_PATH) as f:
        return json.load(f)


def conditional(a: str, b: str) -> str:
    return f'If {a}, then {b}.'


def statement(c: dict) -> str:
    """The question text of a context: intro, 'if P, then Q', outro."""
    return f"{c['intro']}{c['if_word']} {c['P']}, then {c['Q']}{c['outro']}"


def build_options(c: dict) -> dict[str, str]:
    P, Q, nP, nQ = c['P'], c['Q'], c['notP'], c['notQ']
    return {'key': conditional(nQ, nP), 'converse': conditional(Q, P), 'inverse': conditional(nP, nQ),
            'p_notq': conditional(P, nQ)}


def _templates() -> list[ct.CodeTemplate]:
    return [ct.CodeTemplate(parts=(statement(c),), type='other', domain=c['domain'], difficulty_level=2,
                            spec={'context': k}) for k, c in enumerate(bank())]


@function_registry.register(ELEMENT)
def fill_equiv_contrapositive(metadata: dict) -> dict:
    c = bank()[ct.spec_for(ELEMENT, metadata['question_text'])['context']]
    opts = build_options(c)
    kinds = ['converse', 'inverse', 'p_notq']
    n_dis = 3 if rng.random() < FOUR_OPTION_RATE else 2
    chosen = [kinds[i] for i in sorted(rng.choice(len(kinds), size=n_dis, replace=False))]
    options = [opts['key']] + [opts[k] for k in chosen]
    return {
        'speaker_name': str(rng.choice(NAMES)),
        'options_list': [options],
        'correct_answer': [[i == 0 for i in range(len(options))]],
    }


ct.register(ELEMENT, _templates())
