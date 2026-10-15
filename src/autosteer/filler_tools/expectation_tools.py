"""Fillers for the v1-era STEER elements compute_expectations and compute_expected_utility (paper version, see
the section at the end). compute_expectations (recovered 2026-10; templates and families:
audit/steer/regen/recover_expectation_templates.py, data/templates/compute_expectations/template_labels.csv).

Families and formulas (checked against every released row): coin (A - B)/2; faces (k A - (6 - k) B)/6 for k listed
faces of a six-sided die; sides (N + 1)/2; spinner (a + b)/2 with b = a + N - 1; bag (n1 + n2 + n3)/n1 draws
with replacement until the first-named item; days p D / 100. Number ranges follow the released rows. The answer
has 2 decimals with the template's unit (points / times / days / none).

Distractors (re-derived; the released ones were random numbers, some negative counts): the usual mistakes of the
family, e.g. forgetting the loss's sign, swapping the face counts, N/2 instead of (N+1)/2, the probability n1/total
instead of its reciprocal, the other event's days; then perturbations of the answer if fewer than three are
available. Options are at least max(0.1, 5%) apart; counts and days stay positive; every option lies in the range of
the random variable (between the loss and the gain, 1 to N, a to b, 0 to D days, at least 1 draw), where any
expectation lies (review, October 2026: "11.0 days" over a 5-day horizon). Coin and die questions state that
the coin / die is fair (FAIR_COIN, FAIR_DIE).
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache

import numpy as np

from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()
from autosteer.paths import templates_dir  # the data directory, resolved when the labels are read


def _rng():
    return filler_utils.rng


def _int(a, b):
    return int(_rng().integers(a, b + 1))


@lru_cache(maxsize=None)
def labels() -> dict:
    with open(os.path.join(templates_dir(), 'compute_expectations', 'template_labels.csv'), newline='') as f:
        return {r['template']: r for r in csv.DictReader(f)}


# Conventions stated in the question (stated in the stored templates; stripped again for the label
# lookup): a coin or die is fair, and a die has six faces numbered 1 to 6.
FAIR_COIN = ' Assume the coin is fair.'
FAIR_DIE = ' Assume a fair six-sided die with faces numbered 1 to 6.'


def fairness_note(text: str) -> str:
    lab = labels().get(text)
    if lab is None or lab['family'] not in ('coin', 'faces') or text.endswith((FAIR_COIN, FAIR_DIE)):
        return text
    return text.rstrip() + (FAIR_COIN if 'coin' in text.lower() else FAIR_DIE)


def _strip_note(text: str) -> str:
    for note in (FAIR_COIN, FAIR_DIE):
        if text.endswith(note):
            return text[:-len(note)]
    return text


def _draw(family: str):
    """(slot values, answer, distractor candidates, positive-only, (lo, hi) range of the random variable: an
    expectation, and so every option, lies in it)"""
    if family == 'coin':
        A, B = _int(1, 40), _int(1, 40)
        return {'A': A, 'B': B}, (A - B) / 2, [(A + B) / 2, A - B, (B - A) / 2, A / 2], False, (-B, A)
    if family == 'faces':
        k = _int(2, 5)
        faces = [int(x) + 1 for x in _rng().permutation(6)[:k]]
        A, B = _int(1, 40), _int(1, 40)
        ans = (k * A - (6 - k) * B) / 6
        return ({'A': A, 'B': B, 'faces': ', '.join(map(str, faces))}, ans,
                [((6 - k) * A - k * B) / 6, (A - B) / 2, (k * A + (6 - k) * B) / 6, k * A / 6 - B], False, (-B, A))
    if family == 'sides':
        N = _int(2, 14)
        return {'N': N}, (N + 1) / 2, [N / 2, (N - 1) / 2, N / 2 + 1, N + 1], True, (1, N)
    if family == 'spinner':
        a, N = _int(1, 5), _int(2, 11)
        b = a + N - 1
        return {'a': a, 'b': b, 'N': N}, (a + b) / 2, [(N + 1) / 2, (b - a) / 2, N / 2 + a, b / 2, (a + b - 1) / 2,
                                                          (a + b + 1) / 2], True, (a, b)
    if family == 'bag':
        n1, n2, n3 = _int(5, 250), _int(1, 200), _int(1, 200)
        tot = n1 + n2 + n3
        return ({'n1': n1, 'n2': n2, 'n3': n3}, tot / n1,
                [n1 / tot * 10, tot / n2, tot / n3, (n2 + n3) / n1, tot / n1 + 1], True, (1, float('inf')))
    if family == 'days':
        p, D = _int(40, 60), _int(5, 20)
        return {'p': p, 'q': 100 - p, 'D': D}, p * D / 100, [(100 - p) * D / 100, D / 2, p / D, p * D / 100 + 1,
                                                        p * D / 100 - 1], True, (0, D)
    raise ValueError(family)


def _apart(v, vals):
    return all(abs(v - w) >= max(0.1, 0.05 * max(abs(v), abs(w))) for w in vals)


@function_registry.register('compute_expectations')
def fill_compute_expectations(metadata: dict) -> dict:
    text = metadata.get('question_text')
    lab = labels().get(_strip_note(text)) if text else None
    if lab is None:
        if text:
            raise KeyError(f'compute_expectations: unknown template {text[:120]!r}')
        # called without a template (the sampler probing which keys the filler returns): every family's slots
        out = {}
        for fam in ('coin', 'faces', 'sides', 'spinner', 'bag', 'days'):
            out.update(_draw(fam)[0])
        return {**out, 'options_list': [['1.00', '2.00']], 'correct_answer': [[1, 0]]}
    while True:
        slots, ans, cands, positive, (lo, hi) = _draw(lab['family'])
        vals = [round(ans, 2)]

        def ok(c):
            return _apart(c, vals) and (c > 0 or not positive) and lo <= c <= hi
        for i in _rng().permutation(len(cands)):
            c = round(cands[int(i)], 2)
            if len(vals) < 4 and ok(c):
                vals.append(c)
        for _ in range(50):
            if len(vals) == 4:
                break
            c = round(ans * float(_rng().uniform(0.5, 1.6)) + float(_rng().choice([-1, 1])) * float(_rng().uniform(0, 2)), 2)
            if ok(c):
                vals.append(c)
        if len(vals) == 4:
            break
    unit = (' ' + lab['unit']) if lab['unit'] else ''
    return {**slots, 'options_list': [[f'{v:.2f}{unit}' for v in vals]], 'correct_answer': [[1, 0, 0, 0]]}


# ------------------------------------------------------------------ compute_expected_utility (paper version)
# Decision 2026-10-02: the element is the ICML / paper version (2.4.a). Templates and per-template labels (utility
# slots with their released ranges, implicit last probability): audit/steer/regen/recover_eu_templates.py.
# Probabilities are positive integer percentages summing to 100 over the 2 or 3 outcomes; utilities are integers in
# the released range of their slot, distinct across outcomes; a literal 0 utility stays 0. The answer is
# sum(p_i u_i) / 100 with 2 decimals. Distractors (2, as released 3 options in all) come from plausible mistakes:
# leaving one outcome out of the sum, the unweighted mean of the utilities, and weighting the utilities with the
# probabilities of the wrong outcomes (one swap); options are at least max(0.5, 5%) apart, and no two options share
# their hundredths (v * 100 mod 100). (The released options were the key plus or minus a whole
# number, so every option shared the key's decimals.) Every option lies between the smallest and the largest utility
# (where any expected utility lies) and is printed with 2 decimals. Draws where fewer than 2 such distractors exist
# are redrawn.
@lru_cache(maxsize=None)
def eu_labels() -> dict:
    import json
    with open(os.path.join(templates_dir(), 'compute_expected_utility', 'template_labels.csv'), newline='') as f:
        return {r['template']: json.loads(r['labels']) for r in csv.DictReader(f)}


def _composition(n: int) -> list:
    while True:
        cuts = sorted(int(x) for x in _rng().choice(np.arange(1, 100), size=n - 1, replace=False))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [100])]
        if min(parts) >= 1:
            return parts


def eu_distractors(P: list, U: list) -> list:
    ans = sum(p * u for p, u in zip(P, U)) / 100
    c = [ans - p * u / 100 for p, u in zip(P, U) if u != 0]
    c.append(sum(U) / len(U))
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            Q = list(P)
            Q[i], Q[j] = Q[j], Q[i]
            c.append(sum(p * u for p, u in zip(Q, U)) / 100)
    return c


def _cents(v: float) -> int:
    return round(v * 100) % 100


def eu_apart(v: float, vals: list) -> bool:
    """v is at least max(0.5, 5%) from every option in vals and has different hundredths digits (no shared
    decimal pattern among the options)."""
    return all(abs(v - w) >= max(0.5, 0.05 * max(abs(v), abs(w))) and _cents(v) != _cents(w) for w in vals)


@function_registry.register('compute_expected_utility')
def fill_compute_expected_utility(metadata: dict) -> dict:
    text = metadata.get('question_text')
    lab = eu_labels().get(text) if text else None
    if lab is None:
        if text:
            raise KeyError(f'compute_expected_utility: unknown template {text[:120]!r}')
        lab = {'utilities': [['u1', 1, 100], ['u2', 1, 100], ['u3', 1, 100]], 'implicit_last': False, 'n_outcomes': 3}
    n = lab['n_outcomes']
    while True:
        P = _composition(n)
        U = [0 if name == '0' else _int(lo, hi) for name, lo, hi in lab['utilities']]
        if len(set(U)) < n:
            continue
        ans = round(sum(p * u for p, u in zip(P, U)) / 100, 2)
        vals = [ans]
        cands = eu_distractors(P, U)
        for i in _rng().permutation(len(cands)):
            v = round(cands[int(i)], 2)
            # an expected utility lies between the smallest and the largest utility; a distractor outside that
            # range (leaving an outcome out of the sum) is ruled out without computing, and one at its end ("0.00" when the
            # other outcome is left out) is not a weighted mean (review, October 2026)
            if len(vals) < 3 and eu_apart(v, vals) and min(U) < v < max(U):
                vals.append(v)
        if len(vals) == 3:
            break
    out = {f'p{i + 1}': p for i, p in enumerate(P)}
    out.update({name: u for (name, _, _), u in zip(lab['utilities'], U) if name != '0'})
    return {**out, 'options_list': [[f'{v:.2f}' for v in vals]], 'correct_answer': [[1, 0, 0]]}
