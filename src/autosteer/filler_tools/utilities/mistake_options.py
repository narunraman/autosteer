"""Distractors from named calculation mistakes (Review, October 2026).

The default numeric distractors (`filler_utils.gen_answer_and_options`: rand * answer + 0 or 1) are almost always
below the answer, so "pick the largest option" found the key. The helpers here take the values a respondent gets by
a specific mistake, pick some of them at random, and only fall back to random multiples of the key, drawn above or
below it with equal chance, when there are too few distinct mistakes. Which candidates are larger than the key varies
with the mistake and the draw, so the key's rank among the options carries no information.
"""
from __future__ import annotations

import math

from .filler_utils import rng
from autosteer.filler_tools.rng import pyrng


def _apart(x: float, y: float, rel: float, absolute: float) -> bool:
    return abs(x - y) > max(absolute, rel * max(abs(x), abs(y)))


def pick_mistakes(key: float, mistakes, n: int = 3, rel: float = 0.05, absolute: float = 0.015,
                  max_factor: float = 8.0, same_sign: bool = True, no_negation: bool = False,
                  fill: bool = True, ndigits: int = 2, allow_zero: bool = False) -> list[float] | None:
    """n distractor values: mistakes in random order that are finite, within max_factor of the key (by size), of
    the key's sign (same_sign), and apart from the key and from each other after rounding to ndigits. With
    no_negation, no chosen value is the negative of the key or of another chosen value (a sign-flip pair gives the
    key away); allow_zero admits a 0 distractor ("no change"). If fewer than n qualify and fill is set, the rest are
    key * f with f in [0.55, 0.85] or [1.2, 1.8]; otherwise None."""
    key_r = round(float(key), ndigits)
    chosen: list[float] = []

    def ok(c):
        if not math.isfinite(c):
            return False
        c = round(c, ndigits)
        if same_sign and key_r != 0 and (c == 0 or (c > 0) != (key_r > 0)):
            return False
        if key_r != 0 and c != 0 and not (abs(key_r) / max_factor <= abs(c) <= abs(key_r) * max_factor):
            return False
        if abs(c) < 0.5 * 10 ** -ndigits and key_r != 0 and not allow_zero:
            return False
        for o in [key_r] + chosen:
            if not _apart(c, o, rel, absolute):
                return False
            if no_negation and o != 0 and abs(c + o) <= absolute:
                return False
        return True

    cands = [float(m) for m in mistakes]
    for i in [int(j) for j in rng.permutation(len(cands))]:
        if ok(cands[i]):
            chosen.append(round(cands[i], ndigits))
            if len(chosen) == n:
                return chosen
    if not fill:
        return None
    for _ in range(200):
        if len(chosen) == n:
            return chosen
        f = float(rng.uniform(1.2, 1.8)) if rng.random() < 0.5 else float(rng.uniform(0.55, 0.85))
        c = float(key) * f
        if ok(c):
            chosen.append(round(c, ndigits))
    return chosen if len(chosen) == n else None


def options_from_values(key, distractors, fmt=None):
    """([mask], [options]) as gen_answer_and_options returns them, the key formatted like the distractors."""
    fmt = fmt or (lambda v: str(round(float(v), 2)))
    texts = [fmt(key)] + [fmt(d) for d in distractors]
    correct = texts[0]
    pyrng.shuffle(texts)
    return [[1 if t == correct else 0 for t in texts]], [texts]


def mistake_options(key, mistakes, fmt=None, **kw):
    """Key plus three distractors from `mistakes` (see pick_mistakes), or None when there are too few."""
    d = pick_mistakes(key, mistakes, **kw)
    if d is None:
        return None
    return options_from_values(key, d, fmt)
