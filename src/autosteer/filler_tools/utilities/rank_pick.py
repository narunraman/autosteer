"""Distractors chosen from plausible mistakes so that the key's rank among the options carries no information.

`pick_ranked` picks n mistake values that are clearly apart from the key and from each other (as
story_utils.pick_apart) and aims for a uniformly random number of them below the key, so that the key is the
smallest, the largest or in between about equally often when the mistakes allow it (review, October 2026:
in several elements the key was always the largest option, or always the second smallest).
"""
from __future__ import annotations

import math

from .story_utils import pick_apart


def pick_ranked(r, key: float, candidates, n: int = 3, rel: float = 0.1, abs_gap: float = 0.0,
                lo: float | None = None, hi: float | None = None, shown=None, tries: int = 40):
    """n candidates apart from the key and each other, with a random target number below the key.

    Returns None when fewer than n candidates qualify at all."""
    cands = [float(c) for c in candidates if c is not None and math.isfinite(c)
             and (lo is None or c >= lo) and (hi is None or c <= hi)]
    targets = [int(t) for t in r.permutation(n + 1)]
    first = None
    for target in targets:
        for _ in range(tries):
            got = pick_apart(r, key, cands, n=n, rel=rel, abs_gap=abs_gap, shown=shown)
            if got is None:
                return None
            first = first or got
            if sum(g < key for g in got) == target:
                return got
    return first
