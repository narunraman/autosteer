"""Options for the two-group aggregation elements (aggregation_of_consumer_demand, aggregation_of_offer_curve).

Review, October 2026: the numerical distractors were random fractions of the key (plus 0 or 1), so the
key was always the largest option, and the function distractors had coefficients on another scale than the key.
Here every distractor is the result of a natural aggregation mistake, and the key's rank among the options is drawn
at random.

Two groups of n_a and n_b members; each member of group i has the linear curve q_i(P) = a_i + s c_i P (s = -1 for
demand, +1 for an offer curve).
"""
from __future__ import annotations

import itertools

from .rounding import near_tie


def _apart(v, others, rel=0.05):
    return all(abs(v - w) >= rel * max(abs(v), abs(w)) for w in others)


def total_mistakes(n_a, n_b, a, c_a, b, c_b, s, price):
    """The total quantity at `price` under natural mistakes."""
    qa, qb = a + s * c_a * price, b + s * c_b * price
    n = n_a + n_b
    return [
        n_b * qa + n_a * qb,                                  # group sizes swapped
        n_a * qa,                                             # group a only
        n_b * qb,                                             # group b only
        n_a * a + n_b * b,                                    # price term left out
        n_a * (a - s * c_a * price) + n_b * (b - s * c_b * price),   # sign of the price term flipped
        n * (qa + qb) / 2,                                    # all members at the average of the two curves
        n * qa,                                               # all members on group a's curve
        n * qb,                                               # all members on group b's curve
    ]


def numeric_options(r, key, mistakes, n_options=4):
    """The key and n_options - 1 mistakes (positive, at least 5% from the key and from each other), with the number
    of options below the key drawn uniformly from 0..n_options-1. When the mistakes on one side run short, the
    remaining ones are arithmetic slips 6-40% from the key on that side. Returns (texts, mask), texts rounded to 2
    decimals, or None when no well-separated set was found (the caller redraws)."""
    if near_tie(key):
        return None               # exactly on a rounding tie: the printed key would depend on the rule
    key = round(key, 2)
    below_n = int(r.integers(0, n_options))
    above_n = n_options - 1 - below_n
    cands = [round(m, 2) for m in mistakes if m > 0]
    order = [int(i) for i in r.permutation(len(cands))]
    below, above = [], []
    for i in order:
        c = cands[i]
        side = below if c < key else above
        want = below_n if c < key else above_n
        if len(side) < want and _apart(c, [key] + below + above):
            side.append(c)
    for side, want, sign in ((below, below_n, -1), (above, above_n, 1)):
        for _ in range(1000):
            if len(side) >= want:
                break
            c = round(key * (1 + sign * float(r.uniform(0.06, 0.4))), 2)
            if c > 0 and _apart(c, [key] + below + above):
                side.append(c)
    vals = [key] + below + above
    if len(vals) != n_options:
        return None
    perm = [int(i) for i in r.permutation(n_options)]
    texts = [str(vals[i]) for i in perm]
    mask = [1 if i == 0 else 0 for i in perm]
    return texts, mask


def curve_text(intercept, slope, s):
    return f"{round(intercept, 2)} {'-' if s < 0 else '+'} {round(slope, 2)}P"


def function_options(r, n_a, n_b, a, c_a, b, c_b, s):
    """The aggregate curve A + s B P (A = n_a a + n_b b, B = n_a c_a + n_b c_b) and three distractors forming a
    2 x 2 grid {A, A'} x {B, B'} with the key: A' and B' are each an aggregation mistake (group sizes swapped; all
    members at the average of the two coefficients; all members with group a's or group b's coefficient), so every
    distractor is on the key's scale and each coefficient occurs in two options. Returns (texts, mask) or None if
    no mistake is at least 5% from the key's coefficient."""
    n = n_a + n_b

    def mistakes(u, v):
        return [n_b * u + n_a * v, n * (u + v) / 2, n * u, n * v]

    A, B = n_a * a + n_b * b, n_a * c_a + n_b * c_b
    ia = [m for m in mistakes(a, b) if _apart(round(m, 2), [round(A, 2)])]
    sb = [m for m in mistakes(c_a, c_b) if _apart(round(m, 2), [round(B, 2)])]
    if not ia or not sb:
        return None
    A2 = ia[int(r.integers(0, len(ia)))]
    B2 = sb[int(r.integers(0, len(sb)))]
    vals = [curve_text(i, j, s) for i, j in itertools.product((A, A2), (B, B2))]   # key first
    if len(set(vals)) != 4:
        return None
    perm = [int(i) for i in r.permutation(4)]
    return [vals[i] for i in perm], [1 if i == 0 else 0 for i in perm]
