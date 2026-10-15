"""Voting rules and social welfare functions shared by the 4.1 axiom elements (code templates).

A profile is a list of (count, ranking) pairs, a ranking a tuple of candidate letters, best first. Every rule
returns None when it meets a tie anywhere (a tied elimination, a tied runoff, a tied score), so that the fillers can
redraw: questions only use profiles on which every listed rule is decided without tie-breaking.

Ranking rules (sc_monotonicity_rules; design of October 2026), each a strict ranking of all options, best first:
    borda_rank      by Borda score                      plurality_rank   by first places
    antiplurality_rank  by last places, fewest first   top2_rank        by first-or-second places (four options)
    copeland_rank   by the number of pairwise majority wins
    irv_order       instant-runoff elimination order (fewest first places out, until one is left), reversed
    coombs_order    Coombs elimination order (most last places out, until one is left), reversed
    runoff_rank     made up: the two with most first places on top, ordered by their head-to-head majority; the rest
                    by first places
    borda_finalists_rank  made up: the two highest Borda scores on top, ordered by first places; the rest by Borda
    plural_finalists_rank made up: the two with most first places on top, ordered by Borda score; the rest by first
                    places
    The first five are monotonic (raising an option never lowers it below an option it was above); the others are not.

Relation rules (sc_transitivity_rules), each a strict complete relation {(x, y): True if x above y} over pairs:
    majority        x above y if more voters rank x above y
    adjacency       x above y if more voters rank x immediately above y than y immediately above x
    restricted      for each pair, only the voters who rank neither of the two last count; majority among them
    restricted_first   the same with the voters who rank neither of the two first
    borda_rank, plurality_rank, antiplurality_rank, top2_rank   orders by a score (higher first; antiplurality:
                    fewer last places first)
"""
from __future__ import annotations

from itertools import combinations


def total(prof) -> int:
    return sum(n for n, _ in prof)


def restrict(prof, remaining):
    return [(n, tuple(c for c in r if c in remaining)) for n, r in prof]


def firsts(prof, cands):
    f = {c: 0 for c in cands}
    for n, r in prof:
        f[r[0]] += n
    return f


def lasts(prof, cands):
    f = {c: 0 for c in cands}
    for n, r in prof:
        f[r[-1]] += n
    return f


def borda_scores(prof, cands):
    m = len(cands)
    s = {c: 0 for c in cands}
    for n, r in prof:
        for i, c in enumerate(r):
            s[c] += n * (m - 1 - i)
    return s


def top2_counts(prof, cands):
    s = {c: 0 for c in cands}
    for n, r in prof:
        for c in r[:2]:
            s[c] += n
    return s


def _unique_max(scores):
    top = max(scores.values())
    w = [c for c, v in scores.items() if v == top]
    return w[0] if len(w) == 1 else None


def _unique_min(scores):
    return _unique_max({c: -v for c, v in scores.items()})


def _top_two(scores):
    """The two highest scorers, or None if the second place is tied with the third (or the first two are not a set)."""
    order = sorted(scores, key=lambda c: -scores[c])
    if len(order) > 2 and scores[order[1]] == scores[order[2]]:
        return None
    return order[:2]


def _majority(prof, x, y):
    nx = sum(n for n, r in prof if r.index(x) < r.index(y))
    ny = sum(n for n, r in prof if r.index(y) < r.index(x))
    if nx == ny:
        return None
    return x if nx > ny else y


def _order(scores, higher_better=True):
    """Options by score (best first), or None if two have the same score."""
    if len(set(scores.values())) < len(scores):
        return None
    return tuple(sorted(scores, key=lambda c: -scores[c] if higher_better else scores[c]))


def borda_rank(prof, cands):
    return _order(borda_scores(prof, cands))


def plurality_rank(prof, cands):
    return _order(firsts(prof, cands))


def antiplurality_rank(prof, cands):
    return _order(lasts(prof, cands), higher_better=False)


def top2_rank(prof, cands):
    return _order(top2_counts(prof, cands)) if len(cands) == 4 else None


def copeland_rank(prof, cands):
    wins = {c: 0 for c in cands}
    for x, y in combinations(cands, 2):
        w = _majority(prof, x, y)
        if w is None:
            return None
        wins[w] += 1
    return _order(wins)


def _elimination_order(prof, cands, pick):
    remaining, out = list(cands), []
    while len(remaining) > 1:
        loser = pick(restrict(prof, remaining), remaining)
        if loser is None:
            return None
        out.append(loser)
        remaining.remove(loser)
    return tuple(remaining + out[::-1])


def irv_order(prof, cands):
    return _elimination_order(prof, cands, lambda p, rem: _unique_min(firsts(p, rem)))


def coombs_order(prof, cands):
    return _elimination_order(prof, cands, lambda p, rem: _unique_max(lasts(p, rem)))


def _finalists_rank(top_scores, order_two, rest_scores):
    two = _top_two(top_scores)
    if two is None:
        return None
    a, b = two
    if order_two[a] == order_two[b]:
        return None
    pair = (a, b) if order_two[a] > order_two[b] else (b, a)
    rest = {c: v for c, v in rest_scores.items() if c not in pair}
    tail = _order(rest) if rest else ()
    return None if tail is None else pair + tail


def runoff_rank(prof, cands):
    f = firsts(prof, cands)
    two = _top_two(f)
    if two is None:
        return None
    w = _majority(prof, *two)
    if w is None:
        return None
    return _finalists_rank(f, {c: int(c == w) for c in cands}, f)


def borda_finalists_rank(prof, cands):
    return _finalists_rank(borda_scores(prof, cands), firsts(prof, cands), borda_scores(prof, cands))


def plural_finalists_rank(prof, cands):
    return _finalists_rank(firsts(prof, cands), borda_scores(prof, cands), firsts(prof, cands))


RANKING_RULES = {
    'borda_rank': borda_rank, 'plurality_rank': plurality_rank, 'antiplurality_rank': antiplurality_rank,
    'top2_rank': top2_rank, 'copeland_rank': copeland_rank, 'irv_order': irv_order, 'coombs_order': coombs_order,
    'runoff_rank': runoff_rank, 'borda_finalists_rank': borda_finalists_rank,
    'plural_finalists_rank': plural_finalists_rank,
}
MONOTONE_RANKINGS = ('borda_rank', 'plurality_rank', 'antiplurality_rank', 'top2_rank', 'copeland_rank')
NON_MONOTONE_RANKINGS = ('irv_order', 'coombs_order', 'runoff_rank', 'borda_finalists_rank', 'plural_finalists_rank')


def violates(before, after, raised) -> bool:
    """True if some option ranked below `raised` before the change is ranked above it after."""
    below = set(before[before.index(raised) + 1:])
    return bool(below & set(after[:after.index(raised)]))


# ---------------------------------------------------------------------------------------------------- relations
def _score_relation(scores, higher_better=True):
    if len(set(scores.values())) < len(scores):
        return None
    sign = 1 if higher_better else -1
    return {(x, y): sign * scores[x] > sign * scores[y] for x, y in combinations(sorted(scores), 2)}


def majority_rel(prof, cands):
    rel = {}
    for x, y in combinations(sorted(cands), 2):
        w = _majority(prof, x, y)
        if w is None:
            return None
        rel[(x, y)] = w == x
    return rel


def adjacency_rel(prof, cands):
    rel = {}
    for x, y in combinations(sorted(cands), 2):
        nx = sum(n for n, r in prof if r.index(x) + 1 == r.index(y))
        ny = sum(n for n, r in prof if r.index(y) + 1 == r.index(x))
        if nx == ny:
            return None
        rel[(x, y)] = nx > ny
    return rel


def restricted_rel(prof, cands):
    rel = {}
    for x, y in combinations(sorted(cands), 2):
        sub = [(n, r) for n, r in prof if r[-1] not in (x, y)]
        w = _majority(sub, x, y) if sub else None
        if w is None:
            return None
        rel[(x, y)] = w == x
    return rel


def restricted_first_rel(prof, cands):
    rel = {}
    for x, y in combinations(sorted(cands), 2):
        sub = [(n, r) for n, r in prof if r[0] not in (x, y)]
        w = _majority(sub, x, y) if sub else None
        if w is None:
            return None
        rel[(x, y)] = w == x
    return rel


def borda_rel(prof, cands):
    return _score_relation(borda_scores(prof, cands))


def plurality_rel(prof, cands):
    return _score_relation(firsts(prof, cands))


def antiplurality_rel(prof, cands):
    return _score_relation(lasts(prof, cands), higher_better=False)


def top2_rel(prof, cands):
    return _score_relation(top2_counts(prof, cands))


RELATION_RULES = {
    'majority': majority_rel, 'adjacency': adjacency_rel, 'restricted': restricted_rel,
    'restricted_first': restricted_first_rel, 'borda_rank': borda_rel,
    'plurality_rank': plurality_rel, 'antiplurality_rank': antiplurality_rel, 'top2_rank': top2_rel,
}
CYCLIC_CAPABLE = ('majority', 'adjacency', 'restricted', 'restricted_first')
ALWAYS_TRANSITIVE = ('borda_rank', 'plurality_rank', 'antiplurality_rank', 'top2_rank')


def above(rel, x, y) -> bool:
    return rel[(x, y)] if (x, y) in rel else not rel[(y, x)]


def is_transitive(rel, cands) -> bool:
    for x in cands:
        for y in cands:
            for z in cands:
                if len({x, y, z}) == 3 and above(rel, x, y) and above(rel, y, z) and not above(rel, x, z):
                    return False
    return True


def ranking_of(rel, cands):
    """The ranking (best first) of a transitive strict relation."""
    return tuple(sorted(cands, key=lambda c: -sum(above(rel, c, d) for d in cands if d != c)))
