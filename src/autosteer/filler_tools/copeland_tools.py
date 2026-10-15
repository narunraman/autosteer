"""copeland_method (STEER 4.2.c, Copeland's Method): hand-written templates.

Decision of 5 October 2026. The paper's definition: each candidate is compared with every other candidate in a
one-on-one (majority) contest; it receives one point for each victory and half a point for each tie, and the
candidate with the highest total is the winner. `condorcet_criterion` stays its own element (it asks for the
Condorcet winner, not the Copeland winner).

A group ranks three, four or five alternatives (full rankings only). Two question types:
  profile: the ballots are given as lines "12 members voted for X > Y > Z";
  table: only the one-on-one results are given ("Book B vs Book C: 41 to 37").
Tags: `defined` (the question states the rule) or `named` (it only names Copeland's method).
Difficulty (STEER scale, as condorcet_criterion): 2 for three alternatives, 3 for four or five.

The profile is drawn (3 to 6 distinct ballot lines, 1 to 30 members each; ties in a one-on-one contest are possible
when the number of members is even) until the Copeland winner is unique. Outcomes are controlled so that counting
something simpler does not solve the element: in about 60% of questions the Copeland winner is not a Condorcet
winner (it loses or ties some contest; with three alternatives this needs a tie), and where possible it is neither
the plurality (most first places) leader nor the Borda winner. Options: one per alternative, in the order the
question lists them. Pairwise counts reuse `voting_tools.pairwise`.
"""
from __future__ import annotations

from itertools import combinations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .voting_tools import pairwise

function_registry = FillRegister()

ELEMENT = 'copeland_method'
LETTERS = 'BCDEFGHJKLMNPQRSTUVWXYZ'
NUMBER_WORDS = {3: 'three', 4: 'four', 5: 'five'}
NOT_CONDORCET_RATE = 0.6
# intro subject, short name, thing chosen, option noun, label, domain
GROUPS = [
    ('A book club', 'club', 'its next book', 'book', 'Book', 'leisure'),
    ('A city council', 'council', 'a design for the new park', 'design', 'Design', 'politics'),
    ('A band', 'band', 'the opening song for its tour', 'song', 'Song', 'entertainment'),
    ('A family', 'family', 'a holiday destination', 'destination', 'Destination', 'family'),
    ('An office team', 'team', 'a venue for its end-of-year lunch', 'venue', 'Venue', 'workplace'),
    ('A school board', 'board', 'a new logo for the school', 'logo', 'Logo', 'education'),
]


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, (intro, short, thing, noun, Noun, domain) in enumerate(GROUPS):
        for m in (3, 4, 5):
            head = f"{intro} is choosing {thing} from {NUMBER_WORDS[m]} {noun}s: {{alternatives}}. "
            bodies = {
                'profile': f"Each member ranked all the {noun}s, with these results:\n{{ballots}}\n",
                'table': (f"Each member ranked all the {noun}s. In the one-on-one comparisons, the number of members "
                          f"preferring each {noun} was:\n{{table}}\n"),
            }
            asks = {
                'defined': (f"The {short} uses Copeland's method: each {noun} is compared with every other {noun} in a "
                            f"one-on-one contest decided by majority, and receives one point for each contest it wins "
                            f"and half a point for each tie. The {noun} with the most points wins. Which {noun} does "
                            f"the {short} choose?"),
                'named': f"The {short} decides by Copeland's method. Which {noun} does the {short} choose?",
            }
            for qtype, body in bodies.items():
                for tag, ask in asks.items():
                    out.append(ct.CodeTemplate(parts=(head + body + ask,), type=qtype, domain=domain, tags=tag,
                                               difficulty_level=2 if m == 3 else 3,
                                               spec={'group': k, 'm': m, 'type': qtype}))
    return out


def copeland_scores(P, cands) -> dict[str, float]:
    s = {c: 0.0 for c in cands}
    for x, y in combinations(cands, 2):
        if P[x][y] > P[y][x]:
            s[x] += 1
        elif P[x][y] < P[y][x]:
            s[y] += 1
        else:
            s[x] += 0.5
            s[y] += 0.5
    return s


def unique_top(scores: dict):
    top = max(scores.values())
    w = [c for c, v in scores.items() if v == top]
    return w[0] if len(w) == 1 else None


def first_places(profile, cands):
    f = {c: 0 for c in cands}
    for n, r in profile:
        f[r[0]] += n
    return f


def borda(profile, cands):
    m = len(cands)
    s = {c: 0 for c in cands}
    for n, r in profile:
        for i, c in enumerate(r):
            s[c] += n * (m - 1 - i)
    return s


def is_condorcet(P, cands, x) -> bool:
    return all(P[x][y] > P[y][x] for y in cands if y != x)


def draw_profile(r, cands, want_condorcet: bool, tries: int = 4000):
    fallback = None
    m = len(cands)
    for _ in range(tries):
        k = int(r.integers(3, 7))
        rankings = []
        while len(rankings) < k:
            rk = tuple(cands[int(i)] for i in r.permutation(m))
            if rk not in rankings:
                rankings.append(rk)
        prof = [(int(r.integers(1, 31)), rk) for rk in rankings]
        P = pairwise(prof, cands)
        w = unique_top(copeland_scores(P, cands))
        if w is None or is_condorcet(P, cands, w) != want_condorcet:
            continue
        simple = {unique_top(first_places(prof, cands)), unique_top(borda(prof, cands))}
        if w in simple:
            fallback = fallback or (prof, w)
            continue
        return prof, w
    if fallback is not None:
        return fallback
    raise RuntimeError(f'copeland: no profile for {m} alternatives (condorcet={want_condorcet})')


def _alternatives(Noun, letters):
    names = [f'{Noun} {x}' for x in letters]
    return ', '.join(names[:-1]) + ' and ' + names[-1]


@function_registry.register(ELEMENT)
def fill_copeland(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    intro, short, thing, noun, Noun, domain = GROUPS[spec['group']]
    r = filler_utils.rng
    m = spec['m']
    cands = [LETTERS[i] for i in sorted(r.choice(len(LETTERS), size=m, replace=False))]
    want_condorcet = bool(r.random() >= NOT_CONDORCET_RATE)
    prof, w = draw_profile(r, cands, want_condorcet)
    P = pairwise(prof, cands)
    ballots = '\n'.join(f"{n} member{'s' if n != 1 else ''} voted for {' > '.join(rk)}" for n, rk in prof)
    pairs = [(cands[i], cands[j]) for i, j in combinations(range(m), 2)]
    order = [pairs[int(i)] for i in r.permutation(len(pairs))]
    table = '\n'.join(f'{Noun} {x} vs {Noun} {y}: {P[x][y]} to {P[y][x]}' for x, y in order)
    options = [f'{Noun} {c}' for c in cands]
    return {'alternatives': _alternatives(Noun, cands), 'ballots': ballots, 'table': table,
            'options_list': [options], 'correct_answer': [[int(c == w) for c in cands]]}


ct.register(ELEMENT, _templates())
