"""Filler for the STEER element condorcet_criterion (rebuilt 2026-09; the original generator is lost and 19% of the
released answers were wrong).

Templates reuse the released scenario wordings (audit/steer/regen/recover_condorcet_templates.py); per-template
option label and list style are in data/templates/condorcet_criterion/template_labels.csv, looked up by the
template text the sampler passes in metadata['question_text'].

A question is a random profile of full rankings (no truncated ballots, so no convention for unranked candidates is
needed): `type` = number of distinct ballot lines (3/4/5), difficulty 2 = 3 candidates, difficulty 3 = 4 or 5
candidates (the released grades). Options: one per candidate plus "There is no Condorcet winner". The answer is the
candidate who beats every other candidate in a strict pairwise majority, else the no-winner option. Profiles with a
pairwise tie are redrawn. The outcome is controlled: P(no Condorcet winner) = NO_WINNER_RATE, and when there is a
winner it is, where possible, not the plurality (first-choice) leader in NOT_PLURALITY_RATE of the questions, so
counting first choices does not solve the element.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache
from itertools import combinations
from string import ascii_uppercase

from .fill_registry import FillRegister
from .utilities.filler_utils import rng

function_registry = FillRegister()

from autosteer.paths import templates_dir  # the data directory, resolved when the labels are read
NO_WINNER = 'There is no Condorcet winner'
NO_WINNER_RATE = 0.25
NOT_PLURALITY_RATE = 0.5
LETTERS = [c for c in ascii_uppercase if c not in 'AI']  # 'A'/'I' read as words in running text
NUMBER_WORDS = {3: 'three', 4: 'four', 5: 'five'}
VOTERS = (1, 60)  # voters per ballot line


@lru_cache(maxsize=None)
def labels() -> dict[str, dict]:
    path = os.path.join(templates_dir(), 'condorcet_criterion', 'template_labels.csv')
    with open(path, newline='') as f:
        return {r['template']: r for r in csv.DictReader(f)}


def pairwise(profile: list[tuple[int, tuple[str, ...]]], cands: list[str]) -> dict[str, dict[str, int]]:
    """P[x][y] = number of voters ranking x above y (full rankings)."""
    P = {x: {y: 0 for y in cands} for x in cands}
    for n, ranking in profile:
        pos = {c: i for i, c in enumerate(ranking)}
        for x, y in combinations(cands, 2):
            if pos[x] < pos[y]:
                P[x][y] += n
            else:
                P[y][x] += n
    return P


def condorcet_winner(P, cands):
    w = [x for x in cands if all(P[x][y] > P[y][x] for y in cands if y != x)]
    return w[0] if w else None


def has_pairwise_tie(P, cands) -> bool:
    return any(P[x][y] == P[y][x] for x, y in combinations(cands, 2))


def plurality_leaders(profile, cands) -> set[str]:
    first = {c: 0 for c in cands}
    for n, r in profile:
        first[r[0]] += n
    top = max(first.values())
    return {c for c in cands if first[c] == top}


def random_profile(cands: list[str], k: int) -> list[tuple[int, tuple[str, ...]]]:
    rankings: list[tuple[str, ...]] = []  # a list, not a set: set order depends on PYTHONHASHSEED
    while len(rankings) < k:
        r = tuple(cands[i] for i in rng.permutation(len(cands)))
        if r not in rankings:
            rankings.append(r)
    return [(int(rng.integers(VOTERS[0], VOTERS[1] + 1)), r) for r in rankings]


def draw_profile(cands: list[str], k: int, want_winner: bool, want_not_plurality: bool, tries: int = 5000):
    fallback = None
    for _ in range(tries):
        prof = random_profile(cands, k)
        P = pairwise(prof, cands)
        if has_pairwise_tie(P, cands):
            continue
        w = condorcet_winner(P, cands)
        if (w is not None) != want_winner:
            continue
        if want_winner and want_not_plurality and w in plurality_leaders(prof, cands):
            fallback = fallback or (prof, w)
            continue
        return prof, w
    if fallback is not None:
        return fallback
    raise RuntimeError(f'condorcet: no profile with {len(cands)} candidates, {k} ballots, winner={want_winner}')


def candidate_list(cands: list[str], lab: dict) -> str:
    items = [f"{lab['label']} {c}" if lab['labelled'] == 'True' else c for c in cands]
    conj = lab.get('conj') or ''
    if not conj:
        return ', '.join(items)
    sep = ', ' if lab['oxford'] == 'True' else ' '
    return ', '.join(items[:-1]) + f'{sep}{conj} ' + items[-1]


def ballot_lines(profile) -> str:
    return '\n'.join(f"- {n} voter{'s' if n != 1 else ''} voted for {' > '.join(r)}" for n, r in profile)


@function_registry.register('condorcet_criterion')
def fill_condorcet(metadata: dict) -> dict:
    text = metadata.get('question_text')
    lab = labels().get(text) if text else None
    if lab is None:
        if text:
            raise KeyError(f'condorcet_criterion: unknown template {text[:120]!r}')
        lab = {'label': 'Candidate', 'labelled': 'False', 'conj': 'and', 'oxford': 'True'}
    level = int(metadata.get('difficulty_level', 2))
    m = 3 if level <= 2 else int(rng.choice([4, 5]))
    k = int(metadata.get('type', 3))
    cands = sorted(str(c) for c in rng.choice(LETTERS, size=m, replace=False))
    want_winner = rng.random() >= NO_WINNER_RATE
    prof, w = draw_profile(cands, k, want_winner, rng.random() < NOT_PLURALITY_RATE)
    order = [cands[i] for i in rng.permutation(m)]  # order in which the scenario lists the candidates
    options = [f"{lab['label']} {c}" for c in order] + [NO_WINNER]
    correct = [int(c == w) for c in order] + [int(w is None)]
    return {
        'candidates': candidate_list(order, lab),
        'n_word': NUMBER_WORDS[m],
        'N_word': NUMBER_WORDS[m].capitalize(),
        'ballots': ballot_lines(prof),
        'options_list': [options],
        'correct_answer': [correct],
    }


# ------------------------------------------------------------------ plurality_voting / borda_count (recovered 2026-10)
# v1 social_choice.ipynb (generate_random_votes, compute_plurality, borda_count): a profile of ballots that rank 3 of
# the 4 or 5 candidates; type = number of distinct ballot lines (4 / 5); plurality difficulty 1 / 2 and Borda
# difficulty 5 / 6 = 4 / 5 candidates, as released. Templates (recover_voting_templates.py) are the released
# sentences with {n}, {candidates} and {ballots}.
# Borda scoring with truncated ballots is convention-dependent: the profile is redrawn until the asked place (winner,
# or runner-up for tag 'second') is the same unique candidate under all three conventions in use: m-1-i for the
# i-th ranked candidate with unranked candidates scoring 0 (the paper's), the same with the unranked candidates
# sharing the leftover points equally, and ballot-length scoring (2, 1, 0). Plurality: unique most first places.
TRUNCATED = 3
BALLOT_VOTERS = (1, 120)


def truncated_profile(cands: list[str], k: int) -> list[tuple[int, tuple[str, ...]]]:
    rankings: list[tuple[str, ...]] = []
    while len(rankings) < k:
        r = tuple(cands[int(i)] for i in rng.permutation(len(cands))[:TRUNCATED])
        if r not in rankings:
            rankings.append(r)
    return [(int(rng.integers(BALLOT_VOTERS[0], BALLOT_VOTERS[1] + 1)), r) for r in rankings]


def borda_scores(profile, cands, conv: str) -> dict:
    m = len(cands)
    s = {c: 0.0 for c in cands}
    for n, r in profile:
        for i, c in enumerate(r):
            s[c] += n * ((len(r) - 1 - i) if conv == 'ballot' else (m - 1 - i))
        un = [c for c in cands if c not in r]
        if conv == 'avg' and un:
            for c in un:
                s[c] += n * (len(un) - 1) / 2
    return s


def ranked_place(scores: dict, place: int):
    """candidate in `place` (0 winner, 1 runner-up) if no other candidate has the same score, else None."""
    order = sorted(scores, key=lambda c: -scores[c])
    target = order[place]
    return target if sum(v == scores[target] for v in scores.values()) == 1 else None


def first_place_counts(profile, cands) -> dict:
    cnt = {c: 0 for c in cands}
    for n, r in profile:
        cnt[r[0]] += n
    return cnt


def borda_convention(n_cands: int) -> str:
    """The scoring of the truncated ballots, stated in every borda_count question (review, October 2026: it was
    left to the reader; the key is the same under the three usual conventions, but not under every completion)."""
    pts = [n_cands - 1 - i for i in range(TRUNCATED)]
    return (f"Scoring: each ballot ranks three of the {n_cands} candidates and gives them {pts[0]}, {pts[1]} and "
            f"{pts[2]} points, in the order ranked; a candidate left off a ballot gets 0 points from it.")


def _election(metadata, n_cands, solve, note: str = ''):
    k = int(metadata['type'])
    while True:
        cands = [str(c) for c in rng.choice(LETTERS, size=n_cands, replace=False)]
        prof = truncated_profile(cands, k)
        w = solve(prof, cands)
        if w is not None:
            break
    return {
        'n': n_cands,
        'candidates': ', '.join(cands),
        'ballots': '\n'.join(f"- {n} voter{'s' if n != 1 else ''} voted for {' > '.join(r)}" for n, r in prof)
                   + (f'\n{note}' if note else ''),
        'options_list': [[f'Candidate {c}' for c in cands]],
        'correct_answer': [[int(c == w) for c in cands]],
    }


@function_registry.register('plurality_voting')
def fill_plurality_voting(metadata: dict) -> dict:
    def solve(prof, cands):
        cnt = first_place_counts(prof, cands)
        top = max(cnt.values())
        win = [c for c in cands if cnt[c] == top]
        return win[0] if len(win) == 1 else None
    return _election(metadata, 4 if int(metadata['difficulty_level']) <= 1 else 5, solve)


@function_registry.register('borda_count')
def fill_borda_count(metadata: dict) -> dict:
    place = 1 if 'second' in str(metadata.get('tags', '')).split(';') else 0

    def solve(prof, cands):
        got = {ranked_place(borda_scores(prof, cands, conv), place) for conv in ('paper', 'avg', 'ballot')}
        return got.pop() if len(got) == 1 else None
    n_cands = 4 if int(metadata['difficulty_level']) <= 5 else 5
    return _election(metadata, n_cands, solve, borda_convention(n_cands))
