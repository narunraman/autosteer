"""sc_transitivity_rules (STEER 4.1.c, Transitivity in Social Welfare Functions): hand-written templates.

Design of 6 October 2026 (replaces the held sc_transitivity). A group tries four unnamed rules ("Rule 1" to "Rule 4")
that turn the members' ranked ballots into a group verdict on every pair of options (which one the group places
above the other). A verdict is a well-defined ranking exactly when it is transitive (no cycle such as B above C,
C above D, D above B). All verdicts here are complete and strict: profiles with a tie in any listed rule are redrawn.

Rules (utilities/social_choice_utils.py), unnamed in the questions:
    pairwise rules that can cycle: majority (pairwise majority); adjacency (more members rank x immediately above y
        than y immediately above x); restricted (only the members who rank neither of the two last count);
        restricted_first (only the members who rank neither of the two first count);
    score rankings (always transitive): Borda points, first places, fewest last places, first-or-second places
        (four options only; with three it orders as fewest last places).
Profiles: 3 or 4 options, 3 to 7 ballot lines of 1 to 12 members; in about 60% of draws the profile is seeded with
a Condorcet cycle over three of the options (the cyclic ballots get extra weight), so cycles are common.

Types:
    consequences (easier, difficulty 1): the rules are not described; each option lists one rule's verdict on every
        pair ("Rule 2: C above B, B above D, D above C"), in a random order of pairs.
    procedures (harder, difficulty 2): the rules are described in words; the options are "Rule 1" to "Rule 4" and
        the reader works out every rule's verdicts. In the intransitive direction at least one other pairwise rule
        is listed (and is transitive on these ballots), and in the transitive direction the key is a pairwise rule
        too, so recognising a pairwise or a score rule does not give the answer.
Direction (second tag): `intransitive` asks which rule's verdict is not a well-defined (transitive) ranking (exactly
one is cyclic); `transitive` says only one is a well-defined ranking and asks which (exactly one is transitive). In
consequences questions the four listed verdicts differ; with three options there are only two cyclic verdicts, so the
transitive direction of the consequences type uses four options only.
"""
from __future__ import annotations

from itertools import combinations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.sc_common import (GROUPS, LETTERS, NUMBER_WORDS, PERSPECTIVES, ballot_lines, borda_points,
                                  comparing_intro, letters_list)
from .utilities.social_choice_utils import CYCLIC_CAPABLE, RELATION_RULES, is_transitive

function_registry = FillRegister()

ELEMENT = 'sc_transitivity_rules'
TYPES = ('consequences', 'procedures')
DIRECTIONS = ('intransitive', 'transitive')
CYCLE_SEED_RATE = 0.6


def rule_text(rule: str, noun: str, m: int) -> str:
    return {
        'majority': f'For each pair of {noun}s, the group places one above the other if more members rank it higher.',
        'adjacency': (f'For each pair of {noun}s, the group places one above the other if more members rank it '
                      f'immediately above the other (with nothing in between) than the other way round.'),
        'restricted': (f'For each pair of {noun}s, only the members who rank neither of the two last are counted, '
                       f'and the group places one above the other if more of those members rank it higher.'),
        'restricted_first': (f'For each pair of {noun}s, only the members who rank neither of the two first are '
                             f'counted, and the group places one above the other if more of those members rank it '
                             f'higher.'),
        'borda_rank': f'Each ballot gives {borda_points(m)}; the {noun}s are ranked by their total points.',
        'plurality_rank': f'The {noun}s are ranked by the number of members who rank them first.',
        'antiplurality_rank': f'The {noun}s are ranked by the number of members who rank them last, fewest first.',
        'top2_rank': f'The {noun}s are ranked by the number of members who rank them first or second.',
    }[rule]


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for g, group in enumerate(GROUPS):
        noun = group[3]
        for m in (3, 4):
            for k, persp in enumerate(PERSPECTIVES):
                intro = comparing_intro(persp, g + k, group,
                                        "comparing four rules that turn the members' ballots into a group ranking "
                                        "for choosing",
                                        "compare four rules that turn the members' ballots into a group ranking for "
                                        "choosing")
                setup = (f"{intro} Each member ranks all {NUMBER_WORDS[m]} {noun}s ({{alternatives}}). The ballots "
                         f"are:\n{{ballots}}\n")
                for qtype in TYPES:
                    body = (f"For every pair of {noun}s, each rule decides which one the group places above the "
                            f"other. The rules' verdicts are listed in the options. " if qtype == 'consequences' else
                            f"For every pair of {noun}s, each rule decides which one the group places above the "
                            f"other. The four rules are:\n{{rules}}\n")
                    for direction in DIRECTIONS:
                        if qtype == 'consequences' and direction == 'transitive' and m == 3:
                            continue  # three options have only two cyclic verdicts: the odd one out would be the key
                        if direction == 'intransitive':
                            ask = (f"Which rule's verdicts on these ballots do not form a well-defined (transitive) "
                                   f"ranking of the {noun}s?")
                        else:
                            ask = (f"On these ballots, the verdicts of only one rule form a well-defined (transitive) "
                                   f"ranking of the {noun}s. Which rule is it?")
                        out.append(ct.CodeTemplate(
                            parts=(setup + body + ask,), type=qtype, domain=group[5], tags=f'{persp};{direction}',
                            difficulty_level=1 if qtype == 'consequences' else 2,
                            spec={'group': g, 'm': m, 'type': qtype, 'direction': direction}))
    return out


def _draw_profile(r, cands):
    m = len(cands)
    rankings, prof = [], []
    if r.random() < CYCLE_SEED_RATE:
        x, y, z = (cands[int(i)] for i in r.choice(m, size=3, replace=False))
        rest = [c for c in cands if c not in (x, y, z)]
        for cyc in ((x, y, z), (y, z, x), (z, x, y)):
            rk = list(cyc)
            for c in rest:
                rk.insert(int(r.integers(0, len(rk) + 1)), c)
            rk = tuple(rk)
            if rk not in rankings:
                rankings.append(rk)
                prof.append((int(r.integers(5, 13)), rk))
    k = min(len(rankings) + int(r.integers(1 if rankings else 3, 5)), 6 if m == 3 else 24)
    while len(rankings) < k:
        rk = tuple(cands[int(i)] for i in r.permutation(m))
        if rk not in rankings:
            rankings.append(rk)
            prof.append((int(r.integers(1, 9)), rk))
    order = [int(i) for i in r.permutation(len(prof))]
    return [prof[i] for i in order]


def _pick_rules(r, m, direction, qtype):
    pool = [q for q in RELATION_RULES if not (m == 3 and q == 'top2_rank')]
    if direction == 'intransitive':
        key = CYCLIC_CAPABLE[int(r.integers(len(CYCLIC_CAPABLE)))]
        rest = [q for q in pool if q != key]
        others = [rest[int(i)] for i in r.choice(len(rest), size=3, replace=False)]
        if qtype == 'procedures' and not any(q in CYCLIC_CAPABLE for q in others):
            return None
    else:
        keys = CYCLIC_CAPABLE if qtype == 'procedures' else pool  # described score rules would give it away
        key = keys[int(r.integers(len(keys)))]
        rest = [q for q in CYCLIC_CAPABLE if q != key]
        others = [rest[int(i)] for i in r.choice(len(rest), size=3, replace=False)]
    return key, others


def find_example(r, cands, direction, qtype, tries: int = 20000):
    for _ in range(tries):
        picked = _pick_rules(r, len(cands), direction, qtype)
        if picked is None:
            continue
        key, others = picked
        prof = _draw_profile(r, cands)
        rels = {q: RELATION_RULES[q](prof, cands) for q in [key] + others}
        if any(v is None for v in rels.values()):
            continue
        trans = {q: is_transitive(v, cands) for q, v in rels.items()}
        want_key = direction == 'transitive'
        if qtype == 'consequences' and (direction == 'intransitive' or len(cands) == 4) and \
                len({tuple(sorted(v.items())) for v in rels.values()}) < 4:
            continue  # the listed verdicts differ (three options have only two cyclic verdicts)
        if trans[key] == want_key and all(trans[q] != want_key for q in others):
            return prof, key, others, rels
    raise RuntimeError(f'{ELEMENT}: no example for {direction} {qtype} with {len(cands)} options')


def verdict_text(r, rel, cands) -> str:
    pairs = list(combinations(sorted(cands), 2))
    order = [pairs[int(i)] for i in r.permutation(len(pairs))]
    return ', '.join(f'{x} above {y}' if rel[(x, y)] else f'{y} above {x}' for x, y in order)


@function_registry.register(ELEMENT)
def fill_sc_transitivity_rules(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    group = GROUPS[spec['group']]
    noun, m = group[3], spec['m']
    r = filler_utils.rng
    cands = [LETTERS[i] for i in sorted(r.choice(len(LETTERS), size=m, replace=False))]
    prof, key, others, rels = find_example(r, cands, spec['direction'], spec['type'])
    rules = [key] + others
    order = [rules[int(i)] for i in r.permutation(4)]
    out = {'alternatives': letters_list(cands), 'ballots': ballot_lines(prof)}
    if spec['type'] == 'consequences':
        options = [f'Rule {k + 1}: {verdict_text(r, rels[q], cands)}' for k, q in enumerate(order)]
    else:
        out['rules'] = '\n'.join(f'Rule {k + 1}: {rule_text(q, noun, m)}' for k, q in enumerate(order))
        options = [f'Rule {k + 1}' for k in range(4)]
    out['options_list'] = [options]
    out['correct_answer'] = [[int(q == key) for q in order]]
    return out


ct.register(ELEMENT, _templates())
