"""sc_monotonicity_rules (STEER 4.1.b, Monotonicity in Social Welfare Functions): hand-written templates.

Design of October 2026 (replaces the winner-based design of 6 October, which tested whether a winner
stays the winner rather than the paper's axiom on social rankings). A group tries four unnamed rules ("Rule 1" to
"Rule 4") that turn the members' ranked ballots into a group ranking of all options, first on one set of ballots P
and then on P', which is P with one option (the raised option) moved up on some ballots, nothing else changed. A
rule violates monotonicity on the example when the group ranked the raised option above some other option under P
and ranks that option above the raised one under P' (the paper's definition: society prefers A to B, A is raised in
some individual rankings, and A must still be preferred to B).

Rules (utilities/social_choice_utils.py, RANKING_RULES): real ones (Borda ranking, plurality ranking, Copeland
ranking, instant-runoff elimination order, Coombs elimination order) and made-up but well-defined ones (ranking by
fewest last places; by first-or-second places; the two with most first places on top in head-to-head order; the two
highest Borda scores on top ordered by first places; the two with most first places on top ordered by Borda score).
The first five rankings by score or by pairwise wins are monotonic and never violate; the others can. (Kemeny is
not used: as a procedure it is too long to apply by hand, and in the consequences type a rule's identity is not
shown.)

Types:
    consequences (easier, difficulty 1): the rules are not described; each option shows one rule's group ranking
        before and after the change ("Rule 2: C > B > D before the change, B > C > D after it"). The key is the
        rule under which some option overtakes the raised option although the raised option was above it before.
        The four (before, after) pairs differ, at least one other rule's ranking changes and in at least one other
        rule the raised option moves up (options_ok).
    procedures (harder, difficulty 2): the rules are described in words; the options are "Rule 1" to "Rule 4" and
        the reader applies each rule to both sets of ballots. At least one of the other three rules is also of a
        kind that can violate monotonicity (it does not on this example), so knowing which rules are monotonic in
        general does not give the key.
In both types the raised option keeps or improves its place in the key rule's ranking (it overtakes one option and is
overtaken by another), so the violation is seen only by comparing pairs, not by the raised option's position.
Tags: `defined` (the question states what monotonicity requires) or `named`.
Generator: the violating rule is chosen first (uniformly among the non-monotonic ones), three others at random (with
three options never both instant runoff and the head-to-head finalists rule, which then give the same ranking);
profiles (3 or 4 options, 3 to 6 ballot lines of 1 to 12 members) and raises (any option, on one ballot line, all or
some of its members, moved up one or more places) are searched until exactly one rule violates and every rule gives a
ranking without ties on both ballot sets.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.sc_common import (GROUPS, LETTERS, NUMBER_WORDS, PERSPECTIVES, ballot_lines, borda_points,
                                  comparing_intro, letters_list)
from .utilities.social_choice_utils import NON_MONOTONE_RANKINGS, RANKING_RULES, violates

function_registry = FillRegister()

ELEMENT = 'sc_monotonicity_rules'
TYPES = ('consequences', 'procedures')
DEFINITION = ("A rule that ranks the {noun}s is monotonic if, whenever the group ranks one {noun} above another and "
              "some members move the first one up on their ballots, with nothing else changed, the group still ranks "
              "it above the other. ")


def rule_text(rule: str, noun: str, m: int) -> str:
    pts = borda_points(m)
    return {
        'borda_rank': f'Each ballot gives {pts}; the {noun}s are ranked by their total points, most first.',
        'plurality_rank': f'The {noun}s are ranked by the number of members who rank them first, most first.',
        'antiplurality_rank': f'The {noun}s are ranked by the number of members who rank them last, fewest first.',
        'top2_rank': f'The {noun}s are ranked by the number of members who rank them first or second, most first.',
        'copeland_rank': (f'For each pair of {noun}s, the one ranked higher by more members wins the pair; the '
                          f'{noun}s are ranked by the number of pairs they win, most first.'),
        'irv_order': (f'The {noun} ranked first by the fewest members is eliminated, each ballot then counts for its '
                      f'highest-ranked remaining {noun}, and this is repeated until one {noun} is left. The {noun}s '
                      f'are ranked in the reverse order of elimination: the last one left first, the first one '
                      f'eliminated last.'),
        'coombs_order': (f'The {noun} ranked last by the most members is eliminated, the ballots are read again '
                         f'without it, and this is repeated until one {noun} is left. The {noun}s are ranked in the '
                         f'reverse order of elimination: the last one left first, the first one eliminated last.'),
        'runoff_rank': (f'The two {noun}s ranked first by the most members are placed first and second, the one '
                        f'ranked higher on more ballots first; the other {noun}s follow, ordered by the number of '
                        f'members who rank them first.'),
        'borda_finalists_rank': (f'Each ballot gives {pts}. The two {noun}s with the most points are placed first and '
                                 f'second, the one ranked first by more members first; the other {noun}s follow in '
                                 f'order of points.'),
        'plural_finalists_rank': (f'The two {noun}s ranked first by the most members are placed first and second, the '
                                  f'one with more points first, where each ballot gives {pts}; the other {noun}s '
                                  f'follow, ordered by the number of members who rank them first.'),
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
                setup = (f"{intro} Each member ranks all {NUMBER_WORDS[m]} {noun}s ({{alternatives}}). In a trial "
                         f"vote the ballots were:\n{{ballots}}\nThen {{change}}. Nothing else changed.\n")
                for qtype in TYPES:
                    for tag in ('defined', 'named'):
                        if qtype == 'consequences':
                            body = (f"Each of the four rules ranked the {noun}s from the first ballots and from "
                                    f"the changed ballots, as listed in the options (best first). ")
                        else:
                            body = "The four rules are:\n{rules}\n"
                        ask = (DEFINITION.format(noun=noun) if tag == 'defined' else '') + \
                            'Which rule violates monotonicity in this example?'
                        out.append(ct.CodeTemplate(
                            parts=(setup + body + ask,), type=qtype, domain=group[5],
                            tags=f'{persp};{tag}',
                            difficulty_level=1 if qtype == 'consequences' else 2,
                            spec={'group': g, 'm': m, 'type': qtype}))
    return out


def _draw_profile(r, cands):
    m = len(cands)
    k = int(r.integers(3, 7))
    rankings = []
    while len(rankings) < k:
        rk = tuple(cands[int(i)] for i in r.permutation(m))
        if rk not in rankings:
            rankings.append(rk)
    return [(int(r.integers(1, 13)), rk) for rk in rankings]


def _raises(prof, cands):
    """Every raise: (line index, members moved, new ranking, raised option)."""
    out = []
    for i, (n, rk) in enumerate(prof):
        for a in cands:
            pos = rk.index(a)
            for new in range(pos):
                rest = [c for c in rk if c != a]
                new_rk = tuple(rest[:new] + [a] + rest[new:])
                for moved in sorted({n, max(1, n // 2)}):
                    out.append((i, moved, new_rk, a))
    return out


def apply_raise(prof, raise_):
    i, moved, new_rk = raise_[:3]
    n, rk = prof[i]
    out = [p for j, p in enumerate(prof) if j != i]
    if moved < n:
        out.insert(i, (n - moved, rk))
        out.insert(i + 1, (moved, new_rk))
    else:
        out.insert(i, (moved, new_rk))
    return out


def options_ok(rules, key_rule, a, before, after) -> bool:
    """consequences options: the four (before, after) pairs are distinct, at least one other rule's ranking changes,
    and in at least one other rule the raised option moves up (so neither a change nor a move of the raised option
    singles out the key)."""
    pairs = [(before[q], after[q]) for q in rules]
    others = [q for q in rules if q != key_rule]
    return (len(set(pairs)) == len(pairs) and any(before[q] != after[q] for q in others)
            and any(after[q].index(a) < before[q].index(a) for q in others))


def find_example(r, cands, key_rule, others, tries: int = 3000, distinct: bool = False):
    rules = [key_rule] + list(others)
    for _ in range(tries):
        prof = _draw_profile(r, cands)
        before = {q: RANKING_RULES[q](prof, cands) for q in rules}
        if any(w is None for w in before.values()):
            continue
        raises = _raises(prof, cands)
        order = [int(i) for i in r.permutation(len(raises))]
        for j in order:
            a = raises[j][3]
            new = apply_raise(prof, raises[j])
            after = {q: RANKING_RULES[q](new, cands) for q in rules}
            if any(w is None for w in after.values()):
                continue
            if not violates(before[key_rule], after[key_rule], a) or \
                    after[key_rule].index(a) > before[key_rule].index(a):
                continue
            ok = not any(violates(before[q], after[q], a) for q in others)
            if ok and distinct:
                ok = options_ok(rules, key_rule, a, before, after)
            if ok:
                return prof, raises[j], a, before, after
    return None


def _change_text(prof, raise_, a):
    i, moved, new_rk = raise_[:3]
    n, rk = prof[i]
    if n == 1:
        who, became = f"the member who voted {' > '.join(rk)}", 'the ballot became'
    elif moved == n:
        who, became = f"the {n} members who voted {' > '.join(rk)}", 'their ballots became'
    else:
        who = f"{moved} of the {n} members who voted {' > '.join(rk)}"
        became = 'that ballot became' if moved == 1 else 'their ballots became'
    return f"{who} moved {a} up, so {became} {' > '.join(new_rk)}"


@function_registry.register(ELEMENT)
def fill_sc_monotonicity_rules(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    group = GROUPS[spec['group']]
    noun, m = group[3], spec['m']
    r = filler_utils.rng
    while True:
        cands = [LETTERS[i] for i in sorted(r.choice(len(LETTERS), size=m, replace=False))]
        key_rule = NON_MONOTONE_RANKINGS[int(r.integers(len(NON_MONOTONE_RANKINGS)))]
        pool = [q for q in RANKING_RULES if q != key_rule and not (m == 3 and q == 'top2_rank')]
        others = []
        for i in r.permutation(len(pool)):
            q = pool[int(i)]
            if m == 3 and {'irv_order', 'runoff_rank'} <= {q, key_rule, *others}:
                continue
            others.append(q)
            if len(others) == 3:
                break
        if spec['type'] == 'procedures' and not set(others) & set(NON_MONOTONE_RANKINGS):
            continue
        found = find_example(r, cands, key_rule, others, distinct=spec['type'] == 'consequences')
        if found is not None:
            break
    prof, raise_, a, before, after = found
    rules = [key_rule] + others
    order = [rules[int(i)] for i in r.permutation(4)]
    out = {'alternatives': letters_list(cands), 'ballots': ballot_lines(prof), 'change': _change_text(prof, raise_, a)}
    if spec['type'] == 'consequences':
        options = [f"Rule {k + 1}: {' > '.join(before[q])} before the change, {' > '.join(after[q])} after it"
                   for k, q in enumerate(order)]
    else:
        out['rules'] = '\n'.join(f'Rule {k + 1}: {rule_text(q, noun, m)}' for k, q in enumerate(order))
        options = [f'Rule {k + 1}' for k in range(4)]
    out['options_list'] = [options]
    out['correct_answer'] = [[int(q == key_rule) for q in order]]
    return out


ct.register(ELEMENT, _templates())
