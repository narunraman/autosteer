"""sc_transitivity_rules: cycle detection on verdicts recomputed from the ballots in the question text.

Independent of the generator (its rules are in filler_tools/utilities/social_choice_utils.py; these are written
again from the rule descriptions). Ballots are parsed from "12 members voted B > C > D" lines.
  procedures    each "Rule k: ..." description is mapped to its rule by its wording and the rule's verdict on every
                pair is recomputed (no ties); a verdict is a well-defined ranking iff it has no cycle (checked by
                searching for x above y above z above x). Intransitive direction: exactly one verdict is cyclic and
                it is the keyed rule; transitive direction: exactly one is acyclic and it is the keyed rule.
  consequences  each option's listed verdicts are parsed; every pair appears once, the verdict equals one of the
                candidate rules' verdicts on these ballots, and the cycle test picks out exactly the keyed option.
Also both directions, both types, 3 and 4 options, and seeded reproducibility.
"""
import collections
import re
from itertools import combinations, permutations

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'sc_transitivity_rules'
BALLOT = re.compile(r'^(\d+) members? voted ([A-Z](?: > [A-Z])+)$', re.M)
RULE = re.compile(r'^Rule (\d): (.+)$', re.M)
PAIR = re.compile(r'([A-Z]) above ([A-Z])')


def pairwise(prof, alts, keep):
    """x above y iff more of the kept ballots rank x above y; None on a tie."""
    out = set()
    for x, y in combinations(alts, 2):
        kept = [(n, r) for n, r in prof if keep(r, x, y)]
        a = sum(n for n, r in kept if r.index(x) < r.index(y))
        b = sum(n for n, r in kept if r.index(y) < r.index(x))
        if a == b:
            return None
        out.add((x, y) if a > b else (y, x))
    return frozenset(out)


def by_score(score, alts):
    if len(set(score.values())) < len(alts):
        return None
    return frozenset((x, y) for x in alts for y in alts if score[x] > score[y])


def verdicts(prof, alts):
    m = len(alts)
    first = {a: sum(n for n, r in prof if r[0] == a) for a in alts}
    last = {a: sum(n for n, r in prof if r[-1] == a) for a in alts}
    top2 = {a: sum(n for n, r in prof if a in r[:2]) for a in alts}
    pts = {a: sum(n * (m - 1 - r.index(a)) for n, r in prof) for a in alts}
    adj = set()
    for x, y in combinations(alts, 2):
        a = sum(n for n, r in prof if r.index(x) + 1 == r.index(y))
        b = sum(n for n, r in prof if r.index(y) + 1 == r.index(x))
        adj = None if adj is None or a == b else adj | {(x, y) if a > b else (y, x)}
    return {
        'majority': pairwise(prof, alts, lambda r, x, y: True),
        'not_last': pairwise(prof, alts, lambda r, x, y: r[-1] not in (x, y)),
        'not_first': pairwise(prof, alts, lambda r, x, y: r[0] not in (x, y)),
        'adjacent': None if adj is None else frozenset(adj),
        'points': by_score(pts, alts),
        'firsts': by_score(first, alts),
        'fewest_lasts': by_score({a: -v for a, v in last.items()}, alts),
        'top_two': by_score(top2, alts),
    }


def cyclic(rel, alts):
    return any((x, y) in rel and (y, z) in rel and (z, x) in rel for x, y, z in permutations(alts, 3))


def which_rule(desc):
    if 'neither of the two last' in desc:
        return 'not_last'
    if 'neither of the two first' in desc:
        return 'not_first'
    if 'immediately above' in desc:
        return 'adjacent'
    if 'more members rank it higher' in desc:
        return 'majority'
    if 'points' in desc:
        return 'points'
    if 'first or second' in desc:
        return 'top_two'
    if 'rank them last, fewest first' in desc:
        return 'fewest_lasts'
    assert 'rank them first' in desc, desc
    return 'firsts'


def test_keys_by_cycle_detection():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        q = s.sample_question(E, 72000 + seed)
        (row,) = build_rows(q, 'test')
        text = row['question_text']
        direction = 'transitive' if 'only one rule form' in text else 'intransitive'
        prof = [(int(n), tuple(r.split(' > '))) for n, r in BALLOT.findall(text)]
        alts = sorted(prof[0][1])
        computed = verdicts(prof, alts)
        if row['type'] == 'procedures':
            rules = {int(k): which_rule(d) for k, d in RULE.findall(text)}
            assert sorted(rules) == [1, 2, 3, 4] and len(set(rules.values())) == 4, rules
            rels = {f'Rule {k}': computed[name] for k, name in rules.items()}
        else:
            rels = {}
            for o in row['options']:
                label, listed = o.split(': ', 1)
                pairs = PAIR.findall(listed)
                assert len(pairs) == len(alts) * (len(alts) - 1) // 2
                assert {frozenset(p) for p in pairs} == {frozenset(p) for p in combinations(alts, 2)}
                rel = frozenset(pairs)
                assert rel in computed.values(), (o, computed)
                rels[label] = rel
            if direction == 'intransitive' or len(alts) == 4:
                assert len(set(rels.values())) == 4
        assert all(v is not None for v in rels.values()), (text, rels)
        flags = {k: cyclic(v, alts) for k, v in rels.items()}
        odd = [k for k, c in flags.items() if c == (direction == 'intransitive')]
        assert len(odd) == 1, (text, flags)
        key = row['options'][row['correct_index']].split(':')[0]
        assert key == odd[0], (text, row['options'], flags)
        seen[(row['type'], direction)] += 1
        seen[len(alts)] += 1
    for t in ('procedures', 'consequences'):
        for d in ('intransitive', 'transitive'):
            assert seen[(t, d)], seen
    assert seen[3] and seen[4], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
