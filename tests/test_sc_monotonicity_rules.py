"""sc_monotonicity_rules: every ranking rule recomputed on both ballot sets parsed from the question text.

Independent of the generator (its rules live in filler_tools/utilities/social_choice_utils.py; the ones here are
written again from the descriptions). The first ballots are parsed from "12 members voted B > C > D" lines, the
changed ballots from the "Then ... moved X up, so ... became ..." sentence. A rule violates monotonicity when some
option ranked below the raised option on the first ballots is ranked above it on the changed ones (the paper's
definition on social rankings).
  procedures    each "Rule k: ..." description is mapped to its rule by its wording; all four are recomputed on both
                ballot sets; exactly one violates, and it is the keyed rule; every rule ranks without ties; at least
                one other listed rule is of a kind that can violate.
  consequences  each option's (before, after) rankings must be the result of one of the ten candidate rules on the
                two ballot sets; the keyed option is the only violation; the pairs differ, another ranking changes and
                in another rule the raised option moves up.
  both          the raised option does not lose its place in the key's ranking (the violation is not visible from its
                position alone).
Also: the change only moves the raised option up; both types, 3 and 4 options; every non-monotonic rule is a key;
seeded reproducibility.
"""
import pytest
import collections
import re

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'sc_monotonicity_rules'
BALLOT = re.compile(r'^(\d+) members? voted ([A-Z](?: > [A-Z])+)$', re.M)
CHANGE = re.compile(r'Then (?:the (\d+) members who|(\d+) of the (\d+) members who|the member who) voted '
                    r'([A-Z](?: > [A-Z])+) moved ([A-Z]) up, so (?:their ballots|their ballot|the ballot|that ballot) '
                    r'became ([A-Z](?: > [A-Z])+)\.')
RULE = re.compile(r'^Rule (\d): (.+)$', re.M)
OPTION = re.compile(r'^Rule (\d): ([A-Z](?: > [A-Z])+) before the change, ([A-Z](?: > [A-Z])+) after it$')


def counts(prof, alts, pos):
    c = {a: 0 for a in alts}
    for n, r in prof:
        c[r[pos]] += n
    return c


def points(prof, alts):
    c = {a: 0 for a in alts}
    for n, r in prof:
        for i, a in enumerate(r):
            c[a] += n * (len(r) - 1 - i)
    return c


def head_to_head(prof, x, y):
    nx = sum(n for n, r in prof if r.index(x) < r.index(y))
    ny = sum(n for n, r in prof if r.index(x) > r.index(y))
    return None if nx == ny else (x if nx > ny else y)


def by_score(score, fewest=False):
    if len(set(score.values())) < len(score):
        return None
    return tuple(sorted(score, key=lambda a: score[a] if fewest else -score[a]))


def eliminate_all(prof, alts, worst):
    alts, gone = list(alts), []
    while len(alts) > 1:
        sub = [(n, tuple(a for a in r if a in alts)) for n, r in prof]
        out = worst(sub, alts)
        if out is None:
            return None
        gone.append(out)
        alts.remove(out)
    return tuple(alts + gone[::-1])


def single(d, pick):
    v = pick(d.values())
    w = [k for k, x in d.items() if x == v]
    return w[0] if len(w) == 1 else None


def top_two_then(score, decide, rest_score):
    order = sorted(score, key=lambda a: -score[a])
    if len(order) > 2 and score[order[1]] == score[order[2]]:
        return None
    x, y = order[:2]
    w = decide(x, y)
    if w is None:
        return None
    pair = (w, y if w == x else x)
    tail = by_score({a: rest_score[a] for a in score if a not in pair}) if len(score) > 2 else ()
    return None if tail is None else pair + tail


def r_borda(p, al):
    return by_score(points(p, al))


def r_plurality(p, al):
    return by_score(counts(p, al, 0))


def r_fewest_last(p, al):
    return by_score(counts(p, al, -1), fewest=True)


def r_first_or_second(p, al):
    c = {a: 0 for a in al}
    for n, r in p:
        for a in r[:2]:
            c[a] += n
    return by_score(c)


def r_copeland(p, al):
    wins = {a: 0 for a in al}
    for i, x in enumerate(al):
        for y in al[i + 1:]:
            w = head_to_head(p, x, y)
            if w is None:
                return None
            wins[w] += 1
    return by_score(wins)


def r_irv(p, al):
    return eliminate_all(p, al, lambda sub, a: single(counts(sub, a, 0), min))


def r_coombs(p, al):
    return eliminate_all(p, al, lambda sub, a: single(counts(sub, a, -1), max))


def r_runoff(p, al):
    f = counts(p, al, 0)
    return top_two_then(f, lambda x, y: head_to_head(p, x, y), f)


def r_points_firsts(p, al):
    b, f = points(p, al), counts(p, al, 0)
    return top_two_then(b, lambda x, y: None if f[x] == f[y] else (x if f[x] > f[y] else y), b)


def r_firsts_points(p, al):
    b, f = points(p, al), counts(p, al, 0)
    return top_two_then(f, lambda x, y: None if b[x] == b[y] else (x if b[x] > b[y] else y), f)


RULES = {'borda': r_borda, 'plurality': r_plurality, 'fewest_last': r_fewest_last, 'first_or_second': r_first_or_second,
         'copeland': r_copeland, 'irv': r_irv, 'coombs': r_coombs, 'runoff': r_runoff,
         'points_firsts': r_points_firsts, 'firsts_points': r_firsts_points}
CAN_VIOLATE = {'irv', 'coombs', 'runoff', 'points_firsts', 'firsts_points'}


def which_rule(desc):
    if 'eliminated' in desc:
        return 'irv' if 'ranked first by the fewest' in desc else 'coombs'
    if 'placed first and second' in desc:
        if 'ranked higher on more ballots' in desc:
            return 'runoff'
        return 'points_firsts' if desc.index('points') < desc.index('placed') else 'firsts_points'
    if 'wins the pair' in desc:
        return 'copeland'
    if 'points' in desc:
        return 'borda'
    if 'rank them last, fewest first' in desc:
        return 'fewest_last'
    if 'first or second' in desc:
        return 'first_or_second'
    assert 'rank them first, most first' in desc, desc
    return 'plurality'


def violates(before, after, raised):
    below = set(before[before.index(raised) + 1:])
    return bool(below & set(after[:after.index(raised)]))


def parse(text):
    prof = [(int(n), tuple(r.split(' > '))) for n, r in BALLOT.findall(text)]
    m = CHANGE.search(text)
    assert m, text
    whole, part, of, old, raised, new = m.groups()
    old, new = tuple(old.split(' > ')), tuple(new.split(' > '))
    moved = int(whole) if whole else int(part) if part else 1
    line = next(i for i, (n, r) in enumerate(prof) if r == old and n == (int(of) if of else moved))
    # the raised option moves up, the others keep their order
    assert [a for a in old if a != raised] == [a for a in new if a != raised]
    assert new.index(raised) < old.index(raised)
    n = prof[line][0]
    changed = prof[:line] + ([(n - moved, old)] if n > moved else []) + [(moved, new)] + prof[line + 1:]
    return prof, changed, raised


@pytest.mark.slow
def test_keys_recomputed():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question(E, 71000 + seed), 'test')
        text = row['question_text']
        prof, changed, raised = parse(text)
        alts = sorted(prof[0][1])
        seen[row['type']] += 1
        seen[len(alts)] += 1
        if row['type'] == 'procedures':
            rules = {int(k): which_rule(d) for k, d in RULE.findall(text)}
            assert sorted(rules) == [1, 2, 3, 4] and len(set(rules.values())) == 4, rules
            violating = []
            for k, name in rules.items():
                before, after = RULES[name](prof, alts), RULES[name](changed, alts)
                assert before and after, (name, text)
                if violates(before, after, raised):
                    violating.append(k)
                    assert after.index(raised) <= before.index(raised)
            assert len(violating) == 1, (violating, text)
            assert row['options'][row['correct_index']] == f'Rule {violating[0]}'
            assert any(n in CAN_VIOLATE for k, n in rules.items() if k != violating[0]), rules
            seen['violator_' + rules[violating[0]]] += 1
        else:
            results = {(f(prof, alts), f(changed, alts)) for f in RULES.values()}
            pairs = []
            for i, o in enumerate(row['options']):
                k, before, after = OPTION.match(o).groups()
                before, after = tuple(before.split(' > ')), tuple(after.split(' > '))
                assert (before, after) in results, (o, results)
                bad = violates(before, after, raised)
                assert bad == (i == row['correct_index']), (row['options'], raised)
                if bad:
                    assert after.index(raised) <= before.index(raised)
                pairs.append((before, after))
            others = [p for i, p in enumerate(pairs) if i != row['correct_index']]
            assert len(set(pairs)) == 4
            assert any(b != a for b, a in others) and any(a.index(raised) < b.index(raised) for b, a in others)
    assert seen['procedures'] and seen['consequences'] and seen[3] and seen[4], seen
    for name in CAN_VIOLATE:
        assert seen['violator_' + name], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
