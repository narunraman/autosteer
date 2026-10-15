"""ambiguity_aversion: Ellsberg pairs graded by the ellsberg_seu consistency rule.

For each sampled question the urn is read from the part-0 text: n balls of the known colour (the colour named right
after the count n) and 2n of the two unknown colours (named in the clause after 2n) (and, where stated, 3n in total). Each option's event is read
from its text (the colours in its winning clause) and every option pays the same reward. Then, for subjective
beliefs p = P(first unknown colour) on a fine grid in [0, 2/3] (p = 1/3 excluded: indifference), the pair of
strictly better bets (part 0, part 1) is collected; the rule in the released rows must list exactly those pairs.
The scorer must call the known-probability bet in both parts "ambiguity_aversion".
"""
import re
from fractions import Fraction

from autosteer.release.consistency import ellsberg_known, parse_rule, score_ellsberg
from autosteer.filler_tools.ambiguity_tools import INDIFFERENT, RANDOM_DRAW
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

COLOURS = r'red|blue|green|yellow|black|white|purple|orange|pink|silver|gold|golden|brown|grey|gray|violet|turquoise|crimson|emerald|sapphire|ruby|azure|indigo|amber'
COINS = re.compile(r'\b(gold|golden|silver)\s+(coin|coins|treasure|tokens?|pieces?)\b', re.I)
LOSING = re.compile(r'(\band 0\b|, 0\b|\b0 if\b|\b0 otherwise|\bnothing\b|\bno prize|\bno points|\botherwise)', re.I)


def colours(text):
    """Colour words in order (plurals such as 'sapphires' / 'rubies' read as the colour)."""
    text = re.sub(r'\brubies\b', 'ruby', text, flags=re.I)
    return [c.lower() for c in re.findall(rf'\b({COLOURS})(?:e?s)?\b', text, re.I)]


def event(option):
    """Colours in the bet's condition: "If you draw a red gem, ..." or "... if you pick a blue ball and 0 ..."."""
    t = COINS.sub('', option)
    if re.match(r'\s*if\b', t, re.I):
        cond = t.split(',', 1)[0]
    else:
        cond = re.split(r'\b(?:if|for)\b', t, maxsplit=1, flags=re.I)[1]
        m = LOSING.search(cond)
        cond = cond[:m.start()] if m else cond
    return frozenset(colours(cond))


def reward(option):
    return re.search(r'\b(\d+)\b', option).group(1)


def test_rule_marks_exactly_the_seu_consistent_pairs():
    s = ElementSampler()
    domains, outcomes = set(), set()
    assert len(s.template_ids('ambiguity_aversion')) >= 400
    for seed in range(600):
        q = s.sample_question('ambiguity_aversion', 7000 + seed)
        rows = build_rows(q, 'test')
        assert [r['grading'] for r in rows] == ['consistency', 'consistency']
        assert rows[0]['consistency_rule'] == rows[1]['consistency_rule'] and rows[0]['consistency_group']
        text = rows[0]['question_text']
        nums = [int(x) for x in re.findall(r'\b(\d+)\b', text)]
        n = min(nums)
        assert 2 * n in nums and set(nums) <= {n, 2 * n, 3 * n}, text
        known = colours(re.search(rf'\b{n}\b((?:\s+\S+){{1,3}})', text).group(1))
        assert len(known) == 1, text
        k = known[0]
        unknown = [c for c in dict.fromkeys(colours(re.search(rf'\b{2 * n}\b([^.]*)', text).group(1))) if c != k]
        assert len(unknown) == 2, text
        # each part ends with an indifference option (review, October 2026); `idx` maps the two bets of a
        # part to their released indices
        assert all(r['options'].count(INDIFFERENT) == 1 for r in rows)
        assert RANDOM_DRAW in text, text
        ind = [r['options'].index(INDIFFERENT) for r in rows]
        idx = [[i for i, o in enumerate(r['options']) if o != INDIFFERENT] for r in rows]
        opts = [[r['options'][i] for i in ix] for r, ix in zip(rows, idx)]
        assert len({reward(o) for p in opts for o in p}) == 1, opts
        events = [[event(o) for o in p] for p in opts]
        assert all(len(e) == 1 for e in events[0]) and all(len(e) == 2 for e in events[1]), opts
        u1 = next(iter(e for e in events[0] if e != frozenset([k]))) - {k}
        assert len(u1) == 1
        u1 = next(iter(u1))
        u2 = next(c for c in unknown if c != u1)

        def prob(ev, p):
            pr = {k: Fraction(1, 3), u1: p, u2: Fraction(2, 3) - p}
            return sum(pr[c] for c in ev)
        consistent = set()
        for i in range(0, 601):
            p = Fraction(2, 3) * Fraction(i, 600)
            if p == Fraction(1, 3):
                continue
            pick = []
            for ev in events:
                a, b = prob(ev[0], p), prob(ev[1], p)
                assert a != b
                pick.append(idx[len(pick)][0 if a > b else 1])
            consistent.add(tuple(pick))
        rule = rows[0]['consistency_rule']
        rid, pairs = parse_rule(rule)
        # strict preferences: the pairs one belief p != 1/3 justifies; p = 1/3: indifferent in both parts
        assert rid == 'ellsberg_seu' and set(pairs) == consistent | {tuple(ind)}, (rule, consistent, opts)
        # the known-probability bets: the known colour in part 0, the two unknown colours together in part 1
        k0, k1 = ellsberg_known(rule)
        released = [[event(o) if o != INDIFFERENT else None for o in r['options']] for r in rows]
        assert released[0][k0] == {k} and released[1][k1] == {u1, u2}
        o0, o1 = (next(i for i in ix if i != kk) for ix, kk in zip(idx, (k0, k1)))
        assert score_ellsberg(rule, k0, k1) == 'ambiguity_aversion'
        assert score_ellsberg(rule, o0, o1) == 'ambiguity_seeking'
        assert score_ellsberg(rule, ind[0], ind[1]) == 'consistent'
        assert score_ellsberg(rule, ind[0], k1) == 'indifference_mismatch' == score_ellsberg(rule, o0, ind[1])
        outcomes |= {score_ellsberg(rule, a, b) for a, b in consistent}
        domains.add(q['metadata']['domain'])
    assert outcomes == {'consistent'}
    assert domains == {'objects', 'mythical'}
