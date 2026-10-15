"""independence: asymmetric decoy pairs graded by the same_brand consistency rule.

From the released rows alone: part 0 offers brands A and B (price, quality) with a trade-off (neither dominates);
part 1 repeats A and B verbatim and adds brand C. C must be worse than exactly one of A, B on both attributes (a
higher price and a lower quality rating; the target) and better than the other on at least one. The rule must list
exactly the pairs that choose the same brand in both parts, the target first; the scorer must call target-then-
target "consistent", other-then-target "decoy_effect", target-then-other "reverse" and C "decoy_chosen". Both
targets, both domains and "two" / "three" in the texts occur.
"""
import collections
import re

from autosteer.release.consistency import parse_rule, score_same_brand
from autosteer.filler_tools.decoy_tools import NO_PREFERENCE
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

OPT = re.compile(r'^Brand ([ABC]): price is \$([\d,]+), with quality rating (\d+)$')


def parse(options):
    out = {}
    for i, o in enumerate(options):
        if o == NO_PREFERENCE:
            out['none'] = (i,)
            continue
        m = OPT.match(o)
        assert m, o
        out[m.group(1)] = (i, int(m.group(2).replace(',', '')), int(m.group(3)))
    return out


def worse_on_both(x, y):
    """x is dearer and of lower quality than y."""
    return x[1] > y[1] and x[2] < y[2]


def test_decoy_construction_and_rule():
    s = ElementSampler()
    targets, domains = collections.Counter(), set()
    for seed in range(800):
        q = s.sample_question('independence', 9000 + seed)
        rows = build_rows(q, 'test')
        assert [r['grading'] for r in rows] == ['consistency', 'consistency']
        assert re.search(r'\b[Tt]wo\b', rows[0]['question_text']) and re.search(r'\b[Tt]hree\b', rows[1]['question_text'])
        assert not re.search(r'[.!?] (two|three)\b', rows[0]['question_text'] + rows[1]['question_text'])   # capitalised at a sentence start
        p0, p1 = parse(rows[0]['options']), parse(rows[1]['options'])
        # a no-preference option in both parts (review, October 2026)
        assert set(p0) == {'A', 'B', 'none'} and set(p1) == {'A', 'B', 'C', 'none'}
        for x in 'AB':
            assert p0[x][1:] == p1[x][1:]
        assert not worse_on_both(p0['A'], p0['B']) and not worse_on_both(p0['B'], p0['A'])
        dominated_by = [x for x in 'AB' if worse_on_both(p1['C'], p1[x])]
        assert len(dominated_by) == 1, rows[1]['options']
        target = dominated_by[0]
        other = 'B' if target == 'A' else 'A'
        assert p1['C'][1] < p1[other][1] or p1['C'][2] > p1[other][2]  # not worse than the other on both
        rule = rows[0]['consistency_rule']
        rid, pairs = parse_rule(rule)
        assert rid == 'same_brand' and pairs == [(p0[target][0], p1[target][0]), (p0[other][0], p1[other][0]),
                                                 (p0['none'][0], p1['none'][0])], rule
        assert score_same_brand(rule, p0['none'][0], p1['none'][0]) == 'consistent'
        assert score_same_brand(rule, p0['none'][0], p1[target][0]) == 'indifference_mismatch'
        assert score_same_brand(rule, p0['none'][0], p1['C'][0]) == 'decoy_chosen'
        assert score_same_brand(rule, p0[target][0], p1[target][0]) == 'consistent'
        assert score_same_brand(rule, p0[other][0], p1[other][0]) == 'consistent'
        assert score_same_brand(rule, p0[other][0], p1[target][0]) == 'decoy_effect'
        assert score_same_brand(rule, p0[target][0], p1[other][0]) == 'reverse'
        assert score_same_brand(rule, p0[target][0], p1['C'][0]) == 'decoy_chosen'
        targets[target] += 1
        domains.add(q['metadata']['domain'])
    assert set(targets) == {'A', 'B'}
    assert domains == {'everyday_products', 'technological_products'}
