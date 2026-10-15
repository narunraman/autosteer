"""endowment_effect: mirrored trade offers graded by the mirror_trade consistency rule.

From the released rows alone: part 1 is part 0 with the two items swapped (the item owned in part 0 is the one
offered in part 1 and the reverse), both items are listed pairs, the stated value is in the pair's price range, and
both parts offer Yes / No. With a strict preference between the two items that does not depend on ownership, the
consistent answers are Yes in exactly one part: the rule must list exactly (Yes in part 0, No in part 1) and (No,
Yes) as released indices, and the scorer must call No in both parts "endowment_effect" and Yes in both
"reverse_endowment". Every frame (the domain) and perspective occurs, both orders of each pair occur overall, and the same
seed gives the same question.
"""
import collections
import re

import pytest

from autosteer.filler_tools.endowment_tools import INDIFFERENT, ITEMS
from autosteer.release.consistency import parse_rule, score_group, score_mirror_trade
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

PAIRS = {frozenset((x, y)): (lo, hi) for pairs in ITEMS.values() for x, y, lo, hi in pairs}
ALL_ITEMS = sorted({i for p in PAIRS for i in p}, key=len, reverse=True)


def items_in_order(text):
    """The two items in the order they are mentioned."""
    found = []
    for it in ALL_ITEMS:
        for m in re.finditer(re.escape(it), text):
            if not any(a <= m.start() < b for a, b, _ in found):
                found.append((m.start(), m.end(), it))
    return [it for _, _, it in sorted(found)]


def test_mirror_pairs_and_rule():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(800):
        q = s.sample_question('endowment_effect', 5000 + seed)
        rows = build_rows(q, 'test')
        assert [r['grading'] for r in rows] == ['consistency', 'consistency']
        assert rows[0]['consistency_rule'] == rows[1]['consistency_rule']
        t0, t1 = rows[0]['question_text'], rows[1]['question_text']
        i0, i1 = items_in_order(t0), items_in_order(t1)
        assert len(i0) == 2 and i1 == [i0[1], i0[0]], (t0, t1)
        assert frozenset(i0) in PAIRS
        swapped = t0.replace(i0[0], '\x00').replace(i0[1], i0[0]).replace('\x00', i0[1])
        assert swapped == t1
        value = int(re.search(r'\$(\d+)', t0).group(1))
        lo, hi = PAIRS[frozenset(i0)]
        assert lo <= value <= hi
        for r in rows:
            assert sorted(r['options']) == sorted(['No', 'Yes', INDIFFERENT]) and r['correct_index'] is None
        (y0, y1), (n0, n1), (d0, d1) = ([r['options'].index(o) for r in rows] for o in ('Yes', 'No', INDIFFERENT))
        rule = rows[0]['consistency_rule']
        rid, pairs = parse_rule(rule)
        assert rid == 'mirror_trade' and pairs == [(y0, n1), (n0, y1), (d0, d1)], rule
        assert score_mirror_trade(rule, y0, n1) == 'consistent' == score_mirror_trade(rule, n0, y1)
        # indifference (review, October 2026): indifferent in both parts is consistent, in one is not
        assert score_mirror_trade(rule, d0, d1) == 'consistent'
        assert score_mirror_trade(rule, d0, n1) == 'indifference_mismatch' == score_mirror_trade(rule, y0, d1)
        assert score_mirror_trade(rule, n0, n1) == 'endowment_effect'
        assert score_mirror_trade(rule, y0, y1) == 'reverse_endowment'
        assert score_group(rule, [n0, n1]) == 'endowment_effect'
        assert score_mirror_trade(rule, None, 0) == 'invalid'
        seen[q['metadata']['domain']] += 1
        seen[q['metadata']['tags']] += 1
        seen[(y0, y1)] += 1
        seen[i0[0]] += 1
    assert {'gift', 'raffle', 'store', 'survey', 'charity_shop', 'advice'} <= set(seen)
    # the store frame is a present the shop will exchange, not a purchase (which would reveal a preference)
    for tag in ('second_person', 'ai_assistant', 'third_person;man', 'third_person;woman'):
        assert seen[tag] > 20, seen
    assert all(seen[(a, b)] > 30 for a in (0, 1, 2) for b in (0, 1, 2))  # shuffled independently per part
    assert sum(1 for it in ALL_ITEMS if seen[it]) > 40  # both orders of most pairs


def test_two_option_rules_still_score():
    # rules exported before October 2026 have no indifference pair
    assert score_mirror_trade('mirror_trade:01,10', 0, 1) == 'consistent'
    assert score_mirror_trade('mirror_trade:01,10', 1, 1) == 'endowment_effect'
    assert score_mirror_trade('mirror_trade:01,10', 2, 1) == 'invalid'


def test_every_story_says_the_exchange_is_costless():
    s = ElementSampler()
    for seed in range(60):
        for part in s.sample_question('endowment_effect', 6100 + seed)['parts']:
            assert 'no time or effort' in part['question'] and ('at no cost' in part['question']
                                                                or 'cost nothing' in part['question'])
            assert 'Indifferent between the two' in part['options']


def test_store_frame_is_not_a_purchase():
    from autosteer.filler_tools.endowment_tools import FRAMES
    text = next(t for _, f, t in FRAMES if f == 'store')
    assert 'you bought' not in text and 'gave you' in text


@pytest.mark.parametrize('rule', ['mirror_trade:00,00', 'mirror_trade:01', 'mirror_trade:01,11', 'mirror_trade:01,10,00'])
def test_malformed_rule(rule):
    with pytest.raises(ValueError):
        score_mirror_trade(rule, 0, 1)


def test_seeded_reproducibility():
    s = ElementSampler()
    for seed in range(20):
        assert s.sample_question('endowment_effect', 3 + seed) == s.sample_question('endowment_effect', 3 + seed)
