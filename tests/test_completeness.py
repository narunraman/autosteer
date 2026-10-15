"""completeness and completeness_risk: every pair of a set of alternatives, graded together (design of October 2026).

Checks on seeded draws, reading only the released texts: 4 or 5 alternatives described with labels A..E and a note
that the description is complete; one part per pair in combinations order, every part with the same description and
the four options (prefer each, indifferent, cannot decide) for its own pair; no option marked correct; the rows are
consistency-graded with a `pairwise_complete` rule whose per-part tokens point at those option texts. The rule's
scorers are checked on answers generated from weak orders (complete and transitive), with a "cannot decide" (not
complete) and with a strict cycle (intransitive).
"""
import collections
import itertools
import random
import re

import pytest

from autosteer.release.consistency import (CONSISTENT, INCOMPLETE, INTRANSITIVE, INVALID, TRANSITIVE,
                                           PAIRWISE_COMPLETE, pairwise_complete_rule, parse_pairwise_complete,
                                           score_completeness, score_group, score_transitivity)
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

SAMPLER = ElementSampler()
ELEMENTS = ['completeness', 'completeness_risk']
LINE = re.compile(r'^([A-Z][a-z]+(?: [a-z]+)?) ([A-E]): (.+)\.$', re.M)


def draws(element, n=120, base=7000):
    return [SAMPLER.sample_question(element, base + i) for i in range(n)]


def read(question):
    """(label noun, [letters], {letter: description}) from the description lines."""
    found = LINE.findall(question)
    nouns = {f[0] for f in found}
    assert len(nouns) == 1, question
    return nouns.pop(), [f[1] for f in found], {f[1]: f[2] for f in found}


def option_roles(options, x, y):
    roles = {}
    for i, o in enumerate(options):
        if o == f'I prefer {x}':
            roles['a'] = i
        elif o == f'I prefer {y}':
            roles['b'] = i
        elif o == f'I am indifferent between {x} and {y}':
            roles['i'] = i
        elif o == f'I cannot decide between {x} and {y}':
            roles['c'] = i
    return roles


@pytest.mark.parametrize('element', ELEMENTS)
def test_structure_and_rule(element):
    seen = collections.Counter()
    for q in draws(element):
        parts = q['parts']
        noun, letters, desc = read(parts[0]['question'])
        n = len(letters)
        assert n in (4, 5) and letters == list('ABCDE'[:n])
        assert len(set(desc.values())) == n                       # no two alternatives are the same
        pairs = list(itertools.combinations(range(n), 2))
        assert len(parts) == len(pairs)
        head = parts[0]['question'].rsplit('\n', 1)[0]
        assert 'These descriptions are complete' in parts[-1]['question']
        assert 'no more information is available' in parts[0]['question']
        rows = build_rows(q, generator='test')
        rule = rows[0]['consistency_rule']
        assert rule.startswith(PAIRWISE_COMPLETE + ':')
        tokens = parse_pairwise_complete(rule)
        for (a, b), p, r, tok in zip(pairs, parts, rows, tokens):
            x, y = f'{noun} {letters[a]}', f'{noun} {letters[b]}'
            assert p['question'].rsplit('\n', 1)[0] == head          # same description in every part
            last = p['question'].rsplit('\n', 1)[1]
            assert last.endswith(f'{x} and {y}, which do you prefer?') or last.endswith(f'{x} and {y}, which would you choose?'), last
            assert sum(p['correct']) == 0 and len(p['options']) == 4
            roles = option_roles(p['options'], x, y)
            assert set(roles) == {'a', 'b', 'i', 'c'}, p['options']
            assert tok == (a, b, roles['a'], roles['b'], roles['i'], roles['c'])
            assert r['grading'] == 'consistency' and r['correct_index'] is None
            assert r['consistency_rule'] == rule and r['consistency_group'] == rows[0]['consistency_group']
        seen[n] += 1
        seen[q['metadata']['domain']] += 1
    assert seen[4] > 20 and seen[5] > 20, seen
    assert len([k for k in seen if isinstance(k, str)]) == 6, seen


def _sure_numbers(text):
    return [float(x.replace(',', '')) for x in re.findall(r'\d[\d,]*(?:\.\d+)?', text)]


def test_sure_alternatives_trade_off():
    """completeness: no alternative is at least as good as all others on every attribute (signs per domain read
    from the attribute words), so no pair set is decided by a single dominant option."""
    better = {'jobs': (1, -1, 1), 'housing': (-1, 1, -1), 'technology': (-1, 1, -1), 'travel': (-1, 1, -1),
              'medical': (-1, -1, -1), 'finance': (1, -1)}
    for q in draws('completeness', 200, 9100):
        _, letters, desc = read(q['parts'][0]['question'])
        signs = better[q['metadata']['domain']]
        vals = {k: _sure_numbers(v) for k, v in desc.items()}
        for k, v in vals.items():
            assert len(v) == len(signs), (q['metadata']['domain'], desc[k])
            assert not all(all(s * (v[j] - w[j]) >= 0 for j, s in enumerate(signs))
                           for k2, w in vals.items() if k2 != k), desc


def test_lotteries_are_fully_specified():
    """completeness_risk: every alternative states its chances (percentages below 100) and its outcomes."""
    for q in draws('completeness_risk', 200, 9300):
        text = q['parts'][0]['question']
        _, _, desc = read(text)
        for d in desc.values():
            if q['metadata']['domain'] == 'agriculture':
                assert re.search(r'probability \d+%', text) and 'if the summer is dry' in d
            else:
                pcts = [int(p) for p in re.findall(r'(\d+)%', d)]
                assert pcts and all(0 < p < 100 for p in pcts) and sum(pcts) < 100, d
                assert 'otherwise' in d


def _answers_from_weak_order(rule, rank):
    """Choices (released indices) from a weak order: rank[x] smaller = better, equal = indifferent."""
    out = []
    for a, b, pa, pb, ind, _ in parse_pairwise_complete(rule):
        out.append(pa if rank[a] < rank[b] else pb if rank[b] < rank[a] else ind)
    return out


def test_scoring_on_released_rules():
    rnd = random.Random(3)
    for element in ELEMENTS:
        for q in draws(element, 40, 9500):
            rule = build_rows(q, generator='test')[0]['consistency_rule']
            toks = parse_pairwise_complete(rule)
            n = max(t[1] for t in toks) + 1
            rank = [rnd.randint(0, 2) for _ in range(n)]
            ch = _answers_from_weak_order(rule, rank)
            assert score_group(rule, ch) == CONSISTENT and score_transitivity(rule, ch) == TRANSITIVE
            k = rnd.randrange(len(toks))
            ch2 = list(ch)
            ch2[k] = toks[k][5]                                   # "cannot decide" on one pair
            assert score_completeness(rule, ch2) == INCOMPLETE
            assert score_transitivity(rule, ch2) == TRANSITIVE     # the other pairs are still a weak order
            # a strict cycle A > B, B > C, C > A
            ch3 = list(ch)
            for i, (a, b, pa, pb, _, _) in enumerate(toks):
                if (a, b) == (0, 1):
                    ch3[i] = pa
                elif (a, b) == (1, 2):
                    ch3[i] = pa
                elif (a, b) == (0, 2):
                    ch3[i] = pb
            assert score_group(rule, ch3) == CONSISTENT and score_transitivity(rule, ch3) == INTRANSITIVE
            assert score_group(rule, ch[:-1]) == INVALID


def test_rule_validation():
    good = [(a, b, 0, 1, 2, 3) for a, b in itertools.combinations(range(4), 2)]
    assert pairwise_complete_rule(good) == 'pairwise_complete:' + ','.join(f'{a}{b}0123' for a, b, *_ in good)
    with pytest.raises(ValueError):
        pairwise_complete_rule(good[:-1])                          # a pair missing
    with pytest.raises(ValueError):
        pairwise_complete_rule([good[0][:2] + (0, 1, 1, 3)] + good[1:])   # repeated option index
    rule = pairwise_complete_rule(good)
    # indifference everywhere is complete and transitive; indifference with one strict pair inside a triple is not
    assert score_transitivity(rule, [2] * 6) == TRANSITIVE
    ch = [2] * 6
    ch[0] = 0                                                      # A > B, but A ~ C and B ~ C
    assert score_transitivity(rule, ch) == INTRANSITIVE
    ch = [3] * 6
    ch[0] = 0
    assert score_completeness(rule, ch) == INCOMPLETE and score_transitivity(rule, ch) == TRANSITIVE


def test_seeded_reproducibility():
    for element in ELEMENTS:
        for seed in range(5):
            assert SAMPLER.sample_question(element, 300 + seed) == SAMPLER.sample_question(element, 300 + seed)
