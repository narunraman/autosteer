"""maximize_expected_utility: with utility linear in money, the key is the option with the highest expected value.

Recomputed from the texts alone: every question says utility is linear in money; options are a sure amount
("A guaranteed $76") and one or two lotteries ("$118 with a 35% chance and $52 with a 65% chance"), or two jobs
(a sure salary; a lower salary with a stated chance of a bonus). Probabilities add up to 100%, each lottery has one
outcome above and one below the sure amount, the best expected value beats the others by at least 5% (2% for jobs),
and the marked option is the best. Options are distinct and both the sure side and a lottery are keyed.
"""
import collections
import re

from helpers import sample

SURE = re.compile(r'^A guaranteed \$([\d,]+)$')
LOT = re.compile(r'^\$([\d,]+) with a (\d+)% chance and \$([\d,]+) with a (\d+)% chance$')


def n(s):
    return float(s.replace(',', ''))


def test_gambles():
    seen = collections.Counter()
    for s in sample('maximize_expected_utility', 600, seed=71, domain='gambling'):
        assert 'Your utility is linear in money' in s.question
        assert len(set(s.options)) == len(s.options) in (2, 3)
        assert ('three options' in s.question) == (len(s.options) == 3)
        sure = [n(SURE.match(o).group(1)) for o in s.options if SURE.match(o)]
        assert len(sure) == 1, s.options
        evs = []
        for o in s.options:
            if SURE.match(o):
                evs.append(sure[0])
                continue
            m = LOT.match(o)
            assert m, o
            hi, p, lo, q = n(m.group(1)), int(m.group(2)), n(m.group(3)), int(m.group(4))
            assert p + q == 100 and 0 < p < 100
            assert hi > sure[0] > lo, (o, sure)
            evs.append((hi * p + lo * q) / 100)
        best = max(range(len(evs)), key=lambda i: evs[i])
        assert all(evs[best] >= 1.05 * v for i, v in enumerate(evs) if i != best), (s.options, evs)
        assert s.answer_index == best, (s.question, s.options, evs)
        seen['sure' if SURE.match(s.answer) else 'lottery'] += 1
    assert set(seen) == {'sure', 'lottery'}, seen


def test_job_offers():
    seen = collections.Counter()
    for s in sample('maximize_expected_utility', 300, seed=72, domain='job_offer'):
        q = s.question
        assert 'Your utility is linear in money' in q
        m = re.search(r'Job (\w) offers a salary of \$([\d,]+) per year with no chance of a bonus\. Job (\w) offers a '
                      r'salary of \$([\d,]+) per year but with a (\d+)% chance of a \$([\d,]+) bonus\.', q)
        a, sure, b, base, p, bonus = m.group(1), n(m.group(2)), m.group(3), n(m.group(4)), int(m.group(5)), n(m.group(6))
        assert base < sure < base + bonus
        ev = base + bonus * p / 100
        assert max(sure, ev) >= 1.02 * min(sure, ev)
        want = f'Job {a}' if sure > ev else f'Job {b}'
        assert sorted(s.options) == sorted([f'Job {a}', f'Job {b}'])
        assert s.answer == want, (q, s.options, sure, ev)
        seen[want == f'Job {a}'] += 1
    assert set(seen) == {True, False}, seen
