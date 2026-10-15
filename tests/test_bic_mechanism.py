"""bic_mechanism: a property table for the described mechanisms and numeric best responses for the bid formulas.

Independent of the generator: the mechanism is recognised from its description in the question text, and
  - the Yes/No key must match the table below (Bayes-Nash incentive compatibility with independent private values);
  - for "pays <fraction> of their own report" (and the first-price rule, fraction 1) the best report of a bidder
    with value v against truthful opponents is found on a grid, U(r) = (v - k r) (r / V)^(n - 1): truthful
    reporting is a best response exactly when the key is Yes;
  - for the equilibrium-bid questions the keyed bid is the best response on a $1 grid to opponents who bid
    (n - 1) / n of their values, and differs from the value (truthful bidding is not an equilibrium).
Also Yes and No are about equally common, every mechanism occurs, and sampling is reproducible from the seed.
"""
import collections
import re

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'bic_mechanism'
WORDS = {'two': 2, 'three': 3, 'four': 4, 'five': 5}
FRACTION = {'half': 1 / 2, 'two-thirds': 2 / 3, 'three-quarters': 3 / 4, 'four-fifths': 4 / 5}
TABLE = [  # (phrase in the description, truthful reporting is a Bayes-Nash equilibrium)
    ('pays the second-highest bid', True),
    ('pays the third-highest bid', False),
    ('pays their own bid', False),
    ('the price at which the last other bidder dropped out', True),
    ('one is drawn at random', True),
    ('expected total net benefit that the other', True),
    ('pays exactly the amount they pledged', False),
    ('share of the cost proportional to their', False),
]


def best_report(v, k, n, vmax, steps=4000):
    grid = [vmax * i / steps for i in range(steps + 1)]
    return max(grid, key=lambda r: (v - k * r) * (r / vmax) ** (n - 1))


def dollars(s):
    return int(s.replace('$', '').replace(',', ''))


def test_keys():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(700):
        (row,) = build_rows(s.sample_question(E, 75000 + seed), 'test')
        text = row['question_text']
        n = WORDS[re.search(r'There are (\w+) |one of (\w+) bidders', text).group(1) or
                  re.search(r'one of (\w+) bidders', text).group(1)]
        vmax = dollars(re.search(r'and (-?\$[\d,]+)[,.]', text).group(1).lstrip('-'))
        key = row['options'][row['correct_index']]
        seen[row['type']] += 1
        if row['type'] == 'bne_bid':
            v = dollars(re.search(r'values? the \w+ at (\$[\d,]+)\.', text).group(1))

            def payoff(b):
                return (v - b) * min(1.0, b * n / (n - 1) / vmax) ** (n - 1)
            best = max(range(0, v + 1), key=payoff)
            assert abs(best - dollars(key)) <= 1, (best, key)
            assert dollars(key) < v
            continue
        m = re.search(r'pays (\w[\w-]*) of their own report', text)
        if m:
            k = FRACTION[m.group(1)]
            v = 0.6 * vmax
            truthful = abs(best_report(v, k, n, vmax) - v) <= vmax / 400
            assert (key == 'Yes') == truthful, (text, key)
        else:
            hits = [ans for phrase, ans in TABLE if phrase in text]
            assert len(hits) == 1, text
            assert (key == 'Yes') == hits[0], (text, key)
            if 'pays their own bid' in text:
                assert best_report(0.6 * vmax, 1.0, n, vmax) < 0.6 * vmax - vmax / 100
            if 'third-highest' in text:
                assert n >= 3
        seen[key] += 1
    assert 0.4 <= seen['Yes'] / (seen['Yes'] + seen['No']) <= 0.6, seen
    for t in ('second_price', 'english', 'posted_report', 'scaled_payment', 'first_price', 'third_price', 'agv',
              'pay_your_bid', 'proportional_share', 'bne_bid'):
        assert seen[t], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
