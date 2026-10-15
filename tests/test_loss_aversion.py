"""loss_aversion: keys recomputed as the higher expected value from the amounts in the question text.

Independent of the generator: the gain $G, the loss $L, the probability of the gain (a fair coin = 1/2, otherwise
"a 40% chance") and, for sure_amount, the guaranteed $S are parsed from the text. EV = p G - (1 - p) L. The key must
be the gamble when EV exceeds the alternative (0, or S) and the sure option otherwise, with a gap of at least 10%
of the probability-weighted loss (accept_reject) or 15% of the smaller of EV and S (sure_amount). Also: EV is always positive and the
gamble is framed in the loss-aversion zone (probability-weighted gain 1.1 to 2 times the weighted loss); in
sure_amount both keys occur (the sure option about a third of the time); seeded reproducibility.
"""
import collections
import re

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'loss_aversion'


def money(s):
    return float(s.replace(',', ''))


def test_keys_by_expected_value():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question(E, 78000 + seed), 'test')
        text = row['question_text']
        amounts = re.findall(r'\$([\d,]+)', text)
        if 'fair coin' in text:
            p = 0.5
        else:
            p = int(re.search(r'a (\d+)% chance', text).group(1)) / 100
        G, L = money(amounts[0]), money(amounts[1])
        ev = p * G - (1 - p) * L
        ratio = p * G / ((1 - p) * L)
        if ev < 0:  # negative-EV control (accept_reject only, review October 2026)
            assert row['type'] == 'accept_reject' and 0.5 - 1e-9 <= ratio <= 0.87 + 1e-9, text
        else:
            assert ev > 0 and 1.1 - 1e-9 <= ratio <= 2.0 + 1e-9, text
        key = row['options'][row['correct_index']]
        gamble = next(o for o in row['options'] if not o.startswith('Take the guaranteed'))
        if row['type'] == 'sure_amount':
            S = money(re.search(r'a guaranteed \$([\d,]+) instead', text).group(1))
            assert abs(ev - S) >= 0.15 * min(ev, S) - 1e-9
            want = gamble if ev > S else next(o for o in row['options'] if o.startswith('Take the guaranteed'))
            seen['sure' if ev < S else 'gamble'] += 1
        else:
            assert abs(ev) >= 0.1 * (1 - p) * L - 1e-9
            accept = next(o for o in row['options'] if not o.startswith(('Decline', 'Do not', 'Stay')))
            want = accept if ev > 0 else next(o for o in row['options'] if o != accept)
            seen['ar_reject' if ev < 0 else 'ar_accept'] += 1
        assert key == want, (text, row['options'], ev)
        seen[row['type']] += 1
    assert seen['accept_reject'] and seen['sure_amount'], seen
    assert 0.2 <= seen['sure'] / seen['sure_amount'] <= 0.45, seen
    # "always take the gamble" fails on accept_reject too
    assert 0.2 <= seen['ar_reject'] / seen['accept_reject'] <= 0.45, seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
