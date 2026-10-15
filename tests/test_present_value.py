"""present_value: present values recomputed by direct summation of the payments stated in the text.

From the released rows alone: the payments (one amount in t years; a list "$X one year from now, ..."; or $C every
year forever from one year from now) and either an annual interest rate r (discount factor 1/(1 + r)) or a discount
factor d are parsed; the present value is the sum of payment * factor^year (a perpetuity summed over 6,000 years,
where the remaining tail is below a millionth of a cent). Checks: exactly one option is within half a cent of it and
it is the marked one; four distinct options at least 2% apart; every type, both ways of stating the discounting,
every domain and perspective occur; sampling is reproducible from the seed.
"""
import collections
import re

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'present_value'
WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
RATE = re.compile(r'interest rate of (\d+)% per year')
DELTA = re.compile(r'discount factor of ([\d.]+) per year')
AMOUNT = re.compile(r'\$([\d,]+) (?:when it matures in|that will be paid out in|that will be paid in|that will arrive in) (\d+) years')
STREAM_ITEM = re.compile(r'\$([\d,]+) (one|two|three|four|five) years? from now')
PERP = re.compile(r'\$([\d,]+) (?:a year|every year)')


def money(o):
    return float(o.replace('$', '').replace(',', ''))


def test_templates():
    assert len(ct.templates(E)) == 8 * 5 * 3 * 2


def test_present_values_by_summation():
    s = ElementSampler()
    seen = collections.Counter()
    domains, persp = set(), set()
    for seed in range(900):
        (row,) = build_rows(s.sample_question(E, 71000 + seed), 'test')
        text = row['question_text']
        m = RATE.search(text)
        f = 1 / (1 + int(m.group(1)) / 100) if m else float(DELTA.search(text).group(1))
        how = 'rate' if m else 'factor'
        if row['type'] == 'amount':
            x, t = AMOUNT.search(text).groups()
            pays = {int(t): float(x.replace(',', ''))}
        elif row['type'] == 'stream':
            items = STREAM_ITEM.findall(text)
            assert 3 <= len(items) <= 5
            pays = {WORDS[w]: float(x.replace(',', '')) for x, w in items}
            assert sorted(pays) == list(range(1, len(items) + 1))
        else:
            assert 'with the first payment one year from now' in text
            c = float(PERP.search(text).group(1).replace(',', ''))
            pays = {k: c for k in range(1, 6001)}
        value = sum(v * f ** k for k, v in pays.items())
        vals = [money(o) for o in row['options']]
        assert len(set(row['options'])) == 4
        hits = [i for i, v in enumerate(vals) if abs(v - value) <= 0.0051]
        assert hits == [row['correct_index']], (text, row['options'], value)
        for i in range(4):
            for j in range(i):
                assert abs(vals[i] - vals[j]) >= 0.02 * max(vals[i], vals[j]) - 0.011, row['options']
        seen[(row['type'], how)] += 1
        domains.add(row['domain'])
        persp.add(row['perspective'])
    assert set(seen) == {(t, h) for t in ('amount', 'stream', 'perpetuity') for h in ('rate', 'factor')}
    assert len(domains) == 8
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
