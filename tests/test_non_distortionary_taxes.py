"""non_distortionary_taxes: keys recomputed from the question text.

identify: each option is classified by what its tax base is, from a hand-written table of descriptions: a charge that
does not depend on anything the payer does, or a tax on the rent of land whose amount is fixed, is non-distortionary;
taxes on wages, meals, savings interest, imported cars, overtime hours, hiring and hotel nights are not. Exactly one
option is non-distortionary and it is the marked one; the lump-sum and land taxes never appear together.
zero_dwl: each market's demand, supply and tax are parsed. The untaxed quantity solves Q_d(P) = Q_s(P), the taxed one
Q_d(P_b) = Q_s(P_b - t) (brentq); the deadweight loss is the integral (scipy quad) of the inverse demand minus the
inverse supply between the two quantities (zero when they are equal, as with fixed supply). Exactly one tax has zero
deadweight loss and it is the marked one; it is also the largest tax and raises the most revenue.
Every type, key kind, domain and perspective occurs; sampling is reproducible.
"""
import collections
import re

from scipy.integrate import quad
from scipy.optimize import brentq

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'non_distortionary_taxes'
NEUTRAL = [r'^A tax of \$[\d,]+ a year on every adult resident$',
           r'^A tax of \d+% on the rental value of bare land$']
DISTORTING = [r'^A tax of \d+% on all wage income$', r'^A tax of \$[\d.]+ on every restaurant meal sold$',
              r'^A tax of \d+% on interest earned on savings$', r'^A tax of \d+% on all imported cars$',
              r'^A tax of \$[\d.]+ on every hour of overtime worked$',
              r'^A tax of \d+% on the wages employers pay$',
              r'^A tax of \$[\d.]+ on every hotel room let per night$']
MARKET = re.compile(r'market ([A-D]), [^:]+: Q_d = ([\d,]+) - (\d+)P and Q_s = ([\d,]+)(?: \+ (\d+)P)?, '
                    r'tax \$([\d.]+) per unit')


def n(s):
    return float(s.replace(',', ''))


def dwl(a, b, c, d, t):
    qd = lambda p: a - b * p                     # noqa: E731
    qs = lambda p: c + d * p                     # noqa: E731
    if d == 0:
        return 0.0, t * c
    p0 = brentq(lambda p: qd(p) - qs(p), -1e6, 1e6)
    pb = brentq(lambda p: qd(p) - qs(p - t), -1e6, 1e6)
    q0, qt = qd(p0), qd(pb)
    loss = quad(lambda q: (a - q) / b - (q - c) / d, qt, q0)[0]
    return loss, t * qt


def test_non_distortionary_taxes():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(400):
        (row,) = build_rows(s.sample_question(E, 211000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        if row['type'] == 'identify':
            kinds = []
            for o in opts:
                neutral = [i for i, p in enumerate(NEUTRAL) if re.match(p, o)]
                dist = [p for p in DISTORTING if re.match(p, o)]
                assert len(neutral) + len(dist) == 1, o
                kinds.append(neutral[0] if neutral else None)
            assert [j for j, z in enumerate(kinds) if z is not None] == [k], opts
            seen['lump_sum' if kinds[k] == 0 else 'land'] += 1
        else:
            markets = MARKET.findall(text)
            assert [m[0] for m in markets] == list('ABCD')
            res = {}
            for name, a, b, c, d, t in markets:
                res[name] = dwl(n(a), n(b), n(c), n(d) if d else 0.0, float(t))
            zero = [x for x in res if res[x][0] < 1e-9]
            assert len(zero) == 1 and all(res[x][0] >= 5 - 1e-9 for x in res if x not in zero), res
            assert opts[k] == f'The tax in market {zero[0]}' and len(set(opts)) == 4
            taxes = {m[0]: float(m[5]) for m in markets}
            assert len(set(taxes.values())) == 4
            # review: the zero-DWL tax was always the largest; its rank is now uniform
            seen[f'rank{sorted(taxes.values()).index(taxes[zero[0]])}'] += 1
        seen[row['type']] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('identify', 'zero_dwl', 'lump_sum', 'land', 'first_person', 'second_person', 'man',
                                 'woman', 'anon', 'rank0', 'rank1', 'rank2', 'rank3')), seen
    ranks = [seen[f'rank{i}'] for i in range(4)]
    assert max(ranks) < 0.45 * sum(ranks), ranks


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
