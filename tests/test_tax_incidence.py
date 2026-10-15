"""tax_incidence: the same tax collected from sellers or from buyers has the same effect, recomputed from the text.

From the released rows alone: both parts state Q_d = a - bP, Q_s = c + dP (or dP - c) and a per-unit tax t; part 0
says sellers pay it, part 1 that buyers pay it. The independent check solves, with numpy, the linear system of the
part's own reading: with sellers liable, Q_d(P) = Q_s(P - t) for the market price P (buyers pay P, sellers keep
P - t); with buyers liable, Q_d(P + t) = Q_s(P) (buyers pay P + t, sellers keep P). Checks: exactly one option
matches the recomputed value (buyers' total price, sellers' net price, or the quantity) and it is the marked one;
the two parts have the same key; four distinct options, at least 2% (1.5% for quantities) apart; the untaxed
equilibrium price is never the key; every type, market domain and perspective
occurs; sampling is reproducible from the seed.
"""
import collections
import re

import numpy as np

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'tax_incidence'
NUM = r'[\d,]+'
DEMAND = re.compile(rf'Q_d = ({NUM}) - ({NUM})?P and')
SUPPLY = re.compile(rf'Q_s = (?:({NUM}) \+ ({NUM})?P|({NUM})?P - ({NUM})),')
TAX = re.compile(r'a tax of \$([\d.,]+) per')


def n(x, default=1.0):
    return default if x is None else float(x.replace(',', ''))


def market(text):
    m = DEMAND.search(text)
    a, b = n(m.group(1)), n(m.group(2))
    m = SUPPLY.search(text)
    if m.group(1) is not None:
        c, d = n(m.group(1)), n(m.group(2))
    else:
        c, d = -n(m.group(4)), n(m.group(3))
    return a, b, c, d, n(TAX.search(text).group(1))


def solve(a, b, c, d, t, payer):
    """(buyers' total price, sellers' net price, quantity) from the shifted curve of the liable side."""
    # unknowns: market price P and quantity Q;  Q + b*Pb = a,  Q - d*Ps = c
    if payer == 'sellers':      # Pb = P, Ps = P - t
        A = np.array([[b, 1.0], [-d, 1.0]])
        y = np.array([a, c - d * t])
        P, Q = np.linalg.solve(A, y)
        return P, P - t, Q
    A = np.array([[b, 1.0], [-d, 1.0]])   # Pb = P + t, Ps = P
    y = np.array([a - b * t, c])
    P, Q = np.linalg.solve(A, y)
    return P + t, P, Q


def value(o):
    return float(o.replace('$', '').replace(',', ''))


def test_templates():
    ts = ct.templates(E)
    assert len(ts) == 8 * 5 * 3
    for t in ts:
        assert len(t.parts) == 2
        assert 'sellers must pay' in t.parts[0] and 'buyers must pay' in t.parts[1]


def test_keys_recomputed_from_the_text():
    s = ElementSampler()
    types, domains, persp = collections.Counter(), set(), set()
    for seed in range(600):
        rows = build_rows(s.sample_question(E, 41000 + seed), 'test')
        assert [r['grading'] for r in rows] == ['answer', 'answer']
        mk = market(rows[0]['question_text'])
        assert market(rows[1]['question_text']) == mk
        a, b, c, d, t = mk
        qtype = rows[0]['type']
        keys = []
        for row, payer in zip(rows, ('sellers', 'buyers')):
            pb, ps, q = solve(a, b, c, d, t, payer)
            x, tol = {'buyer_price': (pb, 0.0051), 'seller_price': (ps, 0.0051), 'quantity': (q, 0.51)}[qtype]
            opts = row['options']
            assert len(opts) == 4 and len(set(opts)) == 4
            vals = [value(o) for o in opts]
            hits = [i for i, v in enumerate(vals) if abs(v - x) <= tol]
            assert hits == [row['correct_index']], (row['question_text'], opts, x)
            rel, slack = (0.015, 1.0) if qtype == 'quantity' else (0.02, 0.011)  # slack: rounding of the shown values
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= rel * max(vals[i], vals[j]) - slack, opts
            keys.append(opts[row['correct_index']])
            assert 0 < ps < pb and q > 0
        assert keys[0] == keys[1]
        # untaxed equilibrium is never the key
        p0 = (a - c) / (b + d)
        if qtype != 'quantity':
            assert abs(value(keys[0]) - p0) > 0.01
        types[qtype] += 1
        domains.add(rows[0]['domain'])
        persp.add(rows[0]['perspective'])
    assert set(types) == {'buyer_price', 'seller_price', 'quantity'}
    assert domains == {'consumer_goods', 'entertainment', 'education', 'environmental_policy', 'sports',
                       'technology', 'medical', 'politics'}
    assert persp == {'first_person', 'second_person', 'man', 'woman', 'anon'}


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
