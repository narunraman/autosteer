"""Review fixes (October 2026), labor and finance elements: capital_market_distortions, labor_supply_distortions,
labor_supply_elasticity, price_of_risk (rebuilt as code templates), deriving_labor_supply, aggregation_of_labor_supply,
arbitrage (pipeline templates with conventions and filters).

Each test re-derives the key from the question text alone, under the model the question states, and checks the
structural cues the review found (the key always the largest option, or always the second smallest).
"""
import collections
import re

import pytest

from helpers import sample

NUM = r'-?\d+(?:\.\d+)?'


def f(pattern, text):
    m = re.search(pattern, text, re.S)
    assert m, (pattern, text)
    return [float(g) for g in m.groups()] if len(m.groups()) > 1 else float(m.group(1))


def vals(options):
    return [float(re.search(NUM, o.replace('$', '')).group(0)) for o in options]


def keyed(s, expected, tol=0.006):
    """Exactly the marked option is within rounding of `expected`."""
    v = vals(s.options)
    hits = [i for i, x in enumerate(v) if abs(x - expected) <= tol + 1e-9 * abs(expected)]
    assert hits == [s.answer_index], (s.question, s.options, expected)


def rank_spread(samples, element, min_share=0.05, max_share=0.5):
    """The key's rank among the numeric options carries little information: each of the 4 ranks occurs in at least
    min_share and at most max_share of the questions."""
    ranks = collections.Counter(sorted(vals(s.options)).index(vals(s.options)[s.answer_index]) for s in samples)
    n = sum(ranks.values())
    assert all(min_share * n <= ranks[k] <= max_share * n for k in range(4)), (element, ranks)


# ------------------------------------------------------------------ capital_market_distortions
def test_capital_market_distortions_model_and_key():
    got = sample('capital_market_distortions', 300, seed=11)
    assert {s.type for s in got} == {'capital_stock', 'capital_shortfall', 'output'}
    for s in got:
        q = s.question
        assert 'the whole after-tax return is reinvested' in q and 'does not depreciate' in q
        K0 = f(r'is worth \$(\d+) thousand today', q)
        g = f(r'pre-tax return of (' + NUM + ')%', q) / 100
        t = f(r'taxes that return at (\d+)%', q) / 100
        n = f(r'after (\d+) years', q)
        K = K0 * (1 + g * (1 - t)) ** n
        if s.type == 'capital_stock':
            keyed(s, K)
        elif s.type == 'capital_shortfall':
            keyed(s, K0 * (1 + g) ** n - K)
        else:
            A, alpha, beta = f(r'Y = (\d+)L\^(' + NUM + r')K\^(' + NUM + ')', q)
            L = f(r'employs (\d+) workers', q)
            keyed(s, A * L ** alpha * K ** beta)
        if s.type != 'capital_shortfall':
            # the capital grows, so no option may sit at or below the initial capital (or its output)
            floor = K0 if s.type == 'capital_stock' else A * L ** alpha * K0 ** beta
            assert min(vals(s.options)) > floor, (q, s.options)
    rank_spread(got, 'capital_market_distortions')
    for typ in ('capital_stock', 'output'):
        rank_spread([s for s in got if s.type == typ], 'capital_market_distortions ' + typ)


# ------------------------------------------------------------------ labor_supply_distortions
def test_labor_supply_distortions_model_and_key():
    got = sample('labor_supply_distortions', 300, seed=12)
    assert {s.type for s in got} == {'hours_worked', 'labor_supply', 'equilibrium_wage_quantity'}
    for s in got:
        q = s.question
        assert not re.search(r'LS = ' + NUM + r' \+ ' + NUM + r'T\b', q)        # supply rising with the tax
        if s.type == 'hours_worked':
            assert 'currently pays no income tax' in q or 'currently pay no income tax' in q
            assert 'gross wage stays at' in q and 'linear approximation' in q and 'percentage' in q
            assert all(o.endswith('%') for o in s.options)
            t = f(r'income tax of (\d+)%', q)
            e = f(r'after-tax wage, is (' + NUM + ')', q)
            keyed(s, -e * t)
        else:
            assert 'T is a tax per hour worked, in dollars' in q and 'W is the hourly wage employers pay' in q
            a, b = f(r'LS = (\d+) \+ (\d+)\(W - T\)', q)
            if s.type == 'labor_supply':
                assert 'Employers keep paying' in q and 'change' in q
                t0, t1 = f(r'rises from \$(' + NUM + r') to \$(' + NUM + ')', q)
                keyed(s, -b * (t1 - t0))
            else:
                c, d = f(r'LD = (\d+) - (\d+)W', q)
                T = f(r'The tax is \$(' + NUM + ')', q)
                W = (c - a + b * T) / (b + d)
                pairs = [tuple(float(x) for x in re.findall(NUM, o.replace('$', ''))) for o in s.options]
                hits = [i for i, (w, h) in enumerate(pairs) if abs(w - W) < 0.006 and abs(h - (c - d * W)) < 0.006]
                assert hits == [s.answer_index], (q, s.options, W)
    rank_spread([s for s in got if s.type == 'hours_worked'], 'labor_supply_distortions hours_worked')


# ------------------------------------------------------------------ labor_supply_elasticity
def test_labor_supply_elasticity_units_and_key():
    got = sample('labor_supply_elasticity', 300, seed=13)
    for s in got:
        q = s.question
        assert 'wage elasticity of labor supply' in q and 'unchanged labor supply curve' in q
        assert 'linear approximation' in q and all(o.endswith('%') for o in s.options)
        e = f(r'labor supply of [^:]*? is (' + NUM + '):', q)
        if s.type == 'quantity_supplied':
            d, x = re.search(r'wage (rises|falls) by (' + NUM + ')%', q).groups()
            key = (1 if d == 'rises' else -1) * e * float(x)
        else:
            d, y = re.search(r'to (rise|fall) by (' + NUM + ')%', q).groups()
            key = (1 if d == 'rise' else -1) * float(y) / e
        keyed(s, key)
        assert abs(key) <= 40          # small changes, never a fall of 100% or more
    rank_spread(got, 'labor_supply_elasticity')


# ------------------------------------------------------------------ price_of_risk
def test_price_of_risk_conventions_and_key():
    got = sample('price_of_risk', 400, seed=14)
    asks = collections.Counter()
    for s in got:
        q = s.question
        assert 'U = E[r] - (A/2) Var(r)' in q and 'decimals' in q
        borrow = 'can borrow at the risk-free rate' in q
        assert borrow != ('cannot borrow or sell short' in q)
        A = f(r'A = (\d+)', q)
        if s.type == 'risk_allocation':
            rf = f(r'paying (' + NUM + ')%', q)
            mu = f(r'expected value of (' + NUM + ')%', q)
            sd = f(r'standard deviation of (\d+)%', q)
            w = (mu - rf) / 100 / (A * (sd / 100) ** 2)
            keyed(s, 100 * (w if borrow else min(1.0, w)))
            asks['level>100' if w > 1 else 'level'] += 1
        else:
            assert 'Sharpe ratio S = (expected return - risk-free rate) / standard deviation' in q
            sd = f(r'standard deviation is (\d+)% and stays the same', q)
            s0, s1 = f(r'rises from (' + NUM + ') to (' + NUM + ')', q)
            if 'By how many percentage points' in q:
                keyed(s, 100 * (s1 - s0) / (A * sd / 100))
                asks['change'] += 1
            else:
                assert 'after the change' in q
                keyed(s, 100 * s1 / (A * sd / 100))
                asks['new level'] += 1
    assert min(asks.values()) >= 20, asks
    rank_spread(got, 'price_of_risk')


# ------------------------------------------------------------------ deriving_labor_supply
def test_deriving_labor_supply_stories_and_options():
    got = sample('deriving_labor_supply', 300, seed=15)
    for s in got:
        q = s.question
        assert not re.search(r'week|month|salar|per item|price|match|tournament|practi[cs]', q, re.I), q
        assert 'w is the hourly wage in dollars' in q and 'non-labor income in dollars per day' in q
        a, b = (float(x) for x in re.search(r'C\^\(?(' + NUM + r')\)?L\^\(?(' + NUM + r')\)?', q).groups())
        if s.type == 'hours':
            assert all(0 < x < 24 for x in vals(s.options)), s.options       # every option is a feasible day
        else:
            assert 'chooses to work some hours (h > 0)' in q
            parts = [tuple(float(x) for x in re.fullmatch(r'\((' + NUM + r')w - (' + NUM + r')\)/\((' + NUM + r')w\)',
                                                          o).groups()) for o in s.options]
            key = parts[s.answer_index]
            assert abs(key[0] - 24 * a) < 0.011 and abs(key[2] - (a + b)) < 0.011
            # the key is not the per-part majority of the options
            assert not all(sum(p[k] == key[k] for p in parts) > max(
                [sum(p[k] == x[k] for p in parts) for x in parts if x[k] != key[k]] or [0]) for k in range(3)), s.options
    rank_spread([s for s in got if s.type == 'hours'], 'deriving_labor_supply hours')


# ------------------------------------------------------------------ aggregation_of_labor_supply
def test_aggregation_of_labor_supply_conventions_and_distractors():
    got = sample('aggregation_of_labor_supply', 300, seed=16)
    for s in got:
        q = s.question
        assert 'P is the hourly wage in dollars' in q and 'no worker supplies labor to more than one' in q
        assert not re.search(r'strateg|recruit|committee|council', q, re.I), q
        fa, fb = re.findall(r'(' + NUM + r') \+ (' + NUM + r')P', q)[:2]
        if s.type == 'supply_function':
            key = s.options[s.answer_index]
            ki, ks = (float(x) for x in re.fullmatch(r'(' + NUM + r') \+ (' + NUM + r')P', key).groups())
            for o in s.options:
                if o != key:
                    i, sl = (float(x) for x in re.fullmatch(r'(' + NUM + r') \+ (' + NUM + r')P', o).groups())
                    assert i != ki and sl != ks, s.options     # no distractor shares a part with the key
    rank_spread([s for s in got if s.type == 'numerical'], 'aggregation_of_labor_supply numerical')


# ------------------------------------------------------------------ arbitrage
def test_arbitrage_states_currency_and_both_directions():
    for s in sample('arbitrage', 100, seed=17):
        assert 'All prices and costs are in US dollars, and the same transport cost applies to shipping in either ' \
               'direction.' in s.question
