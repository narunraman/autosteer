"""pure_nash and dominant_strategies: every key re-derived by brute force from the question text alone.

The payoff table is parsed from what the question shows (the 3x3 bimatrix, or the prices / units / salaries /
raises stated in the story), the payoffs are recomputed under the objective the question states (profit = price x
units; profit = hires x (value per hire - salary); one's own raise), and:
  pure_nash: the game has exactly one pure Nash equilibrium (weak best responses, all cells checked), both best
    responses are strict there, and the marked option names it (the asked player's action or the pair);
  dominant_strategies: the asked player has exactly one strictly dominant action and the marked option names it.
Options are distinct and every equilibrium / dominance class occurs.
"""
import collections
import re

import numpy as np
import pytest

from helpers import sample

MONEY = r'\$([\d,]+(?:\.\d+)?)'


def money(s):
    return float(s.replace(',', ''))


def ne_cells(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if all(A[i, j] >= A[k, j] for k in range(A.shape[0])) and all(B[i, j] >= B[i, k] for k in range(A.shape[1]))]


def strict(A, B, i, j):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return all(A[i, j] > A[k, j] for k in range(A.shape[0]) if k != i) and \
        all(B[i, j] > B[i, k] for k in range(A.shape[1]) if k != j)


def dominant(U):
    """strictly dominant rows of U (rows = chooser's actions)."""
    U = np.asarray(U, float)
    return [a for a in range(U.shape[0]) if all((U[a] > U[b]).all() for b in range(U.shape[0]) if b != a)]


# ------------------------------------------------------------------------------------------------ matrix questions
CELL = re.compile(r'\(([\d.]+), ([\d.]+)\)')


def parse_matrix(q):
    head, _, tab = q.partition('\n\n')
    lines = [ln for ln in tab.splitlines() if ln.strip()]
    cols = re.findall(r'Action [A-Z]', lines[0])
    rows, A, B = [], [], []
    for ln in lines[1:]:
        rows.append(re.match(r'(Action [A-Z])', ln).group(1))
        cells = CELL.findall(ln)
        assert len(cells) == len(cols), ln
        A.append([float(a) for a, _ in cells])
        B.append([float(b) for _, b in cells])
    return head, rows, cols, np.array(A), np.array(B)


def test_pure_nash_matrix():
    seen = collections.Counter()
    for side in ('row', 'column'):
        for s in sample('pure_nash', 300, seed=61, type=side):
            head, rows, cols, A, B = parse_matrix(s.question)
            assert (len(rows), len(cols)) == (3, 3)
            assert f'You are the {side} player' in head
            ne = ne_cells(A, B)
            assert len(ne) == 1 and strict(A, B, *ne[0]), (s.question, ne)
            i, j = ne[0]
            assert len(set(s.options)) == len(s.options)
            if 'action pair' in head:
                want = f'({rows[i]}, {cols[j]})'
                assert len(s.options) == 9
                kind = 'pair'
            elif 'should your opponent play' in head:
                want = cols[j] if side == 'row' else rows[i]
                kind = 'opponent'
            else:
                assert 'should you play' in head
                want = rows[i] if side == 'row' else cols[j]
                kind = 'self'
            assert s.options[s.answer_index] == want, (s.question, s.options, want)
            assert s.options.count(want) == 1
            seen[(side, kind)] += 1
    assert len(seen) == 6, seen


# ------------------------------------------------------------------------------------------------ pricing stories
def parse_pricing(q):
    m = re.search(rf'^(.+?) and (.+?) are the only two .*? Each can charge either {MONEY} or {MONEY} per', q)
    a, b, lo, hi = m.group(1), m.group(2), money(m.group(3)), money(m.group(4))
    assert lo < hi
    p = {'l': lo, 'h': hi}
    units = {}
    m = re.search(rf'If both charge \${m.group(4)}, {re.escape(a)} sells (\d+) \w+ and {re.escape(b)} sells (\d+)\.', q)
    units['hh'] = (int(m.group(1)), int(m.group(2)))
    for x, y in (('h', 'l'), ('l', 'h')):
        px, py = (m_ for m_ in (f'{p[x]:.2f}', f'{p[y]:.2f}'))
        m = re.search(rf'If {re.escape(a)} charges \${px} and {re.escape(b)} charges \${py}, {re.escape(a)} sells (\d+)'
                      rf' and {re.escape(b)} sells (\d+)\.', q)
        units[x + y] = (int(m.group(1)), int(m.group(2)))
    m = re.search(rf'If both charge \${lo:.2f}, {re.escape(a)} sells (\d+) and {re.escape(b)} sells (\d+)\.', q)
    units['ll'] = (int(m.group(1)), int(m.group(2)))
    assert 'profit is its price times the number of' in q and 'maximise its own profit' in q
    # profit tables, index 0 = low, 1 = high; rows A's price, columns B's price
    idx = {0: 'l', 1: 'h'}
    PA = np.array([[p[idx[x]] * units[idx[x] + idx[y]][0] for y in (0, 1)] for x in (0, 1)])
    PB = np.array([[p[idx[y]] * units[idx[x] + idx[y]][1] for y in (0, 1)] for x in (0, 1)])
    return a, b, lo, hi, PA, PB


def price_text(v):
    return f'${v:,.2f}'


@pytest.mark.parametrize('typ', ['symmetric', 'non-symmetric'])
def test_pure_nash_pricing(typ):
    seen = collections.Counter()
    for s in sample('pure_nash', 300, seed=62, type=typ, domain='finance'):
        a, b, lo, hi, PA, PB = parse_pricing(s.question)
        ne = ne_cells(PA, PB)
        assert len(ne) == 1 and strict(PA, PB, *ne[0]), (s.question, ne)
        x, y = ne[0]
        pr = {0: price_text(lo), 1: price_text(hi)}
        if 'what price does' in s.question:
            name = re.search(r'what price does (.+?) charge\?', s.question).group(1)
            mine = x if name == a else y
            want = f'{name} charges {pr[mine]}'
        elif x == y:
            want = f'Both charge {pr[x]}'
        else:
            want = f'{a} charges {pr[x]} and {b} charges {pr[y]}'
        assert len(set(s.options)) == len(s.options)
        assert s.options[s.answer_index] == want, (s.question, s.options, want)
        seen[(x, y)] += 1
    assert set(seen) == ({(0, 0), (1, 1)} if typ == 'symmetric' else {(0, 0), (1, 1), (0, 1), (1, 0)}), seen


def test_pure_nash_hiring():
    seen = collections.Counter()
    for s in sample('pure_nash', 200, seed=63, domain='job_market'):
        q = s.question
        a = re.match(r'(.+?) and (.+?) are the only two', q).group(1)
        n = int(re.search(r'and ([\d,]+) [\w ]+ are looking for jobs', q).group(1).replace(',', ''))
        lo, hi = (money(v) for v in re.search(rf'offer either {MONEY} or {MONEY} a year', q).groups())
        h = int(re.search(r'it hires (\d+)% of the', q).group(1))
        assert re.search(rf'the remaining {100 - h}%', q)
        value = money(re.search(rf'hired brings in {MONEY} a year', q).group(1))
        assert 'maximise its own profit' in q
        hires = {(1, 1): n / 2, (0, 0): n / 2, (1, 0): n * h / 100, (0, 1): n * (100 - h) / 100}
        sal = {0: lo, 1: hi}
        U = np.array([[hires[(o, t)] * (value - sal[o]) for t in (0, 1)] for o in (0, 1)])
        ne = ne_cells(U, U.T)
        assert len(ne) == 1 and strict(U, U.T, *ne[0]), (q, ne)
        x, y = ne[0]
        st = {0: f'${lo:,.0f}', 1: f'${hi:,.0f}'}
        if 'what salary does' in q:
            want = f'{a} offers {st[x]}'
        else:
            assert x == y
            want = f'Both offer {st[x]}'
        assert s.options[s.answer_index] == want, (q, s.options, want)
        seen[(x, y)] += 1
    assert set(seen) == {(0, 0), (1, 1)}, seen


# ------------------------------------------------------------------------------------------------ dominant strategies
@pytest.mark.parametrize('typ', ['symmetric', 'non-symmetric'])
def test_dominant_pricing(typ):
    seen = collections.Counter()
    for s in sample('dominant_strategies', 300, seed=64, type=typ, domain='finance'):
        a, b, lo, hi, PA, PB = parse_pricing(s.question)
        m = re.search(r'dominant strategy for (.+?)\?$', s.question) or \
            re.search(r'Which price should (.+?) charge to maximise its profit', s.question)
        name = m.group(1)
        own = PA if name == a else PB.T
        d = dominant(own)
        assert len(d) == 1, (s.question, own)
        want = f'{name} should charge {price_text(lo if d[0] == 0 else hi)}'
        assert len(s.options) == 2 and len(set(s.options)) == 2
        assert s.options[s.answer_index] == want, (s.question, s.options, want)
        seen[(name == a, d[0])] += 1
    assert {k[1] for k in seen} == {0, 1}, seen
    if typ == 'non-symmetric':
        assert {k[0] for k in seen} == {True, False}, seen


def test_dominant_negotiation():
    seen = collections.Counter()
    for s in sample('dominant_strategies', 200, seed=65, domain='job_market'):
        q = s.question
        x = re.match(r'(\w+) and (\w+) have just been offered jobs', q).group(1)
        both_neg = money(re.search(rf'If both negotiate, each gets a raise of {MONEY}\.', q).group(1))
        one_neg, alone = (money(v) for v in re.search(
            rf'the one who negotiates gets a raise of {MONEY} and the one who accepts gets {MONEY}\.', q).groups())
        both_acc = money(re.search(rf'If both accept, each gets a raise of {MONEY}\.', q).group(1))
        assert 'largest raise for themselves' in q
        U = np.array([[both_acc, alone], [one_neg, both_neg]])  # rows: own accept / negotiate
        d = dominant(U)
        assert len(d) == 1, (q, U)
        want = f'{x} should negotiate' if d[0] == 1 else f'{x} should accept the offer as it is'
        assert s.options[s.answer_index] == want, (q, s.options, want)
        seen[d[0]] += 1
    assert set(seen) == {0, 1}, seen
