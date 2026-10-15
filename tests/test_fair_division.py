"""fair_division: every procedure simulated again from the valuations parsed from the question text.

Independent of the generator (floats and bisection here, exact fractions there).
  divider_chooser   the divider's cut is found by bisection on the divider's value of the left piece; the chooser
                    takes the piece worth more to them; the key is the cut, the chooser's piece and its value, or
                    the divider's piece value (half the divider's total), as asked.
  last_diminisher   each round, every remaining person's point worth 1/k of the rest (k people left) is found by bisection from the current start; the
                    piece ends at the smallest one, taken by the first person in turn order who reaches it; the key
                    states the first piece or the whole allocation (rounded to 2 decimals).
  adjusted_winner   items go to the higher bidder, the leader hands over items by increasing ratio, the crossing item
                    is split; the key names that item and the shares (1 decimal).
  envy_check        proportional (each share >= 100/3 by own points) and envy-free (no one prefers another's share);
                    the key is the matching statement, and "envy-free but not proportional" is never the key.
Also every type and ask occurs, and sampling is reproducible from the seed.
"""
import collections
import re

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'fair_division'


def bisect(f, target, lo, hi):
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fmt(x, places=2):
    return f'{round(x, places):.{places}f}'.rstrip('0').rstrip('.')


NUM = re.compile(r'\d+(?:\.\d+)?')


def same(shown, want):
    """Equal texts, numbers equal up to rounding of the last shown decimal (the generator rounds exact values)."""
    a, b = NUM.split(shown), NUM.split(want)
    if a != b:
        return False
    for x, y in zip(NUM.findall(shown), NUM.findall(want)):
        places = len(x.split('.')[1]) if '.' in x else 0
        if abs(float(x) - float(y)) > 0.5 * 10 ** -places + 0.006:
            return False
    return True


def check_divider_chooser(text, options, key):
    L = 12 if '12-inch' in text else 60
    unit = 'inches' if L == 12 else 'metres'
    a = int(re.search(r'the first (\d+) ', text).group(1))
    vals = re.findall(r'(\w+) values \w[\w ]* at \$(\d+) per \w+ and \w[\w ]* at \$(\d+) per', text)
    (div, d1, d2), (cho, c1, c2) = [(n, int(x), int(y)) for n, x, y in vals]

    def value(t, v1, v2):
        return v1 * min(t, a) + v2 * max(0, t - a)
    t = bisect(lambda x: value(x, d1, d2), value(L, d1, d2) / 2, 0, L)
    cl, cr = value(t, c1, c2), value(L, c1, c2) - value(t, c1, c2)
    if text.endswith('does ' + div + ' cut?'):
        assert same(key, f'{fmt(t)} {unit}'), (key, t)
        return 'cut'
    if 'Which piece does' in text:
        side = f'the left piece (0 to {fmt(t)} {unit})' if cl > cr else f'the right piece ({fmt(t)} to {L} {unit})'
        assert same(key, f'{side}, worth ${fmt(max(cl, cr))}'), (key, cl, cr)
        return 'chooser'
    assert key == f'${fmt(value(L, d1, d2) / 2)}', key
    return 'divider_share'


def check_last_diminisher(text, options, key):
    n = 4 if '1/4 of the whole' in text else 3
    bounds = [int(x) for x in re.search(r'chocolate from 0 to (\d+) inches, vanilla from \d+ to (\d+) inches',
                                        text).groups()]
    edges = [0] + bounds + [12]
    pts = {p: [int(a), int(b), int(c)] for p, a, b, c in
           re.findall(r'^(\w+): chocolate (\d+), vanilla (\d+), lemon (\d+)$', text, re.M)}
    order = re.search(r'turn order ([\w, ]+?): ', text).group(1).replace(' and ', ', ').split(', ')
    assert len(order) == n == len(pts)

    def worth(p, x):  # value of [0, x]
        return sum(v * max(0, min(x, hi) - lo) / (hi - lo) for v, lo, hi in zip(pts[p], edges, edges[1:]))
    left, start, pieces = list(order), 0.0, []
    while len(left) > 1:
        # standard last diminisher: 1/k of the rest of the cake by each person's own points, k = people left
        xs = {p: bisect(lambda x, p=p: worth(p, x) - worth(p, start), (sum(pts[p]) - worth(p, start)) / len(left),
                        start, 12) for p in left}
        best = min(xs.values())
        taker = next(p for p in left if xs[p] - best < 1e-9)
        pieces.append((taker, start, best))
        left.remove(taker)
        start = best
    pieces.append((left[0], start, 12))
    if 'Who gets the first piece' in text:
        p, s, e = pieces[0]
        assert same(key, f'{p} gets the piece from 0 to {fmt(e)} inches'), (key, pieces)
        return 'first'
    want = '; '.join(f'{p}: {fmt(s)} to {fmt(e)} inches' for p, s, e in pieces)
    assert same(key, want), (key, want)
    return 'final'


def check_adjusted_winner(text, options, key):
    rows = re.findall(r'^([A-Z][\w ]+): (\w+) (\d+), (\w+) (\d+)$', text, re.M)
    a, b = rows[0][1], rows[0][3]
    va = {it.lower(): int(x) for it, _, x, _, _ in rows}
    vb = {it.lower(): int(y) for it, _, _, _, y in rows}
    assert sum(va.values()) == sum(vb.values()) == 100
    ta = sum(va[i] for i in va if va[i] > vb[i])
    tb = sum(vb[i] for i in vb if vb[i] > va[i])
    lead, other, vl, vo = (a, b, va, vb) if ta > tb else (b, a, vb, va)
    hi, lo = max(ta, tb), min(ta, tb)
    for it in sorted((i for i in va if vl[i] > vo[i]), key=lambda i: vl[i] / vo[i]):
        if hi - vl[it] > lo + vo[it]:
            hi, lo = hi - vl[it], lo + vo[it]
            continue
        give = (hi - lo) / (vl[it] + vo[it])
        keep = 1 - give
        assert abs((hi - give * vl[it]) - (lo + give * vo[it])) < 1e-9
        want = (f'The {it} is split: {lead} keeps {fmt(100 * keep, 1)}% of it and {other} gets '
                f'{fmt(100 * give, 1)}%')
        assert same(key, want), (key, want)
        return 'adjusted_winner'
    raise AssertionError('no split')


def check_envy(text, options, key):
    vals = {}
    for p, rest in re.findall(r'^(\w+): ((?:[a-z][a-z ]+ \d+(?:, )?)+)$', text, re.M):
        vals[p] = {it: int(v) for it, v in re.findall(r'([a-z][a-z ]+?) (\d+)', rest)}
    alloc = {}
    for p, items in re.findall(r'(\w+) gets ((?:the [a-z ]+?)(?:(?:, | and )the [a-z ]+?)*)(?=;|\. )', text):
        alloc[p] = re.findall(r'the ([a-z ]+?)(?=, | and |$)', items)
    assert set(alloc) == set(vals) and sum(len(v) for v in alloc.values()) == len(next(iter(vals.values())))
    share = {p: {q: sum(vals[p][i] for i in alloc[q]) for q in alloc} for p in vals}
    prop = all(3 * share[p][p] >= 100 for p in vals)
    ef = all(share[p][p] >= share[p][q] for p in vals for q in vals)
    want = ('It is both envy-free and proportional' if ef else 'It is proportional but not envy-free' if prop
            else 'It is neither envy-free nor proportional')
    assert same(key, want), (key, share)
    return want


def test_procedures_simulated():
    s = ElementSampler()
    seen = collections.Counter()
    checks = {'divider_chooser': check_divider_chooser, 'last_diminisher': check_last_diminisher,
              'adjusted_winner': check_adjusted_winner, 'envy_check': check_envy}
    for seed in range(800):
        (row,) = build_rows(s.sample_question(E, 74000 + seed), 'test')
        assert len(set(row['options'])) == len(row['options']) == (3 if row['type'] == 'envy_check' else 4)
        key = row['options'][row['correct_index']]
        seen[checks[row['type']](row['question_text'], row['options'], key)] += 1
    for k in ('cut', 'chooser', 'divider_share', 'first', 'final', 'adjusted_winner',
              'It is both envy-free and proportional', 'It is proportional but not envy-free',
              'It is neither envy-free nor proportional'):
        assert seen[k], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q
