"""dsic_mechanism (filler_tools/dsic_tools.py): keys from a property table and brute force, independent of the filler.

1. The mechanism is recognised from the question's own wording (phrases of the rule, not the type label), and the key
   of the Yes/No part must be the textbook answer of the property table below; the question must state its
   assumptions (private values; value minus payment where money changes hands; single-peaked preferences; strict
   rankings) and must not name the mechanism.
2. Brute force, with the question's own number of participants and, for public projects, its cost and share:
   for the Yes mechanisms no deviation from truthful reporting pays on random instances (all reports on a grid, or
   every ranking for matching); for the No mechanisms a profitable deviation is found. The rules are implemented
   here from their descriptions.
3. Follow-up parts (winner and payment, revenue, a bidder's payoff) are recomputed from the stated bids, and the
   counterexample type (location_misreport) is decided by a search over reports.
4. Yes and No are balanced; every type occurs; sampling is reproducible.
"""
import collections
import itertools
import random
import re

import pytest

from autosteer.sampler import ElementSampler

WORDS = {'three': 3, 'four': 4, 'five': 5, 'six': 6}
TABLE = {  # mechanism -> is truthful reporting a dominant strategy for every participant (on the side asked about)
    'second': True, 'reserve': True, 'clock': True, 'posted': True, 'first': False, 'third': False, 'allpay': False,
    'lottery': False, 'pivot': True, 'pledge': False, 'equal': False, 'median': True, 'mean': False, 'serial': True,
    'cycles': True, 'immediate': False, 'da_proposers': True, 'da_receivers': False}
NAMES = re.compile(r'Vickrey|second-price|first-price|third-price|Clarke|VCG|Groves|pivot|median|Boston|'
                   r'deferred acceptance|Gale|Shapley|serial dictator|top trading|English|Dutch|all-pay|'
                   r'immediate acceptance|strategy-?proof|incentive compatib', re.I)


def classify(text):
    t = text
    if 'sealed bid' in t:
        if re.search(r'minimum price of|provided the bid is at least', t):
            return 'reserve'
        if re.search(r'third-highest|ranked third', t):
            return 'third'
        if re.search(r'highest of the other bids|best competing bid', t):
            return 'second'
        if re.search(r'every bidder, whether they win or not, pays|Every bidder must pay their own bid', t):
            return 'allpay'
        if re.search(r'drawn at random|random draw', t):
            return 'lottery'
        if re.search(r'pays the amount of their own bid|pays exactly what they offered', t):
            return 'first'
    if re.search(r'price starts at \$0 and rises|keeps going up from zero|rises continuously from zero', t):
        return 'clock'
    if re.search(r'fixes a price of|take-it-or-leave-it price', t):
        return 'posted'
    if 'deciding whether to buy' in t:
        if re.search(r'\bfee\b', t):
            return 'pivot'
        if re.search(r'in proportion to the reports|share of the reported total', t):
            return 'pledge'
        return 'equal'
    if 'shared' in t and 'favourite spot' in t:
        return 'median' if re.search(r'in the middle|middle one', t) else 'mean'
    if re.search(r'random order of the|a lottery then sets an order', t):
        return 'serial'
    if re.search(r'points? to (?:the owner|whoever holds)', t):
        return 'cycles'
    if re.search(r'Round one|In the first round', t):
        return 'immediate'
    m = re.search(r'pairs \w+ (.+?) with \w+ (.+?), one each|matches \w+ (.+?) to \w+ (.+?)\. ', t)
    if m:
        props = m.group(1) or m.group(3)
        q = t.rsplit('. ', 1)[-1]
        asked = re.search(r'for (?:every|each) ([\w -]+?)(?: under| to report)', q).group(1)
        return 'da_proposers' if props.endswith(asked) or asked + 's' == props.split()[-1] or \
            props.rstrip('s').endswith(asked) else 'da_receivers'
    raise AssertionError(t[:200])


def assumptions_stated(mech, t):
    assert re.search(r'alone knows|only they know', t), t
    if mech in ('median', 'mean'):
        assert 'the nearer the better' in t and 'No money' in t
    elif mech in ('serial', 'cycles', 'immediate', 'da_proposers', 'da_receivers'):
        assert 'strict ranking' in t and 'No money' in t
    else:
        assert 'minus whatever they pay' in t


# ------------------------------------------------------------------------------------------- the rules
def auction(mech, bids, reserve=0.0, price=0.0):
    """Expected (allocation probabilities, payments) of a single-item rule with distinct bids."""
    n = len(bids)
    order = sorted(range(n), key=lambda i: -bids[i])
    win, pay = [0.0] * n, [0.0] * n
    w = order[0]
    if mech in ('second', 'clock'):  # clock: bids are drop-out prices; the last one in pays the second drop-out
        win[w], pay[w] = 1, bids[order[1]]
    elif mech == 'first':
        win[w], pay[w] = 1, bids[w]
    elif mech == 'third':
        win[w], pay[w] = 1, bids[order[2]]
    elif mech == 'reserve':
        if bids[w] >= reserve:
            win[w], pay[w] = 1, max(reserve, bids[order[1]])
    elif mech == 'allpay':
        win[w] = 1
        pay = list(bids)
    elif mech == 'lottery':
        tot = sum(bids)
        for i in range(n):
            win[i] = bids[i] / tot
            pay[i] = bids[i] * bids[i] / tot
    elif mech == 'posted':  # bids are 1 (accept) / 0; random order: everyone is first with chance 1/n
        acc = [i for i in range(n) if bids[i]]
        for perm in itertools.permutations(range(n)):
            first = next((i for i in perm if i in acc), None)
            if first is not None:
                win[first] += 1
                pay[first] += price
        k = sum(1 for _ in itertools.permutations(range(n)))
        win, pay = [x / k for x in win], [x / k for x in pay]
    return win, pay


def auction_utility(mech, v, i, bids, **kw):
    win, pay = auction(mech, bids, **kw)
    return v * win[i] - pay[i]


def project(mech, reports, cost, share):
    n = len(reports)
    built = sum(reports) >= cost
    pay = [0.0] * n
    if built:
        if mech == 'pledge':
            pay = [cost * r / sum(reports) for r in reports]
        else:
            pay = [share] * n
    if mech == 'pivot':
        for i in range(n):
            others = sum(reports[j] - share for j in range(n) if j != i)
            if (others >= 0) != built:
                pay[i] += abs(others)
    return built, pay


def facility(mech, reports):
    s = sorted(reports)
    return s[len(s) // 2] if mech == 'median' else sum(reports) / len(reports)


def serial(prefs, order):
    taken, out = set(), {}
    for i in order:
        out[i] = next(o for o in prefs[i] if o not in taken)
        taken.add(out[i])
    return out


def cycles(prefs):
    """Top trading cycles; agent i owns object i."""
    left, out = set(range(len(prefs))), {}
    while left:
        point = {i: next(o for o in prefs[i] if o in left) for i in left}
        i, seen = next(iter(left)), []
        while i not in seen:
            seen.append(i)
            i = point[i]
        cyc = seen[seen.index(i):]
        for j in cyc:
            out[j] = point[j]
        left -= set(cyc)
    return out


def immediate(prefs, priority, seats):
    out, free = {}, list(seats)
    for rnd in range(len(seats)):
        for s in range(len(seats)):
            applicants = [i for i in priority[s] if i not in out and prefs[i][rnd] == s]
            for i in applicants[:free[s]]:
                out[i] = s
            free[s] -= min(free[s], len(applicants))
    return out


def deferred(prop_prefs, rec_prefs):
    """Proposers propose; returns proposer -> receiver."""
    n = len(prop_prefs)
    nxt, held, free = [0] * n, {}, list(range(n))
    while free:
        p = free.pop()
        r = prop_prefs[p][nxt[p]]
        nxt[p] += 1
        cur = held.get(r)
        if cur is None:
            held[r] = p
        elif rec_prefs[r].index(p) < rec_prefs[r].index(cur):
            held[r] = p
            free.append(cur)
        else:
            free.append(p)
    return {p: r for r, p in held.items()}


# ------------------------------------------------------------------------------------------- brute force
def profitable_deviation(mech, n, params, rnd, tries=60):
    """True if some instance has a participant who gains by misreporting (everyone else truthful)."""
    for _ in range(tries):
        if mech in ('second', 'clock', 'first', 'third', 'reserve', 'allpay', 'lottery'):
            vals = [rnd.uniform(1, 100) for _ in range(n)]
            kw = {'reserve': 50.0} if mech == 'reserve' else {}  # values 1..100, the reserve in the middle
            base = auction_utility(mech, vals[0], 0, vals, **kw)
            for b in [x / 4 for x in range(1, 600)]:
                if b in vals[1:]:
                    continue
                if auction_utility(mech, vals[0], 0, [b] + vals[1:], **kw) > base + 1e-9:
                    return True
        elif mech == 'posted':
            price = 50
            vals = [rnd.uniform(1, 100) for _ in range(n)]
            truth = [int(v >= price) for v in vals]
            base = auction_utility(mech, vals[0], 0, truth, price=price)
            dev = list(truth)
            dev[0] = 1 - dev[0]
            if auction_utility(mech, vals[0], 0, dev, price=price) > base + 1e-9:
                return True
        elif mech in ('pivot', 'pledge', 'equal'):
            cost = params['cost']
            share = params.get('share', cost // n)  # the pledge rule states no share; a scale for the grid
            vals = [rnd.choice(range(0, 3 * share + 1, max(1, share // 10))) for _ in range(n)]

            def u(i, reports):
                built, pay = project(mech, reports, cost, share)
                return (vals[i] if built else 0) - pay[i]
            for i in range(n):
                base = u(i, vals)
                for r in range(0, 4 * share + 1, max(1, share // 20)):
                    rep = list(vals)
                    rep[i] = r
                    if u(i, rep) > base + 1e-9:
                        return True
        elif mech in ('median', 'mean'):
            peaks = [rnd.randint(0, 20) for _ in range(n)]
            for i in range(n):
                base = abs(facility(mech, peaks) - peaks[i])
                for r in [x / 4 for x in range(0, 81)]:
                    rep = list(peaks)
                    rep[i] = r
                    if abs(facility(mech, rep) - peaks[i]) < base - 1e-9:
                        return True
        else:
            if matching_deviation(mech, n, rnd):
                return True
    return False


def matching_deviation(mech, n, rnd):
    def ranking():
        x = list(range(n))
        rnd.shuffle(x)
        return x
    prefs = [ranking() for _ in range(n)]
    perms = list(itertools.permutations(range(n)))
    if mech == 'serial':
        order = ranking()
        res = serial(prefs, order)
        for i in range(n):
            for lie in perms:
                alt = serial(prefs[:i] + [list(lie)] + prefs[i + 1:], order)
                if prefs[i].index(alt[i]) < prefs[i].index(res[i]):
                    return True
    elif mech == 'cycles':
        res = cycles(prefs)
        for i in range(n):
            for lie in perms:
                alt = cycles(prefs[:i] + [list(lie)] + prefs[i + 1:])
                if prefs[i].index(alt[i]) < prefs[i].index(res[i]):
                    return True
    elif mech == 'immediate':
        schools = 3
        sp = [list(p) for p in itertools.permutations(range(schools))]
        prefs = [rnd.choice(sp) for _ in range(n)]
        priority = [ranking() for _ in range(schools)]
        seats = [1, 1, n - 2]
        res = immediate(prefs, priority, seats)
        for i in range(n):
            for lie in sp:
                alt = immediate(prefs[:i] + [lie] + prefs[i + 1:], priority, seats)
                if i in alt and (i not in res or prefs[i].index(alt[i]) < prefs[i].index(res[i])):
                    return True
    else:
        rec = [ranking() for _ in range(n)]
        res = deferred(prefs, rec)
        if mech == 'da_proposers':
            for i in range(n):
                for lie in perms:
                    alt = deferred(prefs[:i] + [list(lie)] + prefs[i + 1:], rec)
                    if prefs[i].index(alt[i]) < prefs[i].index(res[i]):
                        return True
        else:
            inv = {r: p for p, r in res.items()}
            for j in range(n):
                for lie in perms:
                    alt = deferred(prefs, rec[:j] + [list(lie)] + rec[j + 1:])
                    alt_inv = {r: p for p, r in alt.items()}
                    if rec[j].index(alt_inv[j]) < rec[j].index(inv[j]):
                        return True
    return False


def params_of(text):
    out = {}
    for k, pat in (('cost', r'which costs \$([\d,]+)'), ('share', r'(?:equal share of|equally,) \$([\d,]+)'),
                   ('reserve', r'minimum price of \$([\d,]+)|at least \$([\d,]+), an amount'),
                   ('price', r'price of \$([\d,]+)')):
        m = re.search(pat, text)
        if m:
            out[k] = int(next(g for g in m.groups() if g).replace(',', ''))
    return out


_CACHE = {}


def draws():
    if not _CACHE:
        s = ElementSampler()
        types = sorted(s._manifest['elements']['dsic_mechanism']['types'])
        _CACHE['types'] = types
        _CACHE['q'] = [s.sample_question('dsic_mechanism', 7000 + 37 * k + i, type=t)
                       for k, t in enumerate(types) for i in range(30)]
    return _CACHE['q']


def test_types():
    draws()
    assert set(_CACHE['types']) == {f'auction_{x}' for x in ('second', 'reserve', 'clock', 'posted', 'first', 'third',
                                                              'allpay', 'lottery')} | \
        {'project_pivot', 'project_pledge', 'project_equal', 'location_median', 'location_mean', 'match_serial',
         'match_cycles', 'match_immediate', 'match_da_proposers', 'match_da_receivers', 'location_misreport'}


def test_property_table_and_wording():
    seen = collections.Counter()
    for q in draws():
        t = q['metadata']['type']
        if t == 'location_misreport':
            continue
        p0 = q['parts'][0]
        mech = classify(p0['question'])
        assert t.endswith(mech), (t, mech)
        assumptions_stated(mech, p0['question'])
        assert not NAMES.search(p0['question']), p0['question']
        assert p0['options'] and set(p0['options']) == {'Yes', 'No'}
        key = p0['options'][p0['correct'].index(True)]
        assert key == ('Yes' if TABLE[mech] else 'No'), (t, p0['question'])
        seen[key] += 1
    assert abs(seen['Yes'] - seen['No']) <= 0.1 * sum(seen.values()), seen


@pytest.mark.parametrize('mech', sorted(TABLE))
def test_brute_force(mech):
    """With the parameters of a sampled question of that mechanism: Yes iff no profitable deviation is found."""
    qs = [q for q in draws() if q['metadata']['type'] != 'location_misreport' and
          classify(q['parts'][0]['question']) == mech]
    assert qs
    rnd = random.Random(11)
    for q in qs[:3]:
        text = q['parts'][0]['question']
        n = WORDS[re.search(r'\b(three|four|five|six)\b', text).group(1)]
        if mech in ('serial', 'cycles', 'da_proposers', 'da_receivers'):
            n = 3  # every ranking of every agent is tried; 3 agents keep it small
        if mech == 'immediate':
            n = 4
        found = profitable_deviation(mech, n, params_of(text), rnd, tries=40 if TABLE[mech] else 400)
        assert found == (not TABLE[mech]), (mech, text[:200])


def test_followups():
    kinds = collections.Counter()
    for q in draws():
        if len(q['parts']) == 1:
            continue
        mech = classify(q['parts'][0]['question'])
        kw = {'reserve': params_of(q['parts'][0]['question']).get('reserve', 0)} if mech == 'reserve' else {}
        p1 = q['parts'][1]
        key = p1['options'][p1['correct'].index(True)]
        bids = [(a, int(b.replace(',', ''))) for a, b in re.findall(r'(\w+) bids \$([\d,]+)', p1['question'])]
        if 'payoff' in p1['question']:
            me = re.search(r'(\w+) values the item|(\w+), whose value', p1['question'])
            me = me.group(1) or me.group(2)
            v = int(re.search(r'(?:values the item at|value for the item is) \$([\d,]+)', p1['question']).group(1)
                    .replace(',', ''))
            own = int(re.search(rf'{me}[^.]*? bids \$([\d,]+)', p1['question']).group(1).replace(',', ''))
            other = re.search(r'The other bids are: ([^.]*)\.', p1['question']).group(1)
            vals = [own] + [int(b.replace(',', '')) for _, b in re.findall(r'(\w+) bids \$([\d,]+)', other)]
            u = auction_utility(mech, v, 0, vals, **kw)
            assert key == (f'-${-int(u):,}' if u < 0 else f'${int(u):,}'), (p1['question'], key)
            kinds['payoff'] += 1
            continue
        names, vals = [a for a, _ in bids], [b for _, b in bids]
        win, pay = auction(mech, vals, **kw)
        if 'Who wins' in p1['question'] or 'who wins' in p1['question']:
            w = win.index(1)
            assert key == f'{names[w]} wins and pays ${int(pay[w]):,}', (p1['question'], key)
            kinds['winner'] += 1
        else:
            assert key == f'${int(sum(pay)):,}', (p1['question'], key)
            kinds['revenue'] += 1
        assert len(set(p1['options'])) == 4
    assert set(kinds) == {'payoff', 'winner', 'revenue'}, kinds


def test_misreport_by_search():
    seen = collections.Counter()
    for q in draws():
        if q['metadata']['type'] != 'location_misreport':
            continue
        p = q['parts'][0]
        t = p['question']
        mech = 'median' if re.search(r'in the middle|middle one', t) else 'mean'
        spots = {a: int(b) for a, b in re.findall(r'(\w+) at (\d+)', t)}
        who = re.search(r'everyone except (\w+) reports', t).group(1)
        length = int(re.search(r'from 0 to (\d+)', t).group(1))
        names = list(spots)
        i = names.index(who)
        peaks = [spots[a] for a in names]
        base = abs(facility(mech, peaks) - peaks[i])
        dirs = set()
        for k in range(0, 100 * length + 1):
            rep = list(peaks)
            rep[i] = k / 100
            if abs(facility(mech, rep) - peaks[i]) < base - 1e-9:
                dirs.add('east' if rep[i] > peaks[i] else 'west')
        assert len(dirs) <= 1
        want = {'east': 'further east', 'west': 'further west'}.get(next(iter(dirs), None), 'No:')
        key = p['options'][p['correct'].index(True)]
        assert want in key, (t, key)
        assert len(p['options']) == 3
        seen[(mech, want)] += 1
    assert seen[('median', 'No:')] and seen[('mean', 'further east')] and seen[('mean', 'further west')], seen


def test_reproducible():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question('dsic_mechanism', seed)
        assert b.sample_question('dsic_mechanism', seed) == q
        assert a.sample_question('dsic_mechanism', seed, template_id=q['template_id']) == q
