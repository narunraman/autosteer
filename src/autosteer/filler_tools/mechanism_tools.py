"""Fillers for the regenerated STEER mechanism-design elements (paper elements 4.3.b, 4.3.c, 4.4.b).

ir_mechanism   -- individual rationality: which participant would be better off not participating?
budget_balance -- does the mechanism collect exactly the cost (balanced / surplus / deficit)?
auctions_risk  -- revenue ranking under risk attitudes; expected revenue and equilibrium bids with
                  uniform independent private values.

Design: STEER-generation's audit/DECISIONS.md, "Behalf-of-others redesign". Templates: data/templates/<element>/, built by
src/autosteer/filler_tools/mechanism_templates.py. All fillers are Mode A (they return options_list/correct_answer)
and draw only from filler_utils.rng (the question's own generator), so they are reproducible. Every amount shown is
a whole number of dollars; option order is shuffled by the sampler.
"""
from math import lcm

from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

NAMES = [
    'Ana', 'Bo', 'Cy', 'Dee', 'Eli', 'Fay', 'Gus', 'Hal', 'Ivy', 'Jo', 'Kai', 'Lena', 'Max', 'Nia',
    'Omar', 'Pia', 'Raj', 'Sol', 'Tess', 'Uma', 'Vic', 'Wen', 'Yara', 'Zed', 'Ines', 'Kofi', 'Mei',
    'Luis', 'Noor', 'Tomas', 'Aiko', 'Priya', 'Dmitri', 'Zara', 'Hugo', 'Leila', 'Femi', 'Rosa',
    'Sven', 'Amara',
]
NUM_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
SHARE_WORDS = {3: 'one-third', 4: 'one-quarter', 5: 'one-fifth'}

EVERYONE = 'Everyone is better off participating'
P_EVERYONE = 0.25  # share of ir_mechanism items whose answer is EVERYONE

IR_TYPES = ('equal_split', 'proportional', 'outside_option', 'vcg')
BB_TYPES = ('direct_transfers', 'fees', 'clarke')
AR_TYPES = ('risk_ranking', 'expected_revenue', 'equilibrium_bid')
# STEER grade levels (released: ir_mechanism 7, budget_balance 9, auctions_risk 10).
GRADES = {
    'ir_mechanism': {'equal_split': 6, 'proportional': 7, 'outside_option': 7, 'vcg': 8},
    'budget_balance': {'direct_transfers': 8, 'fees': 9, 'clarke': 10},
    'auctions_risk': {'risk_ranking': 10, 'equilibrium_bid': 10, 'expected_revenue': 11},
}


# ------------------------------------------------------------------ helpers
def _rng():
    return filler_utils.rng  # the generator bound for the question being filled


def _randint(a: int, b: int) -> int:
    """Uniform integer in [a, b]."""
    return int(_rng().integers(a, b + 1))


def _choice(seq):
    return seq[int(_rng().integers(len(seq)))]


def _coin(p: float) -> bool:
    return float(_rng().random()) < p


def _shuffled(seq) -> list:
    seq = list(seq)
    order = _rng().permutation(len(seq))
    return [seq[int(i)] for i in order]


def money(x: int) -> str:
    x = int(x)
    return f'-${-x:,}' if x < 0 else f'${x:,}'


def join_names(names: list) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def _names(n: int) -> list:
    idx = _rng().choice(len(NAMES), size=n, replace=False)
    return [NAMES[int(i)] for i in idx]


def _count_words(n: int) -> dict:
    w = NUM_WORDS[n]
    return {'n': w, 'n_cap': w.capitalize()}


def _value_list(names, values) -> str:
    return join_names([f'{a} {money(v)}' for a, v in zip(names, values)])


def _clarke_rule(n: int) -> str:
    return (f'Everyone reports their value truthfully, and a Clarke (pivotal) mechanism is used. '
            f"Each person's cost share is an equal {SHARE_WORDS[n]} of the cost. The project goes ahead if the "
            f'values minus the cost shares add up to zero or more, and then everyone pays their cost share. '
            f'A person is pivotal if leaving out both their value and their cost share would change the decision '
            f'(that is, if the decision based on everyone else\'s values minus cost shares alone would be '
            f'different); each pivotal person '
            f"also pays a Clarke tax equal to the absolute value of the sum, over everyone else, of value minus "
            f'cost share.')


def _clarke_taxes(r: list) -> list:
    """Clarke taxes given net reports r_i = v_i - share_i (build iff sum >= 0)."""
    total = sum(r)
    built = total >= 0
    taxes = []
    for ri in r:
        others = total - ri
        taxes.append(abs(others) if (others >= 0) != built else 0)
    return taxes


# ------------------------------------------------------------------ ir_mechanism
def _ir_equal(n: int, outside: int, everyone: bool, trap: bool):
    """Equal split; outside option o (0 = get nothing). Returns (cost, values, loser or None)."""
    s = 5 * _randint(4, 40)
    loser = None if everyone else _randint(0, n - 1)
    while True:
        vals = []
        for i in range(n):
            if i == loser:
                if outside and trap:  # positive surplus that is still below the outside option
                    d = 5 * _randint(1, outside // 5 - 1)
                else:
                    d = 5 * _randint(max(1, outside // 5 + 1) if outside else 1, min(16, (s + outside) // 5 - 1))
                vals.append(s + outside - d)
            else:
                vals.append(s + outside + 5 * _randint(1, 30))
        if len(set(vals)) == n:
            return n * s, vals, loser


def _ir_usage(n: int):
    """Proportional to usage points; exactly one loser. Prefer draws where equal split names someone else."""
    for attempt in range(40):
        w = [_randint(1, 6) for _ in range(n)]
        k = 5 * _randint(2, 10)
        pay = [k * wi for wi in w]
        cost = sum(pay)
        cand = [i for i in range(n) if pay[i] >= 15]
        if not cand:
            continue
        loser = _choice(cand)
        vals = []
        for i in range(n):
            if i == loser:
                vals.append(pay[i] - 5 * _randint(1, min(12, pay[i] // 5 - 1)))
            else:
                vals.append(pay[i] + 5 * _randint(1, 24))
        if len(set(vals)) < n:
            continue
        eq_losers = [i for i in range(n) if vals[i] < cost / n]
        if eq_losers != [loser] or attempt == 39:
            return cost, vals, loser, w
    raise RuntimeError('unreachable')


def _ir_prop_value(n: int):
    """Proportional to value: individually rational whenever the values cover the cost."""
    while True:
        vals = [5 * _randint(4, 50) for _ in range(n)]
        if len(set(vals)) < n:
            continue
        total = sum(vals)
        cost = 10 * ((total * _randint(45, 90) // 100) // 10)
        if cost < 50:
            continue
        # net_i = v_i * (1 - cost/total) >= 5 for everyone
        if min(vals) * (total - cost) < 5 * total:
            continue
        # prefer a trap: someone below the equal share
        if min(vals) < cost / n or _coin(0.2):
            return cost, vals


def _ir_vcg(n: int, everyone: bool):
    """Clarke mechanism with equal shares. Returns (cost, values, loser or None)."""
    while True:
        s = 5 * _randint(6, 30)
        if everyone:
            r = [5 * _randint(1, 16) for _ in range(n)]
            loser = None
        else:
            loser = _randint(0, n - 1)
            rl = -5 * _randint(1, min(14, s // 5 - 1))
            total = 5 * _randint(1, 6) if _coin(0.75) else 5 * _randint(7, 30)  # small totals -> pivots
            rest = total - rl  # sum of the others' r, each >= 5
            k = rest // 5
            if k < n - 1:
                continue
            cuts = sorted(int(c) for c in _rng().choice(range(1, k), size=n - 2, replace=False)) if n > 2 else []
            parts = [b - a for a, b in zip([0] + cuts, cuts + [k])]
            others = [5 * p for p in parts]
            r = others[:loser] + [rl] + others[loser:]
        total = sum(r)
        if total < 5 or any(total - ri == 0 for ri in r):
            continue
        vals = [s + ri for ri in r]
        if len(set(vals)) < n or min(vals) < 5:
            continue
        return n * s, vals, loser


@function_registry.register('ir_mechanism')
def fill_ir_mechanism(metadata: dict) -> dict:
    t = metadata['type']
    n = _randint(3, 5)
    names = _names(n)
    everyone = _coin(P_EVERYONE)
    out = {}
    if t == 'equal_split':
        cost, vals, loser = _ir_equal(n, 0, everyone, False)
        rule = _choice([
            f'The cost is split equally, so each of the {NUM_WORDS[n]} pays the same share.',
            'Everyone pays an equal share of the total cost.',
            f'Each person pays an equal {SHARE_WORDS[n]} share of the cost.',
        ])
    elif t == 'outside_option':
        o = 5 * _randint(2, 12)
        cost, vals, loser = _ir_equal(n, o, everyone, trap=_coin(0.5))
        out['outside_option'] = money(o)
        rule = _choice([
            f'The cost is split equally, so each of the {NUM_WORDS[n]} pays the same share.',
            'Everyone pays an equal share of the total cost.',
            f'Each person pays an equal {SHARE_WORDS[n]} share of the cost.',
        ])
    elif t == 'proportional':
        if everyone:
            cost, vals = _ir_prop_value(n)
            loser = None
            rule = ('Everyone reports their value truthfully, and each person pays a share of the cost '
                    'proportional to their value (their value divided by the total of all values, times the cost).')
        else:
            cost, vals, loser, w = _ir_usage(n)
            pts = ', '.join(f'{a}: {wi} point{"s" if wi > 1 else ""}' for a, wi in zip(names, w))
            rule = f'Each person pays a share of the cost proportional to their usage points ({pts}).'
    elif t == 'vcg':
        cost, vals, loser = _ir_vcg(n, everyone)
        rule = _clarke_rule(n)
    else:
        raise ValueError(f'ir_mechanism: unknown type {t!r}')
    out.update(_count_words(n))
    out.update({'cost': money(cost), 'value_list': _value_list(names, vals), 'rule_text': rule})
    opts = list(names) + [EVERYONE]
    ans = [int(loser is not None and i == loser) for i in range(n)] + [int(loser is None)]
    out['options_list'] = [opts]
    out['correct_answer'] = [ans]
    return out


# ------------------------------------------------------------------ budget_balance
DEFINITIONS = {
    'exact': [
        '(Here a mechanism counts as budget balanced only if it collects exactly the cost: no surplus and no deficit.)',
        '(In this question, "budget balanced" means the money collected equals the cost exactly.)',
    ],
    'no_deficit': [
        '(Here a mechanism counts as budget balanced if it never runs a deficit; collecting more than the cost is allowed.)',
        '(In this question, "budget balanced" means weakly balanced: the money collected is at least the cost.)',
    ],
}
FEE_NAMES = ['processing fee', 'late sign-up fee', 'administration fee', 'membership fee', 'booking fee']
CREDIT_NAMES = ['volunteer credit', 'hardship rebate', 'early-bird rebate', 'loyalty credit', 'setup-help credit']


def bb_option(delta: int, definition: str) -> str:
    if delta == 0:
        return 'Budget balanced: it collects exactly the cost'
    if delta > 0:
        label = 'Budget balanced' if definition == 'no_deficit' else 'Not budget balanced'
        return f'{label}: it collects a surplus of {money(delta)}'
    return f'Not budget balanced: it runs a deficit of {money(-delta)}'


def _bb_options(delta: int, slips: list, definition: str) -> tuple:
    """Correct option + 3 distractors (slip deltas first, then the other outcome categories)."""
    cand = list(slips) + ([-delta, 0] if delta else [])
    extra = [5 * _randint(1, 20) * _choice([-1, 1]) for _ in range(30)]
    chosen = [delta]
    for d in _shuffled(cand) + extra:
        if d not in chosen:
            chosen.append(d)
        if len(chosen) == 4:
            break
    # ensure every outcome category (balanced / surplus / deficit) is on offer
    kinds = {(d > 0) - (d < 0) for d in chosen}
    for need in (0, 1, -1):
        if need not in kinds:
            repl = 0 if need == 0 else need * 5 * _randint(1, 20)
            for j in range(3, 0, -1):
                kj = (chosen[j] > 0) - (chosen[j] < 0)
                if sum(((d > 0) - (d < 0)) == kj for d in chosen) > 1 and repl not in chosen:
                    chosen[j] = repl
                    kinds = {(d > 0) - (d < 0) for d in chosen}
                    break
    opts = [bb_option(d, definition) for d in chosen]
    return opts, [1, 0, 0, 0]


def _bb_direct(n: int, names: list, outcome: int):
    for _ in range(200):
        cost = 10 * _randint(20, 120)
        n_reb = _randint(0, 2) if n > 3 else _randint(0, 1)
        payers = n - n_reb
        rebates = [5 * _randint(1, 8) for _ in range(n_reb)]
        target = cost + outcome * 5 * _randint(1, 16) + sum(rebates)  # total paid in
        base = target // payers
        pays = [5 * round((base + 5 * _randint(-8, 8)) / 5) for _ in range(payers - 1)]
        last = target - sum(pays)
        if last < max(20, base // 2) or last > 2 * base or last % 5:
            continue
        pays.append(last)
        if len(set(pays)) < 2 or min(pays) < 5:   # every payer pays something (rebates are the ones paid out)
            continue
        delta = sum(pays) - sum(rebates) - cost
        slips = [sum(pays) - cost, sum(pays) + sum(rebates) - cost, delta - pays[0]]
        clauses = [f'{a} pays {money(p)}' for a, p in zip(names[:payers], pays)]
        text = 'Under the proposed payment plan, ' + join_names(clauses)
        if rebates:
            text += ', while ' + join_names([f'{a} receives a rebate of {money(b)}'
                                              for a, b in zip(names[payers:], rebates)]) + ' from the fund'
        return cost, text + '.', delta, slips
    raise RuntimeError('budget_balance direct_transfers: no draw')


def _bb_fees(n: int, names: list, outcome: int):
    for _ in range(200):
        cost = n * 5 * _randint(8, 60)
        k_fee = _randint(1, n - 1)
        k_cred = _randint(1, n - k_fee)
        f = 5 * _randint(1, 8)
        c = 5 * _randint(1, 8)
        if outcome == 0:
            # k_fee * f == k_cred * c
            g = lcm(k_fee, k_cred)
            unit = 5 * _randint(1, 3)
            f, c = g * unit // k_fee, g * unit // k_cred
            if f == c and _coin(0.5):
                continue
        elif outcome > 0 and _coin(0.3):
            k_cred = 0
        elif outcome < 0 and _coin(0.3):
            k_fee = 0
        delta = k_fee * f - k_cred * c
        if (delta > 0) - (delta < 0) != outcome:
            continue
        order = _shuffled(range(n))
        fee_who = [names[i] for i in sorted(order[:k_fee])]
        cred_who = [names[i] for i in sorted(order[k_fee:k_fee + k_cred])]
        parts = ['Each person pays an equal share of the cost.']
        if k_fee:
            fee = _choice(FEE_NAMES)
            verb = 'each pay' if k_fee > 1 else 'pays'
            parts.append(f'In addition, {join_names(fee_who)} {verb} a {money(f)} {fee} into the same fund.')
        if k_cred:
            cr = _choice(CREDIT_NAMES)
            verb = 'each receive' if k_cred > 1 else 'receives'
            lead = 'Also' if k_fee else 'In addition'
            parts.append(f'{lead}, {join_names(cred_who)} {verb} a {money(c)} {cr} paid out of the fund.')
        slips = [-k_cred * c, k_fee * f, f * bool(k_fee) - k_cred * c, k_fee * f + k_cred * c,
                 k_fee * f - c * bool(k_cred)]
        return cost, ' '.join(parts), delta, slips
    raise RuntimeError('budget_balance fees: no draw')


def _bb_clarke(n: int, names: list, outcome: int):
    """outcome 0: nobody pivotal (exactly balanced); outcome 1: someone pivotal (surplus)."""
    for _ in range(500):
        s = 5 * _randint(6, 40)
        r = [5 * _randint(-10, 16) for _ in range(n)]
        total = sum(r)
        if total < 5 or any(total - ri == 0 for ri in r) or any(ri == 0 for ri in r):
            continue
        vals = [s + ri for ri in r]
        if min(vals) < 5 or len(set(vals)) < n:
            continue
        taxes = _clarke_taxes(r)
        delta = sum(taxes)
        if (delta > 0) != bool(outcome):
            continue
        others = [abs(total - ri) for ri in r]
        non_piv = [others[i] for i in range(n) if taxes[i] == 0]
        slips = [0, -delta] + [delta + x for x in non_piv[:2]] + [sum(others)]
        text = f'Values: {_value_list(names, vals)}. ' + _clarke_rule(n) \
            + ' Clarke taxes are kept by the mechanism, not refunded.'
        return n * s, text, delta, slips
    raise RuntimeError('budget_balance clarke: no draw')


@function_registry.register('budget_balance')
def fill_budget_balance(metadata: dict) -> dict:
    t = metadata['type']
    n = _randint(3, 5)
    names = _names(n)
    definition = _choice(['exact', 'no_deficit'])
    if t == 'direct_transfers':
        cost, text, delta, slips = _bb_direct(n, names, _choice([-1, 0, 1]))
    elif t == 'fees':
        cost, text, delta, slips = _bb_fees(n, names, _choice([-1, 0, 1]))
    elif t == 'clarke':
        cost, text, delta, slips = _bb_clarke(n, names, _choice([0, 1]))
    else:
        raise ValueError(f'budget_balance: unknown type {t!r}')
    opts, ans = _bb_options(delta, slips, definition)
    out = _count_words(n)
    out.update({'cost': money(cost), 'mechanism_text': text,
                'definition_text': _choice(DEFINITIONS[definition]),
                'options_list': [opts], 'correct_answer': [ans]})
    return out


# ------------------------------------------------------------------ auctions_risk
FIRST_CLASS = ['a first-price sealed-bid auction', 'a Dutch (descending-price) auction']
SECOND_CLASS = ['a second-price sealed-bid (Vickrey) auction', 'an English (ascending-price) auction']
STANDARD_FORMATS = FIRST_CLASS + SECOND_CLASS
SAME_REVENUE = 'Both give the same expected revenue'
ATTITUDES = {
    'averse': ['All bidders are risk-averse.',
               'The bidders are risk-averse: each prefers a sure payoff to a gamble with the same expected value.'],
    'neutral': ['All bidders are risk-neutral.',
                'The bidders are risk-neutral: they care only about their expected payoff.'],
    'seeking': ['All bidders are risk-seeking.',
                'The bidders are risk-seeking: each prefers a gamble to a sure payoff with the same expected value.'],
}


def format_option(fmt: str) -> str:
    """'a first-price sealed-bid auction' -> 'The first-price sealed-bid auction'."""
    return 'The ' + fmt.split(' ', 1)[1]


def _ar_options(correct: int, pool: list) -> tuple:
    chosen = [correct]
    for x in _shuffled(pool):
        if x not in chosen and x > 0:
            chosen.append(x)
        if len(chosen) == 4:
            break
    if len(chosen) < 4:
        raise RuntimeError('auctions_risk: not enough distinct distractors')
    return [money(x) for x in chosen], [1, 0, 0, 0]


@function_registry.register('auctions_risk')
def fill_auctions_risk(metadata: dict) -> dict:
    t = metadata['type']
    out = {}
    if t == 'risk_ranking':
        n = _randint(2, 6)
        V = 10 * _randint(5, 300)
        att = _choice(['averse', 'neutral', 'seeking'])
        a, b = _choice(FIRST_CLASS), _choice(SECOND_CLASS)
        if _coin(0.5):
            a, b = b, a
        winner = {'averse': 'first', 'neutral': 'same', 'seeking': 'second'}[att]
        first = a if a in FIRST_CLASS else b
        second = b if first is a else a
        opts = [format_option(first), format_option(second), SAME_REVENUE]
        ans = [int(winner == 'first'), int(winner == 'second'), int(winner == 'same')]
        out.update({'format_a': a, 'format_b': b, 'attitude_text': _choice(ATTITUDES[att])})
    elif t == 'expected_revenue':
        n = _randint(3, 7)
        step = lcm(n, n + 1)
        V = step * _randint(max(1, -(-60 // step)), max(2, 3000 // step))
        rev = V * (n - 1) // (n + 1)
        pool = [V * n // (n + 1), V // (n + 1), V * (n - 1) // n]
        opts, ans = _ar_options(rev, pool)
        out['format'] = _choice(STANDARD_FORMATS)
    elif t == 'equilibrium_bid':
        n = _randint(2, 6)
        step = lcm(2, n, n + 1)
        v = step * _randint(max(1, -(-30 // step)), max(2, 1500 // step))
        V = n * _randint(-(-(v + 10) // n), max(-(-(v + 10) // n), 2 * v // n + 10))
        bid = v * (n - 1) // n
        pool = [v, v // n, v * (n - 1) // (n + 1), v * n // (n + 1), v // 2, V * (n - 1) // n]
        opts, ans = _ar_options(bid, pool)
        out['v'] = money(v)
    else:
        raise ValueError(f'auctions_risk: unknown type {t!r}')
    out.update(_count_words(n))
    out['V'] = money(V)
    out['options_list'] = [opts]
    out['correct_answer'] = [ans]
    return out


# dsic_mechanism: rebuilt with code templates in dsic_tools.py (October 2026 cleanup).
