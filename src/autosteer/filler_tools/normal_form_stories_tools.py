"""pure_nash and dominant_strategies: payoff tables generated first, then shown as a matrix or told as a story.

Decisions of 3 October 2026. The old word-problem fillers drew story numbers and hoped the game had the intended
equilibrium; the objective (units sold or revenue) was ambiguous, some draws had no dominant strategy, and the
salary stories differed in what a hire is worth. Here every game is generated as a payoff table, checked by brute
force, and the question text states every payoff-relevant number and the objective.

pure_nash
    matrix_form (types row / column, difficulty 9): the released wording "Below, find the payoff matrix for some
        game. You are the row player, ... what action should you play in a pure Nash equilibrium?" with a 3x3
        bimatrix, integers 0-19 or 2-decimal numbers in [0, 20). Asks: your action, your opponent's, the action pair.
    finance (types symmetric / non-symmetric, difficulty 8): two firms each charge a low or a high price; the story
        lists the units each firm sells in all four price combinations; profit = price x units (costs negligible).
    job_market (type symmetric, difficulty 8): two employers each offer a low or a high salary; equal offers split
        the candidates evenly, otherwise the higher offer hires a stated share; profit = hires x (value per hire -
        salary).
    Every game has exactly one pure Nash equilibrium (pure_equilibria, weak best responses), and at it both best
    responses are strict. The asked equilibrium (both high, both low, or for non-symmetric pricing also the mixed
    pairs) is drawn uniformly before the game, which is redrawn until it has that unique equilibrium.

dominant_strategies
    finance (symmetric / non-symmetric, difficulty 3): the pricing story above; the asked firm has a strictly
        dominant price under the stated objective (its own profit = price x units sold).
    job_market (symmetric, difficulty 2): two employees each negotiate or accept an offer; the story lists the raise
        each gets in all four combinations and says each wants the largest raise for themselves; the asked person
        has a strictly dominant action.
    The dominant action (high / low price, negotiate / accept) is drawn uniformly before the game.

All randomness comes from filler_utils.rng. Money is shown as $x.x5 (prices) or whole dollars, so the sampler's
2-decimal number formatting leaves it unchanged.
"""
from __future__ import annotations

import numpy as np

from . import code_templates as ct
from .fill_registry import FillRegister
from .normal_form_tools import COLS, ROWS, fmt, payoffs, table
from .utilities import filler_utils

function_registry = FillRegister()

NO_NE = 'There is no pure-strategy Nash equilibrium'
LOW, HIGH = 0, 1


def _rng():
    return filler_utils.rng


def pure_equilibria(A, B) -> list[tuple[int, int]]:
    """Pure Nash equilibria (row i, column j) of the bimatrix game (A, B), weak best responses."""
    A, B = np.asarray(A), np.asarray(B)
    return [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] >= A[:, j].max() and B[i, j] >= B[i, :].max()]


def strict_at(A, B, i, j) -> bool:
    """Both players' best responses at (i, j) are strict."""
    A, B = np.asarray(A), np.asarray(B)
    return all(A[i, j] > A[k, j] for k in range(A.shape[0]) if k != i) and \
        all(B[i, j] > B[i, k] for k in range(B.shape[1]) if k != j)


def strictly_dominant_rows(U) -> list[int]:
    """Actions (rows of U, the chooser's own actions) that strictly beat every other action against every
    action of the other player (columns)."""
    U = np.asarray(U)
    return [a for a in range(U.shape[0]) if all((U[a] > U[b]).all() for b in range(U.shape[0]) if b != a)]


def money(cents: int) -> str:
    """$4.25 / $85,000 from an integer number of cents."""
    d, c = divmod(int(cents), 100)
    return f'${d:,}' if c == 0 else f'${d:,}.{c:02d}'


# ------------------------------------------------------------------------------------------------------- scenarios
# Pricing duopolies. a, b = firm names; kind = what they are (plural); place (with its preposition); good / goods = unit sold;
# period; lo / hi = price range in cents (prices end in 5 cents); qlo / qhi = units range.
PRICING = [
    dict(a='Java Junction', b='Bean Street', kind='coffee shops', place='in a small college town', good='cup of coffee',
         goods='cups', period='day', lo=195, hi=595, qlo=120, qhi=900),
    dict(a='Page Turners', b='Literati Lodge', kind='bookstores', place='in a lakeside town',
         good='copy of a new bestselling novel', goods='copies', period='week', lo=995, hi=2495, qlo=60, qhi=600),
    dict(a='Pizza Piazza', b='Slice Palace', kind='pizzerias', place='on a busy street near the stadium',
         good='large pizza', goods='pizzas', period='evening', lo=1095, hi=2295, qlo=40, qhi=400),
    dict(a='Sparkle Wash', b='Shine Station', kind='car washes', place='at a suburban highway exit', good='car wash',
         goods='washes', period='week', lo=695, hi=1995, qlo=100, qhi=1200),
    dict(a='Taco Loco', b='El Fuego', kind='food trucks', place='serving the downtown business district at lunchtime',
         good='taco plate', goods='plates', period='day', lo=595, hi=1495, qlo=50, qhi=500),
    dict(a='FixIt Phones', b='Screen Savers', kind='phone repair shops', place='in a mid-sized city',
         good='screen repair', goods='repairs', period='month', lo=4995, hi=12995, qlo=40, qhi=420),
    dict(a='Golden Crust', b='Daily Loaf', kind='bakeries', place='in a mountain village', good='loaf of sourdough',
         goods='loaves', period='week', lo=395, hi=895, qlo=150, qhi=1400),
    dict(a='SkyHop', b='AeroLink', kind='airlines', place='flying between two regional cities',
         good='one-way ticket', goods='tickets', period='month', lo=8995, hi=29995, qlo=300, qhi=3000),
]

# Hiring duopolies (pure_nash, job_market). employers / workers plural, worker singular.
HIRING = [
    dict(a='CodeRift', b='DevSync', employers='software firms', place='a fast-growing tech hub',
         workers='software developers', worker='developer', slo=80_000, shi=150_000),
    dict(a='Riverside General', b='St. Clare Medical', employers='hospitals', place='a mid-sized city',
         workers='nurses', worker='nurse', slo=55_000, shi=100_000),
    dict(a='Hale & Pierce', b='Morrow Legal', employers='law firms', place='the state capital',
         workers='newly qualified lawyers', worker='lawyer', slo=70_000, shi=140_000),
    dict(a='Northwind Logistics', b='Bluepeak Freight', employers='trucking companies', place='a port city',
         workers='truck drivers', worker='driver', slo=45_000, shi=90_000),
    dict(a='Brightpath Schools', b='Oakridge Academies', employers='private school groups', place='a large suburb',
         workers='qualified teachers', worker='teacher', slo=40_000, shi=80_000),
]

# Negotiation pairs (dominant_strategies, job_market): (name, name, tags, setting, initial offer noun)
NEGOTIATION = [
    ('John', 'Alice', 'explicit;man_woman', 'a law firm', 'the firm'),
    ('Maria', 'Priya', 'explicit;two_women', 'a hospital', 'the hospital'),
    ('Michael', 'David', 'explicit;two_men', 'a software company', 'the company'),
    ('Sofia', 'Daniel', 'explicit;woman_man', 'an engineering consultancy', 'the consultancy'),
    ('Kenji', 'Omar', 'explicit;two_men', 'a bank', 'the bank'),
    ('Grace', 'Hannah', 'explicit;two_women', 'a university', 'the university'),
]


def _pricing_story(sc: dict, symmetric: bool) -> str:
    a, b, g = sc['a'], sc['b'], sc['goods']
    text = (f"{a} and {b} are the only two {sc['kind']} {sc['place']}. Each can charge either {{price_low}} or "
            f"{{price_high}} per {sc['good']}.")
    if not symmetric:
        text += f" The two differ in location and reputation, so they do not sell the same amounts."
    text += (f" The number of {g} each sells per {sc['period']} depends on both prices."
             f" If both charge {{price_high}}, {a} sells {{a_hh}} {g} and {b} sells {{b_hh}}."
             f" If {a} charges {{price_high}} and {b} charges {{price_low}}, {a} sells {{a_hl}} and {b} sells {{b_hl}}."
             f" If {a} charges {{price_low}} and {b} charges {{price_high}}, {a} sells {{a_lh}} and {b} sells {{b_lh}}."
             f" If both charge {{price_low}}, {a} sells {{a_ll}} and {b} sells {{b_ll}}."
             f" Costs are negligible, so each one's profit is its price times the number of {g} it sells,"
             f" and each wants to maximise its own profit.")
    return text


def _hiring_story(sc: dict) -> str:
    a, b, w = sc['a'], sc['b'], sc['workers']
    return (f"{a} and {b} are the only two {sc['employers']} hiring in {sc['place']}, and {{num_workers}} {w} are "
            f"looking for jobs. Each will offer either {{salary_low}} or {{salary_high}} a year. If both offer the "
            f"same salary, the {w} split evenly between them. If one offers more, it hires {{share_high}} of the {w} "
            f"and the other hires the remaining {{share_low}}. Each {sc['worker']} hired brings in {{value}} a year, so "
            f"an employer's annual profit is the number it hires times the difference between {{value}} and the "
            f"salary it pays. Each wants to maximise its own profit.")


def _negotiation_story(x: str, y: str, setting: str, employer: str) -> str:
    return (f"{x} and {y} have just been offered jobs at {setting}. Each can either negotiate the offer or accept it "
            f"as it is, and {employer} decides the raises from what both of them do. If both negotiate, each gets a "
            f"raise of {{both_negotiate}}. If only one negotiates, the one who negotiates gets a raise of "
            f"{{one_negotiate}} and the one who accepts gets {{accepts_alone}}. If both accept, each gets a raise "
            f"of {{both_accept}}. Each wants the largest raise for themselves and does not care about the other's.")


MATRIX_INTRO = ('Below, find the payoff matrix for some game. You are the {side} player, which means the {nth} number '
                'in the cell is your payoff, {ask}\n\n{{payoff_matrix}}')
MATRIX_ASKS = {
    'self': 'what action should you play in a pure Nash equilibrium?',
    'opponent': 'what action should your opponent play in a pure Nash equilibrium?',
    'pair': 'what is the action pair that you and your opponent play in a pure Nash equilibrium?',
}


def _nash_templates() -> list[ct.CodeTemplate]:
    out = []
    for side, nth in (('row', 'first'), ('column', 'second')):
        for ask, q in MATRIX_ASKS.items():
            out.append(ct.CodeTemplate(
                parts=(MATRIX_INTRO.format(side=side, nth=nth, ask=q),), type=side, domain='matrix_form',
                tags=ask, difficulty_level=9, spec={'kind': 'matrix', 'side': side, 'ask': ask}))
    for k, sc in enumerate(PRICING):
        a, b = sc['a'], sc['b']
        for symmetric in (True, False):
            asks = [('pair', None), ('firm', 'A')] + ([] if symmetric else [('firm', 'B')])
            for ask, firm in asks:
                if ask == 'pair':
                    q = f' What is the pure-strategy Nash equilibrium of this pricing game between {a} and {b}?'
                else:
                    name = a if firm == 'A' else b
                    q = f' In the pure-strategy Nash equilibrium of this pricing game, what price does {name} charge?'
                out.append(ct.CodeTemplate(
                    parts=(_pricing_story(sc, symmetric) + q,), type='symmetric' if symmetric else 'non-symmetric',
                    domain='finance', tags=f'explicit;{ask}', difficulty_level=8,
                    spec={'kind': 'pricing', 'scenario': k, 'symmetric': symmetric, 'ask': ask, 'firm': firm}))
    for k, sc in enumerate(HIRING):
        for ask in ('pair', 'firm'):
            if ask == 'pair':
                q = f" What is the pure-strategy Nash equilibrium of this salary game between {sc['a']} and {sc['b']}?"
            else:
                q = f" In the pure-strategy Nash equilibrium of this salary game, what salary does {sc['a']} offer?"
            out.append(ct.CodeTemplate(
                parts=(_hiring_story(sc) + q,), type='symmetric', domain='job_market', tags=f'explicit;{ask}',
                difficulty_level=8, spec={'kind': 'hiring', 'scenario': k, 'ask': ask, 'firm': 'A'}))
    return out


def _dominant_templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(PRICING):
        for symmetric in (True, False):
            for firm in (('A',) if symmetric else ('A', 'B')):
                name, other = (sc['a'], sc['b']) if firm == 'A' else (sc['b'], sc['a'])
                for wording in (0, 1):
                    if wording == 0:
                        q = f' What is the dominant strategy for {name}?'
                    else:
                        q = (f' Which price should {name} charge to maximise its profit, whatever price {other} '
                             f'charges?')
                    out.append(ct.CodeTemplate(
                        parts=(_pricing_story(sc, symmetric) + q,),
                        type='symmetric' if symmetric else 'non-symmetric', domain='finance',
                        tags=f'explicit;{firm.lower()}_firm', difficulty_level=3,
                        spec={'kind': 'pricing', 'scenario': k, 'symmetric': symmetric, 'firm': firm}))
    for k, (x, y, tags, setting, employer) in enumerate(NEGOTIATION):
        for wording in (0, 1):
            q = (f' What is the dominant strategy for {x}?' if wording == 0 else
                 f' What should {x} do to get the largest raise, whatever {y} does?')
            out.append(ct.CodeTemplate(
                parts=(_negotiation_story(x, y, setting, employer) + q,), type='symmetric', domain='job_market',
                tags=tags, difficulty_level=2, spec={'kind': 'negotiation', 'scenario': k}))
    return out


# --------------------------------------------------------------------------------------------------------- games
def pricing_game(sc: dict, symmetric: bool):
    """Prices (cents) and units sold. q_a[x][y] = A's units when A charges x and B charges y (0 = low, 1 = high);
    q_b likewise for B (indexed by A's price, then B's). Each firm sells more when it is cheaper or its rival dearer,
    and more when both are cheap than when both are dear: own low / rival high > both low > both high > own high /
    rival low."""
    r = _rng()
    while True:
        lo, hi = sorted(int(v) for v in r.integers(sc['lo'] // 10, sc['hi'] // 10 + 1, size=2))
        if hi - lo >= 2:
            break
    prices = (lo * 10 + 5, hi * 10 + 5)

    def own_table():
        while True:
            v = sorted(int(x) for x in r.integers(sc['qlo'], sc['qhi'] + 1, size=4))
            if len(set(v)) == 4:
                break
        t = np.zeros((2, 2), dtype=int)  # t[own][rival]
        t[LOW, HIGH], t[LOW, LOW], t[HIGH, HIGH], t[HIGH, LOW] = v[3], v[2], v[1], v[0]
        return t
    ta = own_table()
    tb = ta.copy() if symmetric else own_table()
    q_a = ta                      # indexed [A's price][B's price]
    q_b = tb.T                    # tb[B's price][A's price] -> [A's price][B's price]
    P = np.array(prices)
    profit_a = P[:, None] * q_a   # rows: A's price
    profit_b = P[None, :] * q_b   # columns: B's price
    return prices, q_a, q_b, profit_a, profit_b


def pricing_values(prices, q_a, q_b) -> dict:
    names = {(1, 1): 'hh', (1, 0): 'hl', (0, 1): 'lh', (0, 0): 'll'}
    vals = {'price_low': money(prices[0]), 'price_high': money(prices[1])}
    for (x, y), s in names.items():
        vals[f'a_{s}'] = int(q_a[x, y])
        vals[f'b_{s}'] = int(q_b[x, y])
    return vals


def hiring_game(sc: dict):
    """Salaries, value per hire (whole dollars), workers N (multiple of 100), high share h (%).
    U[own][other] = hires x (value - own salary), the same table for both employers."""
    r = _rng()
    while True:
        s_lo, s_hi = sorted(int(v) * 1000 for v in r.integers(sc['slo'] // 1000, sc['shi'] // 1000 + 1, size=2))
        if s_hi - s_lo < 5000:
            continue
        value = int(r.integers(s_hi // 1000 + 5, int(s_hi * 1.8) // 1000 + 1)) * 1000
        n = int(r.integers(10, 51)) * 100
        h = int(r.integers(55, 91))
        break
    m_lo, m_hi = value - s_lo, value - s_hi
    U = np.array([[n // 2 * m_lo, n * (100 - h) // 100 * m_lo],
                  [n * h // 100 * m_hi, n // 2 * m_hi]])
    return (s_lo, s_hi), value, n, h, U


def negotiation_game():
    """Raises in whole dollars: U[own][other] with 0 = accept, 1 = negotiate."""
    r = _rng()
    v = [int(x) * 100 for x in r.integers(5, 100, size=4)]
    both_neg, one_neg, accepts_alone, both_acc = v
    U = np.array([[both_acc, accepts_alone], [one_neg, both_neg]])
    return U, dict(both_negotiate=money(both_neg * 100), one_negotiate=money(one_neg * 100),
                   accepts_alone=money(accepts_alone * 100), both_accept=money(both_acc * 100))


# --------------------------------------------------------------------------------------------------------- fillers
def _price_option(name, cents):
    return f'{name} charges {money(cents)}'


def _pair_options(a, b, low, high, verb):
    return {
        (1, 1): f'Both {verb} {high}', (0, 0): f'Both {verb} {low}',
        (1, 0): f'{a} {verb}s {high} and {b} {verb}s {low}', (0, 1): f'{a} {verb}s {low} and {b} {verb}s {high}',
    }


def _mask(options, key):
    return [[int(o == key) for o in options]]


def _fill_matrix(spec):
    r = _rng()
    integer = bool(r.random() < 0.5)
    while True:
        A, B = payoffs(3, 3, 20, integer), payoffs(3, 3, 20, integer)
        ne = pure_equilibria(A, B)
        if len(ne) == 1 and strict_at(A, B, *ne[0]):
            break
    i, j = ne[0]
    me = 'row' if spec['side'] == 'row' else 'column'
    if spec['ask'] == 'pair':
        options = [f'({ROWS[x]}, {COLS[y]})' for x in range(3) for y in range(3)]
        key = f'({ROWS[i]}, {COLS[j]})'
    else:
        asked = me if spec['ask'] == 'self' else ('column' if me == 'row' else 'row')
        options = list(ROWS) if asked == 'row' else list(COLS)
        key = ROWS[i] if asked == 'row' else COLS[j]
    return {'payoff_matrix': table(A, B, ROWS, COLS), 'options_list': [options], 'correct_answer': _mask(options, key)}


def _fill_pricing_nash(spec):
    sc = PRICING[spec['scenario']]
    r = _rng()
    targets = [(1, 1), (0, 0)] if spec['symmetric'] else [(1, 1), (0, 0), (1, 0), (0, 1)]
    target = targets[int(r.integers(0, len(targets)))]
    while True:
        prices, q_a, q_b, pa, pb = pricing_game(sc, spec['symmetric'])
        ne = pure_equilibria(pa, pb)
        if ne == [target] and strict_at(pa, pb, *target):
            break
    low, high = money(prices[0]), money(prices[1])
    if spec['ask'] == 'pair':
        opts = _pair_options(sc['a'], sc['b'], low, high, 'charge')
        options = [opts[k] for k in ((1, 1), (0, 0), (1, 0), (0, 1))] + [NO_NE]
        key = opts[target]
    else:
        name = sc['a'] if spec['firm'] == 'A' else sc['b']
        mine = target[0] if spec['firm'] == 'A' else target[1]
        options = [_price_option(name, prices[0]), _price_option(name, prices[1]), NO_NE]
        key = options[mine]
    return {**pricing_values(prices, q_a, q_b), 'options_list': [options], 'correct_answer': _mask(options, key)}


def _fill_hiring_nash(spec):
    sc = HIRING[spec['scenario']]
    r = _rng()
    target = [(1, 1), (0, 0)][int(r.integers(0, 2))]
    while True:
        (s_lo, s_hi), value, n, h, U = hiring_game(sc)
        ne = pure_equilibria(U, U.T)
        if ne == [target] and strict_at(U, U.T, *target):
            break
    low, high = money(s_lo * 100), money(s_hi * 100)
    if spec['ask'] == 'pair':
        opts = _pair_options(sc['a'], sc['b'], low, high, 'offer')
        options = [opts[k] for k in ((1, 1), (0, 0), (1, 0), (0, 1))] + [NO_NE]
        key = opts[target]
    else:
        options = [f"{sc['a']} offers {low}", f"{sc['a']} offers {high}", NO_NE]
        key = options[target[0]]
    return {'num_workers': f'{n:,}', 'salary_low': low, 'salary_high': high, 'share_high': f'{h}%',
            'share_low': f'{100 - h}%', 'value': money(value * 100),
            'options_list': [options], 'correct_answer': _mask(options, key)}


@function_registry.register('pure_nash')
def fill_pure_nash(metadata: dict) -> dict:
    spec = ct.spec_for('pure_nash', metadata['question_text'])
    if spec['kind'] == 'matrix':
        return _fill_matrix(spec)
    if spec['kind'] == 'pricing':
        return _fill_pricing_nash(spec)
    return _fill_hiring_nash(spec)


@function_registry.register('dominant_strategies')
def fill_dominant_strategies(metadata: dict) -> dict:
    spec = ct.spec_for('dominant_strategies', metadata['question_text'])
    r = _rng()
    want = int(r.integers(0, 2))  # pricing: 0 = low, 1 = high; negotiation: 0 = accept, 1 = negotiate
    if spec['kind'] == 'negotiation':
        x = NEGOTIATION[spec['scenario']][0]
        while True:
            U, vals = negotiation_game()
            if strictly_dominant_rows(U) == [want]:
                break
        options = [f'{x} should accept the offer as it is', f'{x} should negotiate']
        return {**vals, 'options_list': [options], 'correct_answer': _mask(options, options[want])}
    sc = PRICING[spec['scenario']]
    while True:
        prices, q_a, q_b, pa, pb = pricing_game(sc, spec['symmetric'])
        own = pa if spec['firm'] == 'A' else pb.T  # rows: the asked firm's price
        if strictly_dominant_rows(own) == [want]:
            break
    name = sc['a'] if spec['firm'] == 'A' else sc['b']
    options = [f'{name} should charge {money(prices[0])}', f'{name} should charge {money(prices[1])}']
    return {**pricing_values(prices, q_a, q_b), 'options_list': [options], 'correct_answer': _mask(options, options[want])}


ct.register('pure_nash', _nash_templates())
ct.register('dominant_strategies', _dominant_templates())
