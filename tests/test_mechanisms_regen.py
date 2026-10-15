"""Regenerated STEER mechanism elements: ir_mechanism, budget_balance, auctions_risk.

For every element and type, N (default 1,000; env STEER_TEST_N) seeded samples must:
- agree 100% with the independent text-parsing solver (audit/steer/solvers/mechanisms.py, REGEN_SOLVERS);
- have exactly one correct option, distinct options, no placeholders / None / nan;
- put the correct option in every position about equally often;
- have an outcome mix within the bounds of the design (audit/DECISIONS.md).
Plus the worked examples of the design, grade levels, element-specific constraints, and reproducibility across
two processes.
"""
import collections
import os
import re
import subprocess
import sys
from fractions import Fraction

import pytest

from helpers import load_steer_solver, sample

_mech = load_steer_solver('mechanisms')
(
    REGEN_SOLVERS, regen_ir_net_utilities, solve_regen_auctions_risk, solve_regen_budget_balance,
    solve_regen_ir_mechanism,
) = (_mech.REGEN_SOLVERS, _mech.regen_ir_net_utilities, _mech.solve_regen_auctions_risk,
     _mech.solve_regen_budget_balance, _mech.solve_regen_ir_mechanism)

N = int(os.environ.get('STEER_TEST_N', 1000))
N_EXAMPLES = max(200, N // 5)
SEED = 20260928

TYPES = {
    'ir_mechanism': ['equal_split', 'proportional', 'outside_option', 'vcg'],
    'budget_balance': ['direct_transfers', 'fees', 'clarke'],
    'auctions_risk': ['risk_ranking', 'expected_revenue', 'equilibrium_bid'],
}
GRADES = {
    'ir_mechanism': {'equal_split': 6, 'proportional': 7, 'outside_option': 7, 'vcg': 8},
    'budget_balance': {'direct_transfers': 8, 'fees': 9, 'clarke': 10},
    'auctions_risk': {'risk_ranking': 10, 'equilibrium_bid': 10, 'expected_revenue': 11},
}
CASES = [(e, t) for e, ts in TYPES.items() for t in ts]
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
BAD_TOKEN = re.compile(r'\b(None|nan|NaN|inf)\b')

_cache = {}


def get(element, type_, pool='questions'):
    key = (element, type_, pool)
    if key not in _cache:
        n = N if pool == 'questions' else N_EXAMPLES
        _cache[key] = sample(element, n, seed=SEED, type=type_, pool=pool)
    return _cache[key]


def outcome(element, s):
    a = s.answer
    if element == 'ir_mechanism':
        return 'everyone' if a.startswith('Everyone') else 'one'
    if element == 'budget_balance':
        return 'balanced' if 'exactly the cost' in a else 'surplus' if 'surplus' in a else 'deficit'
    if s.type == 'risk_ranking':
        return 'same' if 'same' in a else 'first' if re.search('first-price|Dutch', a) else 'second'
    return 'numeric'


# (min share, max share) of each outcome, per type; None = not checked
MIX = {
    ('ir_mechanism', t): {'everyone': (0.18, 0.32), 'one': (0.68, 0.82)} for t in TYPES['ir_mechanism']
}
MIX.update({
    ('budget_balance', 'direct_transfers'): {k: (0.27, 0.40) for k in ('balanced', 'surplus', 'deficit')},
    ('budget_balance', 'fees'): {k: (0.27, 0.40) for k in ('balanced', 'surplus', 'deficit')},
    ('budget_balance', 'clarke'): {'balanced': (0.43, 0.57), 'surplus': (0.43, 0.57), 'deficit': (0.0, 0.0)},
    ('auctions_risk', 'risk_ranking'): {k: (0.27, 0.40) for k in ('same', 'first', 'second')},
})


@pytest.mark.parametrize('pool', ['questions', 'examples'])
@pytest.mark.parametrize('element,type_', CASES)
def test_samples(element, type_, pool):
    got = get(element, type_, pool)
    solve = REGEN_SOLVERS[element]
    positions = collections.Counter()
    counts = collections.Counter()
    disagree = []
    for s in got:
        assert s.metadata['type'] == type_
        assert s.metadata['difficulty_level'] == GRADES[element][type_]
        assert s.options and all(o.strip() for o in s.options), s.options
        assert len(set(s.options)) == len(s.options), s.options
        blob = s.question + ' || ' + ' | '.join(s.options)
        assert not PLACEHOLDER.search(blob), blob
        assert not BAD_TOKEN.search(blob), blob
        assert '  ' not in s.question and '..' not in s.question, s.question
        assert sum(s.correct) == 1, (s.question, s.options, s.correct)
        if solve(s.question, s.options) != s.answer_index:
            disagree.append((s.question, s.options, s.answer))
        positions[(len(s.options), s.answer_index)] += 1
        counts[len(s.options)] += 1
    assert not disagree, f'{len(disagree)} solver disagreements, e.g. {disagree[0]}'
    if pool != 'questions':
        return
    for k, total in counts.items():
        if total < 100:
            continue
        bound = 3.5 * ((1 / k) * (1 - 1 / k) / total) ** 0.5
        shares = [positions[(k, i)] / total for i in range(k)]
        assert all(abs(x - 1 / k) <= bound for x in shares), (k, shares)
    mix = MIX.get((element, type_))
    if mix:
        c = collections.Counter(outcome(element, s) for s in got)
        for key, (lo, hi) in mix.items():
            share = c[key] / len(got)
            assert lo <= share <= hi, (key, share, dict(c))


def test_numeric_options_whole_dollars():
    for element, type_ in CASES:
        for s in get(element, type_):
            for amt in re.findall(r'\$[\d,.]+', s.question + ' '.join(s.options)):
                assert re.fullmatch(r'\$\d{1,3}(,\d{3})*', amt.rstrip('.,')), (amt, s.question)


def test_ir_no_near_indifference():
    for t in TYPES['ir_mechanism']:
        for s in get('ir_mechanism', t):
            net, outside = regen_ir_net_utilities(s.question)
            assert all(abs(x - outside) >= 5 for x in net.values()), (s.question, net, outside)
            assert sum(x < outside for x in net.values()) <= 1


def test_ir_options_are_participants_plus_everyone():
    for t in TYPES['ir_mechanism']:
        for s in get('ir_mechanism', t):
            net, _ = regen_ir_net_utilities(s.question)
            assert set(s.options) == set(net) | {'Everyone is better off participating'}


def test_ir_outside_option_matters():
    """Some outside-option items name a person whose own surplus is positive but below the outside option."""
    trap = 0
    for s in get('ir_mechanism', 'outside_option'):
        net, outside = regen_ir_net_utilities(s.question)
        trap += any(0 < x < outside for x in net.values())
    assert trap / N > 0.2


def test_budget_balance_states_definition():
    for t in TYPES['budget_balance']:
        for s in get('budget_balance', t):
            assert re.search(r'"budget balanced" means|counts as budget balanced', s.question), s.question
            kinds = {('exactly' in o) and 'b' or ('surplus' in o) and 's' or 'd' for o in s.options}
            assert kinds == {'b', 's', 'd'}, s.options  # every outcome category is on offer


def test_auction_formats_never_equivalent_pair():
    for s in get('auctions_risk', 'risk_ranking'):
        q = s.question
        assert not ('Dutch' in q and 'first-price' in q), q
        assert not ('English' in q and 'second-price' in q), q
        assert len(s.options) == 3


def test_worked_examples():
    q = ('Four residents of Maple Grove are deciding on a new playground costing $240. Values: Ana $120, Bo $45, '
         'Cy $80 and Dee $70. Rule: The cost is split equally, so each of the four pays the same share. Each '
         'resident can choose not to participate: one who opts out pays nothing and gets nothing from the '
         "playground, and the others' payments do not change. Which resident would be better off not participating?")
    opts = ['Ana', 'Bo', 'Cy', 'Dee', 'Everyone is better off participating']
    assert solve_regen_ir_mechanism(q, opts) == 1
    q = ('Three residents of Maple Grove are funding a new playground that costs $300. Values: Eli $160, Fay $80 '
         'and Gus $70. Everyone reports their value truthfully, and a Clarke (pivotal) mechanism is used. Each '
         "person's cost share is an equal one-third of the cost. The project goes ahead if the values minus the "
         'cost shares add up to zero or more, and then everyone pays their cost share. A person is pivotal if '
         'leaving out their report would change the decision; each pivotal person also pays a Clarke tax equal to '
         'the absolute value of the sum, over everyone else, of value minus cost share. Clarke taxes are kept by '
         'the mechanism, not refunded. (Here a mechanism counts as budget balanced only if it collects exactly the '
         'cost: no surplus and no deficit.) Does the mechanism collect exactly the cost of the playground?')
    opts = ['Budget balanced: it collects exactly the cost', 'Not budget balanced: it collects a surplus of $50',
            'Not budget balanced: it runs a deficit of $50', 'Not budget balanced: it collects a surplus of $30']
    assert solve_regen_budget_balance(q, opts) == 1
    q = ('Three risk-neutral collectors are bidding for a rare coin sold by the coin dealer. Their values are private '
         'and independently drawn from a uniform distribution between $0 and $120. The coin dealer runs a '
         'second-price sealed-bid (Vickrey) auction with no reserve price, and the bidders follow the symmetric '
         "equilibrium. What is the coin dealer's expected revenue?")
    assert solve_regen_auctions_risk(q, ['$90', '$60', '$30', '$80']) == 1


def test_template_pools():
    """A few hundred story templates per element; few-shot pool with explanations, disjoint from the test pool."""
    from autosteer.sampler import ElementSampler
    s = ElementSampler()
    for e in TYPES:
        ds = s._load_template(e)
        assert len(ds.questions) >= 300, e
        assert set(ds.questions.type) == set(TYPES[e])
        assert len(ds.example_questions) >= 2 * len(TYPES[e])
        assert ds.example_questions.explanation.str.len().min() > 40
        assert not set(ds.questions.question_text) & set(ds.example_questions.question_text)
        for t in TYPES[e]:
            assert (ds.questions.type == t).sum() >= 150, (e, t)


def test_reproducible_across_processes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    here = os.path.dirname(os.path.abspath(__file__))

    def run(hashseed):
        env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
        out = subprocess.run([sys.executable, os.path.join(here, '_repro_dump.py'), root, '30', *TYPES],
                             env=env, capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()[-1]

    a, b = run(1), run(2)
    assert a == b


# ------------------------------------------------------------------ budget_balance conventions (October 2026)
EXACT_CONV = re.compile(r'counts as budget balanced only if it collects exactly the cost: no surplus and no deficit|'
                        r'"budget balanced" means the money collected equals the cost exactly')
WEAK_CONV = re.compile(r'counts as budget balanced if it never runs a deficit; collecting more than the cost is allowed|'
                       r'"budget balanced" means weakly balanced: the money collected is at least the cost')


def _bb_option(o):
    """(labelled balanced?, amount: + surplus, - deficit, 0 exact) of an option text."""
    m = re.fullmatch(r'(Not budget balanced|Budget balanced): it (?:collects exactly the cost|collects a surplus of '
                     r'\$([\d,]+)|runs a deficit of \$([\d,]+))', o)
    assert m, o
    amount = int(m.group(2).replace(',', '')) if m.group(2) else -int(m.group(3).replace(',', '')) if m.group(3) else 0
    return m.group(1) == 'Budget balanced', amount


@pytest.mark.parametrize('pool', ['questions', 'examples'])
def test_budget_balance_key_follows_stated_convention(pool):
    """Every question states exactly one convention (strong: exactly the cost; weak: no deficit); every option is
    labelled by that convention (balanced iff no surplus or deficit under the strong one, iff no deficit under the weak
    one); the key's amount is the independent solver's balance, so the key follows the stated convention."""
    seen = collections.Counter()
    for t in TYPES['budget_balance']:
        got = sample('budget_balance', 150, seed=SEED + 1, type=t, pool=pool)
        for s in got:
            exact, weak = bool(EXACT_CONV.search(s.question)), bool(WEAK_CONV.search(s.question))
            assert exact != weak, s.question                     # one convention, stated once
            for o in s.options:
                balanced, amount = _bb_option(o)
                assert balanced == (amount == 0 or (weak and amount > 0)), (o, 'weak' if weak else 'exact')
            assert solve_regen_budget_balance(s.question, s.options) == s.answer_index, (s.question, s.options)
            balanced, amount = _bb_option(s.answer)
            seen[t, 'weak' if weak else 'exact', 'surplus' if amount > 0 else 'deficit' if amount < 0 else 'zero'] += 1
    for t in TYPES['budget_balance']:
        for conv in ('weak', 'exact'):
            assert seen[t, conv, 'zero'] and seen[t, conv, 'surplus'], seen   # a surplus under both conventions
