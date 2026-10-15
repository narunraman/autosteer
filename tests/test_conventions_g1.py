"""Fixes after the review of October 2026, games / social choice / mechanisms group.

One test per class of problem the review found; each would have failed before the fix.
"""
import re
from itertools import product

import numpy as np
import pytest

from autosteer.filler_tools import multi_agent_tools as ma
from autosteer.filler_tools.auctions_affiliated_tools import SYMMETRIC
from autosteer.sampler import ElementSampler

N = 120


@pytest.fixture(scope='module')
def sampler():
    return ElementSampler()


def draws(sampler, element, n=N, start=7000, **kw):
    return [sampler.sample_question(element, seed, **kw) for seed in range(start, start + n)]


def key(p):
    return p['options'][p['correct'].index(True)]


# ----------------------------------------------------------------------------- bayes_nash
CELL = re.compile(r'\((\d+), (\d+)\)')


def parse_bayes(text):
    p1 = int(re.search(r'Type 1 (?:with (?:probability |a )|is )(\d+)%', text).group(1)) / 100
    cells = [tuple(map(int, c)) for c in CELL.findall(text)]
    assert len(cells) == 8
    U = [np.array([[cells[4 * t + 2 * i + j][0] for j in range(2)] for i in range(2)], float) for t in range(2)]
    V = [np.array([[cells[4 * t + 2 * i + j][1] for j in range(2)] for i in range(2)], float) for t in range(2)]
    return U, V, p1


def weak_pure_equilibria(U, V, p):
    """Pure Bayes-Nash equilibria with weak best responses (ties allowed): (your row, (type-1 col, type-2 col))."""
    out = []
    for x, s in product(range(2), product(range(2), repeat=2)):
        mine = [p * U[0][a, s[0]] + (1 - p) * U[1][a, s[1]] for a in range(2)]
        if mine[x] + 1e-9 < mine[1 - x]:
            continue
        if all(V[t][x, s[t]] + 1e-9 >= V[t][x, 1 - s[t]] for t in range(2)):
            out.append((x, s))
    return out


def test_bayes_unique_rejects_review_games():
    # Q10 of the review: the type-2 opponent is indifferent against S, so (S; K, K) is a second, weak equilibrium
    U = [np.array([[2, 8], [4, 4]], float), np.array([[2, 1], [7, 7]], float)]
    V = [np.array([[0, 5], [4, 0]], float), np.array([[9, 2], [7, 7]], float)]
    assert len(weak_pure_equilibria(U, V, 0.7)) >= 2
    assert not ma.bayes_unique(U, V, 0.7)
    # Q9: a unique pure equilibrium, but also mixed ones
    U = [np.array([[2, 8], [4, 3]], float), np.array([[1, 7], [2, 5]], float)]
    V = [np.array([[1, 5], [8, 3]], float), np.array([[8, 4], [2, 4]], float)]
    assert len(weak_pure_equilibria(U, V, 0.36)) == 1
    assert not ma.bayes_unique(U, V, 0.36)


def test_bayes_nash_single_equilibrium(sampler):
    for q in draws(sampler, 'bayes_nash'):
        text = q['parts'][0]['question']
        U, V, p = parse_bayes(text)
        eq = weak_pure_equilibria(U, V, p)
        assert len(eq) == 1, (q['seed'], eq)
        assert ma.bayes_unique(U, V, p), q['seed']
        x, s = eq[0]
        if q['metadata']['type'] == 'typed_own':
            assert key(q['parts'][0]) == ['Action R', 'Action S'][x]
        else:
            t = int(re.findall(r'Type (\d)', text.rstrip().splitlines()[-1])[-1]) - 1
            assert key(q['parts'][0]) == ['Action K', 'Action L'][s[t]]


# ----------------------------------------------------------------------------- backward_induction
def bi_templates():
    return ElementSampler()._load_template('backward_induction')


def test_backward_induction_drops_and_wording():
    d = bi_templates()
    texts = d.questions['question_text']
    # the review's Q5 (keyed option is the ending action), Q12 (equity "23442.69%")
    assert not any('{p1_round1}%' in t or re.search(r'\}\s*%', t) for t in texts)
    assert not any('continue with her current management plan' in t for t in texts)
    assert not any(re.search(r'\bimprove for both\b', t) for t in texts)
    assert not any(re.search(r'\bimproved (terms|offer|deal)\b', t) for t in texts)
    assert len(texts) % 2 == 0 and len(texts) >= 600


def test_backward_induction_mislabelled_treatment_stories_dropped():
    """Treatment stories where the keyed option is the ending action or the rounds are mis-assigned (review Q5)."""
    texts = list(bi_templates().questions['question_text'])
    for frag in ('can either continue her protocol, receiving {p1_round1}',
                 'Dr. Alice should continue with her current management plan',
                 'continue with her current management plan, which would benefit her by {p1_round1}',
                 'can either retain her treatment, in which case she receives {p2_round2}',
                 'stick with her own treatment, resulting in {p2_round2}',
                 'whether to stick with her experimental protocol, earning {p1_round2}',
                 'persist with it, which will grant Dr. Smith {p1_round1}'):
        assert not any(frag in t for t in texts), frag
    assert set(ma.TEMPLATE_DROPS['backward_induction']) and set(ma.TEMPLATE_DROPS['trigger'])


# ----------------------------------------------------------------------------- trigger
def test_trigger_states_rounding_and_drops(sampler):
    d = sampler._load_template('trigger')
    texts = list(d.questions['question_text'])
    assert all('Round your answer to two decimal places.' in t for t in texts)
    assert not any('secure-gambler' in t or 'strategy of always collaborating' in t
                   or 'strategy of always settling' in t for t in texts)
    for q in draws(sampler, 'trigger', 60):
        t = q['parts'][0]['question']
        if '{' in t:
            continue
        assert 'Round your answer to two decimal places.' in t


def test_trigger_secure_gamble_roles():
    """Gambling stories: the player who secures (the cooperative action, mutual securing gives the both-cooperate
    payoff) must never be given the defector payoff."""
    d = ElementSampler()._load_template('trigger')
    bad = []
    for t in d.questions['question_text']:
        m = re.search(r'one [^.;]*?secures?[^.;]*?while the other gambles[^.;]*?(\{defect\}|\{cooperate\})', t)
        if m and 'respectively' in t[m.start():m.end() + 40] and m.group(1) == '{defect}':
            bad.append(t[m.start():m.end()])
        if re.search(r'secure-gambler gets a payoff of \{defect\}', t):
            bad.append(t)
    assert not bad, bad[:3]


# ----------------------------------------------------------------------------- borda_count
def test_borda_convention_stated_and_key(sampler):
    for q in draws(sampler, 'borda_count'):
        t = q['parts'][0]['question']
        m = re.search(r'gives them (\d), (\d) and (\d) points, in the order ranked; a candidate left off a '
                      r'ballot gets 0', t)
        assert m, t
        pts = list(map(int, m.groups()))
        cands = [o.split()[-1] for o in q['parts'][0]['options']]
        score = {c: 0 for c in cands}
        for n, r in re.findall(r'- (\d+) voters? voted for ([A-Z](?: > [A-Z])+)', t):
            for i, c in enumerate(r.split(' > ')):
                score[c] += int(n) * pts[i]
        order = sorted(cands, key=lambda c: -score[c])
        place = 1 if 'second' in q['metadata']['tags'] else 0
        assert key(q['parts'][0]) == f'Candidate {order[place]}'


# ----------------------------------------------------------------------------- mechanisms
def test_budget_balance_asks_comparison(sampler):
    for q in draws(sampler, 'budget_balance', 150):
        t = q['parts'][0]['question']
        assert 'collect exactly' not in t, t


def test_clarke_pivot_definition(sampler):
    for el in ('ir_mechanism', 'budget_balance'):
        for q in draws(sampler, el, 150):
            t = q['parts'][0]['question']
            if 'Clarke' in t:
                assert 'leaving out their report' not in t
                assert 'both their value and their cost share' in t


def test_dsic_clock_continuous_with_tie_rule(sampler):
    for t in sampler._load_template('dsic_mechanism').questions['question_text']:
        assert 'small steps' not in t
        if 'price clock' in t or 'asking price starts' in t or 'calling out a price' in t:
            assert 'drawn at random' in t, t


# ----------------------------------------------------------------------------- dominance
def dominated_any_independent(U):
    """Weakly dominated by a pure action, or strictly by a mixture of two others (grid over the mixture)."""
    k = U.shape[0]
    out = set()
    for a in range(k):
        o = [b for b in range(k) if b != a]
        if any((U[b] >= U[a]).all() and (U[b] > U[a]).any() for b in o):
            out.add(a)
        for w in np.linspace(0, 1, 2001):
            if len(o) == 2 and ((w * U[o[0]] + (1 - w) * U[o[1]]) > U[a] + 1e-9).all():
                out.add(a)
                break
    return sorted(out)


def parse_matrix(text, which):
    rows = re.findall(r'^Action ([RST]) +(.+)$', text, re.M)
    vals = []
    for _, rest in rows:
        cells = re.findall(r'\(([-\d.]+), ([-\d.]+)\)', rest)
        if cells:
            vals.append([float(c[which]) for c in cells])
        else:
            vals.append([float(x) for x in rest.split()])
    return np.array(vals)


def test_dominated_strategies_unique_under_any_dominance(sampler):
    n = 0
    for q in draws(sampler, 'dominated_strategies', 300):
        md = q['metadata']
        if not str(md['type']).endswith('_dominated'):
            continue
        t = q['parts'][0]['question']
        single = 'find your' in t
        labels = q['parts'][0]['options']
        cols = labels[0] in ('Action X', 'Action Y', 'Action Z')
        if single:
            U = parse_matrix(t, 0)
            U = U.T if cols else U
        else:
            U = parse_matrix(t, 1 if cols else 0)
            U = U.T if cols else U
        names = ['Action X', 'Action Y', 'Action Z'] if cols else ['Action R', 'Action S', 'Action T']
        k = U.shape[0]
        assert dominated_any_independent(U) == [names[:k].index(key(q['parts'][0]))], (q['seed'], t)
        n += 1
    assert n > 20


def test_iterated_removal_part2_states_removal(sampler):
    for q in draws(sampler, 'iterated_removal', 80):
        t = q['parts'][1]['question']
        assert 'remove' in t or 'remaining game' in t, t


# ----------------------------------------------------------------------------- social choice
@pytest.mark.slow
def test_sc_monotonicity_options_not_structural(sampler):
    n = 0
    for q in draws(sampler, 'sc_monotonicity_rules', 120):
        if q['metadata']['type'] != 'consequences':
            continue
        n += 1
        p = q['parts'][0]
        pairs = [re.search(r': ([A-Z >]+) before the change, ([A-Z >]+) after it', o).groups() for o in p['options']]
        raised = re.search(r'moved (\w) up', p['question']).group(1)
        assert len(set(pairs)) == 4, (q['seed'], pairs)
        k = p['correct'].index(True)
        others = [x for i, x in enumerate(pairs) if i != k]
        assert any(b != a for b, a in others), (q['seed'], pairs)
        assert any(a.index(raised) < b.index(raised) for b, a in others), (q['seed'], pairs)
    assert n > 20


def test_sc_nondictatorship_distinct_dictators(sampler):
    n = 0
    for q in draws(sampler, 'sc_nondictatorship', 150):
        if q['metadata']['type'] != 'describe':
            continue
        n += 1
        t = q['parts'][0]['question']
        people = re.findall(r'(\w+) \(the (chair|secretary|treasurer), age (\d+)\)', t)
        role = {r: name for name, r, _ in people}
        oldest = max(people, key=lambda x: int(x[2]))[0]
        reg = re.search(r'(\w+) registered for the committee first', t)
        who = []
        for i, o in enumerate(q['parts'][0]['options']):
            if q['parts'][0]['correct'][i]:
                continue
            if "chair's ballot becomes" in o or o.startswith("The chair's ballot"):
                who.append(role['chair'])
            elif 'registered first' in o:
                who.append(reg.group(1))
            elif 'oldest member' in o:
                who.append(oldest)
            elif "secretary's ballot" in o:
                who.append(role['secretary'])
            elif 'alphabetical order of the members' in o:
                who.append(sorted(name for name, _, _ in people)[0])
        assert len(who) == 3 and len(set(who)) == 3, (q['seed'], who, q['parts'][0]['options'])
    assert n > 20


# ----------------------------------------------------------------------------- auctions / fair division
def test_auctions_affiliated_assumptions_stated(sampler):
    for q in draws(sampler, 'auctions_affiliated', 80):
        t = q['parts'][0]['question']
        assert SYMMETRIC in t
        assert "shops' customers" not in t
        assert 'than a fifty-fifty chance of $100 or nothing.' not in t or 'average amount' in t


def test_fair_division_wording_and_options(sampler):
    for q in draws(sampler, 'fair_division', 150):
        p = q['parts'][0]
        if q['metadata']['type'] == 'envy_check':
            assert len(p['options']) == 3
            assert 'It is envy-free but not proportional' not in p['options']
        if q['metadata']['type'] == 'last_diminisher':
            assert 'with k people left, the piece is worth exactly 1/k of the rest of the cake' in p['question']
            assert 'of the whole cake to its cutter and' not in p['question']


# ----------------------------------------------------------------------------- re-review round 2
NUMW = {'three': 3, 'four': 4, 'five': 5, 'six': 6}


def bimatrix(text):
    rows = re.findall(r'^Action ([RST]) +(.+)$', text, re.M)
    cells = [re.findall(r'\(([-\d.]+), ([-\d.]+)\)', rest) for _, rest in rows]
    A = np.array([[float(c[0]) for c in r] for r in cells])
    B = np.array([[float(c[1]) for c in r] for r in cells])
    return A, B


def test_iterated_removal_both_steps_needed(sampler):
    """Part 1's key is strictly dominated by a pure action; part 2's key is not dominated (pure or mixed) in the
    original game and is strictly dominated by a pure action once part 1's action is removed (re-review Q3/7/9/12)."""
    for q in draws(sampler, 'iterated_removal', 80):
        t0 = q['parts'][0]['question']
        A, B = bimatrix(t0)
        you = 'row' if 'You are the row player' in t0 else 'column'
        first = you if 'one of your actions' in t0 else ('column' if you == 'row' else 'row')
        second = 'column' if first == 'row' else 'row'
        mat = {'row': A, 'column': B.T}
        lab = {'row': ['Action R', 'Action S', 'Action T'], 'column': ['Action K', 'Action L', 'Action M']}
        k1 = lab[first].index(key(q['parts'][0]))
        U1 = mat[first]
        assert any((U1[b] > U1[k1]).all() for b in range(3) if b != k1), q['seed']
        k2 = lab[second].index(key(q['parts'][1]))
        U2 = mat[second]
        assert dominated_any_independent(U2) == [], (q['seed'], U2)
        keep = [b for b in range(3) if b != k1]
        U2r = U2[:, keep]
        assert any((U2r[b] > U2r[k2]).all() for b in range(3) if b != k2), q['seed']


def test_dsic_immediate_acceptance_capacities_bind(sampler):
    n = 0
    for q in draws(sampler, 'dsic_mechanism', 400):
        if q['metadata']['type'] != 'match_immediate':
            continue
        t = q['parts'][0]['question']
        m = re.search(r'(\d+), (\d+) and (\d+) seats respectively', t)
        assert m, t
        k = NUMW[re.search(r'(?:assigns|with) (three|four|five|six) ', t).group(1)]
        caps = list(map(int, m.groups()))
        assert sum(caps) == k and max(caps) < k
        n += 1
    assert n > 3


# ----------------------------------------------------------------------------- re-review round 3
def test_auctions_affiliated_information_environment(sampler):
    from autosteer.filler_tools.auctions_affiliated_tools import APPRAISAL, ENVIRONMENT
    for q in draws(sampler, 'auctions_affiliated', 120):
        t = q['parts'][0]['question']
        assert any(s in t for s in ENVIRONMENT.values()), t
        if q['metadata']['type'] == 'info_policy':
            assert any(s in t for s in APPRAISAL.values()), t
        assert 'would trade any gamble' not in t and 'takes the sure amount' not in t


def test_fair_division_adjusted_winner_shares_vary(sampler):
    n = 0
    for q in draws(sampler, 'fair_division', 200):
        if q['metadata']['type'] != 'adjusted_winner':
            continue
        n += 1
        shares = [re.search(r'keeps ([\d.]+)%', o).group(1) for o in q['parts'][0]['options']]
        assert len(set(shares)) == 4, (q['seed'], q['parts'][0]['options'])
    assert n > 20


# ----------------------------------------------------------------------------- re-review round 4
def test_auctions_affiliated_states_strict_affiliation(sampler):
    from autosteer.filler_tools.auctions_affiliated_tools import ENVIRONMENT
    n = 0
    for q in draws(sampler, "auctions_affiliated", 120):
        t = q["parts"][0]["question"]
        if ENVIRONMENT["affiliated"] in t:
            n += 1
            assert "strictly affiliated" in t
            if q["metadata"]["type"] == "info_policy":
                assert "strictly affiliated with the value and the bidders" in t
    assert n > 20
