"""Review (October 2026), production and factor-market elements: one test per class of finding.

- numeric distractors: the key's rank among the options must not identify it (it was nearly always the largest);
- aggregation of factor demands: no group's demand is negative at the asked price, the price range is stated for the
  demand function, and the key is not singled out by the size of its coefficients;
- factor_demand: both parts offer the same values (the key was the only value common to both);
- tfp_shocks: the shock's direction is stated, no option is the negative of another, the fixed prices are stated;
- conventions stated in the question: cost model and rounding (marginal_cost, average_cost), both inputs scaled
  (returns_to_scale), profit without capital cost (profit_maximization), the varied input and the asked marginal
  product named in L / K (diminishing_marginal_product), output elasticity with bK shown exactly;
- supply_elasticity: distractors are not always slope, intercept and their sum;
- dynamic_profit_max: no story that counts shows but prices tickets.
"""
import re

import pytest

from helpers import sample, num


def _rank_stats(samples):
    top = bottom = 0
    for s in samples:
        vals = [num(o) for o in s.options]
        key = vals[s.answer_index]
        top += key == max(vals)
        bottom += key == min(vals)
    return top / len(samples), bottom / len(samples)


@pytest.mark.parametrize('element,typ', [
    ('cost_minimization_problem', 'compute_labor'), ('cost_minimization_problem', 'compute_capital'),
    ('cost_minimization_problem', 'cost_minimizing'), ('profit_maximization', 'compute_labor'),
    ('profit_maximization', 'compute_profit'), ('aggregation_of_capital_demand', 'numerical'),
    ('supply_elasticity', 'Linear_supply_given'), ('supply_elasticity', 'Quadratic_supply_given'),
])
def test_key_rank_not_informative(element, typ):
    got = sample(element, 120, seed=31, type=typ)
    top, bottom = _rank_stats(got)
    # four options: a random rank would be largest 25% of the time
    assert top <= 0.5 and bottom <= 0.5, (element, typ, top, bottom)


def _lin(f):
    a, c = re.fullmatch(r'([\d.]+) - ([\d.]+)P', f).groups()
    return float(a), float(c)


@pytest.mark.parametrize('element', ['aggregation_of_labor_demand', 'aggregation_of_capital_demand'])
def test_factor_aggregation_well_posed(element):
    biggest = 0
    got = sample(element, 120, seed=5)
    for s in got:
        p = s.params
        (a, ca), (b, cb) = _lin(p['demand_function_a']), _lin(p['demand_function_b'])
        choke = min(a / ca, b / cb)
        if s.metadata['type'] == 'numerical':
            price = float(p['price'])
            assert price < choke, (s.question, price, choke)
            assert f"(P = {p['price']})" in s.question, s.question
        else:
            assert float(p['p_max']) <= choke + 1e-9
            assert f"0 ≤ P ≤ {p['p_max']}" in s.question, s.question
            inter = [float(re.match(r'([\d.]+) - ([\d.]+)P', o).group(1)) for o in s.options]
            biggest += inter[s.answer_index] == max(inter)
    n_fn = sum(s.metadata['type'] == 'demand_function' for s in got)
    assert n_fn and biggest / n_fn <= 0.5, (biggest, n_fn)


def test_factor_demand_parts_share_options():
    from autosteer.sampler import load_manifest
    from autosteer.web.site import ElementSampler
    sampler = ElementSampler(manifest=load_manifest())
    for seed in range(40):
        q = sampler.sample_question('factor_demand', 5000 + seed)
        o0, o1 = q['parts'][0]['options'], q['parts'][1]['options']
        assert sorted(o0) == sorted(o1), (o0, o1)
        k0 = o0[q['parts'][0]['correct'].index(True)]
        k1 = o1[q['parts'][1]['correct'].index(True)]
        assert k0 == k1


@pytest.mark.parametrize('typ', ['increase_productivity', 'decrease_productivity', 'increase_labor',
                                 'decrease_labor'])
def test_tfp_direction_and_options(typ):
    up = typ.startswith('increase')
    word = re.compile(r'increas|rise|rais|boost|improv|grow|gain|enhanc|advanc|jump|surge|climb|higher|hike|up\b'
                      if up else r'decreas|drop|fall|fell|declin|reduc|cut|lower|dip|loss|slump|diminish|shrink|down\b',
                      re.I)
    for s in sample('tfp_shocks', 60, seed=3, type=typ):
        p = s.params
        shock = p.get('productivity_increase') or p.get('productivity_decrease') or \
            p.get('labor_price_increase') or p.get('labor_price_decrease')
        i = s.question.index(str(shock))
        window = re.split(r'[.!?]\s', s.question[:i])[-1] + ' ' + ' '.join(s.question[i:].split()[:5])
        assert word.search(window), window
        assert 'rental price of capital do not change' in s.question, s.question
        vals = [num(o) for o in s.options]
        assert not any(v != 0 and -v in vals for v in vals), s.options


@pytest.mark.parametrize('element', ['marginal_cost', 'average_cost'])
def test_cost_convention_stated(element):
    for s in sample(element, 40, seed=2):
        assert 'rounded to two decimal places' in s.question
        assert f"${s.params['wage']} per unit of L" in s.question, s.question
        assert ('dC/dQ' if element == 'marginal_cost' else 'C(Q)/Q') in s.question


def test_conventions_rts_profit_output_elasticity():
    for s in sample('returns_to_scale', 30, seed=2):
        assert 'both are increased by the same proportion' in s.question
    for s in sample('profit_maximization', 30, seed=2, type='compute_profit'):
        assert 'Ignore the cost of capital' in s.question
    for s in sample('output_elasticity', 40, seed=2, type='linear'):
        a, b = map(float, re.search(r'([\d.]+)L \+ ([\d.]+)K', s.params['p_function']).groups())
        k = float(s.params['fixed_val'].split('=')[1])
        key = s.options[s.answer_index]
        assert k == int(k) and key == f"{a}L/({a}L+{round(b * k, 2)})", (key, a, b, k)
        assert all('/' in o for o in s.options), s.options


@pytest.mark.parametrize('typ', ['linear_labor_fixed', 'quadratic_capital_fixed', 'cobb_douglas_labor_fixed'])
def test_dmp_asks_marginal_product_of_named_input(typ):
    var = 'L' if 'capital_fixed' in typ else 'K'
    for s in sample('diminishing_marginal_product', 40, seed=4, type=typ):
        q = s.question
        assert q.rstrip().endswith(f'marginal product of {var} change?'), q
        assert f"{var} is the input that increases from {s.params['start']} to {s.params['end']}" in q, q
        assert all(f'marginal product of {var} ' in o for o in s.options), s.options


def test_supply_elasticity_distractor_structure():
    hits = 0
    got = sample('supply_elasticity', 80, seed=8, type='Linear_supply_given')
    for s in got:
        a, b = map(float, re.search(r'Q = ([\d.]+)P \+ ([\d.]+)', s.params['s_func']).groups())
        vals = {round(num(o), 2) for i, o in enumerate(s.options) if i != s.answer_index}
        hits += vals == {round(a, 2), round(b, 2), round(a + b, 2)}
    assert hits == 0
    below_one = sum(any(num(o) < 1 for i, o in enumerate(s.options) if i != s.answer_index) for s in got)
    assert below_one / len(got) >= 0.5


def test_dynamic_profit_no_ticket_stories():
    for s in sample('dynamic_profit_max', 60, seed=6):
        assert 'ticket' not in s.question.lower()


# ---- Re-review, round 2 ----

_ABBR = re.compile(r'\b(Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|Inc|Co|Ltd|vs|etc|e\.g|i\.e)\. (?=[A-Z])')
_INSERTED = ('Ignore the cost of capital', 'Here L and K are the two inputs', 'In the production function, ',
             'Here P is the ', 'Assume a competitive market')


def test_sentence_splitter_skips_abbreviations():
    from autosteer.sampler.template_fixes import _SENTENCE_START, before_last_sentence
    t = 'What is the maximum profit that Dr. Anna can achieve? Mr. Lee asks e.g. Ms. Wu.'
    assert [t[m.end():m.end() + 3] for m in _SENTENCE_START.finditer(t)] == ['Mr.']
    assert before_last_sentence('Dr. Anna sells. What can Dr. Anna earn?', 'X.') == 'Dr. Anna sells. X. What can Dr. Anna earn?'


@pytest.mark.parametrize('element,typ', [('profit_maximization', 'compute_profit'), ('returns_to_scale', None),
                                         ('diminishing_marginal_product', None), ('tfp_shocks', None),
                                         ('aggregation_of_labor_demand', None)])
def test_no_convention_after_abbreviation(element, typ):
    kw = {'type': typ} if typ else {}
    for s in sample(element, 150, seed=12, **kw):
        for m in _ABBR.finditer(s.question):
            assert not s.question[m.end():].startswith(_INSERTED), s.question


def test_tfp_options_same_decimals():
    for s in sample('tfp_shocks', 80, seed=13):
        assert all(o == '0%' or re.fullmatch(r'-?\d+\.\d\d%', o) for o in s.options), s.options


def test_dynamic_profit_asks_increase_and_output_quantity():
    for s in sample('dynamic_profit_max', 60, seed=14):
        assert s.question.rstrip().endswith("Tomorrow's revenue is multiplied by the discount factor " + str(s.params['discount']) + '.')
        f = s.params['p_func']
        head = re.split(r'[.!?]\s', s.question[:s.question.index(f)])[-1]
        assert not re.search(r'revenue|profit|benefit|effectiveness|performance\b', head, re.I), head


@pytest.mark.parametrize('typ', ['quadratic_capital_fixed', 'quadratic_labor_fixed'])
def test_dmp_quadratic_output_positive(typ):
    from econ_solvers import parse_terms, eval_terms
    var = 'L' if 'capital_fixed' in typ else 'K'
    other = 'K' if var == 'L' else 'L'
    for s in sample('diminishing_marginal_product', 60, seed=15, type=typ):
        terms = parse_terms(s.params['p_func'], 'LK')
        for v in (s.params['start'], s.params['end']):
            assert eval_terms(terms, {var: v, other: s.params['fixed']}) > 0, (s.params, v)


def test_dmp_letters_mapped():
    # "S represents the specialists, and T, the technology resources, is fixed" -> L and K
    from autosteer.sampler.template_fixes import _dmp_rewrite
    import pandas as pd
    row = pd.Series({'type': 'quadratic_capital_fixed', 'question_text':
                     'The campaign is described by {p_func}, where S represents the number of social media specialists, '
                     'and T, the technology resources, is fixed at {fixed}. If the number of social media specialists '
                     'increases from {start} to {end}, how does the marginal contribution change?'})
    out = _dmp_rewrite(row)
    assert 'where L represents' in out and 'and K, the technology' in out, out


@pytest.mark.parametrize('element', ['marginal_cost', 'average_cost'])
def test_cost_stories_define_output_and_wage(element):
    for s in sample(element, 60, seed=16):
        f = s.params['p_function']
        head = re.split(r'[.!?]\s', s.question[:s.question.index(f)])[-1]
        assert not re.search(r'revenue|profit|quality|effectiveness|performance\b', head, re.I), head


def test_dynamic_profit_discount_is_a_factor_not_a_rate():
    """Re-review: 'discounted by a rate of 0.63' read as a rate (1/1.63) while the key multiplies by 0.63."""
    from autosteer.sampler import load_manifest
    from autosteer.web.site import ElementSampler
    q = ElementSampler(manifest=load_manifest())._load_template('dynamic_profit_max').questions
    for t in q['question_text']:
        for sent in re.split(r'(?<=[.!?])\s+', t):
            if '{discount}' in sent:
                assert not re.search(r'\brate\b', sent, re.I), sent
    for s in sample('dynamic_profit_max', 40, seed=17):
        assert 'Answer with the increase in capital (K_2 minus K_1)' in s.question
