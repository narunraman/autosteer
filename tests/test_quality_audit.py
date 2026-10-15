"""Quality audit (October 2026): generated questions of the elements fixed in the audit have none of the defects a
reader would notice. The checks are those of scripts/quality_audit.py (placeholders, values printed as 0.0 / -0.0 /
nan, long decimals, scientific notation, Python reprs, duplicate or numerically equal options, not exactly one
correct option, double or missing spaces, a repeated name where a possessive belongs), plus a check per fixed
defect.
"""
import re
import sys
from pathlib import Path

import pytest

from autosteer.sampler import ElementSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from quality_audit import check  # noqa: E402

pytestmark = pytest.mark.slow

N = 60
BANNED = {'placeholder', 'neg_zero', 'zero_decimal', 'bad_value', 'long_decimal', 'sci_notation', 'repr',
          'dup_options', 'equal_numeric_options', 'double_space', 'missing_space', 'repeated_name_possessive'}
FIXED = ['feasibility', 'trigger', 'enforceability', 'subgame_nash', 'input_price_response', 'output_elasticity',
         'supply_elasticity', 'dynamic_profit_max', 'price_elasticity', 'marginal_cost', 'average_cost',
         'diminishing_marginal_product', 'capital_market_distortions', 'plurality_voting', 'interpret_games',
         'complement_rule', 'consumption_changes', 'engel_curves', 'income_elasticity', 'mrs_tangency',
         'optimal_portfolio_choice', 'loss_aversion', 'mrs_labor_supply', 'substitutes_complements', 'tfp_shocks',
         'deadweight_loss', 'find_eq_price', 'profit_maximization', 'labor_supply_distortions', 'consumer_surplus',
         'producer_surplus', 'efficient_surplus', 'marginal_production', 'returns_to_scale']


@pytest.fixture(scope='module')
def sampler():
    return ElementSampler()


def draws(sampler, element, n=N):
    q, _ = sampler._pool_frames(sampler._load_template(element), 'questions')
    types = sorted(set(str(t) for t in q['type'].dropna().unique()))
    for i in range(n):
        seed = 9100 + i
        yield (sampler.sample_question(element, seed, type=types[i % len(types)]) if i % 2 == 0
               else sampler.sample_question(element, seed))


@pytest.mark.parametrize('element', FIXED)
def test_no_reader_visible_defects(sampler, element):
    bad = []
    for q in draws(sampler, element):
        for name, part, ex in check(q, consistency=False):
            if name in BANNED or name.startswith('correct_count'):
                bad.append((name, q['template_id'], q['seed'], ex))
    assert not bad, (len(bad), bad[:4])


def test_price_elasticity_convention(sampler):
    """The question states the simple percentage-change convention; the midpoint value is never an option;
    quantities are whole numbers."""
    for q in draws(sampler, 'price_elasticity'):
        p = q['parts'][0]
        text = p['question']
        assert 'relative to the initial price and quantity' in text
        nums = [float(x) for x in re.findall(r'from \$?(\d+(?:\.\d+)?)(?: [a-z]+)? to \$?(\d+(?:\.\d+)?)', text)
                for x in x]
        assert len(nums) == 4, text
        (a0, a1), (b0, b1) = nums[:2], nums[2:]
        # which pair is the price: the one stated first in the template varies; quantities are whole
        prices, qty = ((a0, a1), (b0, b1)) if a0 != int(a0) or a1 != int(a1) else ((b0, b1), (a0, a1))
        assert qty[0] == int(qty[0]) and qty[1] == int(qty[1]), text
        simple = ((qty[1] - qty[0]) / qty[0]) / ((prices[1] - prices[0]) / prices[0])
        arc = ((qty[1] - qty[0]) / (qty[0] + qty[1])) / ((prices[1] - prices[0]) / (prices[0] + prices[1]))
        vals = [float(o) for o in p['options']]
        assert vals[p['correct'].index(True)] == round(simple, 2)
        assert all(abs(v - arc) >= 0.04 for v in vals), (text, vals, arc)


def test_cost_wording(sampler):
    """'the cost of labor per hour is 8.47L': a wage is a number, not a cost function."""
    per_hour = re.compile(r'(per (?:[\w-]+ )?hour|hourly [\w-]+|wages?)\b[^.,;]{0,40}?\b\d+(?:\.\d+)?L\b', re.I)
    for element in ('marginal_cost', 'average_cost'):
        for q in draws(sampler, element):
            assert not per_hour.search(q['parts'][0]['question']), q['parts'][0]['question']


def test_template_wording(sampler):
    for q in draws(sampler, 'diminishing_marginal_product'):
        assert not re.search(r'\bC\b', q['parts'][0]['question']), q['parts'][0]['question']
    for q in draws(sampler, 'capital_market_distortions'):
        assert not re.search(r'\ban? \d+(?:\.\d+)?% on\b', q['parts'][0]['question']), q['parts'][0]['question']
    for q in draws(sampler, 'plurality_voting', 20):
        for line in q['parts'][0]['question'].split('\n'):
            assert line.count('voters voted for') <= 1 and not re.search(r'voted for [A-Z >]+ [A-Z][a-z]', line), line
            if 'voters voted for' in line:
                assert line.startswith('- '), line
    for q in draws(sampler, 'subgame_nash', 20):
        assert "opponents's" not in q['parts'][0]['question']


def test_supply_intercept(sampler):
    for q in draws(sampler, 'supply_elasticity'):
        m = re.search(r'Q = ([\d.]+)P \+ ([\d.]+)\b(?!P)', q['parts'][0]['question'])
        if m:
            assert float(m.group(2)) >= 0.5, q['parts'][0]['question']


def test_dynamic_profit_numbers(sampler):
    for q in draws(sampler, 'dynamic_profit_max'):
        text = q['parts'][0]['question']
        m = re.search(r'factor of (\d\.\d+)', text)
        if m:
            assert float(m.group(1)) >= 0.5, text
        assert not re.search(r'probability 0(?:\.0+)?\b(?!\.)', text), text


def test_hicksian_and_deadweight(sampler):
    """deriving_hicksian_demand: no 3.8e+19 keys (exponents >= 0.1), no "x3 , where"; the linear type's key may be 0
    (a corner). deadweight_loss: no option printed as 0.0 or 0.01-0.04."""
    for q in draws(sampler, 'deriving_hicksian_demand'):
        p = q['parts'][0]
        assert ' ,' not in p['question'] and '  ' not in p['question'], p['question']
        assert not any('e+' in o or 'e-' in o for o in p['options']), p['options']
    for q in draws(sampler, 'deadweight_loss'):
        p = q['parts'][0]
        vals = [float(o) for o in p['options'] if re.fullmatch(r'-?\d+(?:\.\d+)?', o)]
        assert all(v >= 0.05 for v in vals), p["options"]


def test_trigger_matrix_last(sampler):
    """The trigger payoff table is printed after the question text, not inside a sentence ("matrix \\n<table>\\n, what")."""
    seen = 0
    for q in draws(sampler, 'trigger', 200):
        text = q['parts'][0]['question']
        if '\nCooperate' in text or re.search(r'\n\s+Action \w+\s+Action', text):
            seen += 1
            after = text[text.index('\n\n'):]
            assert not re.search(r'\n\s*[,.]|[a-z]{3,}', after.split('\n', 2)[-1].replace('Cooperate', '').replace('Defect', '')
                                 .replace('Action', '')), text
    assert seen
