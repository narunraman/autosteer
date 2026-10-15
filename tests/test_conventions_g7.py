"""Review of October 2026, logic / arithmetic / probability elements: one test per fixed class of problem.

add_sub, mult_div, bayes_rule, complement_rule, compute_expectations (also tests/test_compute_expectations.py),
compute_expected_utility, categorical_syllogism, first/second_order_false_belief (tests/test_false_belief.py),
optimize_continuous, and the option formatting shared by every element (formatting.format_value).
"""
import re
from fractions import Fraction

from autosteer.formatting import format_value
from autosteer.filler_tools import categorical_syllogism_tools as cs
from autosteer.filler_tools.arithmetic_tools import NEG_SLOT, ROUNDING_NOTE, labels
from autosteer.filler_tools.probability_tools import BAYES_NOTE
from conventions import COMPLEMENT_REWORDS
from helpers import get_sampler, sample

N = 300


def _num(s: str) -> float:
    return float(re.sub(r'[^\d.\-]', '', s))


def test_format_value_keeps_two_decimals():
    """A number printed with 2 decimals by its filler stays so ("36.00%" next to "30.34%"; it was "36.0%", and the
    computed option was the only one with 2 decimals). Longer decimals are still rounded."""
    assert format_value('36.00%') == '36.00%'
    assert format_value('$77340.70 and $1.5') == '$77340.70 and $1.5'
    assert format_value('2.345678') == '2.35'
    assert format_value('7') == '7'


def test_add_sub_no_conflicting_negative_stories():
    """No story template writes an amount as a negative number and then asks for a distance or an amount owed
    ("descends -90.43 feet ... how far has the climber descended?", key -152.07)."""
    texts = set(get_sampler()._load_template('add_sub').questions.question_text)
    lab = labels('add_sub')
    for t in texts:
        if 'E' not in (lab[t]['flags'] or '') and NEG_SLOT.search(t):
            assert re.search(r'balance|score|elevation|negative numbers', t), t
    assert not any('descends -{a} feet' in t or 'overdraft of -{a}' in t for t in texts)


def test_mult_div_rounding_and_magnitude():
    """Non-exact questions state the rounding; every option is within a factor 3 of the answer (no quotient offered
    for a product of four-digit numbers)."""
    for s in sample('mult_div', N, seed=8101):
        x = _num(s.answer)
        assert all(1 / 3 <= abs(_num(o)) / abs(x) <= 3 for o in s.options), (s.question, s.options)
        if '.' in s.answer:
            assert s.question.endswith(ROUNDING_NOTE), s.question
        assert not re.search(r'[\w}] If the result', s.question), s.question


def test_bayes_rule_options_uniform():
    """Every option has 2 decimals and the question asks for them; at most one option is a number from the question."""
    for s in sample('bayes_rule', N, seed=8102):
        assert s.question.endswith(BAYES_NOTE), s.question
        assert all(re.fullmatch(r'\d+\.\d\d%', o) for o in s.options), s.options
        given = {float(v) for v in re.findall(r'(\d+)%', s.question)}
        assert sum(_num(o) in given for o in s.options if o != s.answer) <= 1, (s.question, s.options)


def test_complement_rule_wordings():
    """The asked event is the complement of the given ones: no 'will you lose' after a chance of winning (draws), no
    'late' after 'on time' (early), no home run / ground ball / fly ball."""
    texts = list(get_sampler()._load_template('complement_rule').questions.question_text)
    joined = ' '.join(texts)
    for old, new in COMPLEMENT_REWORDS:
        assert old in new or old not in joined, old
        assert new in joined, new
    assert not any('home run' in t for t in texts)


def test_expected_utility_options_in_range():
    """Every option lies between the smallest and the largest utility (an expected utility does), 2 decimals."""
    for s in sample('compute_expected_utility', N, seed=8103):
        us = [float(v) for v in re.findall(r'utility (?:=|of) (-?\d+)', s.question)]
        assert all(re.fullmatch(r'-?\d+\.\d\d', o) for o in s.options), s.options
        if len(us) >= 2:
            assert all(min(us) <= _num(o) <= max(us) for o in s.options), (s.question, s.options)


def test_syllogism_cannot_be_determined_only_if_nothing_follows():
    """'Cannot be determined' is keyed only if no conclusion relates the outer terms in either direction."""
    r = __import__('numpy').random.default_rng(7)
    for _ in range(400):
        prem, (x, y), ent = cs.draw(r, want_valid=False)
        assert not ent and not cs.entailed(prem, y, x), prem
    # the reviewed case: Some S are M; No M are P entails Some S are not P
    assert 'O' in cs.entailed([('I', 0, 1), ('E', 1, 2)], 0, 2)


def test_optimize_continuous_options_feasible():
    """'Which x' options are points of the interval; box cuts and fence sides are feasible lengths."""
    for s in sample('optimize_continuous', N, seed=8104):
        q, vals = s.question, [int(o) for o in s.options]
        m = re.search(r'(-?\d+) <= [xq] <= (-?\d+)|\[(-?\d+), (-?\d+)\]|between (\d+) and (\d+) hundred', q)
        if m and 'volume' not in q and 'area' not in q:
            lo, hi = (int(v) for v in [g for g in m.groups() if g is not None])
            assert all(lo <= v <= hi for v in vals), (q, vals)
        if 'side length x' in q:
            side = int(re.search(r'cardboard (\d+) cm', q).group(1))
            assert all(0 < v < side / 2 for v in vals), (q, vals)
        if 'perpendicular to the river' in q:
            L = int(re.search(r'(\d+) metres of fencing', q).group(1))
            assert all(0 < v < L / 2 for v in vals), (q, vals)
        if 'largest possible volume' in q or 'What is the largest area' in q:
            assert max(vals) != int(s.answer), (q, vals)  # the key is not simply the largest option
