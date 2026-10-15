"""Sampler behaviour: reproducibility, option shuffling, filled template options."""
import collections

import pytest

from autosteer.sampler import ElementSampler, SamplerConfig


def run(seed, element, n=10):
    s = ElementSampler(seed=seed)
    return [s.sample(SamplerConfig(element_name=element)) for _ in range(n)]


@pytest.mark.parametrize('element', ['find_eq_price', 'certainty_effect', 'law_of_demand'])
def test_seed_reproducible(element):
    assert run(7, element) == run(7, element)
    assert run(7, element) != run(8, element)


@pytest.mark.parametrize('element', ['find_eq_price', 'consumer_surplus', 'arbitrage',
                                     'welfare_theorem_1', 'diminishing_marginal_product'])
def test_correct_position_uniform(element):
    s = ElementSampler(seed=3)
    pos = collections.Counter()
    k = None
    n = 400
    for _ in range(n):
        r = s.sample(SamplerConfig(element_name=element))
        mask = [o['correct'] for o in r['options']]
        k = len(mask)
        pos[mask.index(True)] += 1
    for i in range(k):
        share = pos[i] / n
        # includes the last position, which several fillers never used
        assert abs(share - 1 / k) < 4 * ((1 / k) * (1 - 1 / k) / n) ** 0.5, (element, dict(pos))


def test_unshuffled_option_order_is_filler_order(monkeypatch):
    s = ElementSampler(seed=3, shuffle_options=False)
    fixed = lambda name, md: {'d_func': 'Q', 's_func': 'Q', 'options_list': [['1.0', '2.0', '3.0']],
                              'correct_answer': [[0, 0, 1]]}
    monkeypatch.setattr(s, '_call_filler', fixed)
    for _ in range(20):
        r = s.sample(SamplerConfig(element_name='find_eq_price'))
        assert [o['text'] for o in r['options']] == ['1.0', '2.0', '3.0']
        assert r['answer'] == '3.0'


def test_template_options_are_filled():
    s = ElementSampler(seed=0)
    for _ in range(30):
        r = s.sample(SamplerConfig(element_name='certainty_effect'))
        for part in r.get('sub_questions', [r]):
            assert all('{' not in o['text'] for o in part['options']), part['options']


def test_examples_pool():
    s = ElementSampler(seed=0)
    r = s.sample(SamplerConfig(element_name='law_of_demand', pool='examples'))
    assert r['question']
    with pytest.raises(ValueError):
        s.sample(SamplerConfig(element_name='law_of_demand', pool='nope'))


def test_format_value_rounds_each_number_once():
    from autosteer.dataset_filler import format_value
    assert format_value('1.5 and 21.53 and 1.5') == '1.5 and 21.53 and 1.5'
    assert format_value('x = 0.123, y = 10.12345') == 'x = 0.12, y = 10.12'
    assert format_value('{0} costs 3.14159') == '{0} costs 3.14'
