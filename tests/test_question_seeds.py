"""Per-question seeds: (element, template_id, seed) reproduces a question exactly, whatever was sampled before.

- sample_question(e, seed) equals sample_question(e, seed, template_id=<its template>);
- the same draws give the same rows in one process in any order, interleaved with other sampling;
- and in two fresh processes with different PYTHONHASHSEED values, one drawing in reverse order.
"""
import json
import os
import subprocess
import sys

import pytest

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler, SamplerConfig
from autosteer.sampler.generator import SEED_MAX, derive_seed

HERE = os.path.dirname(os.path.abspath(__file__))
N = 5
SAMPLER = ElementSampler()


def sampleable():
    """Elements that sample at all (sc_* produce duplicate options and are not released)."""
    return sorted(e for e in SAMPLER._sampleable if not e.startswith('sc_'))


def test_derive_seed_is_stable_and_in_range():
    assert derive_seed(1, 'a', 2) == derive_seed(1, 'a', 2) != derive_seed(1, 'a', 3)
    assert derive_seed(20261003, 'law_of_demand', 'price_decrease', 'consumer_goods', 0, 0) == \
        derive_seed('20261003', 'law_of_demand', 'price_decrease', 'consumer_goods', '0', '0')
    assert all(0 <= derive_seed(i) <= SEED_MAX for i in range(100))


@pytest.mark.parametrize('seed', [-1, 2 ** 63, 1.5, '3', True])
def test_seed_must_be_int64_nonnegative(seed):
    with pytest.raises(ValueError):
        SAMPLER.sample_question('add_sub', seed)


@pytest.mark.parametrize('element', sampleable())
def test_template_id_and_seed_reproduce_the_question(element):
    for i in range(N):
        seed = derive_seed('tid', element, i)
        q = SAMPLER.sample_question(element, seed)
        again = SAMPLER.sample_question(element, seed, template_id=q['template_id'])
        assert again == q
        assert build_rows(again, 'g') == build_rows(q, 'g')


def test_order_and_other_sampling_do_not_matter():
    elements = ['law_of_demand', 'certainty_effect', 'iterated_removal', 'compute_expectations', 'find_eq_price']
    draws = [(e, derive_seed('order', e, i)) for e in elements for i in range(4)]
    first = {d: SAMPLER.sample_question(*d) for d in draws}
    other = ElementSampler(seed=99)
    for d in reversed(draws):
        other.sample(SamplerConfig(element_name='consumer_surplus'))  # moves the global RNGs
        assert SAMPLER.sample_question(*d) == first[d]
        assert ElementSampler().sample_question(*d) == first[d]


def _dump(hashseed, order, elements):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([sys.executable, os.path.join(HERE, '_question_seed_dump.py'), '3', order, *elements],
                         env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_same_rows_across_processes_and_orders():
    elements = sampleable()
    a, b = _dump(1, 'forward', elements), _dump(2, 'reverse', elements)
    assert a.keys() == b.keys()
    diff = sorted(k for k in a if a[k] != b[k])
    assert not diff, diff[:10]
