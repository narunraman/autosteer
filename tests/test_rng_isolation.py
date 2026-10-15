"""A seeded question depends on (element, seed) only: the same from a fresh sampler, after other draws, and when
drawn concurrently in threads; and filling never touches the stdlib's or numpy's global generators."""
import json
import random
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from autosteer.filler_tools import rng as fill_rng
from autosteer.sampler import ElementSampler, SamplerConfig

SEED = 7


def _q(s, e, seed=SEED):
    try:
        return json.dumps(s.sample_question(e, seed), sort_keys=True, default=str)
    except Exception as x:  # noqa: BLE001
        return type(x).__name__


@pytest.fixture(scope="module")
def sampler():
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    return ElementSampler(seed=0)


def test_seeded_questions_are_independent_of_history_and_threads(sampler):
    elements = sorted(sampler._sampleable)
    fresh = {e: _q(sampler, e) for e in elements}
    other = ElementSampler(seed=3)
    for e in elements:   # unrelated draws first: sequential, and other seeds
        try:
            other.sample(SamplerConfig(element_name=e))
        except Exception:  # noqa: BLE001
            pass
        _q(sampler, e, seed=SEED + 1)
    after = {e: _q(sampler, e) for e in elements}
    with ThreadPoolExecutor(8) as pool:
        threaded = dict(zip(elements, pool.map(lambda e: _q(sampler, e), elements)))
    assert [e for e in elements if after[e] != fresh[e]] == []
    assert [e for e in elements if threaded[e] != fresh[e]] == []


def test_filling_leaves_the_global_generators_alone(sampler):
    before = (random.getstate(), np.random.get_state()[1].tobytes(), fill_rng.SHARED_NP.bit_generator.state)
    for e in sorted(sampler._sampleable):
        _q(sampler, e)
    after = (random.getstate(), np.random.get_state()[1].tobytes(), fill_rng.SHARED_NP.bit_generator.state)
    assert after == before
