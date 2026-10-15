"""Repeated templates: none in the data or after the load-time corrections, and the balance they used to give.

The released data repeated templates to balance a field (independence_risk held its 18 vacation templates 19 times
each, to match its 337 meal templates). The repeats were deleted from the data; the sampler
draws the field's value first instead (SAMPLING_BALANCE). A template text with several fill settings (mult_div's
signs) is drawn as one template, then one of its settings.
"""
import collections

import pytest

from autosteer.sampler import ElementSampler
from autosteer.sampler.generator import SAMPLING_BALANCE
from autosteer.sampler.template_fixes import exact_copies

N = 1200


@pytest.fixture(scope="module")
def sampler():
    return ElementSampler()


@pytest.mark.parametrize("element", ["independence_risk", "compute_expectations", "subgame_nash", "complement_rule",
                                     "welfare_theorem_1", "welfare_theorem_2", "diminishing_marginal_utility"])
def test_no_copies(sampler, element):
    t = sampler._load_template(element)
    assert not exact_copies(t.questions, t.options)


@pytest.mark.parametrize("element", sorted(e for e, (_, w) in SAMPLING_BALANCE.items() if w is None))
def test_balanced(sampler, element):
    field = SAMPLING_BALANCE[element][0]
    seen = collections.Counter(str(sampler.sample_question(element, s)["metadata"][field])
                               for s in range(N))
    assert min(seen.values()) > 0.7 * N / len(seen), seen


def test_mult_div_signs_as_released(sampler):
    seen = collections.Counter(sampler.sample_question("mult_div", s)["metadata"]["type"]
                               for s in range(N))
    assert abs(seen["positive"] / N - 0.5) < 0.05 and abs(seen["negative"] / N - 0.25) < 0.05, seen


def test_a_set_type_is_kept(sampler):
    for s in range(50):
        assert sampler.sample_question("mult_div", s, type="negative")["metadata"]["type"] == "negative"
