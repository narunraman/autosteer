"""Generic checks over every non-strategic element in the manifest.

For each element, sample N questions (default 1,000, env STEER_TEST_N; seeded) and check:
- sampling does not raise;
- every question part has non-empty, unique options;
- exactly one option is marked correct (except consistency-graded elements);
- no unfilled {placeholder}, bare filler key (e.g. num_victims), None, nan
  or inf in question or options;
- the correct index is not degenerate: every position is used and none is
  more than 3.5 binomial standard deviations above 1/k.
"""
import collections
import os
import re

import pytest

from helpers import CONSISTENCY_GRADED, STRATEGIC, get_sampler, sample
from autosteer.taxonomy import is_non_strategic_element

# Large enough that rare bad draws show up; override with STEER_TEST_N.
N = int(os.environ.get('STEER_TEST_N', 1000))

# Elements outside this audit pass whose fillers/templates are still broken.
KNOWN_FAILURES = {
}


def elements():
    s = get_sampler()
    return sorted(e for e in s._sampleable if is_non_strategic_element(e) and e not in STRATEGIC)


NUMERIC = re.compile(r'[-+]?\$?\d+(?:\.\d+)?%?')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
BAD_TOKEN = re.compile(r'\b(None|nan|NaN|inf|No options found)\b')


def params(e):
    if e in KNOWN_FAILURES:
        return pytest.param(e, marks=pytest.mark.xfail(reason=KNOWN_FAILURES[e], strict=False))
    return e


@pytest.mark.parametrize('pool', ['questions', 'examples'])
@pytest.mark.parametrize('element', [params(e) for e in elements()])
def test_element(element, pool):
    try:
        got = sample(element, N, seed=12345, pool=pool)
    except ValueError as err:
        if pool == 'examples' and 'No template questions' in str(err):
            pytest.skip('element has no example templates')
        raise
    positions = collections.Counter()
    counts = collections.Counter()
    for s in got:
        assert s.options and all(str(o).strip() for o in s.options), s.options
        assert len(set(s.options)) == len(s.options), s.options
        # also distinct as numbers ('1' vs '1.0' were two correct options)
        as_num = [float(o.strip().replace('$', '').rstrip('%')) if NUMERIC.fullmatch(o.strip()) else o
                  for o in s.options]
        assert len(set(as_num)) == len(as_num), s.options
        blob = s.question + ' || ' + ' | '.join(s.options)
        assert not PLACEHOLDER.search(blob), blob
        assert not BAD_TOKEN.search(blob), blob
        for key in s.params:
            if '_' in key:
                assert not re.search(rf'\b{re.escape(key)}\b', blob), (key, blob)
        n_correct = sum(s.correct)
        if element in CONSISTENCY_GRADED:
            continue
        assert n_correct == 1, (s.question, s.options, s.correct)
        positions[(len(s.options), s.answer_index)] += 1
        counts[len(s.options)] += 1
    if element in CONSISTENCY_GRADED:
        return
    k, total = counts.most_common(1)[0]
    shares = [positions[(k, i)] / total for i in range(k)]
    # uniform would give 1/k each; allow 3.5 binomial standard deviations
    bound = 1 / k + 3.5 * ((1 / k) * (1 - 1 / k) / total) ** 0.5
    assert min(shares) > 0 and max(shares) <= bound, (k, shares, bound)
