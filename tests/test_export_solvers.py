"""Run the export agent's independent text-only solvers on fresh samples.

The solvers (audit/steer_me/solvers/, written by the export agent)
re-derive each answer from the question text alone. For every element they
cover, sample N questions and require, for every question the solver can
parse and that is well defined, that the marked option is the solver's
answer and that exactly one option matches it. Their tolerance is
max(0.011, 2% relative) because some fillers compute from unrounded values.
"""
import collections
import importlib
import os
import pkgutil
import sys

import pytest

from helpers import sample

# the solvers live in audit/steer_me/solvers (package name `solvers`)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solvers', 'steer_me'))
import solvers  # noqa: E402
from solvers.common import SOLVERS  # noqa: E402

pytestmark = pytest.mark.slow

for m in pkgutil.iter_modules(solvers.__path__):
    if m.name.startswith('el_'):
        importlib.import_module(f'solvers.{m.name}')

N = int(os.environ.get('STEER_SOLVER_N', 2000))
MIN_SCORED = 0.5   # at least half the sampled questions must be parseable

# elements whose solver disagrees for a reason documented in the audit report
KNOWN = {}


def score(element, n=N, seed=2024):
    """(status counts, disagreements, multi-correct cases) for n fresh samples."""
    stats = collections.Counter()
    bad, multi = [], []
    for s in sample(element, n, seed=seed):
        r = SOLVERS[element](s.question, s.options, s.type)
        stats[r.status] += 1
        if r.status != 'ok':
            continue
        marked = [i for i, c in enumerate(s.correct) if c]
        case = (s.type, s.question[:300], s.options, marked, r.expected, r.note)
        if sum(r.correct) > 1:
            multi.append(case)
        elif not (len(marked) == 1 and r.correct[marked[0]]):
            bad.append(case)
    return stats, bad + multi, multi


@pytest.mark.parametrize('element', sorted(SOLVERS))
def test_solver_agrees(element):
    """>= 99.9% agreement on scored questions and no question with two solver-correct options."""
    if element in KNOWN:
        pytest.xfail(KNOWN[element])
    stats, bad, multi = score(element)
    total = sum(stats.values())
    assert stats['ok'] >= MIN_SCORED * total, dict(stats)
    assert not multi, (len(multi), multi[:3])
    assert len(bad) <= 0.001 * stats['ok'], (len(bad), stats, bad[:3])
