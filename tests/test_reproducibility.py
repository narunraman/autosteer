"""Same seed -> identical samples, across processes with different hash seeds.

Each element is sampled (test and few-shot pools) in two fresh interpreters
with different PYTHONHASHSEED values, both with ElementSampler(seed=...) and
with an unseeded sampler after seed_everything() (how the release exporter
seeds). Digests of the full outputs must match.
"""
import json
import os
import subprocess
import sys

from helpers import CONSISTENCY_GRADED, STRATEGIC, get_sampler  # noqa: F401
from autosteer.taxonomy import is_non_strategic_element

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def run(hashseed, elements, n=15):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([sys.executable, os.path.join(HERE, '_repro_dump.py'), ROOT, str(n), *elements],
                         env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_same_seed_same_output_across_processes():
    s = get_sampler()
    elements = sorted(e for e in s._sampleable if is_non_strategic_element(e) and e not in STRATEGIC)
    a, b = run(1, elements), run(2, elements)
    diff = sorted(k for k in a if a[k] != b[k])
    assert not diff, diff
