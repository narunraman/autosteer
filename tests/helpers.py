"""
Test helpers: sample questions from the on-the-fly sampler and expose the
values that were substituted into each question.

`sample(element, n, seed=..., **config)` returns a list of `Sample`s. Each
has the question text, option texts, the index of the option marked
correct, the metadata, and `params`: the filler's placeholder values after
the same 2-decimal formatting that is applied to the question text, i.e. the
numbers a reader of the question actually sees.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from autosteer.sampler import ElementSampler, SamplerConfig
from autosteer.dataset_filler import format_fill_values

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Elements that belong to STEER (strategic) and are not part of STEER-ME.
STRATEGIC = {
    'backward_induction', 'trigger', 'enforceability', 'feasibility',
    'iterated_removal', 'best_response',
}
# Consistency-graded elements: no single correct option required.
CONSISTENCY_GRADED = {
    'certainty_effect', 'ambiguity_aversion', 'endowment_effect',
    'reflection_effect', 'independence', 'completeness', 'completeness_risk',
}


@dataclass
class Sample:
    question: str
    options: list[str]
    correct: list[bool]
    metadata: dict
    params: dict = field(default_factory=dict)

    @property
    def answer_index(self) -> Optional[int]:
        return self.correct.index(True) if sum(self.correct) == 1 else None

    @property
    def answer(self) -> Optional[str]:
        i = self.answer_index
        return None if i is None else self.options[i]

    @property
    def type(self) -> str:
        return self.metadata['type']


class RecordingSampler(ElementSampler):
    """ElementSampler that remembers the placeholder values of the last filler call."""

    def _call_filler(self, element_name, metadata):
        result = super()._call_filler(element_name, metadata)
        shown = {k: v for k, v in result.items() if k not in ('options_list', 'correct_answer')}
        self.last_params = format_fill_values(dict(shown))
        return result


@lru_cache(maxsize=None)
def get_sampler() -> RecordingSampler:
    return RecordingSampler()


def sample(element: str, n: int = 50, seed: int = 0, **config) -> list[Sample]:
    """Sample n single-part questions of `element` (config: type=, domain=, pool=, ...)."""
    s = get_sampler()
    s.seed(seed)
    out = []
    for _ in range(n):
        r = s.sample(SamplerConfig(element_name=element, **config))
        parts = r.get('sub_questions', [r])
        for part in parts:
            out.append(Sample(
                question=part['question'],
                options=[o['text'] for o in part['options']],
                correct=[o['correct'] for o in part['options']],
                metadata=r['metadata'],
                params=dict(s.last_params),
            ))
    return out


SOLVERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solvers')


def load_steer_solver(name: str):
    """Load tests/solvers/steer/solvers/<name>.py by path (its package name would clash with the STEER-ME
    solvers, which tests import as the package `solvers`)."""
    import importlib.util
    import sys
    key = f'steer_solver_{name}'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, os.path.join(SOLVERS_DIR, 'steer', 'solvers', f'{name}.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


NUM = r'[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?'


def num(s) -> float:
    """Parse a number that may carry $, %, commas or surrounding text."""
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(NUM, str(s).replace(',', '').replace('$', ''))
    if not m:
        raise ValueError(f'no number in {s!r}')
    return float(m.group(0))


def matching(options: list[str], value: float, rel: float = 1e-9, abs_: float = 0.006) -> list[int]:
    """
    Indices of numeric options within tolerance of value.

    Default tolerance (0.006 absolute) is for answers recomputed from the
    displayed parameters when the filler also computes from those rounded
    values: both sides then agree up to the final 2-decimal rounding.
    """
    idx = []
    for i, o in enumerate(options):
        try:
            v = num(o)
        except ValueError:
            continue
        if abs(v - value) <= max(abs_, rel * abs(value)):
            idx.append(i)
    return idx
