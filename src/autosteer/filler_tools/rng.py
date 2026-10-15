"""Where fillers draw random numbers.

Fillers draw through three proxies: `pyrng` (stdlib-style: randint, uniform, choice, sample, shuffle, random),
`rng` (a numpy Generator: uniform, integers, random, choice, permutation, shuffle) and `legacy_np` (numpy's older
RandomState functions: rand, randint, uniform, choice). Each forwards to the generators bound for the question being
filled (bound()), so a question's numbers depend on its own seed only, and fills in different threads never share a
generator. With nothing bound, they forward to process-wide generators (fillers called directly, e.g. in tests).
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FillRng:
    py: random.Random
    np: np.random.Generator
    legacy: np.random.RandomState

    @classmethod
    def from_seed(cls, seed: int) -> FillRng:
        from autosteer.sampler.generator import derive_seed   # (late: the sampler imports the fillers)
        return cls(random.Random(derive_seed(seed, 'py')), np.random.default_rng(derive_seed(seed, 'np')),
                   np.random.RandomState(derive_seed(seed, 'legacy') % 2 ** 32))

    @classmethod
    def from_entropy(cls) -> FillRng:
        return cls(random.Random(), np.random.default_rng(), np.random.RandomState())


# With nothing bound: the stdlib's and numpy's global generators, and one shared Generator (what fillers used before
# they had their own; ElementSampler's sequential sampling still seeds these).
SHARED_NP = np.random.default_rng(seed=42)
_UNBOUND = FillRng(random._inst, SHARED_NP, np.random.mtrand._rand)
_CURRENT: ContextVar[FillRng | None] = ContextVar("fill_rng", default=None)


def current() -> FillRng:
    return _CURRENT.get() or _UNBOUND


@contextmanager
def bound(r: FillRng):
    """Fill with r's generators in this context (thread or task)."""
    token = _CURRENT.set(r)
    try:
        yield r
    finally:
        _CURRENT.reset(token)


class _Proxy:
    __slots__ = ("_which",)

    def __init__(self, which: str):
        self._which = which

    def __getattr__(self, name):
        return getattr(getattr(current(), self._which), name)

    def __repr__(self):
        return f"<{self._which} generator of the question being filled>"


pyrng = _Proxy("py")
rng = _Proxy("np")
legacy_np = _Proxy("legacy")


def legacy_state() -> np.random.RandomState:
    """The bound RandomState itself (for APIs that take a random_state, such as DataFrame.sample)."""
    return current().legacy
