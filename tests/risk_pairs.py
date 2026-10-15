"""Risk-pair rows for the certainty_effect / reflection_effect tests, drawn the way the exporter draws them.

Replaces STEER-generation's src/release/export_risk_pairs.generate (which these tests were written against):
n unique items of one type, each from sample_question with its own seed and turned into rows by
autosteer.release.rows, skipping draws that fail row_problems or repeat an item.
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from autosteer.release import schema as schema_mod  # noqa: F401  (tests call schema_mod.validate)
from autosteer.release.rows import QUOTED_RE, build_rows, check_pair, check_reflection, parse_mirror, parse_pair, row_problems  # noqa: F401
from autosteer.release.schema import to_table  # noqa: F401
from autosteer.sampler import ElementSampler, SamplerConfig
from autosteer.sampler.generator import derive_seed

_SAMPLER = None


def _sampler() -> ElementSampler:
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = ElementSampler()
    return _SAMPLER


def generate(element: str, n: int, seed: int, source: str = "", qtype: str = "two_part"):
    """(rows, stats) for n unique items of `element` and type `qtype` (source is ignored; rows carry generator)."""
    s = _sampler()
    rows, keys, stats = [], set(), {"draws": 0, "rejected": 0}
    i = 0
    while len(keys) < n:
        if i > 20 * n:
            raise RuntimeError(f"only {len(keys)} unique items after {i} draws")
        q = s.sample_question(element, derive_seed(seed, element, qtype, i), type=qtype)
        i += 1
        r = build_rows(q, "autosteer@test")
        if row_problems(q, r) or r[0]["base_id"] in keys:
            stats["rejected"] += 1
            continue
        keys.add(r[0]["base_id"])
        rows += r
    stats["draws"] = i
    return rows, stats


class _Sampler(ElementSampler):
    """Sequential sampler in template option order (option 0 = sure side), drawing from `pool`."""

    def __init__(self):
        super().__init__(shuffle_options=False)
        self.pool = "questions"
        self._pool_cache = {}

    def sample(self, config: SamplerConfig | None = None) -> dict:
        config = config or SamplerConfig()
        config.pool = self.pool
        return super().sample(config)


certainty_effect = SimpleNamespace(generate=partial(generate, "certainty_effect"), schema_mod=schema_mod,
                                   to_table=to_table, check_pair=check_pair, parse_pair=parse_pair)
