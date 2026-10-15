"""Digest of seeded samples per (element, pool), printed as one JSON line.

    python tests/_digest_dump.py autosteer <n> [element ...]     # this package
    python tests/_digest_dump.py src <n> [element ...]           # a STEER-generation checkout (run from its root)

Used by test_source_digests.py. The reference file tests/reference/source_digests.json was written by running the
`src` mode in a checkout of the source commit named in it. For each element and pool the
sampler is reseeded with SEED and n questions are drawn with the sequential API; a draw that raises is recorded
as the exception's class name, so an element that fails in both codebases still compares equal.
"""
import hashlib
import json
import os
import sys
import warnings

SEED = 1234


def digests(sampler_cls, config_cls, n, elements=None):
    s = sampler_cls(seed=0)
    out = {}
    for e in elements or sorted(s._sampleable):
        for pool in ('questions', 'examples'):
            s.seed(SEED)
            items = []
            for _ in range(n):
                try:
                    items.append(s.sample(config_cls(element_name=e, pool=pool)))
                except Exception as x:  # noqa: BLE001
                    items.append(type(x).__name__)
            blob = json.dumps(items, sort_keys=True, default=str).encode()
            out[f'{e}/{pool}'] = hashlib.sha256(blob).hexdigest()[:16]
    return out


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    import numpy as np
    np.seterr(all='ignore')
    mode, n = sys.argv[1], int(sys.argv[2])
    if mode == 'src':
        sys.path.insert(0, os.getcwd())
        from src.sampler import ElementSampler, SamplerConfig
    else:
        from autosteer.sampler import ElementSampler, SamplerConfig
    print(json.dumps(digests(ElementSampler, SamplerConfig, n, sys.argv[3:] or None), sort_keys=True))
