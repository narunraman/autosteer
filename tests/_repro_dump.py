"""Helper for test_reproducibility: print a digest of seeded samples per element (run in a subprocess)."""
import hashlib
import json
import random
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path[:0] = [sys.argv[1]]

import numpy as np  # noqa: E402

from autosteer.sampler import ElementSampler, SamplerConfig  # noqa: E402
from autosteer.sampler.generator import seed_everything  # noqa: E402

n = int(sys.argv[2])
elements = sys.argv[3:]
out = {}
seeded = ElementSampler(seed=0)
unseeded = ElementSampler()
for e in elements:
    for mode, s in (('seeded', seeded), ('global', unseeded)):
        if mode == 'seeded':
            s.seed(1234)
        else:
            seed_everything(1234)
        items = []
        for pool in ('questions', 'examples'):
            for _ in range(n):
                try:
                    items.append(s.sample(SamplerConfig(element_name=e, pool=pool)))
                except Exception as x:  # noqa: BLE001
                    items.append(f'{type(x).__name__}')
        out[f'{e}/{mode}'] = hashlib.sha1(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()
print(json.dumps(out))
