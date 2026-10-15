"""Helper for test_reproducibility: print a digest of seeded samples per element (run in a subprocess)."""
import hashlib
import json
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path[:0] = [sys.argv[1]]

from autosteer.sampler import ElementSampler, SamplerConfig  # noqa: E402

n = int(sys.argv[2])
elements = sys.argv[3:]
out = {}
s = ElementSampler(seed=0)
for e in elements:
    s.seed(1234)
    items = []
    for pool in ('questions', 'examples'):
        for _ in range(n):
            try:
                items.append(s.sample(SamplerConfig(element_name=e, pool=pool)))
            except Exception as x:  # noqa: BLE001
                items.append(f'{type(x).__name__}')
    out[f'{e}/seeded'] = hashlib.sha1(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()
    qs = []
    for seed in range(n):
        try:
            qs.append(s.sample_question(e, seed))
        except Exception as x:  # noqa: BLE001
            qs.append(f'{type(x).__name__}')
    out[f'{e}/question'] = hashlib.sha1(json.dumps(qs, sort_keys=True, default=str).encode()).hexdigest()
print(json.dumps(out))
