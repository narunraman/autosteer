"""Helper for test_question_seeds: digest of per-question samples, in the given element order (run in a subprocess).

    python tests/_question_seed_dump.py <n> <forward|reverse> element ...
"""
import hashlib
import json
import sys
import warnings

warnings.filterwarnings('ignore')

from autosteer.release.rows import build_rows  # noqa: E402
from autosteer.sampler import ElementSampler  # noqa: E402
from autosteer.sampler.generator import derive_seed  # noqa: E402

n, order, elements = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
draws = [(e, derive_seed('xproc', e, i)) for e in elements for i in range(n)]
if order == 'reverse':
    draws = draws[::-1]
s = ElementSampler()
out = {}
for e, seed in draws:
    try:
        q = s.sample_question(e, seed)
        blob = json.dumps(build_rows(q, 'autosteer@test'), sort_keys=True)
    except Exception as x:  # noqa: BLE001
        blob = type(x).__name__
    out[f'{e}/{seed}'] = hashlib.sha256(blob.encode()).hexdigest()[:16]
print(json.dumps(out, sort_keys=True))
