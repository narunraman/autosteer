"""Golden digests of what autoSTEER produces, so a refactor that changes any output is caught.

    uv run python tests/golden.py write            # (re)write tests/reference/golden_digests.json
    uv run python tests/golden.py show <key>       # print the outputs behind one key, to diff before and after

Per element: seeded sequential draws for both pools (`seq/`), seeded single questions (`q/`), the template ids of
each pool after the load-time fixes (`ids/`), the manifest entry (`manifest/`) and dataset rows (`rows/`); and a
small export with few-shot rows for a few elements (`export/`). A draw that raises is recorded as the exception's
class name. test_golden.py compares against the file; when an output change is intended, rewrite it in the same
commit and say why.
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

REFERENCE = Path(__file__).with_name("reference") / "golden_digests.json"
SEQ_SEED, N_SEQ, N_Q, N_ROWS = 1234, 8, 8, 4
EXPORT_ELEMENTS = ("consumer_surplus", "independence", "law_of_demand")
EXPORT_SEED = 20261003


def _quiet():
    warnings.filterwarnings("ignore")
    import numpy as np
    np.seterr(all="ignore")


def _try(f):
    try:
        return f()
    except Exception as x:  # noqa: BLE001
        return type(x).__name__


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def outputs(keys=None):
    """{key: output} for every golden key (or only those in `keys`, a set of keys or key prefixes)."""
    _quiet()
    from autosteer.release.export import export_element
    from autosteer.release.rows import build_rows
    from autosteer.sampler import ElementSampler, SamplerConfig
    from autosteer.sampler.manifest import load_manifest

    want = (lambda k: True) if keys is None else (lambda k: any(k == x or k.startswith(x) for x in keys))
    s = ElementSampler(seed=0)
    manifest = load_manifest()
    out = {}
    for e in sorted(s._sampleable):
        if want(f"manifest/{e}"):
            out[f"manifest/{e}"] = manifest["elements"].get(e)
        for pool in ("questions", "examples"):
            if want(f"seq/{e}/{pool}"):
                s.seed(SEQ_SEED)
                out[f"seq/{e}/{pool}"] = [_try(lambda: s.sample(SamplerConfig(element_name=e, pool=pool))) for _ in range(N_SEQ)]
            if want(f"ids/{e}/{pool}"):
                out[f"ids/{e}/{pool}"] = _try(lambda: s.template_ids(e, pool))
        if want(f"q/{e}"):
            out[f"q/{e}"] = [_try(lambda: s.sample_question(e, seed)) for seed in range(N_Q)]
        if want(f"rows/{e}"):
            out[f"rows/{e}"] = [_try(lambda: build_rows(s.sample_question(e, seed), "golden", manifest)) for seed in range(N_ROWS)]
    for e in EXPORT_ELEMENTS:
        if want(f"export/{e}"):
            r = export_element(e, EXPORT_SEED, per_cell=2, floor=10, max_factor=5, generator="golden", few_shot_per_type=1)
            out[f"export/{e}"] = {"rows": r["rows"], "few_shot_rows": r["few_shot_rows"]}
    return out


def digests():
    return {k: _digest(v) for k, v in outputs().items()}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "write":
        d = digests()
        REFERENCE.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(d)} digests to {REFERENCE}")
    elif cmd == "show" and len(sys.argv) > 2:
        print(json.dumps(outputs(set(sys.argv[2:])), indent=1, sort_keys=True, default=str))
    else:
        print(__doc__)
