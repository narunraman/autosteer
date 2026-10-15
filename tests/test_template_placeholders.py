"""Every placeholder in a stored template is one its element's filler returns (the templates that used one it never
returns were removed from data/ in October 2026; the sampler used to drop them at load).

For each metadata combination of an element's templates the filler is called 5 times from a fixed seed, and the keys it
returns are collected; a template needs all its placeholders among them. Elements whose templates are written in code
are built with their filler and are skipped."""
import warnings

import numpy as np
import pytest

from autosteer.filler_tools import code_templates, function_registry
from autosteer.formatting import extract_fstring_keys
from autosteer.sampler import ElementSampler
from autosteer.filler_tools.rng import FillRng, bound
from autosteer.sampler.generator import derive_seed

TRIES = 5


@pytest.fixture(scope="module")
def sampler():
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    return ElementSampler(seed=0)


def elements():
    s = ElementSampler(seed=0)
    return sorted(e for e in s._sampleable if not code_templates.has_code_templates(e) and not code_templates.is_composite(e))


@pytest.mark.parametrize("element", elements())
def test_every_placeholder_is_filled(sampler, element):
    fn = function_registry.get(element)
    t = sampler._load_template(element)
    keys: dict[tuple, set] = {}
    missing = []
    for q in (t.questions, t.example_questions):
        if q is None or len(q) == 0:
            continue
        for row in q.itertuples(index=False):
            md = (row.domain, row.type, row.difficulty_level, row.tags)
            if md not in keys:
                meta = {"domain": md[0], "type": md[1], "difficulty_level": int(md[2]), "tags": md[3]}
                got = set()
                with bound(FillRng.from_seed(derive_seed("placeholders", element, *map(str, md)))):
                    for _ in range(TRIES):
                        try:
                            got |= set(fn(dict(meta)) or {})
                        except Exception:  # noqa: BLE001 - a draw that fails is retried, as at fill time
                            pass
                keys[md] = got
            need = extract_fstring_keys(row.question_text) - keys[md]
            if need:
                missing.append((row.question_id, md, sorted(need)))
    assert not missing, f"{len(missing)} templates use placeholders the filler never returns, e.g. {missing[:3]}"
