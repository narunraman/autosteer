"""compute_expected_utility, paper (ICML) version (recovered 2026-10; src/filler_tools/expectation_tools.py).

On 1,000 seeded questions:
- independent recomputation: the question is matched against its template text (data/templates/.../
  template_labels.csv, used only as a parsing grammar), the probabilities and utilities are read back, the last
  probability is 100 minus the others when the template leaves it implicit, and the expected utility
  sum(p u) / 100 must be the only option within 0.006 and the keyed one;
- probabilities are positive integers summing to 100, utilities distinct; 3 distinct options at least max(0.5, 5%)
  apart, no two sharing their decimals (the released options were the key +/- whole numbers);
- the audit solver (audit/steer/solvers/expected_utility.py) agrees wherever it scores;
- structure vs the staged release rows (STEER HF test split, if present): 2 / 3 outcome mix (difficulty 3 / 4) and
  domain shares within 5 points, type default, 3 options;
- the same seed gives the same questions.
"""
import collections
import importlib.util
import json
import os
import re

import pandas as pd

from helpers import get_sampler, sample
from autosteer.directory_utils import TEMPLATES_PATH
from autosteer.sampler import SamplerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get('STEER_TEST_N', 1000))
# a staged release parquet of compute_expected_utility to compare with (optional)
STAGED = os.environ.get('STEER_STAGED_EU_PARQUET', '')
LAB = pd.read_csv(os.path.join(TEMPLATES_PATH, 'compute_expected_utility', 'template_labels.csv'))
_C = []


def got():
    if not _C:
        _C.extend(sample('compute_expected_utility', N, seed=20261003))
    return _C


def grammar(t):
    out, names = '', []
    for part in re.split(r'(\{\w+\})', t):
        m = re.fullmatch(r'\{(\w+)\}', part)
        if m:
            if m.group(1) in names:
                out += f'(?P={m.group(1)})'
            else:
                names.append(m.group(1))
                out += rf'(?P<{m.group(1)}>-?\d+)'
        else:
            out += re.escape(part)
    return re.compile(out + '$')


GRAMMARS = [(grammar(t), json.loads(lab)) for t, lab in zip(LAB.template, LAB.labels)]


def recompute(q):
    hits = [(m, lab) for g, lab in GRAMMARS for m in [g.match(q)] if m]
    assert len(hits) == 1, q
    m, lab = hits[0]
    vals = {k: int(v) for k, v in m.groupdict().items()}
    P = [vals[k] for k in sorted(k for k in vals if k.startswith('p'))]
    if lab['implicit_last']:
        P.append(100 - sum(P))
    U = [0 if name == '0' else vals[name] for name, _, _ in lab['utilities']]
    assert len(P) == len(U) == lab['n_outcomes'] and sum(P) == 100 and min(P) >= 1 and len(set(U)) == len(U)
    return sum(p * u for p, u in zip(P, U)) / 100, len(U)


def test_recomputed_key():
    for s in got():
        v, _ = recompute(s.question)
        vals = [float(o) for o in s.options]
        assert [i for i, x in enumerate(vals) if abs(x - v) <= 0.006] == [s.answer_index], (s.question, s.options, v)
        assert len(set(vals)) == 3
        assert all(abs(a - b) >= max(0.5, 0.05 * max(abs(a), abs(b))) - 0.011 for i, a in enumerate(vals)
                   for b in vals[i + 1:]), s.options
        assert len({round(x * 100) % 100 for x in vals}) == 3, s.options


def test_structure_vs_release():
    n_out = collections.Counter(recompute(s.question)[1] for s in got())
    assert set(n_out) == {2, 3} and 0.15 < n_out[3] / N < 0.45, n_out
    for s in got():
        assert s.metadata['difficulty_level'] == (4 if recompute(s.question)[1] == 3 else 3)
        assert s.metadata['type'] == 'default' and len(s.options) == 3
    doms = collections.Counter(s.metadata['domain'] for s in got())
    assert set(doms) == {'job', 'vacation', 'medical', 'investment', 'lottery'}, doms
    if os.path.exists(STAGED):
        st = pd.read_parquet(STAGED)
        assert abs(n_out[3] / N - (st.difficulty == 4).mean()) < 0.05, (n_out, (st.difficulty == 4).mean())
        ref = st.domain.value_counts(normalize=True)
        assert all(abs(doms[d] / N - ref[d]) < 0.05 for d in ref.index), (doms, ref.to_dict())
        # 28 of 995 staged rows have 2 options (a duplicate option removed in repair); the generator's count is 3
        assert (st.options.map(len) == 3).mean() > 0.95 and set(st.type) == {'default'}


def test_audit_solver():
    spec = importlib.util.spec_from_file_location('audit_eu', os.path.join(ROOT, 'tests/solvers/steer/solvers/expected_utility.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.solve_compute_eu(pd.DataFrame({'question_text': [s.question for s in got()],
                                             'options': [s.options for s in got()]}))
    ok = [(p, s.answer_index) for p, st, s in zip(res.pred, res.status, got()) if st == 'ok']
    assert len(ok) > 0.9 * N and all(p == k for p, k in ok), (len(ok), collections.Counter(res.status))


def test_reproducible():
    s = get_sampler()
    out = []
    for _ in range(2):
        s.seed(7)
        out.append([s.sample(SamplerConfig(element_name='compute_expected_utility')) for _ in range(30)])
    assert out[0] == out[1]
