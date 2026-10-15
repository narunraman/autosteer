"""pareto_sc decision (2026-10-02): Yes/No questions without Pareto wording ask a majority question and are dropped
by the audit solver (audit/steer/solvers/voting.py, ill_posed:majority_wording); Yes/No Pareto questions and the
"which option should be chosen" questions are kept, and on the kept staged rows the solver agrees with the key."""
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGED = Path.home() / 'UBC/steer-release/hf/STEER/data/behalf_of_others/social_choice_axioms/pareto_sc/test.parquet'


@pytest.mark.skipif(not STAGED.exists(), reason='staged release not available')
def test_majority_wording_dropped():
    spec = importlib.util.spec_from_file_location('audit_voting', os.path.join(ROOT, 'tests/solvers/steer/solvers/voting.py'))
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    d = pd.read_parquet(STAGED)
    r = v.solve_pareto(d)
    yn = d.options.map(lambda o: sorted(o) == ['No', 'Yes'])
    pareto = d.question_text.str.contains('pareto', case=False)
    dropped = r.status == 'ill_posed:majority_wording'
    assert (dropped == (yn & ~pareto)).all()
    assert dropped.sum() > 0
    kept = r.status == 'ok'
    assert (kept | dropped).all()
    assert (r.pred[kept] == d.correct_index[kept]).all()
    assert (~yn & kept).sum() > 0  # "which option should be chosen" questions stay
