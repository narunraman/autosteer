"""pareto_sc: hand-written Yes/No and "which should be chosen" questions; the audit solver agrees with every key.

For each sampled question: the existing audit solver (tests/solvers/steer/solvers/voting.py, solve_pareto, which
reads only the text and options) returns "ok" and the marked option; and by brute force over the ballots in the
text, a Yes/No key is "No" exactly when every member ranks some other alternative above the chosen one, which is
also exactly when one alternative is ranked above it by every member; a "which" key is the alternative every
member ranks first. Two and three alternatives, Yes and No keys and both types occur.
"""
import collections
import re

import pandas as pd

from helpers import load_steer_solver, sample

VOTING = load_steer_solver('voting')
BALLOT = re.compile(r'^(\d+) voters voted for ([A-Z](?: > [A-Z])+)$', re.M)


def test_pareto_sc_keys():
    got = sample('pareto_sc', 800, seed=91)
    seen = collections.Counter()
    df = pd.DataFrame({'question_text': [s.question for s in got], 'options': [s.options for s in got],
                       'tags': [s.metadata['tags'] for s in got]})
    res = VOTING.solve_pareto(df)
    for s, (_, row) in zip(got, res.iterrows()):
        assert row['status'] == 'ok' and row['pred'] == s.answer_index, (row['status'], row['pred'], s.question, s.options)
        profile = [(int(n), r.split(' > ')) for n, r in BALLOT.findall(s.question)]
        assert profile
        alts = profile[0][1]
        assert all(sorted(r) == sorted(alts) for _, r in profile) and len(alts) in (2, 3)
        assert 'A' not in alts and 'I' not in alts
        if s.type == 'yes_no':
            chosen = re.search(r"final decision was \w+ ([A-Z])\.", s.question).group(1)
            everyone_other = all(r.index(chosen) > 0 for _, r in profile)
            dominated = any(all(r.index(y) < r.index(chosen) for _, r in profile) for y in alts if y != chosen)
            assert everyone_other == dominated
            assert s.answer == ('No' if dominated else 'Yes'), (s.question, s.answer)
            assert sorted(s.options) == ['No', 'Yes']
            seen[('yes_no', len(alts), s.answer)] += 1
        else:
            tops = {r[0] for _, r in profile}
            assert len(tops) == 1
            assert s.answer.endswith(' ' + tops.pop())
            assert len(s.options) == len(alts)
            seen[('which', len(alts))] += 1
    assert set(seen) == {('yes_no', 2, 'Yes'), ('yes_no', 2, 'No'), ('yes_no', 3, 'Yes'), ('yes_no', 3, 'No'),
                         ('which', 2), ('which', 3)}, seen
