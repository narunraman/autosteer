"""Regenerated STEER multi-agent elements: trigger, enforceability, feasibility, backward_induction.

Oracle: the independent solvers (tests/solvers/steer/solvers/repeated_games.py via tests/solvers/steer/multi_agent_oracle.py)
read the game from the question text and options; they must agree with the sampler's correct option on every
sampled question. Also checked on 1,000 seeded samples per element: exactly one correct option, distinct options,
no unfilled placeholders, answer position uniform (chi-square), STEER grade level as in the released element, and
the filler's own numbers (from the recorded placeholder values) are genuine PDs / tie-free trees.
"""
import collections
import re
import sys
from pathlib import Path

import pytest

from helpers import sample, num

sys.path.insert(0, str(Path(__file__).resolve().parent / "solvers" / "steer"))
from multi_agent_oracle import oracle  # noqa: E402

pytestmark = pytest.mark.slow

ELEMENTS = ["trigger", "enforceability", "feasibility", "backward_induction"]
N = 1000
GRADES = {"trigger": {7}, "enforceability": {7}, "feasibility": {7, 8}, "backward_induction": {5, 6}}
PLACEHOLDER = re.compile(r"\{[^{}]*\}")


@pytest.fixture(scope="module", params=ELEMENTS)
def samples(request):
    return request.param, sample(request.param, N, seed=2026)


def chi2_sf(x, k):
    from scipy.stats import chi2
    return float(chi2.sf(x, k))


def test_solver_agrees(samples):
    element, got = samples
    if element == "feasibility":
        pytest.skip("code templates since October 2026; keys recomputed in tests/test_feasibility.py")
    status = collections.Counter()
    bad = []
    for s in got:
        pred, st = oracle(element, s.question, s.options, s.metadata["tags"], s.metadata["domain"])
        ok = st == "ok" and pred == s.answer_index
        status["agree" if ok else st if st != "ok" else "disagree"] += 1
        if not ok and len(bad) < 3:
            bad.append((st, pred, s.answer_index, s.question, s.options))
    assert status["agree"] == len(got), (dict(status), bad)


def test_well_formed(samples):
    element, got = samples
    assert len(got) == N
    for s in got:
        assert sum(s.correct) == 1, (s.question, s.options, s.correct)
        assert len(set(s.options)) == len(s.options), s.options
        assert not PLACEHOLDER.search(s.question), s.question
        assert not any(PLACEHOLDER.search(o) for o in s.options), s.options
        assert "np." not in s.question and not any("np." in o for o in s.options)
        assert s.metadata["difficulty_level"] in GRADES[element]
    assert {s.metadata["difficulty_level"] for s in got} == GRADES[element]


def test_answer_position_uniform(samples):
    element, got = samples
    by_n = collections.defaultdict(collections.Counter)
    for s in got:
        by_n[len(s.options)][s.answer_index] += 1
    for k, cnt in by_n.items():
        total = sum(cnt.values())
        exp = total / k
        x = sum((cnt.get(i, 0) - exp) ** 2 / exp for i in range(k))
        assert chi2_sf(x, k - 1) > 1e-3, (k, dict(cnt))


def _pd(R, S, T, P):
    return T > R > P > S and 2 * R > T + S


def test_story_is_pd(samples):
    """The story's own labels: the payoffs the question text gives for (cooperate, cooperate), (defect, defect) and
    the two mixed outcomes, read back by the independent story reader, satisfy T > R > P > S and are not stated as
    losses (a feasibility template told 'both deliver -> each gets the both-defect payoff')."""
    element, got = samples
    if element == "backward_induction":
        pytest.skip("sequential game")
    if element == "feasibility":
        pytest.skip("code templates since October 2026: tests/test_feasibility.py::test_story_is_pd reads their payoff "
                    "sentence")
    from multi_agent_oracle import RG
    is_pd, parse_game = RG.is_pd, RG.parse_game
    bad, unread = [], 0
    for s in got:
        g, why = parse_game(s.question)
        if g is None:      # a few stories restate payoffs, which the reader does not follow
            unread += 1
        elif not is_pd(g) or min(g.values()) < 0:
            bad.append((g, s.question[:400]))
    assert not bad, (len(bad), bad[:3])
    assert unread <= 0.03 * len(got), unread


def test_filler_games(samples):
    """The filler's own numbers: genuine PDs (trigger/enforceability/feasibility), no ties (backward_induction)."""
    element, got = samples
    for s in got:
        p = s.params
        if element == "trigger":
            R, S, T, P = (num(p[k]) for k in ("both_cooperate", "cooperate", "defect", "both_defect"))
            assert _pd(R, S, T, P)
            assert s.answer is not None and abs(num(s.answer) - (T - R) / (T - P)) <= 0.005 + 1e-9
        elif element in ("enforceability", "feasibility"):
            R, S, T, P = (num(p[k]) for k in ("both_y", "y_payoff", "x_payoff", "both_x"))
            assert _pd(R, S, T, P)
        else:
            u = {k: num(v) for k, v in p.items()}
            # round 3 (player 1): end vs third party; round 2 (player 2); round 1 (player 1): strict preferences
            v3 = (u["p1_round3"], u["p2_round3"]) if u["p1_round3"] > u["p1_round3_2"] else (u["p1_round3_2"], u["p2_round3_2"])
            assert u["p1_round3"] != u["p1_round3_2"]
            assert u["p2_round2"] != v3[1]
            v2 = (u["p1_round2"], u["p2_round2"]) if u["p2_round2"] > v3[1] else v3
            assert u["p1_round1"] != v2[0]


def test_trigger_payoffs_shown():
    """Either the complete 2x2 matrix is printed, or all four payoffs appear in the prose."""
    got = sample("trigger", 300, seed=5, domain="matrix_form") + sample("trigger", 300, seed=6)
    n_matrix = 0
    for s in got:
        rows = re.findall(r"^.*\(\s*-?\d+, -?\d+\)\s+\(\s*-?\d+, -?\d+\)\s*$", s.question, re.M)
        if rows:
            n_matrix += 1
            assert len(rows) == 2, s.question
        else:
            nums = set(re.findall(r"\d+", s.question))
            assert all(str(s.params[k]) in nums for k in ("both_cooperate", "cooperate", "defect", "both_defect")), s.question
    assert n_matrix > 0


def test_enforceability_answer_mix():
    got = sample("enforceability", 400, seed=9)
    kinds = collections.Counter()
    for s in got:
        a = s.answer
        kinds["none" if a.startswith("None") else "alt" if "take turns" in a else "DD" if "defect every" in a
              else "CC" if "cooperate every" in a else "other"] += 1
    assert set(kinds) == {"none", "alt", "DD", "CC"}, kinds
    assert min(kinds.values()) > 60, kinds


def test_backward_induction_both_answers_per_domain():
    for domain in ("finance", "medical"):
        got = sample("backward_induction", 200, seed=3, domain=domain)
        tags = collections.Counter(s.metadata["tags"].split(";")[-1] for s in got)
        assert set(tags) == {"continue", "end"}, (domain, tags)


def test_reproducible_across_processes():
    from test_reproducibility import run
    a, b = run(1, ELEMENTS), run(2, ELEMENTS)
    assert a == b
