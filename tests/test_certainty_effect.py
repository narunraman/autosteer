"""certainty_effect: Allais / common-ratio pairs, consistency-graded (autosteer/release/consistency.py).

Checks >= 1,000 exported items (fixed seed) on the *displayed* numbers, independently of the exporter's own
parser, plus unit tests of the scorer.
"""
import collections
import re
from fractions import Fraction

import pytest

import risk_pairs
from autosteer.release.consistency import (
    CERTAINTY_EFFECT, CONSISTENT, INVALID, REVERSE, common_ratio_rule, parse_rule, score_common_ratio, score_group,
    sure_side,
)

N = 1000
SEED = 12345
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}")
BAD_TOKEN = re.compile(r"\b(None|nan|NaN|inf|PROB|AMT)\b|%\(")
ex = risk_pairs.certainty_effect
PRIOR = re.compile(r"\b(previous\w*|earlier|before|already|chose|chosen|prefer\w*|decided|instead)\b", re.I)


@pytest.fixture(scope="module")
def rows():
    r, _ = ex.generate(N, SEED, "test")
    return r


@pytest.fixture(scope="module")
def items(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["base_id"]][r["part"]] = r
    return by


def nums(text):
    pct = [Fraction(m) / 100 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]
    rest = re.sub(r"\d+(?:\.\d+)?%", " ", text)
    amt = [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\d|\d", rest)]
    return pct, amt


def test_count_and_parts(rows, items):
    assert len(items) == N and len(rows) == 2 * N
    assert all(set(v) == {0, 1} for v in items.values())


def test_common_ratio_exact_on_displayed_numbers(items):
    for b, it in items.items():
        a0, c1 = sure_side(it[0]["consistency_rule"])
        A, B = it[0]["options"][a0], it[0]["options"][1 - a0]
        C, D = it[1]["options"][c1], it[1]["options"][1 - c1]
        (pA, xA), (pB, yB), (pC, xC), (pD, yD) = map(nums, (A, B, C, D))
        assert pA == [] and len(xA) == 1, A  # A is the sure option
        assert len(pB) == len(pC) == len(pD) == 1 and len(yB) == len(xC) == len(yD) == 1, (B, C, D)
        X, Y, p, k, kp = xA[0], yB[0], pB[0], pC[0], pD[0]
        assert xC[0] == X and yD[0] == Y, (A, B, C, D)  # same prizes in both parts
        assert Y > X > 0
        assert Fraction(6, 10) <= p <= Fraction(95, 100)
        assert Fraction(1, 10) <= k <= Fraction(1, 2)
        assert kp == k * p, (k, p, kp)  # common ratio exact in the shown numbers
        assert abs(p * Y - X) >= Fraction(1, 20) * X  # no (near) EV tie for a risk-neutral agent


def test_two_distinct_options_no_placeholders(rows):
    for r in rows:
        assert len(r["options"]) == 2 and len(set(r["options"])) == 2, r["options"]
        blob = r["question_text"] + " || " + " | ".join(r["options"])
        assert not PLACEHOLDER.search(blob), blob
        assert not BAD_TOKEN.search(blob), blob
        assert not re.search(r"\ba (8|11|18|8\d)\b", blob), blob  # a/an before percentages


def test_parts_asked_fresh(items):
    """Part 1 is self-contained: it neither states nor refers to the part-0 choice."""
    for it in items.values():
        assert not PRIOR.search(it[1]["question_text"]), it[1]["question_text"]
        assert not PRIOR.search(it[0]["question_text"]), it[0]["question_text"]


def test_balanced_order(items):
    n = len(items)
    pos0 = collections.Counter(sure_side(it[0]["consistency_rule"])[0] for it in items.values())
    pos1 = collections.Counter(sure_side(it[0]["consistency_rule"])[1] for it in items.values())
    joint = collections.Counter(sure_side(it[0]["consistency_rule"]) for it in items.values())
    sd = (0.25 / n) ** 0.5
    for c in (pos0, pos1):
        assert abs(c[0] / n - 0.5) < 4 * sd, c
    sdj = (0.25 * 0.75 / n) ** 0.5
    assert len(joint) == 4 and all(abs(v / n - 0.25) < 4 * sdj for v in joint.values()), joint


def test_ev_side_and_mix(items):
    n = len(items)
    favors = collections.Counter(t for it in items.values() for t in it[0]["tags"] if t.startswith("ev_favors"))
    assert set(favors) == {"ev_favors_lottery", "ev_favors_sure"}
    assert abs(favors["ev_favors_lottery"] / n - 0.5) < 4 * (0.25 / n) ** 0.5, favors
    dom = collections.Counter(it[0]["domain"] for it in items.values())
    assert set(dom) == {"finance", "medical", "employment", "gambling"}, dom
    assert {it[0]["difficulty"] for it in items.values()} == {1}  # the element's only level (raw 7)


def test_consistency_fields(rows, items):
    for r in rows:
        assert r["grading"] == "consistency" and r["correct_index"] is None
        assert r["consistency_group"] == f"certainty_effect/{r['base_id']}"
        assert r["question_id"] == f"certainty_effect/{r['base_id']}/{r['part']}"
        rid, pairs = parse_rule(r["consistency_rule"])
        assert rid == "common_ratio_same_side" and len(pairs) == 2
        sure_side(r["consistency_rule"])  # well-formed and complementary
    for it in items.values():
        assert it[0]["consistency_rule"] == it[1]["consistency_rule"]
        assert it[0]["consistency_group"] == it[1]["consistency_group"]
        assert it[0]["type"] == it[1]["type"] and it[0]["domain"] == it[1]["domain"]
    assert ex.schema_mod.validate(ex.to_table(rows)) == []


def test_reproducible():
    a, _ = ex.generate(40, 7, "t")
    b, _ = ex.generate(40, 7, "t")
    c, _ = ex.generate(40, 8, "t")
    assert a == b
    assert [r["question_id"] for r in a] != [r["question_id"] for r in c]


# ------------------------------------------------------------------------------------------------ scorer
@pytest.mark.parametrize("s0,s1", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_scorer_all_outcomes(s0, s1):
    rule = common_ratio_rule(s0, s1)
    assert sure_side(rule) == (s0, s1)
    A, B, C, D = s0, 1 - s0, s1, 1 - s1
    assert score_common_ratio(rule, A, C) == CONSISTENT
    assert score_common_ratio(rule, B, D) == CONSISTENT
    assert score_common_ratio(rule, A, D) == CERTAINTY_EFFECT
    assert score_common_ratio(rule, B, C) == REVERSE
    assert score_group(rule, [A, D]) == CERTAINTY_EFFECT


def test_scorer_encoding_example():
    rule = "common_ratio_same_side:10,01"  # A = option 1 of part 0, C = option 0 of part 1
    assert score_common_ratio(rule, 1, 0) == CONSISTENT
    assert score_common_ratio(rule, 0, 1) == CONSISTENT
    assert score_common_ratio(rule, 1, 1) == CERTAINTY_EFFECT
    assert score_common_ratio(rule, 0, 0) == REVERSE


@pytest.mark.parametrize("c0,c1", [(None, 0), (0, None), (2, 0), (0, -1), (True, 0)])
def test_scorer_invalid_choices(c0, c1):
    assert score_common_ratio("common_ratio_same_side:00,11", c0, c1) == INVALID


@pytest.mark.parametrize("rule", ["common_ratio_same_side", "common_ratio_same_side:00,00",
                                  "common_ratio_same_side:00", "same_brand:00,11", ":00,11"])
def test_scorer_malformed_rule(rule):
    with pytest.raises(ValueError):
        score_common_ratio(rule, 0, 0)


def test_score_group_arity_and_unknown():
    with pytest.raises(ValueError):
        score_group("common_ratio_same_side:00,11", [0])
    with pytest.raises(NotImplementedError):
        score_group("no_such_rule:00,11", [0, 0])  # same_brand and ellsberg_seu have scorers since October 2026
