"""reflection_effect: gain / loss mirror pairs with equal EV, consistency-graded (rule reflection_same_side).

Checks >= 1,000 exported items (fixed seed) on the displayed numbers, independently of the exporter's parser,
plus unit tests of the scorer.
"""
import collections
import re
from fractions import Fraction

import pytest

import risk_pairs as ex
from autosteer.release.consistency import (
    CONSISTENT, INVALID, REFLECTION_EFFECT, REVERSE_REFLECTION, parse_rule, reflection_rule, score_common_ratio,
    score_group, score_reflection, sure_side,
)

EL = "reflection_effect"
N = 1000
SEED = 12345
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}")
BAD_TOKEN = re.compile(r"\b(None|nan|NaN|inf|PROB|AMT)\b|%\(")
PRIOR = re.compile(r"\b(previous\w*|earlier|before|already|chose|chosen|prefer\w*|decided|instead)\b", re.I)
GAIN = re.compile(r"\b(profit|pays|paying|recover\w*|gain|bonus|win\w*|prize|save|saving|protect\w*|detect\w*|"
                  r"cure\w*|interest|fee|cash)\b", re.I)
LOSS = re.compile(r"\b(loss|pay|paying|costs?|costing|die|deaths|infected|infections|undetected|repayment|penalty|"
                  r"deducts?|deducting|deduction|give back|giving back|charge|refund|without treatment)\b", re.I)


@pytest.fixture(scope="module")
def items():
    rows, _ = ex.generate(EL, N, SEED, "test")
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["base_id"]][r["part"]] = r
    return rows, by


def nums(text):
    pct = [Fraction(m) / 100 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]
    rest = re.sub(r"\d+(?:\.\d+)?%", " ", text)
    amt = [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\d|\d", rest)]
    return pct, amt


def sides(it):
    a0, c1 = sure_side(it[0]["consistency_rule"], "reflection_same_side")
    return (it[0]["options"][a0], it[0]["options"][1 - a0], it[1]["options"][c1], it[1]["options"][1 - c1])


def test_count_and_parts(items):
    rows, by = items
    assert len(by) == N and len(rows) == 2 * N
    assert all(set(v) == {0, 1} for v in by.values())


def test_mirror_equal_ev_on_displayed_numbers(items):
    for it in items[1].values():
        A, B, C, D = sides(it)
        (pA, xA), (pB, yB), (pC, xC), (pD, yD) = map(nums, (A, B, C, D))
        assert pA == [] and pC == [] and len(xA) == len(xC) == 1, (A, C)  # sure options
        assert len(pB) == len(pD) == 1 and len(yB) == len(yD) == 1, (B, D)
        X, Y, p = xA[0], yB[0], pB[0]
        assert xC[0] == X and yD[0] == Y and pD[0] == p, (A, B, C, D)  # loss frame mirrors gain frame
        assert Y > X > 0 and Fraction(1, 5) <= p <= Fraction(4, 5)
        assert p * Y == X  # exactly equal expected values


def test_frames(items):
    """Part 0 options are gains and part 1 options are losses; the two stories differ only in frame."""
    for it in items[1].values():
        A, B, C, D = sides(it)
        assert GAIN.search(A) and GAIN.search(B), (A, B)
        assert LOSS.search(C) and LOSS.search(D), (C, D)
        assert not re.search(r"\b(loss|die|infected|penalty|costs?)\b", A + B), (A, B)


def test_two_distinct_options_no_placeholders(items):
    for r in items[0]:
        assert len(r["options"]) == 2 and len(set(r["options"])) == 2, r["options"]
        blob = r["question_text"] + " || " + " | ".join(r["options"])
        assert not PLACEHOLDER.search(blob), blob
        assert not BAD_TOKEN.search(blob), blob
        assert not re.search(r"\ba (8|11|18|8\d)\b", blob), blob
        assert not PRIOR.search(r["question_text"]), r["question_text"]  # both parts asked fresh


def test_balanced_order(items):
    by = items[1]
    n = len(by)
    joint = collections.Counter(sure_side(it[0]["consistency_rule"]) for it in by.values())
    for i in (0, 1):
        share = sum(v for k, v in joint.items() if k[i] == 0) / n
        assert abs(share - 0.5) < 4 * (0.25 / n) ** 0.5, joint
    assert len(joint) == 4 and all(abs(v / n - 0.25) < 4 * (0.1875 / n) ** 0.5 for v in joint.values()), joint


def test_mix_and_fields(items):
    rows, by = items
    dom = collections.Counter(it[0]["domain"] for it in by.values())
    assert set(dom) == {"finance", "medical", "employment", "gambling"}, dom
    for r in rows:
        assert r["difficulty"] == 1 and r["grading"] == "consistency" and r["correct_index"] is None
        assert r["consistency_group"] == f"{EL}/{r['base_id']}"
        assert r["question_id"] == f"{EL}/{r['base_id']}/{r['part']}"
        rid, pairs = parse_rule(r["consistency_rule"])
        assert rid == "reflection_same_side" and len(pairs) == 2
        assert "equal_ev" in r["tags"]
    for it in by.values():
        assert it[0]["consistency_rule"] == it[1]["consistency_rule"]
        assert it[0]["domain"] == it[1]["domain"] and it[0]["type"] == it[1]["type"]
    assert ex.schema_mod.validate(ex.to_table(rows)) == []


def test_reproducible():
    a, _ = ex.generate(EL, 40, 7, "t")
    b, _ = ex.generate(EL, 40, 7, "t")
    c, _ = ex.generate(EL, 40, 8, "t")
    assert a == b
    assert [r["question_id"] for r in a] != [r["question_id"] for r in c]


def test_examples_pool_renders():
    s = ex._Sampler()
    s.pool = "examples"
    s._pool_cache.clear()
    from autosteer.sampler import SamplerConfig
    for _ in range(20):
        parts = s.sample(SamplerConfig(element_name=EL, type="two_part"))["sub_questions"]
        texts = [o["text"] for p in parts for o in p["options"]]
        assert len(parts) == 2 and not any(PLACEHOLDER.search(t) for t in texts)
        assert ex.check_reflection(ex.parse_mirror(*texts)) == []
        item = s.sample(SamplerConfig(element_name=EL, type="stated_prior"))
        a, b, _ = ex.QUOTED_RE.findall(item["question"])
        assert ex.check_reflection(ex.parse_mirror(a, b, *[o["text"] for o in item["options"]])) == []


# ------------------------------------------------------------------------------------------------ scorer
@pytest.mark.parametrize("s0,s1", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_scorer_all_outcomes(s0, s1):
    rule = reflection_rule(s0, s1)
    A, B, C, D = s0, 1 - s0, s1, 1 - s1
    assert score_reflection(rule, A, C) == CONSISTENT
    assert score_reflection(rule, B, D) == CONSISTENT
    assert score_reflection(rule, A, D) == REFLECTION_EFFECT
    assert score_reflection(rule, B, C) == REVERSE_REFLECTION
    assert score_group(rule, [A, D]) == REFLECTION_EFFECT
    assert score_group(rule, [B, C]) == REVERSE_REFLECTION


@pytest.mark.parametrize("c0,c1", [(None, 0), (0, None), (2, 0), (False, 1)])
def test_scorer_invalid_choices(c0, c1):
    assert score_reflection("reflection_same_side:01,10", c0, c1) == INVALID


def test_rule_ids_not_interchangeable():
    with pytest.raises(ValueError):
        score_reflection("common_ratio_same_side:00,11", 0, 0)
    with pytest.raises(ValueError):
        score_common_ratio("reflection_same_side:00,11", 0, 0)
    with pytest.raises(ValueError):
        score_reflection("reflection_same_side:00,01", 0, 0)
