"""stated_prior type of certainty_effect and reflection_effect (default release type).

One answer-graded question per item: it states the part-0 choice (sure option or lottery) and asks the part-1
decision; the correct option is the one on the stated side (expected utility: sure -> C, lottery -> D; for
reflection_effect sure gain -> sure loss, lottery -> lottery). Checked on >= 1,000 items per element on the
displayed numbers, independently of the exporter's checks.
"""
import collections
import re
from fractions import Fraction

import pytest

import risk_pairs as ex

N = 1000
SEED = 12345
ELEMENTS = ["certainty_effect", "reflection_effect"]
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}")
BAD_TOKEN = re.compile(r"\b(None|nan|NaN|inf|PROB|AMT)\b|%\(")
LEADING = re.compile(r"\b(consisten\w*|inconsisten\w*|rational\w*|irrational\w*|logical\w*|coheren\w*|"
                     r"same (choice|option|preference|side)|stick\w* with|in line with|should remain|"
                     r"expected utility|expected value|certainty effect|reflection effect)\b", re.I)


@pytest.fixture(scope="module", params=ELEMENTS)
def data(request):
    rows, stats = ex.generate(request.param, N, SEED, "test", "stated_prior")
    return request.param, rows, stats


def nums(text):
    pct = [Fraction(m) / 100 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]
    rest = re.sub(r"\d+(?:\.\d+)?%", " ", text)
    amt = [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\d|\d", rest)]
    return pct, amt


def stated(row):
    """(A, B, chosen) quoted in the question."""
    q = re.findall(r'"([^"]+)"', row["question_text"])
    assert len(q) == 3, row["question_text"]
    return q


def test_fields_and_schema(data):
    el, rows, _ = data
    assert len(rows) == N and len({r["base_id"] for r in rows}) == N
    for r in rows:
        assert r["grading"] == "answer" and r["correct_index"] in (0, 1)
        assert r["consistency_group"] is None and r["consistency_rule"] is None
        assert r["type"] == "stated_prior" and r["part"] == 0 and r["difficulty"] == 1
        assert r["question_id"] == f"{el}/{r['base_id']}/0"
        assert len(r["options"]) == 2 and len(set(r["options"])) == 2
    assert ex.schema_mod.validate(ex.to_table(rows)) == []


def test_stated_choice_in_text_and_answer_follows_rule(data):
    el, rows, _ = data
    for r in rows:
        A, B, chosen = stated(r)
        assert chosen in (A, B) and f'chose "{chosen}"' in r["question_text"]
        (pA, xA), (pB, yB) = nums(A), nums(B)
        assert pA == [] and len(xA) == 1 and len(pB) == 1 and len(yB) == 1, (A, B)  # A = sure, B = lottery
        X, Y, p = xA[0], yB[0], pB[0]
        right, wrong = r["options"][r["correct_index"]], r["options"][1 - r["correct_index"]]
        (pr, xr), (pw, xw) = nums(right), nums(wrong)
        want = X if chosen == A else Y  # prize of the option on the stated side
        assert xr == [want] and xw == [Y if want == X else X], (chosen, right, wrong)
        if el == "certainty_effect":  # C = (X, k), D = (Y, k*p)
            k = (pr if want == X else pw)[0]
            kp = (pw if want == X else pr)[0]
            assert kp == k * p and Fraction(1, 10) <= k <= Fraction(1, 2) and Fraction(3, 5) <= p <= Fraction(19, 20)
            assert abs(p * Y - X) >= Fraction(1, 20) * X
        else:  # C = lose X for sure, D = lose Y with probability p
            sure_new = right if want == X else wrong
            risky_new = wrong if want == X else right
            assert nums(sure_new)[0] == [] and nums(risky_new)[0] == [p]
            assert p * Y == X


def test_stated_side_and_position_balanced(data):
    _, rows, _ = data
    n = len(rows)
    sd = (0.25 / n) ** 0.5
    side = collections.Counter("stated_sure" if stated(r)[2] == stated(r)[0] else "stated_lottery" for r in rows)
    assert abs(side["stated_sure"] / n - 0.5) < 4 * sd, side
    pos = collections.Counter(r["correct_index"] for r in rows)
    assert abs(pos[0] / n - 0.5) < 4 * sd, pos
    # the correct position is not tied to the stated side
    joint = collections.Counter((stated(r)[2] == stated(r)[0], r["correct_index"]) for r in rows)
    assert len(joint) == 4 and min(joint.values()) > n / 4 - 4 * (0.1875 * n) ** 0.5, joint
    # tags record the stated side
    for r in rows:
        tag = "stated_sure" if stated(r)[2] == stated(r)[0] else "stated_lottery"
        assert tag in r["tags"]


def test_wording(data):
    _, rows, _ = data
    for r in rows:
        blob = r["question_text"] + " || " + " | ".join(r["options"])
        assert not PLACEHOLDER.search(blob), blob
        assert not BAD_TOKEN.search(blob), blob
        assert not LEADING.search(blob), blob
        assert not re.search(r"\ba (8|11|18|8\d)\b", blob), blob
        assert re.match(r"Earlier, ", r["question_text"]) and ". Now, " in r["question_text"]


def test_domain_mix(data):
    _, rows, _ = data
    dom = collections.Counter(r["domain"] for r in rows)
    assert set(dom) == {"finance", "medical", "employment", "gambling"}, dom


@pytest.mark.parametrize("el", ELEMENTS)
def test_reproducible(el):
    a, _ = ex.generate(el, 30, 7, "t", "stated_prior")
    b, _ = ex.generate(el, 30, 7, "t", "stated_prior")
    c, _ = ex.generate(el, 30, 8, "t", "stated_prior")
    assert a == b and [r["question_id"] for r in a] != [r["question_id"] for r in c]


def test_export_has_both_types():
    """The exporter's cells are type x domain x difficulty, so both stated_prior and two_part are exported."""
    from autosteer.release.export import element_cells
    for el in ELEMENTS:
        assert {c[0] for c in element_cells(ex._sampler(), el)} == {"stated_prior", "two_part"}
