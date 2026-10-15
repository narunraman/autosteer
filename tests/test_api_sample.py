"""/api/sample: the legacy single-question shape and the dataset-shaped {"questions": [row, ...]} response."""
import pytest
from fastapi.testclient import TestClient

from autosteer.release.export import select_elements
from autosteer.schemas import DATASET_FIELDS, DatasetRow
from autosteer.web.app import app

client = TestClient(app)


def get(**params):
    return client.get("/api/sample", params=params)


def test_legacy_shape_without_n_or_seed():
    """What the element pages read: question, options[].text / .correct (single-part elements)."""
    r = get(element_name="consumer_surplus")
    assert r.status_code == 200, r.text
    j = r.json()
    assert isinstance(j["question"], str) and j["question"]
    assert all(set(o) == {"text", "correct"} for o in j["options"]) and sum(o["correct"] for o in j["options"]) == 1
    assert set(j["metadata"]) == {"element", "domain", "type", "difficulty_level", "tags"}


def test_legacy_shape_serves_steer_elements():
    r = get(element_name="interpret_games")
    assert r.status_code == 200, r.text
    assert "question" in r.json()
    assert "sub_questions" in get(element_name="iterated_removal").json()


def test_rows_with_n_and_seed():
    r = get(element_name="law_of_demand", n=5, seed=42)
    assert r.status_code == 200, r.text
    rows = r.json()["questions"]
    assert len(rows) == 5
    for i, row in enumerate(rows):
        assert tuple(row) == DATASET_FIELDS
        DatasetRow(**row)
        assert row["element"] == "law_of_demand" and row["seed"] == 42 + i
    assert get(element_name="law_of_demand", n=5, seed=42).json() == r.json()
    assert get(element_name="law_of_demand", n=5, seed=43).json()["questions"][:4] == rows[1:]


def test_row_reproduces_from_template_and_seed():
    row = get(element_name="price_elasticity", seed=7).json()["questions"][0]
    again = get(element_name="price_elasticity", seed=row["seed"], template_id=row["template_id"]).json()
    assert again["questions"] == [row]


def test_multi_part_and_consistency_rows():
    rows = get(element_name="certainty_effect", type="two_part", n=2, seed=1).json()["questions"]
    assert len(rows) == 4 and {r["part"] for r in rows} == {0, 1}
    assert all(r["grading"] == "consistency" and r["consistency_rule"] for r in rows)


def test_filters_apply():
    rows = get(element_name="law_of_demand", type="price_increase", n=3, seed=5).json()["questions"]
    assert {r["type"] for r in rows} == {"price_increase"}


def test_steer_and_steer_me_served_and_held_refused():
    chosen, _ = select_elements()
    assert {"interpret_games", "law_of_demand"} <= set(chosen)
    for e in ("interpret_games", "plurality_voting", "law_of_demand"):
        assert get(element_name=e, n=1, seed=3).status_code == 200, e
    assert get(element_name="sc_monotonicity", n=1).status_code == 404
    assert get(element_name="no_such_element", n=1).status_code == 404


def test_random_element_is_deterministic_with_seed():
    a = get(n=3, seed=11).json()["questions"]
    assert a == get(n=3, seed=11).json()["questions"]


@pytest.mark.parametrize("params", [{"n": 0}, {"n": 51}, {"seed": -1}, {"seed": 2 ** 63}])
def test_bounds(params):
    assert get(element_name="law_of_demand", **params).status_code == 422


def test_post_matches_get():
    body = {"element_name": "add_sub", "n": 2, "seed": 9}
    assert client.post("/api/sample", json=body).json() == get(**body).json()
    assert "question" in client.post("/api/sample", json={"element_name": "add_sub"}).json()
