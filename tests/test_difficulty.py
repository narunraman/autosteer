"""Relative difficulty (autosteer.sampler.difficulty) and where it is shown: dataset rows, /api/sample, the site."""
import pytest
from fastapi.testclient import TestClient

from autosteer.release.package import _difficulty_fact
from autosteer.release.rows import build_rows, row_problems
from autosteer.sampler import ElementSampler, load_manifest
from autosteer.sampler.difficulty import (level_count, parse_relative, public_entry, raw_levels, relative_levels,
                                          relative_scale, to_raw, to_relative)
from autosteer.web import site
from autosteer.web.app import app

M = {"elements": {
    "grades": {"difficulty_levels": ["10", "8", "9", "13"]},  # string-sorted in the manifest
    "single": {"difficulty_levels": ["0"]},
    "steer_me": {"difficulty_levels": ["1", "2"]},
}}


def test_scale_is_monotone_and_numbered_from_one():
    assert relative_scale(["10", "8", "9", "13"]) == {8: 1, 9: 2, 10: 3, 13: 4}
    assert relative_scale([2, 2, 1]) == {1: 1, 2: 2}
    assert raw_levels("grades", M) == [8, 9, 10, 13]
    assert relative_levels("grades", M) == [1, 2, 3, 4] and level_count("grades", M) == 4


def test_round_trip_and_single_level():
    for element in M["elements"]:
        for raw in raw_levels(element, M):
            assert to_raw(element, to_relative(element, raw, M), M) == raw
        for rel in relative_levels(element, M):
            assert to_relative(element, to_raw(element, rel, M), M) == rel
    assert to_relative("single", 0, M) == 1 and relative_levels("single", M) == [1]
    assert to_raw("steer_me", "2", M) == 2 and to_relative("steer_me", "1", M) == 1


def test_out_of_range():
    for bad in (0, 5, "x", "-1", "1.5", "", True):
        with pytest.raises(ValueError, match="relative to each element"):
            parse_relative("grades", bad, M)
    with pytest.raises(ValueError, match="must be 1 \\(it has one level\\)"):
        to_raw("single", 2, M)
    with pytest.raises(ValueError):
        to_relative("grades", 11, M)
    with pytest.raises(KeyError):
        raw_levels("nope", M)


def test_public_entry():
    assert public_entry("grades", M["elements"]["grades"])["difficulty_levels"] == ["1", "2", "3", "4"]
    assert public_entry("x", {"types": []}) == {"types": []} and public_entry("x", None) is None


def test_elements_md_fact():
    assert _difficulty_fact([1]) == "difficulty level 1 (one level)"
    assert _difficulty_fact([2, 1, 3]) == "difficulty levels 1–3"


def _multi_level_element():
    m = load_manifest()["elements"]
    return next(e for e in ("subgame_nash", "iterated_removal", "add_sub", "consumer_surplus") if
                e in m and len(m[e]["difficulty_levels"]) > 1)


def test_rows_carry_the_relative_level():
    s = ElementSampler()
    element = _multi_level_element()
    raws = raw_levels(element)
    for rel, raw in enumerate(raws, start=1):
        q = s.sample_question(element, 7, difficulty_level=raw)
        assert q["metadata"]["difficulty_level"] == raw  # the sampler keeps raw levels
        rows = build_rows(q, "test")
        assert {r["difficulty"] for r in rows} == {rel}
        assert "difficulty_not_a_relative_level" not in row_problems(q, rows)


client = TestClient(app)


def test_api_filter_and_rows_are_relative():
    element = _multi_level_element()
    k = level_count(element)
    for rel in range(1, k + 1):
        r = client.get("/api/sample", params={"element_name": element, "difficulty_level": rel, "n": 2, "seed": 3})
        assert r.status_code == 200, r.text
        assert {row["difficulty"] for row in r.json()["questions"]} == {rel}
        legacy = client.get("/api/sample", params={"element_name": element, "difficulty_level": rel})
        assert legacy.status_code == 200 and legacy.json()["metadata"]["difficulty_level"] == rel
    for bad in (0, k + 1, "hard"):
        r = client.get("/api/sample", params={"element_name": element, "difficulty_level": bad, "n": 1, "seed": 1})
        assert r.status_code == 400 and "relative to each element" in r.json()["detail"], r.text
        assert client.get("/api/sample", params={"element_name": element, "difficulty_level": bad}).status_code == 400
    r = client.get("/api/sample", params={"difficulty_level": 99, "n": 1, "seed": 1})
    assert r.status_code == 400 and "relative to each element" in r.json()["detail"]
    r = client.get("/api/sample", params={"difficulty_level": 2, "n": 4, "seed": 5})
    assert r.status_code == 200 and {row["difficulty"] for row in r.json()["questions"]} == {2}
    listed = client.get("/api/elements").json()["elements"]
    assert listed and all(e["manifest"]["difficulty_levels"] == [str(i) for i in range(1, len(
        e["manifest"]["difficulty_levels"]) + 1)] for e in listed)


def test_element_page_offers_relative_levels():
    element = _multi_level_element()
    page = client.get(f"/elements/{element}").text
    k = level_count(element)
    assert f'<option value="{k}">level {k}</option>' in page and '<option value="0">' not in page


def test_eval_difficulty_keys():
    results = {"benchmarks": {"STEER-ME": {
        "a": {"0": {"by": {"difficulty": {"1": [1], "2": [1]}}}, "1": {"by": {"difficulty": {"2": [1]}}}},
        "moved": {"0": {"by": {"difficulty": {"0": [1]}}}},
        "missing": {"0": {"by": {"difficulty": {"0": [1], "7": [1]}}}},
        "deriving_hicksian_demand": {"0": {"by": {"difficulty": {str(i): [1] for i in range(5)}}}},
    }}}
    manifest = {"elements": {"a": {"difficulty_levels": ["1", "2"]}, "moved": {"difficulty_levels": ["2"]},
                             "missing": {"difficulty_levels": ["0", "1"]},
                             "deriving_hicksian_demand": {"difficulty_levels": ["0", "1", "2"]}}}
    keys = site.eval_difficulty_keys(results, manifest)["STEER-ME"]
    assert keys == {"a": {"1": "1", "2": "2"}, "moved": {"1": "0"}}
