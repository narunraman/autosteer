"""Exporter: sizes per cell, schema, reproducibility of rows from (element, template_id, seed), identical re-export."""
import argparse
import json

import pyarrow.parquet as pq
import pytest

from autosteer.release import export
from autosteer.release.rows import build_rows
from autosteer.sampler.difficulty import to_relative
from autosteer.release.schema import validate
from autosteer.sampler import ElementSampler

ELEMENTS = ["certainty_effect", "add_sub", "iterated_removal", "law_of_demand"]


def test_cell_targets():
    assert export.cell_targets([("a",)] * 0) == {}
    t = export.cell_targets([(i,) for i in range(15)])
    assert set(t.values()) == {20} and sum(t.values()) == 300
    t = export.cell_targets([(i,) for i in range(8)])
    assert set(t.values()) == {25} and sum(t.values()) == 200
    t = export.cell_targets([(i,) for i in range(7)])
    assert sum(t.values()) == 200 and max(t.values()) - min(t.values()) <= 1
    assert [t[(i,)] for i in range(7)] == [29, 29, 29, 29, 28, 28, 28]
    t = export.cell_targets([(i,) for i in range(1)])
    assert t == {(0,): 200}


def _run(out):
    args = argparse.Namespace(out=str(out), seed=7, elements=ELEMENTS, per_cell=20, floor=200,
                              max_attempts_per_question=20, few_shot_per_type=5, max_held_out=5, workers=2, allow_dirty=True,
                              allow_unverified_data=True, argv=["--test"])
    assert export.run(args) == 0
    return json.loads((out / "manifest.json").read_text())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    a = tmp_path_factory.mktemp("a")
    b = tmp_path_factory.mktemp("b")
    return (a, _run(a)), (b, _run(b))


def test_counts_and_schema(exported):
    (out, m), _ = exported
    s = ElementSampler()
    for e in ELEMENTS:
        rec = m["elements"][e]
        cells = export.element_cells(s, e)
        assert rec["cells"] == len(cells)
        assert rec["target"] == max(200, 20 * len(cells))
        assert rec["questions"] == rec["target"] and rec["status"] == "ok", rec
        t = pq.read_table(out / rec["file"])
        assert validate(t) == []
        rows = t.to_pylist()
        assert len({r["base_id"] for r in rows}) == rec["questions"] and len(rows) == rec["rows"]
        cnt = {}
        for r in rows:
            if r["part"] == 0:
                cnt[(r["type"], r["domain"], r["difficulty"])] = cnt.get((r["type"], r["domain"], r["difficulty"]), 0) + 1
        # rows carry the relative difficulty of the cell's raw level
        assert cnt == {(t, d, to_relative(e, lvl)): n for (t, d, lvl), n in export.cell_targets(cells).items()}


def test_consistency_graded_two_part(exported):
    (out, m), _ = exported
    rows = pq.read_table(out / m["elements"]["certainty_effect"]["file"]).to_pylist()
    two = [r for r in rows if r["type"] == "two_part"]
    assert two and all(r["grading"] == "consistency" and r["correct_index"] is None for r in two)
    assert all(r["grading"] == "answer" for r in rows if r["type"] == "stated_prior")


def test_rows_reproduce_from_template_and_seed(exported):
    (out, m), _ = exported
    s = ElementSampler()
    for e in ELEMENTS:
        rows = pq.read_table(out / m["elements"][e]["file"]).to_pylist()
        by = {}
        for r in rows:
            by.setdefault(r["base_id"], []).append(r)
        for base, parts in list(by.items())[:25]:
            q = s.sample_question(e, parts[0]["seed"], template_id=parts[0]["template_id"])
            assert build_rows(q, parts[0]["generator"]) == sorted(parts, key=lambda r: r["part"])


def test_reexport_is_byte_identical(exported):
    (a, ma), (b, mb) = exported
    for e in ELEMENTS:
        assert (a / ma["elements"][e]["file"]).read_bytes() == (b / mb["elements"][e]["file"]).read_bytes()
        assert ma["elements"][e]["sha256"] == mb["elements"][e]["sha256"]


def test_held_elements_are_skipped():
    chosen, skipped = export.select_elements()
    assert "sc_monotonicity" not in chosen and skipped["sc_monotonicity"].startswith("held")
    assert "enforceability" in chosen  # take turns kept, 6 October 2026
    assert "dsic_mechanism" in chosen and "equiv_contrapositive" in chosen  # released in the October 2026 cleanup
    assert "prob_basic" in chosen  # generator added in October 2026


def test_few_shot_split(exported):
    """few_shot: 5 per type, from templates (and template texts) test never uses, no question shared with test."""
    (out, m), _ = exported
    s = ElementSampler()
    for e in ELEMENTS:
        rec = m["elements"][e]
        plan = export.few_shot_plan(s, e, 7)
        # holding templates out never removes a test cell
        assert export.element_cells(s, e, exclude=frozenset(plan["held_out"])) == export.element_cells(s, e)
        test = pq.read_table(out / rec["file"]).to_pylist()
        types = [t for t, p in plan["types"].items() if p["templates"]]
        if not types:  # every template is the only one of some test cell: no few_shot
            assert "file" not in rec["few_shot"] and rec["few_shot"]["questions"] == 0
            continue
        few = pq.read_table(out / rec["few_shot"]["file"]).to_pylist()
        assert validate(pq.read_table(out / rec["few_shot"]["file"])) == []
        n = {}
        for r in few:
            if r["part"] == 0:
                n[r["type"]] = n.get(r["type"], 0) + 1
        assert n == {t: 5 for t in types}
        assert not {r["template_id"] for r in test} & {r["template_id"] for r in few}
        assert not {r["base_id"] for r in test} & {r["base_id"] for r in few}
        held = {f"{e}/questions/{i}" for i in plan["held_out"]}
        assert not {r["template_id"] for r in test} & held
        for r in few:
            pool = r["template_id"].split("/")[1]
            q = s.sample_question(e, r["seed"], template_id=r["template_id"], pool=pool)
            assert build_rows(q, r["generator"])[r["part"]] == r


def test_few_shot_zero_keeps_old_export(tmp_path):
    """--few-shot-per-type 0: test from all templates, as before the few_shot split."""
    args = argparse.Namespace(out=str(tmp_path), seed=7, elements=["add_sub"], per_cell=20, floor=200,
                              max_attempts_per_question=20, few_shot_per_type=0, max_held_out=5, workers=1,
                              allow_dirty=True, allow_unverified_data=True, argv=["--test"])
    assert export.run(args) == 0
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert "few_shot" not in m["elements"]["add_sub"]
    assert not (tmp_path / "steer" / "add_sub" / "few_shot.parquet").exists()
