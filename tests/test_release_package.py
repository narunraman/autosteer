"""autosteer release build / check: layout of the Hugging Face repos, cards, element files, provenance."""
import argparse
import json

import pyarrow.parquet as pq
import pytest

from autosteer.release import package

ELEMENTS = ["add_sub", "certainty_effect", "law_of_demand"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("release")
    args = argparse.Namespace(out=str(out), from_export="", seed=7, elements=ELEMENTS, per_cell=20, floor=200,
                              few_shot_per_type=5, max_held_out=5, max_attempts_per_question=20, workers=2,
                              allow_dirty=True, allow_unverified_data=True, allow_failures=False, argv=["--test"])
    assert package.build(args) == 0
    return out


def test_layout(built):
    for bench, elements in (("STEER", ["add_sub", "certainty_effect"]), ("STEER-ME", ["law_of_demand"])):
        repo = built / package.REPO_DIRS[bench]
        for f in ("README.md", "ELEMENTS.md", "elements.csv", "elements.jsonl", "provenance.json"):
            assert (repo / f).is_file()
        assert sorted(p.name for p in (repo / "data").iterdir()) == elements
        for e in elements:
            assert (repo / "data" / e / "test.parquet").is_file() and (repo / "data" / e / "few_shot.parquet").is_file()
        recs = [json.loads(x) for x in (repo / "elements.jsonl").read_text().splitlines()]
        assert [r["element"] for r in recs] == elements  # paper order
        for r in recs:
            t = pq.read_table(repo / "data" / r["element"] / "test.parquet")
            assert r["test_rows"] == t.num_rows and r["description"] and r["grading_rule"]
            assert "explanation" not in t.schema.names and "perspective" in t.schema.names
        prov = json.loads((repo / "provenance.json").read_text())
        assert prov["repo_id"] == package.REPOS[bench] and prov["seed"] == 7 and prov["git"]["sha"]
        assert prov["data_files"]["source"]["commit"] and prov["rule"]["per_cell"] == 20
        assert set(prov["files"]) == {str(p.relative_to(repo)) for p in (repo / "data").rglob("*.parquet")}


def test_card(built):
    text = (built / "steer" / "README.md").read_text()
    front = text.split("---\n")[1]
    assert "license: cc-by-4.0" in front and "- multiple-choice" in front
    configs = package._configs(built / "steer" / "README.md")
    assert [c["config_name"] for c in configs] == ["default", "add_sub", "certainty_effect"]
    assert "@inproceedings{ramansteer" in text and "TODO" not in text
    assert 'load_dataset("narunraman/steer", "add_sub")' in text
    assert "`common_ratio_same_side`" in text  # consistency rules listed
    me = (built / "steer-me" / "README.md").read_text()
    assert "@inproceedings{raman2025steerme" in me and "not comparable" in me
    assert "`add_sub`" in me  # shared Foundations elements point to STEER


def test_check_passes(built):
    assert package.check(argparse.Namespace(dir=str(built), partial=True)) == 0


def test_refuses_dirty_tree(tmp_path, monkeypatch):
    from autosteer.release import rows

    monkeypatch.setattr(rows, "git_state", lambda: {"sha": "x", "short_sha": "x", "dirty": True,
                                                    "dirty_files": ["a.py"]})
    args = argparse.Namespace(out=str(tmp_path), allow_dirty=False, argv=[])
    with pytest.raises(SystemExit, match="uncommitted"):
        package.build(args)
