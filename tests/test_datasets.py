"""What a Dataset does: every stored dataset loads, saves and loads back unchanged, and the authoring operations
(append, instructions, query, display JSON, metadata, reindex) give the outputs in tests/reference/dataset_ops.json.

    uv run python tests/test_datasets.py write     # rewrite the reference after an intended change
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pytest

from autosteer import paths
from autosteer.steer_dataset import Dataset, display_to_dataset_json, get_metadata_df, reindex

REFERENCE = Path(__file__).with_name("reference") / "dataset_ops.json"
FRAMES = ("questions", "options", "example_questions", "example_options", "instruction_set", "validation_df")


def stored_folders():
    root = paths.data_dir()
    return sorted(str(p.parent.relative_to(root)) for p in root.glob("*/*/dataset.json"))


@pytest.mark.parametrize("folder", stored_folders())
def test_round_trip(folder):
    d = Dataset.load(paths.data_dir() / folder)
    with tempfile.TemporaryDirectory() as tmp:
        d.save(tmp)
        e = Dataset.load(tmp)
    for name in FRAMES:
        pd.testing.assert_frame_equal(getattr(d, name), getattr(e, name), check_dtype=True, obj=name)
    assert d.element_name == e.element_name
    for name in ("generated_seed_metadata", "generated_template_metadata"):
        a, b = getattr(d, name, None), getattr(e, name, None)
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, obj=name)
        else:
            assert a is None and b is None or a == b, name


SEEDS = [
    {"question_1": "A {good} costs {p}. What is the demand at {p2}?", "options_1": ["10", "12", "14"], "answer_1": "12",
     "metadata": {"domain": "food", "type": "point", "difficulty_level": 1, "tags": "first_person"},
     "instructions": "Ask for the quantity."},
    {"question_1": "Part one about {x}?", "options_1": ["a", "b"], "answer_1": "b",
     "question_2": "Part two, it's {y}?", "options_2": ["c", "d", "e"], "answer_2": "c",
     "metadata": {"domain": "children's toys", "type": "two_part", "difficulty_level": 2, "tags": "third_person"}},
]


def operations() -> dict:
    d = Dataset("toy")
    rows = display_to_dataset_json(SEEDS)
    d.append(rows, dataframe_type="example_questions")
    d.append(display_to_dataset_json(SEEDS[:1], ignore_options=True), dataframe_type="example_questions", ignore_options=True)
    d.add_instructions({**SEEDS[0]["metadata"], "instructions": "Ask for the quantity."}, check_metadata_exists=True)
    q, o = d.query({"domain": "children's toys", "type": "two_part"}, dataframe_type="example_questions")
    sel = d.query({"type": "point", "difficulty_level": "1"}, dataframe_type="example_questions", return_type="dataset")
    r = reindex(d)
    frame = lambda df: json.loads(df.reset_index(drop=True).to_json(orient="split"))
    return {
        "rows": rows,
        "example_questions": frame(d.example_questions), "example_options": frame(d.example_options),
        "instruction_set": frame(d.instruction_set),
        "query_q": frame(q), "query_o": frame(o),
        "query_dataset_display": sel.to_display_json(),
        "display": d.to_display_json(),
        "metadata": frame(d.metadata("example_questions")),
        "metadata_df": frame(get_metadata_df(d.example_questions)),
        "reindexed_ids": r.example_questions["question_id"].tolist(),
    }


class OperationsTests(unittest.TestCase):
    def test_operations_match_the_reference(self):
        want = json.loads(REFERENCE.read_text())
        got = json.loads(json.dumps(operations(), default=str))
        for k in want:
            self.assertEqual(got[k], want[k], k)
        self.assertEqual(sorted(got), sorted(want))


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        REFERENCE.write_text(json.dumps(operations(), indent=1, default=str) + "\n")
        print("wrote", REFERENCE)
    else:
        unittest.main()
