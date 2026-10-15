"""Release schema: DatasetRow rules, the Arrow schema and table-level validation."""
import pytest
from pydantic import ValidationError

from autosteer.release.schema import SCHEMA, to_table, validate
from autosteer.schemas import DATASET_FIELDS, DatasetRow


def row(**kw):
    r = {
        "element": "law_of_demand", "question_id": "law_of_demand/abcd/0", "base_id": "abcd", "part": 0,
        "template_id": "law_of_demand/questions/3", "question_text": "Q?", "options": ["1", "2", "3"],
        "grading": "answer", "correct_index": 1, "consistency_group": None, "consistency_rule": None,
        "type": "price_decrease", "domain": "consumer_goods", "perspective": "first_person", "tags": [],
        "difficulty": 0, "setting": "S", "module": "M", "generator": "autosteer@0.1.0+abc1234", "seed": 2 ** 63 - 1,
    }
    r.update(kw)
    return r


def test_fields_match_spec_order():
    assert DATASET_FIELDS == tuple(SCHEMA.names)
    assert DATASET_FIELDS == (
        "element", "question_id", "base_id", "part", "template_id", "question_text", "options", "grading",
        "correct_index", "consistency_group", "consistency_rule", "type", "domain", "perspective", "tags",
        "difficulty", "setting", "module", "generator", "seed")
    assert str(SCHEMA.field("seed").type) == "int64"


def test_valid_rows():
    DatasetRow(**row())
    DatasetRow(**row(grading="consistency", correct_index=None, consistency_group="law_of_demand/abcd",
                     consistency_rule="common_ratio_same_side:01,10", options=["a", "b"]))
    assert validate(to_table([row()])) == []


@pytest.mark.parametrize("bad", [
    {"options": ["only one"]},
    {"options": ["1", "1", "2"]},
    {"correct_index": None},
    {"correct_index": 3},
    {"grading": "consistency"},
    {"consistency_group": "x"},
    {"question_id": "law_of_demand/other/0"},
    {"template_id": "other/questions/1"},
    {"type": " "},
    {"seed": -1},
    {"seed": 2 ** 63},
    {"source": "sampler@abc"},
])
def test_invalid_rows(bad):
    with pytest.raises(ValidationError):
        DatasetRow(**row(**bad))


def test_template_id_required():
    with pytest.raises(ValidationError):
        DatasetRow(**{k: v for k, v in row().items() if k != "template_id"})
    assert validate(to_table([row(template_id=None)]))


def test_table_checks():
    a = row()
    b = row(base_id="efgh", question_id="law_of_demand/efgh/0")
    assert validate(to_table([a, a]))  # duplicate question_id
    assert any("duplicate question" in e for e in validate(to_table([a, b])))
    c = row(base_id="efgh", question_id="law_of_demand/efgh/1", part=1, question_text="other")
    assert any("parts are not" in e for e in validate(to_table([a, c])))
