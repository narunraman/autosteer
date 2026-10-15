from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DatasetKind = Literal["example_seeds", "generated_seeds", "templates"]


class Metadata(BaseModel):
    domain: str
    type: str
    difficulty_level: int
    tags: str = ""


class OptionRecord(BaseModel):
    option_id: int
    option_text: str
    correct_answer: bool = False


class QuestionRecord(BaseModel):
    question_id: int
    sub_id: int = 0
    question_text: str
    metadata: Metadata
    options: list[OptionRecord] = Field(default_factory=list)
    explanation: str | None = None
    instructions: str | None = None


class SamplerRequest(BaseModel):
    element_name: str | None = None
    domain: str | None = None
    type: str | None = None
    difficulty_level: int | str | None = None
    tags: str | None = None
    # Any of these switches the response to {"questions": [DatasetRow, ...]}.
    n: int | None = Field(default=None, ge=1, le=50)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    template_id: str | None = None


# ---------------------------------------------------------------------------------------------------------------
# Released dataset rows (STEER and STEER-ME): one row per question part. Same fields, in the same order, in the
# Parquet export (autosteer.release.schema) and in /api/sample responses.
# ---------------------------------------------------------------------------------------------------------------
Grading = Literal["answer", "consistency"]
SEED_MAX = 2**63 - 1


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: str
    question_id: str                      # f"{element}/{base_id}/{part}"
    base_id: str                          # content hash of the question (16 hex digits); shared by its parts
    part: int = Field(ge=0)
    template_id: str                      # f"{element}/{pool}/{n}": the template the question was filled from
    question_text: str
    options: list[str] = Field(min_length=2)  # presented order, no letter labels
    grading: Grading
    correct_index: int | None             # set iff grading == "answer"
    consistency_group: str | None         # set iff grading == "consistency"
    consistency_rule: str | None          # "<rule_id>:<allowed>", set iff grading == "consistency"
    type: str
    domain: str
    perspective: str | None               # first_person, second_person, man, woman, anon, ...; null if none
    tags: list[str]                       # other template descriptors (may be empty)
    difficulty: int                       # per-dataset scale (STEER-ME 0-2, STEER grade levels)
    setting: str
    module: str
    generator: str                        # "autosteer@<version>+<short sha>"
    seed: int = Field(ge=0, le=SEED_MAX)  # with element and template_id, reproduces the question

    @model_validator(mode="after")
    def _check(self) -> "DatasetRow":
        if self.question_id != f"{self.element}/{self.base_id}/{self.part}":
            raise ValueError("question_id must be <element>/<base_id>/<part>")
        if not self.template_id.startswith(f"{self.element}/"):
            raise ValueError("template_id must start with <element>/")
        if not self.type.strip():
            raise ValueError("type must not be empty (use 'default')")
        if len(set(self.options)) != len(self.options):
            raise ValueError("duplicate option texts")
        if self.grading == "answer":
            if self.correct_index is None or not 0 <= self.correct_index < len(self.options):
                raise ValueError("answer-graded rows need correct_index within options")
            if self.consistency_group is not None or self.consistency_rule is not None:
                raise ValueError("answer-graded rows have no consistency fields")
        else:
            if self.correct_index is not None:
                raise ValueError("consistency-graded rows have no correct_index")
            if not self.consistency_group or not self.consistency_rule:
                raise ValueError("consistency-graded rows need consistency_group and consistency_rule")
        return self


DATASET_FIELDS = tuple(DatasetRow.model_fields)
