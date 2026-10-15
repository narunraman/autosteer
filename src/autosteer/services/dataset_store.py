from __future__ import annotations

import re
import os
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import pandas as pd

from autosteer import dataset_io

from autosteer import steer_dataset as sd
from autosteer.directory_utils import DATA_PATH
from autosteer.schemas import DatasetKind, Metadata, OptionRecord, QuestionRecord


DATASET_KINDS: tuple[DatasetKind, ...] = (
    "example_seeds",
    "generated_seeds",
    "templates",
)

DATASET_DIRS: dict[DatasetKind, str] = {
    "example_seeds": "example_seeds",
    "generated_seeds": "generated_seeds",
    "templates": "templates",
}


def _clean_optional_string(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _normalized_question_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s{}]", "", text)


class DatasetStore:
    """Filesystem-backed access to STEER Dataset artifacts."""

    def __init__(self, data_root: str | os.PathLike = DATA_PATH):
        self.data_root = Path(data_root)

    def kind_path(self, kind: DatasetKind) -> Path:
        if kind not in DATASET_KINDS:
            raise ValueError(f"Unsupported dataset kind: {kind}")
        return self.data_root / DATASET_DIRS[kind]

    def dataset_path(self, kind: DatasetKind, element_name: str) -> Path:
        return self.kind_path(kind) / element_name

    def list_elements(self, kind: DatasetKind | None = None) -> list[str]:
        kinds = [kind] if kind else list(DATASET_KINDS)
        names: set[str] = set()
        for current_kind in kinds:
            root = self.kind_path(current_kind)
            if not root.exists():
                continue
            names.update(path.name for path in root.iterdir() if path.is_dir())
        return sorted(names)

    def has_dataset(self, kind: DatasetKind, element_name: str) -> bool:
        return dataset_io.saved_file(self.dataset_path(kind, element_name)) is not None

    def load_dataset(self, kind: DatasetKind, element_name: str) -> sd.Dataset:
        path = self.dataset_path(kind, element_name)
        if dataset_io.saved_file(path) is None:
            raise FileNotFoundError(f"No saved dataset for {kind}/{element_name}")
        return sd.Dataset.load(str(path))

    def save_dataset(self, kind: DatasetKind, element_name: str, dataset: sd.Dataset) -> None:
        dataset.element_name = element_name
        dataset.save(str(self.dataset_path(kind, element_name)))

    def get_or_create_dataset(self, kind: DatasetKind, element_name: str) -> sd.Dataset:
        if self.has_dataset(kind, element_name):
            return self.load_dataset(kind, element_name)
        return sd.Dataset(element_name=element_name)

    def summarize_element(self, element_name: str) -> dict:
        summary = {"element_name": element_name, "kinds": {}}
        for kind in DATASET_KINDS:
            if not self.has_dataset(kind, element_name):
                summary["kinds"][kind] = None
                continue
            dataset = self.load_dataset(kind, element_name)
            summary["kinds"][kind] = {
                "questions": len(dataset.questions),
                "options": len(dataset.options),
                "example_questions": len(dataset.example_questions),
                "example_options": len(dataset.example_options),
                "metadata_rows": len(dataset.metadata("questions")) if not dataset.questions.empty else 0,
            }
        return summary

    def records(
        self,
        kind: DatasetKind,
        element_name: str,
        dataframe_type: str = "questions",
        limit: int = 50,
    ) -> list[QuestionRecord]:
        dataset = self.load_dataset(kind, element_name)
        questions_type, options_type = dataset._get_dataframe_types(dataframe_type)
        questions = dataset[questions_type].head(limit)
        options = dataset[options_type]

        records: list[QuestionRecord] = []
        for _, row in questions.iterrows():
            question_id = int(row["question_id"])
            sub_id = int(row["sub_id"])
            option_rows = options[
                (options["question_id"] == question_id) &
                (options["sub_id"] == sub_id)
            ].sort_values("option_id")

            instructions = None
            metadata_dict = {key: row[key] for key in sd.metadata_keys}
            if dataframe_type == "example_questions" and not dataset.instruction_set.empty:
                matches = dataset.instruction_set[sd.metadata_mask(dataset.instruction_set, metadata_dict)]
                if not matches.empty:
                    instructions = matches["instructions"].iloc[0]

            records.append(
                QuestionRecord(
                    question_id=question_id,
                    sub_id=sub_id,
                    question_text=row["question_text"],
                    metadata=Metadata(
                        domain=str(row["domain"]),
                        type=str(row["type"]),
                        difficulty_level=int(row["difficulty_level"]),
                        tags="" if pd.isna(row["tags"]) else str(row["tags"]),
                    ),
                    options=[
                        OptionRecord(
                            option_id=int(option["option_id"]),
                            option_text="" if pd.isna(option["option_text"]) else str(option["option_text"]),
                            correct_answer=bool(option["correct_answer"]),
                        )
                        for _, option in option_rows.iterrows()
                    ],
                    explanation=_clean_optional_string(row.get("explanation")) if "explanation" in row else None,
                    instructions=_clean_optional_string(instructions),
                )
            )
        return records

    def template_preview(self, element_name: str) -> QuestionRecord | None:
        if not self.has_dataset("templates", element_name):
            return None
        records = self.records("templates", element_name, dataframe_type="questions", limit=1)
        return records[0] if records else None

    def append_seed(
        self,
        element_name: str,
        question_text: str,
        metadata: Metadata,
        options: list[OptionRecord],
        explanation: str = "",
        instructions: str | None = None,
    ) -> sd.Dataset:
        dataset = self.get_or_create_dataset("example_seeds", element_name)
        next_question_id = 0 if dataset.example_questions.empty else int(dataset.example_questions["question_id"].max()) + 1
        row = {
            "question_id": next_question_id,
            "sub_id": 0,
            "question_text": question_text,
            "explanation": explanation,
            "domain": metadata.domain,
            "type": metadata.type,
            "difficulty_level": metadata.difficulty_level,
            "tags": metadata.tags,
            "options": [option.model_dump() for option in options],
        }
        dataset.append(row, dataframe_type="example_questions")

        if instructions:
            metadata_dict = {
                "domain": metadata.domain,
                "type": metadata.type,
                "difficulty_level": metadata.difficulty_level,
                "tags": metadata.tags,
            }
            self._upsert_instructions(dataset, metadata_dict, metadata_dict, instructions)

        self.save_dataset("example_seeds", element_name, dataset)
        return dataset

    def seed_duplicates(
        self,
        element_name: str,
        question_text: str,
        metadata: Metadata,
        question_id: int | None = None,
        sub_id: int = 0,
        similarity_threshold: float = 0.92,
    ) -> list[dict]:
        if not self.has_dataset("example_seeds", element_name):
            return []

        dataset = self.load_dataset("example_seeds", element_name)
        if dataset.example_questions.empty:
            return []

        normalized_question = _normalized_question_text(question_text)
        duplicates = []
        for _, row in dataset.example_questions.iterrows():
            candidate_question_id = int(row["question_id"])
            candidate_sub_id = int(row["sub_id"])
            if question_id is not None and candidate_question_id == question_id and candidate_sub_id == sub_id:
                continue

            candidate_text = str(row["question_text"])
            normalized_candidate = _normalized_question_text(candidate_text)
            exact = normalized_candidate == normalized_question
            metadata_matches = (
                str(row["domain"]) == metadata.domain
                and str(row["type"]) == metadata.type
                and int(row["difficulty_level"]) == metadata.difficulty_level
                and ("" if pd.isna(row["tags"]) else str(row["tags"])) == metadata.tags
            )
            similarity = SequenceMatcher(None, normalized_question, normalized_candidate).ratio()
            if exact or (metadata_matches and similarity >= similarity_threshold):
                duplicates.append(
                    {
                        "question_id": candidate_question_id,
                        "sub_id": candidate_sub_id,
                        "question_text": candidate_text,
                        "exact": exact,
                        "similarity": round(similarity, 3),
                    }
                )
        return duplicates

    def seed_draft(self, element_name: str, question_id: int, sub_id: int = 0) -> dict:
        dataset = self.load_dataset("example_seeds", element_name)
        question_mask = (
            (dataset.example_questions["question_id"] == question_id) &
            (dataset.example_questions["sub_id"] == sub_id)
        )
        if not question_mask.any():
            raise FileNotFoundError(f"No example seed found for {element_name} id {question_id}.{sub_id}")

        question = dataset.example_questions.loc[question_mask].iloc[0]
        option_rows = dataset.example_options[
            (dataset.example_options["question_id"] == question_id) &
            (dataset.example_options["sub_id"] == sub_id)
        ].sort_values("option_id")
        meaningful_options = option_rows[
            option_rows["option_text"].fillna("").astype(str).str.strip() != ""
        ]

        correct_answer = ""
        options_text = ""
        if not meaningful_options.empty:
            option_texts = meaningful_options["option_text"].astype(str).tolist()
            options_text = "\n".join(option_texts)
            correct_positions = [
                i
                for i, (_, option) in enumerate(meaningful_options.iterrows())
                if bool(option["correct_answer"])
            ]
            if correct_positions:
                correct_answer = str(correct_positions[0])

        instructions = ""
        if not dataset.instruction_set.empty:
            metadata_dict = {key: question[key] for key in sd.metadata_keys}
            matches = dataset.instruction_set[sd.metadata_mask(dataset.instruction_set, metadata_dict)]
            if not matches.empty:
                instructions = str(matches["instructions"].iloc[0])

        return {
            "element_name": element_name,
            "question_id": int(question_id),
            "sub_id": int(sub_id),
            "question_text": str(question["question_text"]),
            "domain": str(question["domain"]),
            "type": str(question["type"]),
            "difficulty_level": int(question["difficulty_level"]),
            "tags": "" if pd.isna(question["tags"]) else str(question["tags"]),
            "options_text": options_text,
            "correct_answer": correct_answer,
            "instructions": instructions,
        }

    def save_seed(
        self,
        element_name: str,
        question_text: str,
        metadata: Metadata,
        options: list[OptionRecord],
        explanation: str = "",
        instructions: str | None = None,
        question_id: int | None = None,
        sub_id: int = 0,
    ) -> tuple[sd.Dataset, int]:
        if question_id is None:
            dataset = self.append_seed(
                element_name=element_name,
                question_text=question_text,
                metadata=metadata,
                options=options,
                explanation=explanation,
                instructions=instructions,
            )
            saved_question_id = int(dataset.example_questions["question_id"].max())
            return dataset, saved_question_id

        dataset = self.load_dataset("example_seeds", element_name)
        question_mask = (
            (dataset.example_questions["question_id"] == question_id) &
            (dataset.example_questions["sub_id"] == sub_id)
        )
        if not question_mask.any():
            raise FileNotFoundError(f"No example seed found for {element_name} id {question_id}.{sub_id}")

        old_metadata = {
            key: dataset.example_questions.loc[question_mask, key].iloc[0]
            for key in sd.metadata_keys
        }
        new_metadata = {
            "domain": metadata.domain,
            "type": metadata.type,
            "difficulty_level": metadata.difficulty_level,
            "tags": metadata.tags,
        }

        dataset.example_questions.loc[question_mask, ["question_text", "explanation"]] = [
            question_text,
            explanation,
        ]
        for key, value in new_metadata.items():
            dataset.example_questions.loc[question_mask, key] = value

        option_mask = (
            (dataset.example_options["question_id"] == question_id) &
            (dataset.example_options["sub_id"] == sub_id)
        )
        dataset.example_options = dataset.example_options.loc[~option_mask].copy()
        option_rows = [
            {
                "question_id": question_id,
                "sub_id": sub_id,
                **option.model_dump(),
            }
            for option in options
        ]
        if option_rows:
            options_df = pd.DataFrame(option_rows).astype(
                {
                    "question_id": int,
                    "sub_id": int,
                    "option_id": int,
                    "option_text": str,
                    "correct_answer": bool,
                }
            )
            dataset.example_options = pd.concat(
                [dataset.example_options, options_df],
                ignore_index=True,
            )

        self._upsert_instructions(dataset, old_metadata, new_metadata, instructions)
        self.save_dataset("example_seeds", element_name, dataset)
        return dataset, int(question_id)

    def delete_seed(self, element_name: str, question_id: int, sub_id: int = 0) -> None:
        dataset = self.load_dataset("example_seeds", element_name)
        question_mask = (
            (dataset.example_questions["question_id"] == question_id) &
            (dataset.example_questions["sub_id"] == sub_id)
        )
        if not question_mask.any():
            raise FileNotFoundError(f"No example seed found for {element_name} id {question_id}.{sub_id}")

        old_metadata = {
            key: dataset.example_questions.loc[question_mask, key].iloc[0]
            for key in sd.metadata_keys
        }
        option_mask = (
            (dataset.example_options["question_id"] == question_id) &
            (dataset.example_options["sub_id"] == sub_id)
        )
        dataset.example_questions = dataset.example_questions.loc[~question_mask].copy()
        dataset.example_options = dataset.example_options.loc[~option_mask].copy()

        if not dataset.example_questions.empty:
            old_mask = self._metadata_mask(dataset.example_questions, old_metadata)
        else:
            old_mask = pd.Series([], dtype=bool)
        if not old_mask.any() and not dataset.instruction_set.empty:
            old_instruction_mask = self._metadata_mask(dataset.instruction_set, old_metadata)
            dataset.instruction_set = dataset.instruction_set.loc[~old_instruction_mask].copy()

        self.save_dataset("example_seeds", element_name, dataset)

    def _metadata_mask(self, df: pd.DataFrame, metadata: dict) -> pd.Series:
        if df.empty:
            return pd.Series([], dtype=bool)
        mask = pd.Series(True, index=df.index)
        for key, value in metadata.items():
            if key == "difficulty_level":
                mask &= df[key].astype(int) == int(value)
            else:
                mask &= df[key].fillna("").astype(str) == str(value)
        return mask

    def _upsert_instructions(
        self,
        dataset: sd.Dataset,
        old_metadata: dict,
        new_metadata: dict,
        instructions: str | None,
    ) -> None:
        if dataset.instruction_set.empty:
            if instructions:
                dataset.add_instructions({**new_metadata, "instructions": instructions}, check_metadata_exists=False)
            return

        new_mask = self._metadata_mask(dataset.instruction_set, new_metadata)
        dataset.instruction_set = dataset.instruction_set.loc[~new_mask].copy()

        old_mask = self._metadata_mask(dataset.example_questions, old_metadata)
        if not old_mask.any():
            old_instruction_mask = self._metadata_mask(dataset.instruction_set, old_metadata)
            dataset.instruction_set = dataset.instruction_set.loc[~old_instruction_mask].copy()

        if instructions:
            dataset.add_instructions({**new_metadata, "instructions": instructions}, check_metadata_exists=False)


@lru_cache(maxsize=1)
def get_dataset_store() -> DatasetStore:
    return DatasetStore()
