"""A dataset: one folder of an element's questions or templates, with their options, the instructions the authoring
pipeline writes from, and its review state. Stored by autosteer.dataset_io (dataset.json plus a Parquet file per
table, format autosteer-dataset/1).

The tables:

- questions / options: the question (template) rows, one per part (question_id, sub_id), and their options
  (option_id, option_text, correct_answer).
- example_questions / example_options: the same for the example pool (few-shot, and the seeds of the authoring
  pipeline), with an explanation column.
- instruction_set: instructions per metadata combination (domain, type, difficulty_level, tags).
- validation_df: a review status per question_id.

The sampler only reads the four question and option tables; developer mode (autosteer.services) also appends rows,
adds instructions and selects by metadata.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import ClassVar

import pandas as pd

metadata_keys = ['domain', 'type', 'difficulty_level', 'tags']
id_keys = ['question_id', 'sub_id']
question_keys = ['question_text'] + metadata_keys
example_question_keys = ['question_text', 'explanation'] + metadata_keys
options_keys = ['option_id', 'option_text', 'correct_answer']
instruction_keys = metadata_keys + ['instructions']

# Attributes that older dataset.json files carry (they described the code, not the data); read and ignored.
_KEY_CONSTANTS = {'metadata_keys', 'question_keys', 'options_keys', 'id_keys', 'example_question_keys',
                  'example_options_keys', 'instruction_keys'}
_POOLS = {'questions': ('questions', 'options'), 'example_questions': ('example_questions', 'example_options')}


def empty_questions() -> pd.DataFrame:
    return pd.DataFrame({k: [] for k in id_keys + question_keys}).astype(
        {'question_id': int, 'sub_id': int, 'question_text': str, 'domain': str, 'type': str,
         'difficulty_level': int, 'tags': str})


def empty_example_questions() -> pd.DataFrame:
    return pd.DataFrame({k: [] for k in id_keys + example_question_keys}).astype(
        {'question_id': int, 'sub_id': int, 'question_text': str, 'explanation': str, 'domain': str, 'type': str,
         'difficulty_level': int, 'tags': str})


def empty_options() -> pd.DataFrame:
    return pd.DataFrame({k: [] for k in id_keys + options_keys}).astype(
        {'question_id': int, 'sub_id': int, 'option_id': int, 'option_text': str, 'correct_answer': bool})


def empty_instructions() -> pd.DataFrame:
    return pd.DataFrame({k: [] for k in instruction_keys}).astype(
        {'domain': str, 'type': str, 'difficulty_level': int, 'tags': str, 'instructions': str})


def empty_validation() -> pd.DataFrame:
    return pd.DataFrame({'question_id': [], 'validation_status': []}).astype({'question_id': int, 'validation_status': int})


def _safe_value(value):
    """A value Parquet can store: plain values as they are, containers recursively, anything else as its repr."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _safe_value(item) for key, item in value.items()}
    return repr(value)


def _safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].map(_safe_value)
    return df


@dataclass(eq=False)
class Dataset:
    element_name: str | None = None
    questions: pd.DataFrame = field(default_factory=empty_questions)
    options: pd.DataFrame = field(default_factory=empty_options)
    example_questions: pd.DataFrame = field(default_factory=empty_example_questions)
    example_options: pd.DataFrame = field(default_factory=empty_options)
    instruction_set: pd.DataFrame = field(default_factory=empty_instructions)
    validation_df: pd.DataFrame = field(default_factory=empty_validation)
    # the authoring pipeline's records of what generated which seed or template (a table, or None)
    generated_seed_metadata: pd.DataFrame | None = None
    generated_template_metadata: pd.DataFrame | None = None
    # any other attribute a stored dataset has, written back as it was read
    extra: dict = field(default_factory=dict)

    metadata_keys: ClassVar[list[str]] = metadata_keys
    question_keys: ClassVar[list[str]] = question_keys
    instruction_keys: ClassVar[list[str]] = instruction_keys

    def __repr__(self) -> str:
        return (f"Dataset({self.element_name!r}: {len(self.questions)} question rows, "
                f"{len(self.example_questions)} example rows, {len(self.instruction_set)} instructions)")

    # ------------------------------------------------------------------------------------------------- on disk
    def save(self, folder) -> None:
        """Write the dataset into folder (dataset.json and the Parquet tables, see autosteer.dataset_io)."""
        from autosteer import dataset_io
        attrs = {'element_name': self.element_name, 'questions': self.questions, 'options': self.options,
                 'example_questions': self.example_questions, 'example_options': self.example_options,
                 'instruction_set': self.instruction_set, 'validation_df': self.validation_df,
                 'generated_seed_metadata': self.generated_seed_metadata,
                 'generated_template_metadata': self.generated_template_metadata, **self.extra}
        for key in ('generated_seed_metadata', 'generated_template_metadata'):
            if isinstance(attrs[key], pd.DataFrame):
                attrs[key] = _safe_frame(attrs[key])
        dataset_io.save(attrs, folder)

    @classmethod
    def load(cls, folder) -> Dataset:
        """Read a dataset saved in folder; its element_name defaults to the folder's name."""
        from autosteer import dataset_io
        attrs = {k: v for k, v in dataset_io.load(folder).items() if k not in _KEY_CONSTANTS}
        known = {f for f in cls.__dataclass_fields__ if f != 'extra'}
        ds = cls(**{k: v for k, v in attrs.items() if k in known},
                 extra={k: v for k, v in attrs.items() if k not in known})
        if not isinstance(ds.element_name, str) or not ds.element_name:
            ds.element_name = os.path.basename(os.path.normpath(folder))
        return ds

    # ------------------------------------------------------------------------------------------------ authoring
    @staticmethod
    def frame_names(dataframe_type: str = 'questions') -> tuple[str, str]:
        """The question and option table names of a pool: 'questions' or 'example_questions'."""
        if dataframe_type not in _POOLS:
            raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")
        return _POOLS[dataframe_type]

    def frames(self, dataframe_type: str = 'questions') -> tuple[pd.DataFrame, pd.DataFrame]:
        q, o = self.frame_names(dataframe_type)
        return getattr(self, q), getattr(self, o)

    def append(self, rows, dataframe_type: str = 'questions', ignore_options: bool = False) -> None:
        """Append rows (dicts with question_id, sub_id, question_text, the metadata keys, and unless ignore_options
        'options': [{option_id, option_text, correct_answer}]). The question_ids are renumbered to follow the
        dataset's largest one (a row with sub_id 0 starts a new question); sub_ids must count up within a question."""
        questions_type, options_type = self.frame_names(dataframe_type)
        if isinstance(rows, dict):
            rows = [rows]
        current = getattr(self, questions_type)
        max_question_id = int(current['question_id'].max()) if not current.empty else -1

        df = pd.DataFrame(rows)
        questions_df = df[id_keys + question_keys]
        sub_ids = questions_df['sub_id'].tolist()
        for i in range(1, len(sub_ids)):
            if not (sub_ids[i] == 0 or sub_ids[i] == sub_ids[i - 1] + 1):
                raise AssertionError(f"sub_id values are not in order or do not reset to zero at index {i}. "
                                     f"A subset: {sub_ids[i - 5: i + 5]}")
        questions_df.loc[:, 'question_id'] = (questions_df['sub_id'] == 0).cumsum() + max_question_id
        nan_rows = questions_df[questions_df.isna().any(axis=1)]
        if not nan_rows.empty:
            raise ValueError(f"Rows {nan_rows} have nan values.")
        questions_df = questions_df.astype({'question_id': int, 'sub_id': int, 'question_text': str, 'domain': str,
                                            'type': str, 'difficulty_level': int, 'tags': str})

        if not ignore_options:
            options_df = df[id_keys + ['options']]
            options_df = options_df.explode('options').reset_index(drop=True)
            options_df = pd.concat([options_df, pd.json_normalize(options_df['options'])], axis=1).drop(columns=['options'])
            # the options follow their questions' new ids, whatever ids the caller passed
            qid_map = dict(zip(df['question_id'].tolist(), questions_df['question_id'].tolist()))
            options_df['question_id'] = options_df['question_id'].map(qid_map)
            nan_rows = options_df[options_df.isna().any(axis=1)]
            if not nan_rows.empty:
                raise ValueError(f"Rows {nan_rows['question_id'].tolist()} have nan values.")
            options_df = options_df.astype({'question_id': int, 'sub_id': int, 'option_id': int, 'correct_answer': bool})

        setattr(self, questions_type, pd.concat([current, questions_df], ignore_index=True))
        if not ignore_options:
            setattr(self, options_type, pd.concat([getattr(self, options_type), options_df], ignore_index=True))

    def add_instructions(self, instructions, check_metadata_exists: bool = True) -> None:
        """Add instruction rows (dicts with the metadata keys and 'instructions'). With check_metadata_exists, each
        row's metadata must occur in the example questions."""
        if isinstance(instructions, list):
            for instruction in instructions:
                self.add_instructions(instruction, check_metadata_exists=check_metadata_exists)
            return
        if not isinstance(instructions, dict):
            raise ValueError("instructions must be either a list of dictionaries or a single dictionary")
        for key in instruction_keys:
            if key not in instructions:
                raise ValueError(f"a row has no key {key}. Every row needs to have a {key} key.")
        if check_metadata_exists:
            questions, _ = self.query({key: instructions[key] for key in metadata_keys}, dataframe_type='example_questions')
            if questions.empty:
                raise ValueError("Metadata of the instructions does not exist in the example questions dataset.")
        self.instruction_set = pd.concat([self.instruction_set, pd.DataFrame.from_records([instructions])])

    def query(self, command, dataframe_type: str = 'questions', return_type: str = 'tuple'):
        """The questions of a pool (with their options) whose metadata match `command` (a dict, or a list of dicts
        for any of them; see metadata_mask), or a pandas query string over the joined tables. Returns (questions,
        options), or with return_type='dataset' a Dataset with that pool replaced by the selection."""
        if return_type not in ('tuple', 'dataset'):
            raise ValueError("return_type must be either 'tuple' or 'dataset'")
        df = combine(*self.frames(dataframe_type))
        df = df.query(command) if isinstance(command, str) else df[metadata_mask(df, command)]
        questions, options = split(df, dataframe_type=dataframe_type)
        if return_type == 'tuple':
            return questions, options
        if dataframe_type == 'questions':
            return from_tuple(questions, options, self.example_questions, self.example_options)
        return from_tuple(self.questions, self.options, questions, options)

    def to_display_json(self, dataframe_type: str = 'example_questions') -> list[dict]:
        """The pool as display JSON: per question, question_k / options_k / answer_k for each part, and metadata."""
        q, o = self.frames(dataframe_type)
        options = o.sort_values(['question_id', 'sub_id', 'option_id'])   # a sorted copy
        json_list = []
        for question_id, group in q.groupby('question_id'):
            json_object = {}
            for sub_id, sub_group in group.groupby('sub_id'):
                json_object[f'question_{sub_id + 1}'] = sub_group['question_text'].values[0]
                opts = options[(options['question_id'] == question_id) & (options['sub_id'] == sub_id)]
                if opts.empty:
                    json_object[f'options_{sub_id + 1}'] = ['No options found']
                    json_object[f'answer_{sub_id + 1}'] = 'No answer found'
                else:
                    json_object[f'options_{sub_id + 1}'] = opts['option_text'].to_list()
                    correct = opts[opts['correct_answer'] == True]['option_text'].to_list()  # noqa: E712
                    json_object[f'answer_{sub_id + 1}'] = correct[0] if correct else 'No answer found'
            json_object['metadata'] = sub_group[metadata_keys].to_dict('records')[0]
            json_list.append(json_object)
        return json_list

    def metadata(self, dataframe_type: str = 'questions') -> pd.DataFrame:
        """The distinct metadata combinations of a table."""
        return getattr(self, dataframe_type)[metadata_keys].drop_duplicates().reset_index(drop=True)


def _remove_gaps(ids: list[int]) -> list[int]:
    """Renumber sorted ids 0, 1, 2, ... keeping repeats (the rows of one question share its id)."""
    new = {v: i for i, v in enumerate(sorted(set(ids)))}
    return [new[v] for v in ids]


def reindex(dataset: Dataset, in_place: bool = False) -> Dataset | None:
    """Renumber the question_ids of every question and option table from 0 without gaps (instructions are kept).
    Returns a new Dataset unless in_place."""
    def renumber(df: pd.DataFrame) -> pd.DataFrame:
        df = flatten_index(df)
        if df.empty:
            return df
        df['question_id'] = _remove_gaps(df['question_id'].astype(int).to_list())
        return df.reset_index(drop=True)

    instruction_set = dataset.instruction_set
    if not in_place:
        dataset = from_tuple(dataset.questions.copy(), dataset.options.copy(),
                             dataset.example_questions.copy(), dataset.example_options.copy())
    dataset.questions = renumber(dataset.questions)
    dataset.options = renumber(dataset.options)
    dataset.example_questions = renumber(dataset.example_questions)
    dataset.example_options = renumber(dataset.example_options)
    dataset.instruction_set = instruction_set
    if not in_place:
        return dataset
    return None


def flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of df with question_id / sub_id as ordinary columns (not the index)."""
    if not isinstance(df, pd.DataFrame):
        return df
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex) and {'question_id', 'sub_id'}.issubset(df.index.names):
        index_df = df.index.to_frame(index=False)
        insert_at = 0
        for column in ['question_id', 'sub_id']:
            if column not in df.columns:
                df.insert(insert_at, column, index_df[column].to_numpy())
                insert_at += 1
        return df.reset_index(drop=True)
    if df.index.name in {'question_id', 'sub_id'} and df.index.name not in df.columns:
        return df.reset_index()
    return df.reset_index(drop=True)


def from_tuple(questions, options, example_questions=None, example_options=None) -> Dataset:
    if isinstance(example_questions, pd.DataFrame) ^ isinstance(example_options, pd.DataFrame):
        raise ValueError("Must provide both example_questions and example_options or neither")
    dataset = Dataset(questions=questions, options=options)
    if isinstance(example_questions, pd.DataFrame):
        dataset.example_questions, dataset.example_options = example_questions, example_options
    return dataset


def get_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """The distinct metadata combinations of df's rows, by difficulty."""
    return df[metadata_keys].drop_duplicates().sort_values('difficulty_level')


def combine(questions: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """Questions joined with their options (one row per option); without options, a placeholder option per row."""
    if options.empty:
        combined_df = questions.copy()
        combined_df['option_id'] = [0] * len(questions)
        combined_df['option_text'] = ['No options'] * len(questions)
        combined_df['correct_answer'] = [True] * len(questions)
        return combined_df
    try:
        return pd.merge(left=questions, right=options, how='inner', on=['question_id', 'sub_id'])
    except KeyError:
        try:
            return questions.join(options, how='inner')   # question_id / sub_id as the index
        except KeyError as e:
            raise KeyError(f"{e}. Ensure that both DataFrames contain the columns 'question_id' and 'sub_id' or "
                           "that they are set as the index.")


def split(combined_df: pd.DataFrame, dataframe_type: str = 'questions') -> tuple[pd.DataFrame, pd.DataFrame]:
    """The questions and options tables of a joined table (see combine)."""
    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df must be a pandas DataFrame.")
    if dataframe_type not in _POOLS:
        raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")
    keys = question_keys if dataframe_type == 'questions' else example_question_keys
    questions = combined_df[id_keys + keys].drop_duplicates()
    try:
        options = combined_df[id_keys + options_keys]
    except KeyError:
        options = pd.DataFrame()
    return questions, options


def _literal(value):
    """A difficulty_level as a number, as a query string would have read it ("3" and 3 are both 3)."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return float(value)
    return value


def metadata_mask(df: pd.DataFrame, metadata: list[dict] | dict) -> pd.Series:
    """Rows of df whose metadata equals `metadata` (key: value; difficulty_level compared as a number, every other
    key as text); a list of dicts selects the rows matching any of them. Values are compared, never parsed, so text
    with quotes in it is fine."""
    if isinstance(metadata, list):
        mask = pd.Series(False, index=df.index)
        for one in metadata:
            mask |= metadata_mask(df, one)
        return mask
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be either a list of dictionaries or a dictionary")
    mask = pd.Series(True, index=df.index)
    for key, value in metadata.items():
        mask &= df[key] == (_literal(value) if key == 'difficulty_level' else str(value))
    return mask


def display_to_dataset_json(display_json, ignore_options: bool = False) -> list[dict]:
    """Rows for Dataset.append from display JSON (one dict, or a list): per question, question_k / options_k /
    answer_k for each part k = 1, 2, ..., 'metadata' (domain, type, difficulty_level, tags) and optionally
    'instructions'. The question_ids number the questions from 0; with ignore_options each part gets one placeholder
    option."""
    if isinstance(display_json, dict):
        display_json = [display_json]
    dataset_json = []
    for question_id, question in enumerate(display_json):
        sub_question_keys = sorted(key for key in question if key.startswith('question_'))
        for sub_id, sub_question_key in enumerate(sub_question_keys):
            if ignore_options:
                options = [{"option_id": 0, "option_text": "No options found", "correct_answer": False}]
            else:
                options = [{"option_id": option_id, "option_text": option,
                            "correct_answer": option == question[f'answer_{sub_id + 1}']}
                           for option_id, option in enumerate(question[f'options_{sub_id + 1}'])]
            json_object = {
                "question_id": question_id, "sub_id": sub_id, "question_text": question[sub_question_key],
                "domain": question['metadata']['domain'], "type": question['metadata']['type'],
                "difficulty_level": question['metadata']['difficulty_level'], "tags": question['metadata']['tags'],
                "options": options,
            }
            if 'instructions' in question:
                json_object['instructions'] = question['instructions']
            dataset_json.append(json_object)
    return dataset_json


def get_instructions(instruction_set: pd.DataFrame, metadata: dict, get_closest_match: bool = True) -> pd.DataFrame:
    """
    The instruction rows (with instructions) for `metadata`. With get_closest_match, when none match exactly, the closest:
    leaving out one key, then two, and so on, the least important key first (tags, then domain, then
    difficulty_level, then type); the first combination that matches wins.
    """
    def matching(md):
        rows = instruction_set[metadata_mask(instruction_set, md)]
        return rows[rows['instructions'].notna()]

    rows = matching(metadata)
    if not rows.empty or not get_closest_match:
        return rows
    priority = ['type', 'difficulty_level', 'domain', 'tags']   # most important first
    least_first = sorted(metadata, key=lambda k: -(priority.index(k) if k in priority else len(priority)))
    for n in range(1, len(metadata)):
        for left_out in combinations(least_first, n):
            closer = matching({k: v for k, v in metadata.items() if k not in left_out})
            if not closer.empty:
                return closer
    return rows
