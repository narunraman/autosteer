# src/steer_generation/steer_dataset.py

# Built-in packages
from itertools import combinations
import os
from typing import Union
# External packages
import pandas as pd
from tqdm import tqdm
# Local packages


# Constants
metadata_keys = ['domain', 'type', 'difficulty_level', 'tags']
id_keys = ['question_id', 'sub_id']
question_keys = ['question_text'] + metadata_keys
example_question_keys = ['question_text', 'explanation'] + metadata_keys
options_keys = ['option_id', 'option_text', 'correct_answer']
instruction_keys = metadata_keys + ['instructions']
# TODO: Should add template generation metadata so that we can store what was used to generate what


def _pickle_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_pickle_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pickle_safe_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _pickle_safe_value(item) for key, item in value.items()}
    return repr(value)


def _pickle_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].map(_pickle_safe_value)
    return df


def pickle_safe_attributes(attributes: dict) -> dict:
    attributes = attributes.copy()
    for key in ['generated_seed_metadata', 'generated_template_metadata']:
        value = attributes.get(key)
        if isinstance(value, pd.DataFrame):
            attributes[key] = _pickle_safe_dataframe(value)
    return attributes


class Dataset:
    def __init__(self, element_name = None):
        self.element_name = element_name
        self.metadata_keys = metadata_keys
        self.question_keys = question_keys
        self.options_keys = options_keys
        self.id_keys = id_keys
        self.example_question_keys = example_question_keys
        self.example_options_keys = options_keys
        self.instruction_keys = instruction_keys
        self.questions = self._init_questions_df()
        self.options = self._init_options_df()
        self.example_questions = self._init_example_questions_df()
        self.example_options = self._init_options_df()
        self.instruction_set = self._init_instructions_df()
        self.validation_df = self._init_validation_df()
        self.generated_seed_metadata = None
        self.generated_template_metadata = None


    def _init_questions_df(self):
        d = {key: [] for key in self.id_keys + self.question_keys}
        
        column_types = {
            'question_id': int,
            'sub_id': int,
            'question_text': str,
            'domain': str,
            'type': str,
            'difficulty_level': int,
            'tags': str
        }

        return pd.DataFrame(data = d).astype(column_types)

    def _init_example_questions_df(self):
        d = {key: [] for key in self.id_keys + self.example_question_keys}
        column_types = {
            'question_id': int,
            'sub_id': int,
            'question_text': str,
            'explanation': str,
            'domain': str,
            'type': str,
            'difficulty_level': int,
            'tags': str
        }
        return pd.DataFrame(data = d).astype(column_types)

    def _init_options_df(self):
        d = {key: [] for key in self.id_keys + self.options_keys}
        column_types = {
            'question_id': int,
            'sub_id': int,
            'option_id': int,
            'option_text': str,
            'correct_answer': bool
        }
        return pd.DataFrame(data = d).astype(column_types)

    def _init_instructions_df(self):
        d = {key: [] for key in self.instruction_keys}
        column_types = {
            'domain': str,
            'type': str,
            'difficulty_level': int,
            'tags': str,
            'instructions': str,
        }
        return pd.DataFrame(data = d).astype(column_types)

    def _init_validation_df(self):
        d = {'question_id': [], 'validation_status': []}
        column_types = {
            'question_id': int,
            'validation_status': int
        }
        return pd.DataFrame(data = d).astype(column_types)

    @staticmethod
    def _remove_gaps(sorted_list: list[int]):
        """
        Remove gaps from a sorted list while keeping duplicates.

        Args:
            sorted_list (list): List of sorted integers.

        Returns:
            list: List of integers without gaps.
        """

        if not sorted_list:
            return []
        
        # create a dictionary where the keys are the unique elements from the list
        # and the values are the new values without gaps
        unique_elements = sorted(set(sorted_list))
        element_to_new_value = {element: i for i, element in enumerate(unique_elements)}

        # replace the elements in the original list using the dictionary
        return [element_to_new_value[element] for element in sorted_list]

    def _get_dataframe_types(self, dataframe_type):
        if dataframe_type == 'questions':
            return 'questions', 'options'
        elif dataframe_type == 'example_questions':
            return 'example_questions', 'example_options'
        else:
            raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")
    
    def append(self, rows, dataframe_type = 'questions', ignore_options = False, suppress_warnings = True):
        """
        Appends rows to the dataset.

        Parameters:
        - rows: A list of dictionaries or a single dictionary representing the rows to be appended.
                If a list is provided, each element in the list should be a dictionary representing a row.
                If a dictionary is provided, it should represent a single row.
        
        Required keys:
        - 'question_id': The unique identifier for the question. Should start from 0. Append will automatically reindex the question_id.
        - 'sub_id': The unique identifier for the sub-question. Should start from 0.
        - 'question_text': The text of the question.
        - 'options': A list of dictionaries representing the options for the question.
            - 'option_id': The unique identifier for the option. Should start from 0.
            - 'option_text': The text of the option.
            - 'correct_answer': A boolean indicating whether the option is the correct answer.
        Optional keys:
        - 'explanation': The explanation for the question. Only required for example questions.
        - 'instructions': The instructions for the question. 

        Raises:
        - AssertionError: If a row is missing a required key or if an option within a row is missing a required key.
        """
        questions_type, options_type = self._get_dataframe_types(dataframe_type)

        if isinstance(rows, dict):
            rows = [rows]
        
        max_question_id = self[questions_type]['question_id'].max() if not self[questions_type].empty else -1
        max_question_id = int(max_question_id)

        df = pd.DataFrame(rows)
        # questions
        questions_df = df[self.id_keys + self.question_keys]
        

        sub_ids = questions_df['sub_id'].tolist()
        # Check if sub_ids are in order or go to zero
        for i in range(1, len(sub_ids)):
            if not (sub_ids[i] == 0 or sub_ids[i] == sub_ids[i - 1] + 1):
                raise AssertionError(f"sub_id values are not in order or do not reset to zero at index {i}. A subset: {sub_ids[i - 5: i+ 5]}")
        
        questions_df.loc[:, 'question_id'] = (questions_df['sub_id'] == 0).cumsum() + max_question_id

        nan_rows = questions_df[questions_df.isna().any(axis=1)]
        if not nan_rows.empty:
            raise ValueError(f"Rows {nan_rows} have nan values.")
        questions_df = questions_df.astype({'question_id': int, 'sub_id': int, 'question_text': str, 'domain': str, 'type': str, 'difficulty_level': int, 'tags': str})


        # options
        if not ignore_options:
            options_df = df[self.id_keys + ['options']]
            options_df = options_df.explode('options').reset_index(drop=True)
            options_df = pd.concat([options_df, pd.json_normalize(options_df['options'])], axis=1).drop(columns=['options'])
            # Remap question_ids using the same mapping computed for questions_df,
            # so options always align regardless of what question_ids the caller passed in.
            qid_map = dict(zip(
                df['question_id'].tolist(),
                questions_df['question_id'].tolist()
            ))
            options_df['question_id'] = options_df['question_id'].map(qid_map)

            # get all rows with nan values
            nan_rows = options_df[options_df.isna().any(axis=1)]
            # raise an error on all rows with nan values if there are any
            if not nan_rows.empty:
                raise ValueError(f"Rows {nan_rows['question_id'].tolist()} have nan values.")
            else:
                options_df = options_df.astype({'question_id': int, 'sub_id': int, 'option_id': int, 'correct_answer': bool})

        self[questions_type] = pd.concat([self[questions_type], questions_df], ignore_index=True)
        if not ignore_options:
            self[options_type] = pd.concat([self[options_type], options_df], ignore_index=True)


    def update(self, new_rows, dataframe_type='questions'):
        """
        new rows is of the form {'question_id': question_id, 'sub_id': sub_id, 'key': value}
        """
        if isinstance(new_rows, list):
            for row in new_rows:
                self.update(row, dataframe_type=dataframe_type)
        elif isinstance(new_rows, dict):
            self._validate_new_rows(new_rows)
            questions_type, _ = self._get_dataframe_types(dataframe_type)
            self._update_rows(new_rows, questions_type)
        else:
            raise ValueError("new_rows must be either a list of dictionaries or a single dictionary")

    def _validate_new_rows(self, new_rows):
        if 'question_id' not in new_rows:
            raise ValueError("Must provide a question_id.")
        if 'question_text' in new_rows and 'sub_id' not in new_rows:
            raise ValueError("Must provide a sub_id if updating question text.")
        if 'option_text' in new_rows and ('sub_id' not in new_rows or 'option_id' not in new_rows):
            raise ValueError("Must provide a sub_id and option_id if updating option text.")
        if 'instructions' in new_rows and any([key not in new_rows for key in self.metadata_keys]):
            raise ValueError("Must provide metadata keys if updating instructions.")

    def _update_rows(self, new_rows, questions_type):
        if 'sub_id' in new_rows:
            self._update_question_text(new_rows, questions_type)
            options_type = questions_type.replace('questions', 'options')
            self._update_option_text(new_rows, options_type)
            self._update_explanation(new_rows)
        if 'question_id' in new_rows:
            self._update_instructions(new_rows)
        
        self._update_metadata(new_rows, questions_type)

    def _update_question_text(self, new_rows, questions_type):
        if 'question_text' in new_rows:
            mask = (self[questions_type]['question_id'] == new_rows['question_id']) & \
                   (self[questions_type]['sub_id'] == new_rows['sub_id'])
            self[questions_type].loc[mask, 'question_text'] = new_rows['question_text']

    def _update_explanation(self, new_rows):
        if 'explanation' in new_rows:
            mask = (self['example_questions']['question_id'] == new_rows['question_id']) & \
                   (self['example_questions']['sub_id'] == new_rows['sub_id'])
            self['example_questions'].loc[mask, 'explanation'] = new_rows['explanation']
    
    # TODO: make this extensible for changes in metadata (i.e., use metadata_keys)
    def _update_instructions(self, new_rows):
        if 'instructions' in new_rows:
            # add instructions on metadata 
            condition = (self.instruction_set['domain'] == new_rows['domain']) & \
                        (self.instruction_set['type'] == new_rows['type']) & \
                        (self.instruction_set['difficulty_level'] == new_rows['difficulty_level'])
            self.instruction_set.loc[condition, 'instructions'] = new_rows['instructions']

    def _update_option_text(self, new_rows, options_type):
        if 'option_text' in new_rows:
            criteria = (self[options_type]['question_id'] == new_rows['question_id']) & \
                       (self[options_type]['sub_id'] == new_rows['sub_id']) & \
                       (self[options_type]['option_id'] == new_rows['option_id'])
            self[options_type].loc[criteria, 'option_text'] = new_rows['option_text']

    def _update_metadata(self, new_rows, questions_type):
        new_metadatas = [key for key in new_rows if key in self.metadata_keys]
        mask = self[questions_type]['question_id'] == new_rows['question_id']
        for new_metadata in new_metadatas:
            self[questions_type].loc[mask, new_metadata] = new_rows[new_metadata]

    def _get_dataframes(self, dataframe_type):
        if dataframe_type == 'questions':
            return self.questions, self.options
        elif dataframe_type == 'example_questions':
            return self.example_questions, self.example_options
        else:
            raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")

    def add_instructions(self, instructions, check_metadata_exists = True):
        """
        Add instructions to the dataset. The instructions will be added to the instruction_set DataFrame. Instructions are only relevant to the example questions.

        Parameters:
        - instructions: A list of dictionaries or a dictionary representing the instructions to be added.
        - check_metadata_exists: If True, the function will check if the metadata of the instructions exists in the dataset.
        
        Required keys:
        - 'domain': The domain of the instructions.
        - 'type': The type of the instructions.
        - 'difficulty_level': The difficulty level of the instructions.
        - 'tags': The tags of the instructions.
        - 'instructions': The text of the instructions.
        """
        if isinstance(instructions, list):
            for instruction in instructions:
                self.add_instructions(instruction, check_metadata_exists = check_metadata_exists)
        elif isinstance(instructions, dict):
            for key in self.instruction_keys:
                if key not in instructions:
                    raise ValueError(f"a row has no key {key}. Every row needs to have a {key} key.")
            if check_metadata_exists:
                metadata = {key: instructions[key] for key in self.metadata_keys}
                questions, options = self.query(metadata, dataframe_type='example_questions')
                if questions.empty:
                    raise ValueError("Metadata of the instructions does not exist in the example questions dataset.")
            self.instruction_set = pd.concat([self.instruction_set, pd.DataFrame.from_records([instructions])])
        else:
            raise ValueError("instructions must be either a list of dictionaries or a single dictionary")

    def query(self, command, dataframe_type = 'questions', return_type = 'tuple') -> Union[tuple, 'Dataset']:
        """
        Query the merged DataFrame.
        Parameters:
        - command: metadata to match (a dict, or a list of dicts for any of them; see metadata_mask), or a pandas query string.
        - dataframe_type: The type of DataFrame to query. Can be either 'questions' or 'example_questions'.
        - return_type: The type of the returned DataFrames. Can be either 'tuple' or 'dataset'.
        Returns:
        - A tuple containing the questions and options DataFrames.
        """
        if dataframe_type not in ['questions', 'example_questions']:
            raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")
        if return_type not in ['tuple', 'dataset']:
            raise ValueError("return_type must be either 'tuple' or 'dataset'")
        
        df = combine(*self._get_dataframes(dataframe_type))
        df = df.query(command) if isinstance(command, str) else df[metadata_mask(df, command)]
        
        questions, options = split(combined_df = df, dataframe_type=dataframe_type)
        if return_type == 'tuple':
            return questions, options
        else:
            if dataframe_type == 'questions':
                return from_tuple(questions, options, self.example_questions, self.example_options)
            else:
                return from_tuple(self.questions, self.options, questions, options)
    
    def iterrows(self, dataframe_type = 'questions', pbar = False):
        """
        Iterate over the rows of the questions and options DataFrames.
        If there are two rows with the same question_id, the function will yield both rows.
        Returns:
        - A generator that yields the rows of the questions and options DataFrames.
        """
        questions_type, options_type = self._get_dataframe_types(dataframe_type)
        grouped = self[questions_type].groupby('question_id')
        iterator = grouped if not pbar else tqdm(grouped, desc=f"Iterating over {dataframe_type} rows")
        for question_id, questions_df in iterator:
            if not self[options_type].empty:
                options_df = self[options_type][self[options_type]['question_id'] == question_id]
            else:
                options_df = pd.DataFrame()
            yield question_id, questions_df, options_df

    def loc(self, question_id, sub_id=None, dataframe_type='questions'):
        """
        Retrieve the question and options for a given question_id and/or sub_id.
        """
        questions_type, options_type = self._get_dataframe_types(dataframe_type)
        q_mask = self[questions_type]['question_id'] == question_id
        if sub_id is not None:
            q_mask &= self[questions_type]['sub_id'] == sub_id
        questions = self[questions_type][q_mask]

        o_mask = self[options_type]['question_id'] == question_id
        if sub_id is not None:
            o_mask &= self[options_type]['sub_id'] == sub_id
        options = self[options_type][o_mask]

        return questions, options


    def concat(self, datasets: list, on: str):
        """
        Concatenate multiple datasets.
        ## Args:
        **datasets**: (list) A list of Dataset objects to concatenate.
        **on**: (str) The dataframe_type to concatenate on.
        **suppress_warnings**: (bool) If True, suppresses warnings.
        ## Returns:
        A Dataset object containing the concatenated DataFrames.
        """
        if not isinstance(datasets, list):
            raise ValueError("datasets must be a list of Dataset objects.")
        
        
        questions_type, options_type = self._get_dataframe_types(on)

        self[questions_type] = pd.concat([self[questions_type]] + [dataset[questions_type] for dataset in datasets])
        self[options_type] = pd.concat([self[options_type]] + [dataset[options_type] for dataset in datasets])
        
        reindex(self, is_single_part_question=True, in_place=True)

    def sort_values(self, by, ascending = True, dataframe_type = 'questions'):
        questions_type, options_type = self._get_dataframe_types(dataframe_type)
        df = combine(self[questions_type], self[options_type])
        df = df.sort_values(by = by, ascending = ascending)
        questions, options = split(df)
        if dataframe_type == 'questions':
            return from_tuple(questions, options, self.example_questions, self.example_options)
        else:
            return from_tuple(self.questions, self.options, questions, options)

    


    

    def save(self, data_dir):
        """Write the dataset into data_dir as Parquet tables plus dataset.json (see autosteer.dataset_io)."""
        from autosteer import dataset_io
        dataset_io.save(pickle_safe_attributes(self.__dict__), data_dir)


    @staticmethod
    def load(file_path):
        """Load a Dataset from data_dir (dataset.json and Parquet tables, else the older dataset.pkl)."""
        from autosteer import dataset_io
        attributes = dataset_io.load(file_path)
        dataset = Dataset()
        dataset.__dict__.update(attributes)

        if not isinstance(getattr(dataset, 'element_name', None), str) or not dataset.element_name:
            dataset.element_name = os.path.basename(file_path)

        return dataset


        # return questions_df, questions_metadata_df, options_df, answers_df

    def to_display_json(self, dataframe_type = 'example_questions') -> list[dict]:
        """
        Convert the questions and options DataFrames to a JSON object.
        ## Args:
        **example_questions** (bool): If True, the function will convert the example questions and options DataFrames to a JSON object. If False, the function will convert the questions and options DataFrames to a JSON object.
        ## Returns:
        A JSON object in the display_json format. 
        """
        questions_type, options_type = self._get_dataframe_types(dataframe_type)

        options = self[options_type].sort_values(['question_id', 'sub_id', 'option_id'])   # a sorted copy; self is not changed

        groups = self[questions_type].groupby('question_id')
        json_list = []
        for question_id, group in groups:
            json_object = {}
            for sub_id, sub_group in group.groupby('sub_id'):
                json_object[f'question_{sub_id + 1}'] = sub_group['question_text'].values[0]
                opts = options[(options['question_id'] == question_id) & (options['sub_id'] == sub_id)]
                if opts.empty:
                    json_object[f'options_{sub_id + 1}'] = ['No options found']
                    json_object[f'answer_{sub_id + 1}'] = 'No answer found'
                else:
                    json_object[f'options_{sub_id + 1}'] = opts['option_text'].to_list()
                    correct = opts[opts['correct_answer'] == True]['option_text'].to_list()
                    json_object[f'answer_{sub_id + 1}'] = correct[0] if correct else 'No answer found'

            json_object['metadata'] = sub_group[self.metadata_keys].to_dict('records')[0]
            json_list.append(json_object)
        return json_list
    
    
    

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"'{key}' not found in Dataset")
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def is_empty(self):
        return self.questions.empty and self.example_questions.empty and self.options.empty and self.example_options.empty and self.instruction_set.empty

    
    def size(self, dataframe_type = 'questions'):
        return len(self[dataframe_type])
    
    def metadata(self, dataframe_type = 'questions'):
        return self[dataframe_type][self.metadata_keys].drop_duplicates().reset_index(drop = True)
    
    
    def __repr__(self):
        return f"Dataset(questions={self.questions}, options={self.options}, example_questions={self.example_questions}, example_options={self.example_options})"
    
    def __str__(self):
        return f"Dataset(questions={self.questions}, options={self.options}, example_questions={self.example_questions}, example_options={self.example_options})"
    
    def __iter__(self):
        return self.iterrows()
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __delitem__(self, key):
        del self.__dict__[key]

            


    

def reindex(dataset: Dataset, in_place: bool = False, is_single_part_question: bool = False) -> Dataset | None:
    """
    Reindex the (example) questions and (example) options DataFrames.
    Reindexing the DataFrames will remove gaps in the question_id column and start from 0.
    Parameters:
    - dataset: The Dataset object to be reindexed.
    - in_place: If True, modifies the dataset in place. If False, returns a new Dataset object.
    Returns:
    - A Dataset object containing the reindexed questions and options DataFrames if in_place is False.
    """
    def reindex_dataframe(df: pd.DataFrame):
        df = flatten_index(df)
        if df.empty:
            return df
        df['question_id'] = Dataset._remove_gaps(df['question_id'].astype(int).to_list())
        return df.reset_index(drop=True)

    instruction_set = dataset.instruction_set
    if not in_place:
        dataset = from_tuple(
            questions=dataset.questions.copy(),
            options=dataset.options.copy(),
            example_questions=dataset.example_questions.copy(),
            example_options=dataset.example_options.copy()
        )

    dataset.questions = reindex_dataframe(dataset.questions)
    dataset.options = reindex_dataframe(dataset.options)
    # display(dataset.example_questions)
    dataset.example_questions = reindex_dataframe(dataset.example_questions)
    dataset.example_options = reindex_dataframe(dataset.example_options)
    dataset.instruction_set = instruction_set

    if not in_place:
        return dataset
    


def from_records(records, dataframe_type = 'example_questions'):
    dataset = Dataset()
    dataset.append(records, dataframe_type=dataframe_type)
    return dataset


def flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with question_id/sub_id stored as normal columns.

    This is used by explicit migration/legacy paths. Runtime Dataset pickles are
    expected to already be flat.
    """
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
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    return df


def from_tuple(questions, options, example_questions = None, example_options = None):
    dataset = Dataset()
    dataset.questions = questions
    dataset.options = options
    
    if isinstance(example_questions, pd.DataFrame) ^ isinstance(example_options, pd.DataFrame):
        raise ValueError("Must provide both example_questions and example_options or neither")
    
    if isinstance(example_questions, pd.DataFrame) and isinstance(example_options, pd.DataFrame):
        dataset.example_questions = example_questions
        dataset.example_options = example_options
    return dataset


def get_metadata_df(df: pd.DataFrame):
    """
    Extract metadata from a row.
    Parameters:
    - df: A pandas DataFrame containing the rows.
    Returns:
    - A dataframe containing the metadata of the rows.
    """
    return df[metadata_keys].drop_duplicates().sort_values('difficulty_level')


def combine(questions, options, suppress_warnings = True):
    """
    Merge the questions and options DataFrames.
    ## Args:
    **questions**: (pd.DataFrame) A DataFrame containing the questions.
    **options**: (pd.DataFrame) A DataFrame containing the options.
    **suppress_warnings**: (bool) If True, suppresses warnings.
    ## Returns:
    A DataFrame containing the merged questions and options DataFrames.
    """
    if options.empty:
        # add an empty options DataFrame to the questions DataFrame
        combined_df = questions.copy()
        combined_df['option_id'] = [0] * len(questions)
        combined_df['option_text'] = ['No options'] * len(questions)
        combined_df['correct_answer'] = [True] * len(questions)

        return combined_df
    try:
        merged_df = pd.merge(left = questions, right = options, how='inner', on=['question_id', 'sub_id'])
    except KeyError as e:
        if not suppress_warnings:
            print(f"KeyError: {e}. Attempting to merge on the multi-index ['question_id', 'sub_id'].")
        try:
            # Merge on the multi-index
            merged_df = questions.join(options, how='inner')
        except KeyError as e:
            raise KeyError(f"{e}. Ensure that both DataFrames contain the columns 'question_id' and 'sub_id' or that they are set as the index.")
    return merged_df

def split(combined_df: pd.DataFrame = None, dataframe_type: str = 'questions') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the merged DataFrame into questions and options DataFrames.
    ## Args:
    **combined_df**: (pd.DataFrame) A DataFrame containing the combined questions and options DataFrames.
    ## Returns:
    A tuple containing the questions and options DataFrames
    """

    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df must be a pandas DataFrame.")

    def split_dataframes(combined_df, question_keys, options_keys):
        questions = combined_df[id_keys + question_keys].drop_duplicates()
        try:
            options = combined_df[id_keys + options_keys]
        except KeyError:
            options = pd.DataFrame()
        return questions, options
    
    if dataframe_type not in ['questions', 'example_questions']:
        raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")
    
    if dataframe_type == 'questions':
        curr_question_keys = question_keys
    else:
        curr_question_keys = example_question_keys
    
    questions, options = split_dataframes(combined_df, curr_question_keys, options_keys)
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


def display_to_dataset_json(display_json, ignore_options = False):
    """
    Convert the display JSON to dataset JSON.

    ## Args:
    **display_json** (dict): The display JSON. The format of the display JSON is:
    ```json
    [
        {
            "question_1": "What is the capital of France?",
            "options_1": [
                "Paris",
                "London",
            ],
            "answer_1": "Paris",
            "metadata": {
                "domain": "geography",
                "type": "multiple_choice",
                "difficulty_level": 1,
                "tags": "capital cities;Europe",
            },
        },
        ...
    ]
    ```
    Optionally, the display JSON can contain instructions as a key-value pair

    ## Returns:
    The dataset JSON (dict). The format of the dataset JSON is:
    ```json
    [
        {
            "question_id": 0,
            "sub_id": 0,
            "question_text": "What is the capital of France?",
            "domain": "geography",
            "type": "multiple_choice",
            "difficulty_level": 1,
            "tags": "capital cities;Europe",
            "options": [
                {
                    "option_id": 0,
                    "option_text": "Paris",
                    "correct_answer": true
                },
                {
                    "option_id": 1,
                    "option_text": "London",
                    "correct_answer": false
                },
            ],
        },
        ...
    ]
    ```
    """
    if isinstance(display_json, dict):
        display_json = [display_json]
    dataset_json = []
    for question_id, question in enumerate(display_json):
        # get the number of sub-questions
        sub_question_keys = sorted([key for key in question.keys() if key.startswith('question_')])
        for sub_id, sub_question_key in enumerate(sub_question_keys):
            options = []
            if ignore_options:
                options.append({
                    "option_id": 0,
                    "option_text": "No options found",
                    "correct_answer": False
                })
            else:
                for option_id, option in enumerate(question[f'options_{sub_id+1}']):
                    option_text = option
                    correct_answer = option == question[f'answer_{sub_id+1}']
                    options.append({
                        "option_id": option_id,
                        "option_text": option_text,
                        "correct_answer": correct_answer
                    })
            json_object = {
                "question_id": question_id,
                "sub_id": sub_id,
                "question_text": question[sub_question_key],
                "domain": question['metadata']['domain'],
                "type": question['metadata']['type'],
                "difficulty_level": question['metadata']['difficulty_level'],
                "tags": question['metadata']['tags'] ,
                "options": options
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
