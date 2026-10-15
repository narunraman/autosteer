# src/steer_generation/steer_dataset.py

# Built-in packages
from itertools import combinations
import os
import re
from string import ascii_uppercase
import textwrap
from typing import Optional, Union
from termcolor import colored
# External packages
import pandas as pd
from tqdm import tqdm
import dill
import pickle
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

        if 'instructions' in rows:
            self.add_instructions({k: v for k, v in rows.items() if k in ['instructions'] + metadata_keys}, check_metadata_exists = False)


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
                command = build_command(metadata)
                questions, options = self.query(command, dataframe_type='example_questions')
                if questions.empty:
                    raise ValueError("Metadata of the instructions does not exist in the example questions dataset.")
            self.instruction_set = pd.concat([self.instruction_set, pd.DataFrame.from_records([instructions])])
        else:
            raise ValueError("instructions must be either a list of dictionaries or a single dictionary")

    def query(self, command, dataframe_type = 'questions', return_type = 'tuple') -> Union[tuple, 'Dataset']:
        """
        Query the merged DataFrame.
        Parameters:
        - command: A string representing the query command.
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
        df = df.query(command)
        
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
        
        # if not all(isinstance(dataset, Dataset) for dataset in datasets):
        #     raise ValueError("datasets must contain only Dataset objects.")
        
        # if not all(dataset.is_empty() for dataset in datasets):
        #     raise ValueError("datasets must not be empty.")
        
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

    def _build_mcq(self, question_id, dataframe_type):
        selected_question, selected_options = self.loc(question_id = question_id, dataframe_type = dataframe_type)
        # Metadata
        metadata = selected_question[self.metadata_keys].to_dict('records')[0]
        # Questions
        questions = selected_question['question_text'].to_list()
        # Options
        if selected_options.empty:
            options_list = [['No options found']]*len(questions)
            # Correct answers
            correct_answers = ['No correct answer found'] * len(questions)
        else:
            options_list = [
                selected_options[selected_options['sub_id'] == sub_id]['option_text'].to_list()
                for sub_id in selected_options['sub_id'].unique()
            ]
            # Correct answers
            correct_answers = selected_options.query("correct_answer == True")['option_text'].to_list()
        if not correct_answers:
            correct_answers = ['No correct answer found'] * len(questions)
        # Explanations
        if dataframe_type == 'example_questions':
            try:
                explanations = selected_question['explanation'].to_list()
            except KeyError:
                explanations = ['No explanation found'] * len(questions)
        else:
            explanations = None
        return {
            'question': questions,
            'options_list': options_list,
            'correct_answer': correct_answers,
            'explanation': explanations,
            'metadata': metadata
        }
    
    def _print_mcq(self, question_id, dataframe_type = 'questions'):
        mcq_dict = self._build_mcq(question_id, dataframe_type = dataframe_type)
        print("Metadata:\n  " + "\n  ".join([f"{key}: {value}" for key, value in mcq_dict['metadata'].items()]))
        global_option_index = 0
        for i, question in enumerate(mcq_dict['question']):
            print(f"\n------------------------------ Question {question_id}-{i}: ------------------------------")
            question = re.sub(r'\{(.*?)\}', lambda m: colored(m.group(0), 'light_blue'), question)
            # Split the question into lines, wrap each line, and join them back together
            wrapped_question = "\n".join(textwrap.fill(line, width=80) for line in question.splitlines())
            print(wrapped_question)
            for option in mcq_dict['options_list'][i]:
                if option == mcq_dict['correct_answer'][i]:
                    print(f"  {ascii_uppercase[global_option_index]}. {colored(option, 'green')}")
                else:
                    print(f"  {ascii_uppercase[global_option_index]}. {option}")
                global_option_index += 1

            if dataframe_type == 'example_questions':
                print("\nExplanation:")
                if not isinstance(mcq_dict['explanation'][i], str):
                    print("No explanation generated yet.")
                else:
                    print(mcq_dict['explanation'][i])

    def _flip_question_id(self, question_id, dataframe_type):
        questions_type, _ = self._get_dataframe_types(dataframe_type)
        if question_id < 0:
            question_id = self[questions_type]['question_id'].max() + question_id + 1
        return question_id

    def _check_question_id(self, question_id, dataframe_type):
        questions_type, _ = self._get_dataframe_types(dataframe_type)
        if question_id not in self[questions_type]['question_id'].values:
            raise IndexError(f"question_id not found in {questions_type.replace('_', ' ')} DataFrame.")
        return True


    def print_question(self, question_id, dataframe_type = 'questions'):
        """
        Print the question and options for a given question_id.
        """
        question_id = self._flip_question_id(question_id, dataframe_type = dataframe_type)
        if self._check_question_id(question_id, dataframe_type = dataframe_type):
            self._print_mcq(question_id, dataframe_type = dataframe_type)

    def get_metadata(self, question_id):
        return self.questions[self.questions['question_id'] == question_id][self.metadata_keys].iloc[0].to_dict()
    
    # def get_example_templates(self, return_type = 'tuple'):
    #     questions_df = self.questions[self.questions['explanation'] != False]
    #     options_df = self.options[self.options['question_id'].isin(questions_df['question_id'])]
    #     if return_type == 'tuple':
    #         return questions_df, options_df
    #     elif return_type == 'dataset':
    #         return from_tuple(questions_df, options_df)
    #     else:
    #         raise ValueError("return_type must be either 'tuple' or 'dataset'")
    

    def sample(self, value, suppress_warnings = False):
        """
        Sample a fraction of the dataset.
        Parameters:
        - value: The fraction of the dataset to sample.
        Returns:
        - A tuple containing the sampled questions and options DataFrames.

        Raises:
        - ValueError: If the value is less than or equal to 0.

        """
        
        grouped_questions = self.questions.groupby(["domain", "type", "difficulty_level"])

        if value > 0 and value <= 1:
            sampled_questions = grouped_questions.apply(lambda x: x.sample(frac=value)).reset_index(drop=True)
        elif value > 1:
            if not suppress_warnings:
                small_groups = []
                for i, group in grouped_questions:
                    if len(group) < value:
                        small_groups.append(i)
                if small_groups:
                    print(f"Warning: Groups {small_groups} have less than {value} rows to sample from. Sampling all rows from these groups.")
            sampled_questions = grouped_questions.apply(lambda x: x.sample(n=min(len(x), value))).reset_index(drop=True)
        else:
            raise ValueError("Value must take a value greater than 0!")

        sampled_options = self.options[self.options['question_id'].isin(sampled_questions['question_id'])]

        return from_tuple(sampled_questions, sampled_options, self.example_questions, self.example_options)
    
    def set_validation_sample(self, sample_frac: int | float = 0.1):
        """
        Set a class attribute validation_ids of the question_ids to be used for validation. validation_ids should be a list of question_ids.
        Append to this list if there are not sample_frac number of question_ids in validation_ids for each metadata group.

        """
        # get the questions with ids that are in validation_df
        validation_questions = self.questions[self.questions['question_id'].isin(self.validation_df['question_id'])]
        validation_ids = []

        grouped_validation_questions = validation_questions.groupby(["domain", "type", "difficulty_level"])
        grouped_questions = self.questions.groupby(["domain", "type", "difficulty_level"])
        for group_name, group in grouped_questions:
            if sample_frac > 1:
                if not isinstance(sample_frac, int):
                    raise ValueError("sample_frac must be an integer if it is greater than 1")
                threshold = sample_frac
            elif sample_frac > 0 and sample_frac <= 1:
                threshold = int(len(group) * sample_frac)
            else:
                raise ValueError("sample_frac must be greater than 0")

            if self.validation_df.empty or grouped_validation_questions.get_group(group_name).size < threshold:
                sample_num = threshold - grouped_validation_questions.get_group(group_name).size if not self.validation_df.empty else threshold
                sampled_group = grouped_questions.get_group(group_name).sample(n = int(sample_num))
                validation_ids += sampled_group['question_id'].to_list()
        new_validation_df = pd.DataFrame({'question_id': validation_ids, 'validation_status': [2]*len(validation_ids)})
        if self.validation_df.empty:
            self.validation_df = new_validation_df
        else:
            self.validation_df = pd.concat([self.validation_df, new_validation_df], ignore_index=True).astype({'validation_status': int})


    def save(self, data_dir):
        """Write the dataset into data_dir as Parquet tables plus dataset.json (see autosteer.dataset_io)."""
        from autosteer import dataset_io
        dataset_io.save(pickle_safe_attributes(self.__dict__), data_dir)

    # The old name: datasets used to be pickled. It now writes the Parquet format.
    to_pickle = save

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

    from_pickle = load  # the old name; reads either format

    def to_old_pickles(self, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # def convert_to_old(data_dir):
        # with open(os.path.join(data_dir, 'dataset.pkl'), 'rb') as f:
        #     dataset = dill.load(f)
        # dataset = from_pickle(data_dir)
        # Example Questions
        example_questions = self.example_questions.copy()
        example_questions['question_id'] = example_questions['question_id'].astype(str) + '_' + example_questions['sub_id'].astype(str)
        example_questions.drop(columns='sub_id', inplace=True)

        example_questions_metadata = example_questions[['question_id']+metadata_keys]

        # Questions
        questions = self.questions.copy()
        # add the maximum question_id from example_questions to the question_id of questions
        max_question_id = self.example_questions['question_id'].max()
        questions['question_id'] += int(max_question_id + 1)
        questions['question_id'] = questions['question_id'].astype(str) + '_' + questions['sub_id'].astype(str)
        questions.drop(columns = 'sub_id', inplace = True)
        
        questions_metadata = questions[['question_id']+metadata_keys]

        # Combine questions
        questions_df = pd.concat([example_questions, questions]).reset_index(drop = True)
        # convert all nans in 'explanation' column to False
        questions_df['explanation'] = questions_df['explanation'].fillna(False)
        questions_df.drop(columns = metadata_keys, inplace=True)

        # Combine questions metadata
        questions_metadata_df = pd.concat([example_questions_metadata, questions_metadata]).reset_index(drop = True)

        # Options
        example_options = self.example_options.copy()
        example_options['question_id'] = example_options['question_id'].astype(str) + '_' + example_options['sub_id'].astype(str)
        example_options.drop(columns = 'sub_id', inplace = True)
        # print(self.example_options)

        options = self.options.copy()
        # add the maximum question_id from example_questions to the question_id of questions
        options['question_id'] += int(max_question_id + 1)
        options['question_id'] = options['question_id'].astype(str) + '_' + options['sub_id'].astype(str)
        options.drop(columns = 'sub_id', inplace = True)

        # Combine options
        options_df = pd.concat([example_options, options]).reset_index(drop = True)

        # Answers
        answers_df = options_df[['question_id', 'option_id', 'correct_answer']].copy()
        options_df.drop(columns = 'correct_answer', inplace = True)

        with open(os.path.join(data_dir, 'questions.pkl'), 'wb') as f:
            pickle.dump(questions_df, f)
        with open(os.path.join(data_dir, 'questions_metadata.pkl'), 'wb') as f:
            pickle.dump(questions_metadata_df, f)
        with open(os.path.join(data_dir, 'options.pkl'), 'wb') as f:
            pickle.dump(options_df, f)
        with open(os.path.join(data_dir, 'answers.pkl'), 'wb') as f:
            pickle.dump(answers_df, f)
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

        self[options_type] = self[options_type].sort_values(['question_id', 'sub_id', 'option_id'])

        groups = self[questions_type].groupby('question_id')
        json_list = []
        for question_id, group in groups:
            json_object = {}
            for sub_id, sub_group in group.groupby('sub_id'):
                json_object[f'question_{sub_id + 1}'] = sub_group['question_text'].values[0]
                opts = self[options_type][
                    (self[options_type]['question_id'] == question_id) &
                    (self[options_type]['sub_id'] == sub_id)
                ]
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
    
    def to_dataset_json(self, dataframe_type = 'questions') -> list[dict]:
        """
        Convert the questions and options DataFrames to a JSON object.
        ## Args:
        **example_questions** (bool): If True, the function will convert the example questions and options DataFrames to a JSON object. If False, the function will convert the questions and options DataFrames to a JSON object.
        ## Returns:
        A JSON object in the dataset_json format.
        """
        questions_type, options_type = self._get_dataframe_types(dataframe_type)

        self[options_type] = self[options_type].sort_values(['question_id', 'sub_id', 'option_id'])

        groups = self[questions_type].groupby('question_id')
        json_list = []
        for question_id, group in groups:
            json_object = {}
            for sub_id, sub_group in group.groupby('sub_id'):
                json_object['question_id'] = question_id
                json_object['sub_id'] = sub_id
                json_object['question_text'] = sub_group['question_text'].values[0]
                opts = self[options_type][
                    (self[options_type]['question_id'] == question_id) &
                    (self[options_type]['sub_id'] == sub_id)
                ]
                if opts.empty:
                    json_object['options'] = [{'option_text': 'No options found', 'correct_answer': False}]
                else:
                    json_object['options'] = opts[['option_id', 'option_text', 'correct_answer']].to_dict(orient='records')
            json_object.update(**sub_group[self.metadata_keys].to_dict('records')[0])
            if dataframe_type == 'example_questions':
                command = build_command({k: v for k, v in json_object.items() if k in self.metadata_keys})
                instructions = self.instruction_set.query(command)['instructions']
                json_object['instructions'] = instructions.values[0] if not instructions.empty else 'No instructions found.'
            json_list.append(json_object)
        return json_list
    
    def to_json(self, dataframe_type = 'questions', object_format = 'dataset_json'):
        if object_format == 'dataset_json':
            return self.to_dataset_json(dataframe_type = dataframe_type)
        elif object_format == 'display_json':
            return self.to_display_json(dataframe_type = dataframe_type)
        else:
            raise ValueError("object_format must be either 'dataset_json' or 'display_json'")
    
    def to_mcq_string(self, question_id, dataframe_type = 'questions', format = 'markdown'):
        mcq_json = self._build_mcq(question_id, dataframe_type = dataframe_type)
        mcq_string = ""
        for i, question in enumerate(mcq_json['question']):
            mcq_string += f"\nQuestion {question_id}-{i}:\n"
            mcq_string += question
            mcq_string += "\n"
            for j, option in enumerate(mcq_json['options_list'][i]):
                if option == mcq_json['correct_answer'][i]:
                    mcq_string += f":green[{ascii_uppercase[j]}. {option}]\n"
                else:
                    mcq_string += f"{ascii_uppercase[j]}. {option}\n"
            if mcq_json['explanation']:
                mcq_string += f"\nExplanation:\n{mcq_json['explanation'][i]}\n"
        if format == 'markdown':
            mcq_string = mcq_string.replace('\n', '  \n')
            mcq_string = re.sub(r'(?<!\\)\$', r'\$', mcq_string)
            return mcq_string
        return mcq_string

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"'{key}' not found in Dataset")
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def is_empty(self):
        return self.questions.empty and self.example_questions.empty and self.options.empty and self.example_options.empty and self.instruction_set.empty

    def contains(self, json_object, dataframe_type = 'questions', object_format = 'dataset_json'):
        """
        Check if the dataset contains a JSON object.
        Parameters:
        - json_object: The JSON object to check.
        - dataframe_type: The type of DataFrame to check. Can be either 'questions' or 'example_questions'.
        - object_format: The format of the JSON object. Can be either 'dataset_json' or 'mcq'.
        Returns:
        - True if the dataset contains the JSON object. False otherwise.
        """
        if dataframe_type not in ['questions', 'example_questions']:
            raise ValueError("dataframe_type must be either 'questions' or 'example_questions'")
        if object_format not in ['dataset_json', 'display_json']:
            raise ValueError("object_format must be either 'dataset_json' or 'display_json'")
        
        if object_format == 'dataset_json':
            metadata_command = build_command({k: v for k, v in json_object.items() if k in self.metadata_keys})
            formatted_question_text = json_object['question_text'].replace("'", "''")
            question_command = f"question_text == '{formatted_question_text}'"


            formatted_option_texts = [option['option_text'].replace("'", "''") for option in json_object['options']]
            commands = [f"option_text == '{option}' " for option in formatted_option_texts]

            commands = [f"{metadata_command} and {question_command} and {command}" for command in commands]

            command = "(" + ") and (".join(commands) + ")"

            # command += f" and question_text == '{json_object['question_text']}'"
            # option texts
            # command += " and " + " and ".join([f"option_text == '{option['option_text']}' " for option in json_object['options']])
            
            questions, _ = self.query(command, dataframe_type = dataframe_type, return_type = 'tuple')
            return not questions.empty
            
        elif object_format == 'display_json':
            raise NotImplementedError("object_format 'display_json' is not implemented yet.")
            command = build_command({k: v for k, v in json_object['metadata'].items()})
            command += f" and question_text == '{json_object['question_1']}'"
            questions, _ = self.query(command, dataframe_type = dataframe_type, return_type = 'tuple')
            return not questions.empty
    
    def size(self, dataframe_type = 'questions'):
        return len(self[dataframe_type])
    
    def metadata(self, dataframe_type = 'questions'):
        return self[dataframe_type][self.metadata_keys].drop_duplicates().reset_index(drop = True)
    
    def __len__(self):
        return {'questions': self.size('questions'), 'example_questions': self.size('example_questions')}
    
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
        # if is_single_part_question:
        #     df['question_id'] = range(len(df))
        # else:
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
    

# def rows_to_mcq(questions_df, options_df):
#     """
#     Convert the questions and options DataFrames to a multiple-choice question.
#     Parameters:
#     - questions_df: A DataFrame containing the questions.
#     - options_df: A DataFrame containing the options.
#     Returns:
#     - a list of strings representing the multiple-choice questions.
#     """
#     mcqs = []
#     for question_id, group in questions_df.groupby('question_id'):
#         # Questions
#         questions = group['question_text'].to_list()
#         # Options
#         options_list = [options_df.loc[(question_id, sub_id), :].sort_values('option_id')['option_text'].to_list() for sub_id in options_df.index.get_level_values('sub_id').unique()]
#         display(options_df.loc[(question_id), :])
#         # Correct answers
#         correct_answers = options_df.query("correct_answer == True")['option_text'].to_list()
#         if not correct_answers:
#             correct_answers = ['No correct answer found'] * len(questions)
        
#         mcq_text = f"Question:\n"
#         global_option_index = 0
#         for i, question in enumerate(questions):
#             mcq_text += question + "\n"
#             for option in options_list[i]:
#                 mcq_text += f"{ascii_uppercase[global_option_index]}. {option}\n"
#                 global_option_index += 1
#             mcq_text += f"Correct answer: {correct_answers[i]}\n\n"

#         mcqs.append(mcq_text)
#     return mcqs



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

# TODO: make this from_dill
def from_dill(data_dir, get_dataset = True, is_single_part_question = True):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} does not exist.")
    
    if get_dataset:
        if os.path.exists(os.path.join(data_dir, 'dataset.dill')):
            with open(os.path.join(data_dir, 'dataset.dill'), 'rb') as f:
                dataset = Dataset()
                attributes = dill.load(f)
                dataset.__dict__.update(attributes)
        elif os.path.exists(os.path.join(data_dir, 'questions_metadata.pkl')):
            dataset = from_old_pickles(data_dir, is_single_part_question = is_single_part_question)
        else:
            raise FileNotFoundError(f"{data_dir} does not contain a dataset.pkl file or a questions_metadata.pkl file.")
    else:
        with open(data_dir, 'rb') as f:
            dataset = dill.load(f)

    if not hasattr(dataset, 'element_name') or dataset.element_name is None or dataset.element_name != os.path.basename(data_dir):
        dataset.element_name = os.path.basename(data_dir)
    dataset.set_validation_sample()
    
    dataset = reindex(dataset, is_single_part_question=is_single_part_question)
    # for dataframe_type in ['questions', 'example_questions']:
    #     dataset[dataframe_type] = reindex(dataset[dataframe_type])
    # dataset.questions.index = dataset.questions.index.set_levels(dataset.questions.index.levels[0].astype(int), level = 0)
    return dataset

def from_old_pickles(data_dir, overwrite = False, is_single_part_question = False):
    questions = pd.read_pickle(os.path.join(data_dir, 'questions.pkl'))
    questions_metadata = pd.read_pickle(os.path.join(data_dir, 'questions_metadata.pkl'))
    # merge questions and questions_metadata
    questions = pd.merge(questions, questions_metadata, on = 'question_id')
    questions['sub_id'] = questions['question_id'].str.split('_').str[-1].astype(int)
    
    if questions['sub_id'].sum() == 0:
        questions = questions.reset_index()
        questions['question_id'] = questions['index']
        questions = questions.drop('index', axis = 1)
    # if is_single_part_question:
    #     questions = questions.reset_index()
    #     questions['question_id'] = questions['index']
    #     questions = questions.drop('index', axis = 1)
    else:
        questions['question_id'] = questions['question_id'].str.split('_').str[0].astype(int)
    if 'difficulty_level' in questions.columns:
        questions['difficulty_level'] = questions['difficulty_level'].astype(int)
    
    example_questions = questions.query("explanation != False").copy()
    questions = questions.query("explanation == False").copy()
    
    options = pd.read_pickle(os.path.join(data_dir, 'options.pkl'))
    answers = pd.read_pickle(os.path.join(data_dir, 'answers.pkl'))
    options = pd.merge(options, answers, on = ['question_id', 'option_id'])
    options['sub_id'] = options['question_id'].str.split('_').str[-1].astype(int)
    options['question_id'] = options['question_id'].str.split('_').str[0].astype(int)

    # if correct is a column in the options, rename it to correct_answer
    if 'correct' in options.columns and 'correct_answer' not in options.columns:
        options.rename(columns = {'correct': 'correct_answer'}, inplace = True)
    elif 'correct' in options.columns and 'correct_answer' in options.columns:
        options.drop(columns = 'correct', inplace = True)
    
    
    # Ensure the correct_answer column is of the correct dtype
    # if 'correct_answer' in options.columns:
    #     options['correct_answer'] = options['correct_answer'].astype(str)
    

    example_options = options[options['question_id'].isin(example_questions['question_id'])].copy()
    options = options[options['question_id'].isin(questions['question_id'])].copy()

    dataset = from_tuple(questions, options, example_questions, example_options)
    dataset.element_name = os.path.basename(data_dir)
    # display(dataset.questions)

    if os.path.exists(os.path.join(data_dir, 'instruction_set.pkl')):
        instruction_set = pd.read_pickle(os.path.join(data_dir, 'instruction_set.pkl'))
        # set the data types of the instruction_set DataFrame
        instruction_set['difficulty_level'] = instruction_set['difficulty_level'].astype(int)
        dataset.instruction_set = instruction_set

    # if no tags column in the questions_metadata, add it
    if 'tags' not in dataset.questions.columns:
        dataset.questions['tags'] = None
    if 'tags' not in dataset.example_questions.columns:
        dataset.example_questions['tags'] = None

    dataset = reindex(dataset)
    if overwrite:
        # delete the old files
        os.remove(os.path.join(data_dir, 'questions_metadata.pkl'))
        os.remove(os.path.join(data_dir, 'answers.pkl'))
        os.remove(os.path.join(data_dir, 'questions.pkl'))
        os.remove(os.path.join(data_dir, 'options.pkl'))
        # save the new files
        dataset.to_pickle(data_dir)
    return dataset

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

def get_metadata(rows, suppress_warnings = False):
    """
    Extract metadata from a row.
    ## Args:
    rows: Either a single row or a pandas DataFrame containing the rows.
    ## Returns:
    A dictionary containing the metadata of the row.
    ### Warning:
    If the rows DataFrame contains multiple rows, the function will only return the metadata of the first row.
    """

    # drop duplicates on metadata keys
    rows = rows.drop_duplicates(subset = metadata_keys)

    if isinstance(rows, pd.DataFrame):
        if len(rows) > 1 and not suppress_warnings:
            print("Warning: The function will only return the metadata of the first row.")
        row = rows.iloc[0]
    else:
        row = rows

    return {key: row[key] for key in ['domain', 'type', 'difficulty_level', 'tags']}

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
        except KeyError as e:
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

    

def get_display_json(questions_df: pd.DataFrame, options_df: pd.DataFrame, instruction_set: pd.DataFrame = None):
    """
    Convert the questions and options DataFrames to a JSON object.
    ## Args:
    **questions_df**: (pd.DataFrame) A DataFrame containing the questions.
    **options_df**: (pd.DataFrame) A DataFrame containing the options.
    **instruction_set**: (pd.DataFrame) A DataFrame containing the instructions.
    ## Returns:
    A JSON object in the display_json format.
    """
    display_json = {}
    for question_id, group in questions_df.groupby('question_id'):
        # Questions
        questions = group['question_text'].to_list()
        # Options
        options_list = [
            options_df[
                (options_df['question_id'] == question_id) &
                (options_df['sub_id'] == sub_id)
            ].sort_values('option_id')['option_text'].to_list()
            for sub_id in group['sub_id'].unique()
        ]
        # Correct answers
        correct_answers = options_df[
            (options_df['question_id'] == question_id) &
            (options_df['correct_answer'] == True)
        ]['option_text'].to_list()
        if not correct_answers:
            correct_answers = ['No correct answer found'] * len(questions)
        
        for i, question in enumerate(questions):
            display_json[f"question_{i+1}"] = question
            display_json[f"options_{i+1}"] = options_list[i]
            display_json[f"answer_{i+1}"] = correct_answers[i]
        
        display_json["metadata"] = get_metadata(group)

        if instruction_set is not None:
            metadata = display_json["metadata"]
            instructions = instruction_set.query(" & ".join([f"{key} ==@ value" for key, value in metadata.items()]))['instructions'].to_list()
            display_json['instructions'] = instructions[0] if instructions else None
    return display_json


def build_command(query_json: list[dict] | dict) -> str:
    """
    Build a query command from a JSON object.
    ## Args:
    **query_json**: A JSON object representing the query command.
    ## Returns:
    A string representing the query command.
    """
    if isinstance(query_json, list):
        return "(" + ") or (".join([build_command(query) for query in query_json]) + ")"
    elif isinstance(query_json, dict):
        # if key is difficulty_level do not wrap the value in quotes
        command = []
        for key, value in query_json.items():
            if key == 'difficulty_level':
                command.append(f"{key} == {value}")
            else:
                command.append(f"{key} == '{value}'")
        return " and ".join(command)
    else:
        raise ValueError("query_json must be either a list or a dictionary")


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

def get_json_format(json_object):
    """
    Get the format of a JSON object. This is a necessary check not a sufficient one.
    ## Args:
    **json_object** (dict): The JSON object.
    ## Returns:
    The format of the JSON object. Can be either 'display_json' or 'dataset_json'.
    """
    if 'question_1' in json_object:
        return 'display_json'
    elif 'question_id' in json_object:
        return 'dataset_json'
    else:
        raise ValueError("Invalid JSON object format.")
    

def get_instructions(instruction_set: pd.DataFrame, metadata: dict, get_closest_match: bool = True) -> str:
    """
    Get the instructions for a given metadata. If closest_match is True, the function will find the closest match.
    ## Args:
    **instruction_set** (pd.DataFrame): The instruction set DataFrame.
    **metadata** (dict): The metadata.
    **closest_match** (bool): If True, the function will find the closest match. 
    
    ## Returns:
    The instructions.

    """
    def closest_match(metadata: dict, n: int) -> Optional[list[dict]]:
        """
        Find the closest match by excluding n key-value pairs from the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata dictionary.
            n (int): The number of key-value pairs to exclude.

        Returns:
            Optional[List[Dict[str, Any]]]: The list of closest matching instructions, or None if no match is found.
        """
        instructions = []
        # The priority order
        priority_order = ['type', 'difficulty_level', 'domain', 'tags']
        # Sort keys based on priority order
        keys = sorted(metadata.keys(), key=lambda k: priority_order.index(k) if k in priority_order else len(priority_order))
        for keys_to_exclude in combinations(keys, n):
            partial_metadata = {k: v for k, v in metadata.items() if k not in keys_to_exclude}
            command = build_command(partial_metadata)
            matched_instructions = instruction_set.query(command)
            if not matched_instructions.empty:
                instructions.append(matched_instructions)
        # if instructions list is empty, return None
        if not instructions:
            return None
        return pd.concat(instructions)
    
    command = build_command(metadata)
    instructions = instruction_set.query(command)
    if instructions.empty and get_closest_match:
        # find the closest match
        n = len(metadata) - 1
        while instructions is None or instructions.empty or instructions['instructions'].isna().all():
            if n == 0:
                break
            instructions = closest_match(metadata, n)
            n -= 1
        
    return instructions
