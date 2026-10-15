# src/steer_generation/dataset_filler.py

# Built-in packages
import json
import os
from typing import Callable, Union
from tqdm import tqdm
import re
import numpy as np 
# External packages
import pandas as pd
import dill
import textwrap
# Local packages
from . import steer_dataset as sd
from .llm_client import LanguageModelClient
from autosteer.filler_tools import function_registry
from . import directory_utils as du
from .constants import DOMAINS, TAGS


def construct_fill_text(questions_df: pd.DataFrame, options_list: list[list[str]]):
    # Convert DataFrame to JSON string
    questions_str = json.dumps(questions_df['question_text'].to_list())
    
    # Convert list of lists to JSON string
    options_str = json.dumps(options_list)
    
    # Concatenate both strings with a delimiter
    fill_text = questions_str + '||DELIMITER||' + options_str
    
    return fill_text

def split_fill_text(fill_text):
    # Format fill_text so that it can be read as a JSON string (i.e., \n -> \\n)
    fill_text = re.sub(r'(?<!\\)\n', '\\\\n', fill_text)

    # Split the concatenated string using the delimiter
    questions_str, options_str = fill_text.split('||DELIMITER||')
    
    # Convert JSON string back to list
    question_texts = json.loads(questions_str)
    
    # Convert JSON string back to list of lists
    option_list = json.loads(options_str)
    
    return question_texts, option_list


def validate_fill_values(options_list, correct_answer):
    """
    Validate fill values for options and correct answers.

    Args:
        options_list (list): List of lists of options.
        correct_answer (list): List of lists of correct answers.

    Returns:
        bool: True if fill values are valid, raise ValueError otherwise.
    """
    if not isinstance(options_list, list):
        raise ValueError("Options list must be a list of lists of options")
    if not isinstance(options_list[0], list):
        raise ValueError("Options list must be a list of lists of options")
    if not isinstance(correct_answer, list):
        raise ValueError("Correct answer must be a list containing correct answers or a list of one-hot vectors")
    if len(options_list) != len(correct_answer):
        raise ValueError("Length of correct answer must be equal to the length of options list")
    if isinstance(correct_answer[0], str):
        if not all([correct_ans in options_list[i] for i, correct_ans in enumerate(correct_answer)]):
            raise ValueError("At least one correct answer must be provided for each question")
        if not all([sum([1 if option in correct_ans else 0 for option in options_list[i]]) <= 1 for i, correct_ans in enumerate(correct_answer)]):
            raise ValueError("Each correct answer must have at most one True value")
    elif isinstance(correct_answer[0], list):
        if not all([True in correct_ans or 1 in correct_ans for correct_ans in correct_answer]):
            raise ValueError("At least one correct answer must be provided for each question")
    else:
        raise ValueError("Correct answer must be a list of strings or a list of lists of strings")
    return True

def format_value(value):
    if isinstance(value, float) or isinstance(value, np.float64):
        return round(value, 2)
    elif isinstance(value, str):
        # Round every decimal number in one pass. (The old replace-based
        # version corrupted strings where one number was a substring of
        # another, e.g. '1.5' inside '21.53', and clashed with literal '{0}'.)
        # A number written with at most 2 decimals is left as written: the filler chose that format ("36.00%" next
        # to "30.34%"); rounding it again printed "36.0%", so the computed option was the only one with 2 decimals.
        return re.sub(r'\d+\.\d+', lambda m: m.group(0) if len(m.group(0).split('.')[1]) <= 2
                      else str(round(float(m.group(0)), 2)), value)
    elif isinstance(value, int) or isinstance(value, np.int64):
        return value
    elif isinstance(value, tuple):
        return tuple(format_value(val) for val in value)
    else:
        msg = f"Value must be a tuple, string, float, or int, it is of type: {type(value)}"
        raise ValueError(msg)

def format_fill_values(values):
    """
    Format fill values for filling templates.

    Args:
        values (dict): Dictionary containing fill values.

    Returns:
        dict: Formatted fill values.
    """
    # if there are numbers in the values, ensure they are rounded to two decimals


    for key, value in values.items():
        if isinstance(value, list):
            if isinstance(value[0], list):
                values[key] = [[format_value(lst) for lst in val] for val in value]
            else:
                values[key] = [format_value(val) for val in value]
        else:
            values[key] = format_value(value)

    return values

def extract_fstring_keys(fstring):
    """Return the set of placeholder keys in a format string, ignoring escaped braces."""
    import string
    return {field_name for _, field_name, _, _ in string.Formatter().parse(fstring) if field_name is not None}


def exist_duplicate_options(options_list):
    """
    Check if there are duplicate options in the options list.

    Args:
        options_list (list): List of lists of options.

    Returns:
        bool: True if there are duplicate options, False otherwise.
    """
    all_options = [format_value(option) for options in options_list for option in options]
    return len(all_options) != len(set(all_options))



def fill_dataset(dataset: sd.Dataset, fill_func: Callable[[dict], dict], fill_num: int = 100, sample: Union[int, float] = 100, sub_pbar = False):
    """
    Create filled DataFrames for questions, options, answers, and metadata.
    If fill function returns a dictionary with options_list and correct_answer, then options_list must be a list of lists of options and correct_answer must be a boolean list of lists.
    Furthermore, it uses those values for the options and answers even if there already exists options and answers in the DataFrames.

    Args:
        dataset (Dataset): Dataset containing questions, options, answers, and metadata.
        fill_func (function): Function to generate fill values.
        fill_num (int, optional): Number of filled examples to generate. Defaults to 100.

    Returns:
        tuple: Tuple containing filled questions, options, answers, and metadata DataFrames.
    """

    errors = {
        'format_error': 0,
        'metadata_error': 0,
        'fill_func_error': 0
    }

    sampled_dataset = dataset.sample(value = sample).sort_values(by = ['domain', 'type', 'difficulty_level', 'tags'])
    fill_dataset = sd.Dataset()
    for dataframe_type in ['example_questions', 'questions']:
        global_question_id = 0
        for question_id, dataset_questions_df, options_df in sampled_dataset.iterrows(pbar = True, dataframe_type = dataframe_type):
            metadata = sd.get_metadata(rows = dataset_questions_df, suppress_warnings=True)
            for run_num in range(fill_num):
                questions_df = dataset_questions_df.copy()

                
                if dataframe_type == 'example_questions' and run_num == 5:
                    break

                fill_values = fill_func(metadata)

                if 'options_list' in fill_values:
                    _dedup_attempts = 0
                    while exist_duplicate_options(fill_values['options_list']):
                        fill_values = fill_func(metadata)
                        _dedup_attempts += 1
                        if _dedup_attempts >= 10:
                            errors['fill_func_error'] += 1
                            break
                fill_values = format_fill_values(fill_values)
                
                if ('options_list' in fill_values) ^ ('correct_answer' in fill_values):
                    raise ValueError("Both options_list and correct_answer must be provided")
                
                # If there is an options_list then there do not 
                if 'options_list' in fill_values:
                    options_list = fill_values['options_list']
                    correct_answers = fill_values['correct_answer']

                    try:
                        validate_fill_values(options_list, correct_answers)
                        fill_values.pop('options_list')
                        fill_values.pop('correct_answer')
                    except ValueError:
                        errors['fill_func_error'] += 1
                        break
                else:
                    sub_ids = questions_df['sub_id'].unique()
                    options_list = [
                        options_df[
                            (options_df['question_id'] == question_id) &
                            (options_df['sub_id'] == sub_id)
                        ]['option_text'].tolist()
                        for sub_id in sub_ids
                    ]
                    correct_answers = [
                        options_df[
                            (options_df['question_id'] == question_id) &
                            (options_df['sub_id'] == sub_id)
                        ]['correct_answer'].tolist()
                        for sub_id in sub_ids
                    ]


                ### Next code block is for filling in the text ###

                # Build text to be filled in 
                fill_text = construct_fill_text(questions_df, options_list)

                # Extract keys from the f-string
                fstring_keys = extract_fstring_keys(fill_text)

                # Check for any keys in fill_values that are not in the f-string
                extra_keys = set(fill_values.keys()) - fstring_keys
                if extra_keys:
                    print()
                    errors['format_error'] += 1
                    break
                
                # try to fill in the text with the fill values and catch any errors that occur
                try:
                    filled_text = fill_text.format(**fill_values)
                except TypeError as e:
                    # Increment a counter to keep track of how many times this exception occurs: fill_values is None
                    if str(e) == 'str.format() argument after ** must be a mapping, not NoneType':
                        errors['metadata_error'] += 1
                        print(metadata)
                    else:
                        print(e)
                    break
                except KeyError:
                    # Increment a counter to keep track of how many times this exception occurs
                    errors['format_error'] += 1
                    # display(fill_values)
                    break
                except ValueError as e:
                    if "unexpected '{' in field name" in str(e):
                        errors['format_error'] += 1
                        break

                # Split text back into rows
                question_text, options_text = split_fill_text(filled_text)
                
                rows = []
                for sub_id, question in enumerate(question_text):

                    options_text_list = options_text[sub_id]
                    option_rows = []
                    for option_id, option in enumerate(options_text_list):
                        correct_answer = correct_answers[sub_id][option_id]
                        option_rows.append({
                            'question_id': global_question_id,
                            'sub_id': sub_id,
                            'option_id': option_id,
                            'option_text': option,
                            'correct_answer': correct_answer
                        })
                    
                    row = {
                        'question_id': global_question_id,
                        'sub_id': sub_id,
                        'question_text': question,
                        # 'options': option_rows,
                        'original_question_id': question_id,
                    }
                    row.update(metadata)
                    rows.append(row)

                # fill_dataset.append(rows, dataframe_type = dataframe_type, suppress_warnings = True)

                question_rows = pd.DataFrame(rows)
                option_rows = pd.DataFrame(option_rows)
                if 'example' in dataframe_type:
                    if fill_dataset.example_questions.empty:
                        fill_dataset.example_questions = question_rows
                        fill_dataset.example_options = option_rows
                    else:
                        fill_dataset.example_questions = pd.concat([fill_dataset.example_questions, question_rows], ignore_index=True)
                        fill_dataset.example_options = pd.concat([fill_dataset.example_options, option_rows], ignore_index=True)
                else:
                    if fill_dataset.questions.empty:
                        fill_dataset.questions = question_rows
                        fill_dataset.options = option_rows
                    else:
                        fill_dataset.questions = pd.concat([fill_dataset.questions, question_rows], ignore_index=True)
                        fill_dataset.options = pd.concat([fill_dataset.options, option_rows], ignore_index=True)
                global_question_id += 1
    
    # print number of format errors that occurred and as a percentage of the total number of templates
    print(f"Format errors: {errors['format_error']} ({errors['format_error'] / (len(dataset.questions) + len(dataset.example_questions)) * 100:.2f}%)")
    # print number of metadata errors that occurred and as a percentage of the total number of templates
    print(f"Metadata errors: {errors['metadata_error']} ({errors['metadata_error'] / (len(dataset.questions) + len(dataset.example_questions)) * 100:.2f}%)")
    print(f"Fill function errors: {errors['fill_func_error']} ({errors['fill_func_error'] / (len(dataset.questions) + len(dataset.example_questions)) * 100:.2f}%)")
    # Set id column types while keeping question_id/sub_id as regular columns.
    for df in [fill_dataset.questions, fill_dataset.options, fill_dataset.example_questions, fill_dataset.example_options]:
        if not df.empty:
            df[['question_id', 'sub_id']] = df[['question_id', 'sub_id']].astype(int)

    return fill_dataset


def get_mcq(row):
    """
    Get MCQ from template JSON.

    Args:
        row (dict): Row from template DataFrame.

    Returns:
        str: MCQ string.
    """
    question_num = 0
    global_option_index = 0
    mcq = ''
    while True:
        question_num += 1
        try:
            question = row[f'question_{question_num}']
            options = row[f'options_{question_num}']
            answer = row[f'answer_{question_num}']

            # break if questions is nan
            if pd.isna(question):
                break
            mcq = f"{question}  \n"
            if isinstance(options, list):
                for i, option in enumerate(options):
                    mcq += f"{chr(65 + i + global_option_index)}. {option}  \n"
            else:
                mcq += f"{chr(65 + global_option_index)}. {options}  \n"
            print(row)
            global_option_index += len(options)
            mcq += f"\nAnswer: {answer}  \n  \n"
        except KeyError:
            break
    
    return mcq

def pretty_print(mcq, explanation):
    """
    Pretty print MCQ with explanation.

    Args:
        mcq (str): MCQ string.
        explanation (str): Explanation string.
    """
    q_lines = mcq.split('\n')
    q_lines[0] = textwrap.fill(q_lines[0], width=60)
    mcq = '\n'.join(q_lines)
    print(mcq)
    print('Explanation:')
    print(f"{textwrap.fill(explanation, width=60)}")


def add_explanations(client: LanguageModelClient, data_dir: str, save: bool = True, overwrite = False, debug_print = True):
    """
    Add explanations to questions.

    Args:
        client (GPTClient): GPT client.
        data_dir (str): Data directory.
        save (bool, optional): Save DataFrames to the data directory. Defaults to True.
        overwrite (bool, optional): Overwrite existing explanations. Defaults to False.
    """

    if os.path.exists(os.path.join(data_dir, 'questions_metadata.pkl')):
        dataset = sd.from_old_pickles(data_dir)
    else:
        dataset = sd.from_pickle(data_dir)
    
    for question_id, questions_df, options_df in dataset.iterrows(dataframe_type='example_questions', pbar = True):
        explanations = [{'question_id': question_id, 'sub_id': sub_id, 'explanation': ''} for sub_id in questions_df['sub_id']]
        mcqs = sd.rows_to_mcq(questions_df, options_df)
        for sub_id, mcq in enumerate(mcqs):
            current = questions_df.loc[questions_df['sub_id'] == sub_id, 'explanation']
            if not overwrite and not current.empty and isinstance(current.iloc[0], str):
                continue
            explanation = client.make_request(user_prompts = mcq, request_type = 'explanation')
            explanations[sub_id]['explanation'] = explanation
            # print mcq with explanation wrapped text for debugging
            if debug_print:
                print('-------------------------')
                pretty_print(mcq, explanation)
                print('-------------------------')

    dataset.update(explanations, example_questions=True)
    if save:
        dataset.to_pickle(data_dir)


def fill_batches(data_dir: str):
    files = os.listdir(data_dir)
    dataset_files = [f for f in files if '.pkl' in f]
    dataset = sd.from_pickle(os.path.join(data_dir, dataset_files[0]), get_dataset=False)

    for f in tqdm(dataset_files[1:], desc='Processing dataset.dill files'):
        new_dataset = sd.from_pickle(os.path.join(data_dir, f), get_dataset=False)
        # dataset_json = new_dataset.to_json(dataframe_type='questions')
        dataset.concat([new_dataset], on='questions')
    
    dataset_json_files = [f for f in files if 'dataset' in f and '.json' in f]
    for json_file in tqdm(dataset_json_files, desc='Processing dataset JSON files'):
        with open(os.path.join(data_dir, json_file), 'rb') as f:
            dataset_json = dill.load(f)
        for json_object in dataset_json:
            try:
                dataset.append(json_object, dataframe_type='questions', ignore_options=True)
            except Exception as e:
                print(f'Error with dataset_json {json_object}\n {e}')
    
    display_json_files = [f for f in files if 'display' in f and '.json' in f]
    for json_file in tqdm(display_json_files, desc='Processing display JSON files'):
        with open(os.path.join(data_dir, json_file), 'rb') as f:
            display_json = dill.load(f)
        for json_object in display_json:
            try:
                dataset_json = sd.display_to_dataset_json(json_object)
                dataset.append(dataset_json, dataframe_type='questions')
            except Exception:
                print(f'Error with display_json {json_object}')
    
    print("Filling dataset...")
    filled_dataset = fill_dataset(dataset, function_registry.get(os.path.basename(data_dir)), fill_num=1000)
    element_save_path = du.get_data_directory_path('elements', append_dir=os.path.basename(data_dir))
    filled_dataset.to_old_pickles(element_save_path)
