# Built-in packages
import asyncio
import json
import random
import re
import typing
import copy
# External packages
import pandas as pd
# Local packages
from . import steer_dataset as sd
from . import generation_utils as gu
from . import directory_utils as du
from .constants import TAGS, DOMAINS
from autosteer.filler_tools import function_registry



# TODO: Should create two protocols, expansion and mimic where expansion takes the current metadata and expands it to include all possible combinations of tags/domains and mimic takes the current metadata and mimics the current metadata structure


TEMPERATURE = 1

def set_generation_temperature(temp):
    global TEMPERATURE
    TEMPERATURE = temp

def build_template_metadata(request_object: list | dict, generation_object: list | typing.Any) -> pd.DataFrame:
    """
    Builds the template generation metadata DataFrame.

    ## Args:
    **request_object**: The request object.
    **generation_object**: The generation object.

    Returns:
        pd.DataFrame: The template generation metadata DataFrame.
    """
    def build_row(request_object, generation_object):
        metadata = {
            'request_object': request_object,
            'generation_object': generation_object,
        }
        return metadata
    
    if isinstance(request_object, list):
        metadata_df = [build_row(request, generation) for request, generation in zip(request_object, generation_object)]
        metadata_df = pd.DataFrame.from_records(metadata_df)
    else:
        metadata_df = build_row(request_object, generation_object)
        metadata_df = pd.DataFrame([metadata_df])

    metadata_df.reset_index(names = 'question_id', inplace = True)
    return metadata_df

def select_metadata(metadata_counts: pd.DataFrame, select_only: list | dict) -> pd.DataFrame:
    """
    Selects metadata from metadata_counts based on select_only. If select_only is a list, it will return the union of the metadata selected from each element in the list.

    ## Args:
    **metadata_counts:** (pd.DataFrame) The metadata counts dataframe.
    **select_only:** (list | dict) The metadata to select from.
    
    ## Returns:
    **pd.DataFrame:** The selected metadata.
    """
    if isinstance(select_only, dict):
        command = sd.build_command(select_only)
        return metadata_counts.query(command)
    elif isinstance(select_only, list):
        commands = []
        for select in select_only:
            commands.append(select_metadata(metadata_counts, select))
        return pd.concat(commands)
    else:
        raise ValueError("Invalid type for select_only. Must be dict or list of dicts")

def expand_commands(metadata_df: pd.DataFrame, expansion_key: str) -> pd.DataFrame:
    """
    Expands the tags/domains in metadata_counts to include all tag/domain combinations.

    ## Args:
    **metadata_counts:** (pd.DataFrame) The metadata counts dataframe.
    **expansion_key:** (str) The key to expand. Must be 'tags' or 'domain'.
    
    ## Returns:
    **pd.DataFrame:** The expanded metadata.
    """
    if expansion_key == 'tags':
        expansion_values = TAGS
    elif expansion_key == 'domain':
        expansion_values = DOMAINS
    else:
        raise ValueError("Invalid expansion_key. Must be 'tags' or 'domain'")
    # print(metadata_df)
    metadata_df = metadata_df.groupby(['domain', 'type', 'difficulty_level']).apply(lambda x: x.reindex(TAGS, fill_value=0), include_groups = False).index.to_frame(name = sd.metadata_keys).reset_index(drop=True)
    return metadata_df

def get_dispatch_amount(dataset: sd.Dataset, num_questions: int, resume: bool, expansion_key: str) -> pd.DataFrame:
    """
    Gets the metadata counts for the dataset. If resume is True, it will subtract the num_questions from the metadata_counts.

    ## Args:
    **dataset:** (sd.Dataset) The dataset object.
    **num_questions:** (int) The number of questions to generate.
    **resume:** (bool) Whether to resume generating questions. 
    **expansion_key:** (str) The key to expand. Must be 'tags' or 'domain'.

    ## Returns:
    **pd.DataFrame:** The metadata counts.
    ### Example:
    | type | domain | difficulty_level | tags | count |
    |------|--------|------------------|------|-------|
    | implicit | math | 8 | algebra | 50 |


    """
    metadata_counts = dataset.metadata('example_questions')
    if expansion_key:
        metadata_counts = expand_commands(metadata_counts, expansion_key)
    
    metadata_counts['count'] = [int(num_questions)] * len(metadata_counts)

    metadata_counts = metadata_counts.set_index(['domain', 'type', 'difficulty_level', 'tags'])

    if resume and not dataset.questions.empty:
        # get all existing metadata
        current_counts = dataset.questions[sd.metadata_keys].groupby(sd.metadata_keys).size().reset_index(name='count')
        
        current_counts = current_counts.set_index(['domain', 'type', 'difficulty_level', 'tags'])

        metadata_counts = metadata_counts.subtract(current_counts, fill_value=0)

        metadata_counts = metadata_counts[metadata_counts['count'] > 0]

        # Count should be an integer
        metadata_counts['count'] = metadata_counts['count'].astype(int)
    
    return metadata_counts.reset_index()


def get_dispatch_commands(
        dataset: sd.Dataset, 
        num_questions: int, 
        select_only: dict = None, 
        resume: bool = True,
        expansion_key: str = 'tags') -> pd.DataFrame:
    """
    Gets the dispatch commands for generating questions.

    ## Args:
    **dataset:** (sd.Dataset) The dataset object containing the examples_questions.
    **num_questions:** (int) The number of questions to generate.
    **select_only:** (dict, optional) A dictionary of metadata to select from. Defaults to None.
    - Example: ```json
                {"type": "implicit", "domain": "math", "difficulty_level": 8, "tags": "algebra"}
               ```
    **resume:** (bool, optional) Whether to only generate questions with metadata not present in questions dataset. Defaults to True.
    **tag_expansion:** (bool, optional) Whether to add tag combinations to the dispatch commands. Defaults to True.

    ## Returns:
    **pd.DataFrame:** The dispatch commands.
    """
    dispatch_commands = get_dispatch_amount(dataset, num_questions, resume, expansion_key)

    if select_only is not None:
        dispatch_commands = select_metadata(dispatch_commands, select_only)

    return dispatch_commands

def dispatch_job(dataset: sd.Dataset, dispatch_command: dict, tag_expansion: bool = True, batch: bool = True) -> sd.Dataset:
    # if protocol == ''
    irrelevant_metadata = ['count', 'tags'] if tag_expansion else ['count']  
    dispatch_metadata = {k: v for k,v in dispatch_command.items() if k not in irrelevant_metadata}
    metadata = {k: v for k,v in dispatch_command.items() if k not in ['count']}
    # responses = [json.dumps({'metadata': metadata, 'instructions': None, 'question_1': f'Question {i} text'}) for i in range(dispatch_command['count'])]
    if batch:
        # Get all rows in dataset that match the dispatch_command
        selected_dataset = dataset.query(sd.build_command(dispatch_metadata), dataframe_type='example_questions', return_type='dataset')
        # get templates from the selected_dataset
        seed_templates = selected_dataset.to_display_json()
        for i, example_seed in enumerate(seed_templates):
            
            # TODO: this will not be necessary going forward but also I should reuse code from template_generation instead of doing this bespoke thing here.
            # If the instruction set is empty, then we need to look in the example_seeds directory
            if dataset.instruction_set.empty:
                element_name = dataset.element_name
                example_seed_path = du.get_data_directory_path('example_seeds', append_dir=element_name)
                example_seeds_dataset = sd.Dataset.from_pickle(example_seed_path)
                instruction_set = []
                example_instructions = sd.get_instructions(example_seeds_dataset.instruction_set, metadata, get_closest_match=True).iloc[0]['instructions']
                query_command = sd.build_command({key: value for key, value in metadata.items() if key in ['type', 'difficulty_level']})
                # print(query_command)
                # print(example_seeds_dataset.example_questions)
                # print(example_seeds_dataset.query(query_command, dataframe_type='example_questions', return_type='dataset'))
                # get the question text that matches the current metadata on type and difficulty_level
                template = example_seeds_dataset.query(query_command, dataframe_type='example_questions', return_type='dataset').to_display_json()[0]
                # get the text variables for the instructions
                text_variables = ';'.join(re.findall(r'{.*?}', template['question_1'] + '\n'.join(template['options_1']) + template['answer_1']))
                instructions = gu.build_instructions(metadata, num_options = 0, include_answer=False, task_description=example_instructions.split('.')[0], variables = text_variables)
                instruction_set.append({
                    'instructions': instructions,
                    'domain': metadata['domain'],
                    'type': metadata['type'],
                    'difficulty_level': metadata['difficulty_level'],
                    'tags': metadata['tags'],
                })
                instructions_df = pd.DataFrame(instruction_set)
            else:
                instructions_df = sd.get_instructions(dataset.instruction_set, example_seed['metadata'], get_closest_match=True)
                
            try:
                instructions = instructions_df['instructions'].values[0]
            except IndexError:
                print(f"Error getting instructions on element {dataset.element_name} for {example_seed['metadata']}. Looking in example_seeds/")
                example_seeds_dataset = sd.Dataset.from_pickle(du.get_data_directory_path('example_seeds', append_dir=dataset.element_name))
                instructions_df = sd.get_instructions(example_seeds_dataset.instruction_set, example_seed['metadata'], get_closest_match=True)
                instructions = instructions_df['instructions'].values[0]
            seed_templates[i]['instructions'] = instructions

        # print(dispatch_command)
        # TODO: may want to make this function take in a dataset_json
        prompt, _ = gu.build_generate_template_prompt(seed_templates=seed_templates, metadata=metadata)
        # TODO: to make this function return the request and generation object need to call build_request 
        batched_commands = [{'user_prompts': prompt, 'request_type': 'generate_question', 'return_type': 'string', 'seed': random.randint(1, 5000)} for _ in range(dispatch_command['count'])]
        # TODO: then need to not return the completion string
        responses = asyncio.run(gu.batch_call_language_model_async(batched_commands))
    else:
        raise NotImplementedError
    
    display_json_list = []
    for response in responses:
        display_json = json.loads(response)
        display_json.update({'metadata': metadata, 'instructions': instructions})
        display_json_list.append(display_json)
    return display_json_list


async def dispatch_async_job(dataset: sd.Dataset, dispatch_command: dict, tag_expansion: bool = True, batch: bool = True) -> sd.Dataset:
    irrelevant_metadata = ['count', 'tags'] if tag_expansion else ['count']  
    dispatch_metadata = {k: v for k,v in dispatch_command.items() if k not in irrelevant_metadata}
    metadata = {k: v for k,v in dispatch_command.items() if k not in ['count']}
    
    if batch:
        selected_dataset = dataset.query(sd.build_command(dispatch_metadata), dataframe_type='example_questions', return_type='dataset')
        seed_templates = selected_dataset.to_display_json()
        for i, example_seed in enumerate(seed_templates):
            
            # Set get_closest_match to True to get the closest match because the instructions will be re-built anyway. All we need is the first sentence
            instructions_df = sd.get_instructions(dataset.instruction_set, example_seed['metadata'], get_closest_match=True)
            try:
                instructions = instructions_df['instructions'].values[0]
            except (TypeError, IndexError):
                print(f"Error getting instructions on element {dataset.element_name} for {example_seed['metadata']}. Looking in example_seeds/")
                example_seeds_dataset = sd.from_pickle(du.get_data_directory_path('example_seeds', append_dir=dataset.element_name))
                instructions_df = sd.get_instructions(example_seeds_dataset.instruction_set, example_seed['metadata'], get_closest_match=True)
                instructions = instructions_df['instructions'].values[0]
            seed_templates[i]['instructions'] = instructions
        
        prompt, _ = gu.build_generate_template_prompt(seed_templates=seed_templates, metadata=metadata)
        batched_commands = [{'user_prompts': prompt, 'request_type': 'generate_question', 'return_type': 'string', 'seed': random.randint(1, 5000)} for _ in range(dispatch_command['count'])]
        responses = await gu.batch_call_language_model_async_retry(batched_commands)
    else:
        raise NotImplementedError
    
    display_json_list = []
    for response in responses:
        try:
            display_json = json.loads(response)
        except json.JSONDecodeError:
            print(f"Error decoding response: {response}")
            continue
        display_json.update({'metadata': metadata, 'instructions': instructions})
        display_json_list.append(display_json)
    
    return display_json_list
