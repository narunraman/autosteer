# src/steer_generation/generation_utils.py

# Built-in packages
import json
from itertools import product
from copy import deepcopy
import re
from typing import Dict
import backoff
import asyncio
from concurrent import futures
import traceback
# External packages
import openai
import pandas as pd
# Local packages
from .constants import DOMAINS, TAGS
#  get_seed_template_dir, get_generated_templates_dir, get_templates_dir
from .llm_client import LanguageModelClient

# Constants
generation_prompt = "Below, you will find some metadata that corresponds to a set of instructions and a template MCQ example that abides by these instructions.\n{seed_templates}\n\nGiven the following metadata and instructions please write a template MCQ that abides by the following instructions and metadata. Make sure the template fits the following set of metadata and instructions. \n{starting_content}"

def metadata_combinations(domains: list[str], difficulty_levels: list[str], types: list[str], tags_list: list[str] = ['first_person']) -> list[dict]:
    """
    Generate metadata combinations based on the given lists of domains, difficulty levels, types, and an optional tags list.

    Args:
        domains (List[str]): A list of domain names.
        difficulty_levels (List[str]): A list of difficulty levels.
        types (List[str]): A list of types.
        tags_list (List[str], optional): A list of tags. Defaults to ['first_person'].

    Returns:
        List[dict]: A list of dictionaries representing the metadata combinations. Each dictionary contains the following keys:
            - 'domain': The domain name.
            - 'difficulty_level': The difficulty level.
            - 'type': The type.
            - 'tags': The tags.
    """

    metadata_list = [
        {
            'domain': elm[0],
            'difficulty_level': elm[1],
            'type': elm[2],
            'tags': elm[3],
        }
        for elm in product(domains, difficulty_levels, types, tags_list)
    ]
    return metadata_list


def build_instructions(metadata: dict, num_options: int, include_answer: bool, variables: str, task_description: str = '', json_type = 'display_json') -> str:
    if task_description:
        instruction = task_description + "."
    else:
        instruction = "Please give an MCQ example testing the ability " + '<insert task description here>.'.upper()
    try:
        instruction += " The question should be concerning " + DOMAINS[metadata['domain']] + '. ' if metadata['domain'] != 'base' else " The question should not be phrased in terms of a specific domain. "
        # missing_info['domain'] = None
    except KeyError:
        instruction += ' The question should be concerning' + ' <insert domain description>. '.upper()
        # missing_info['domain'] = metadata['domain']
    
    if 'tags' in metadata and metadata['tags']:
        # tag_substrings = get_tag_substrings(metadata['tags'])
        tags = metadata['tags'].split(';')
        tags_instructions = []
        missing_tags = ''
        for tag in tags:
            try:
                tags_instructions.append(TAGS[tag])
            except KeyError:
                missing_tags += ';' + tag
        if tags_instructions:
            instruction += '. '.join(tags_instructions) + '.'
        
        if len(missing_tags) > 1:
            instruction += ' <insert tag descriptions for the following tags: '.upper() + ', '.join(missing_tags) + '>.'

    if variables:
        text_variables = variables.split(';')
        # remove duplicates
        text_variables = ', '.join(set(text_variables))
        instruction += f" Do not include numbers in the example but leave them as variables as in the example below. Use the following variables in the curly braces: {text_variables}."
    

    if json_type == 'display_json':
        if not include_answer:
            instruction += ' Leave the answer as an empty string.'
        
        if num_options == 1:
            instruction += ' The MCQ should have only one option.'
        elif num_options == 0:
            instruction = instruction.replace(' Leave the answer as an empty string.', ' Leave the answer as an empty string and the options as a list containing a single empty string.')
    elif json_type == 'dataset_json':
        if not include_answer:
            instruction += ' Do not mark any option with correct_answer as True.'
        if num_options == 1:
            instruction += ' The MCQ should have only one option.'
        elif num_options == 0:
            instruction = instruction.replace(' Do not mark any option with correct_answer as True.', ' Do not mark any option with correct_answer as True and the options should be a list containing a single empty string.')
                                            #   ' Do not include an options key in the JSON object. Ignore the previous instruction if it is present, only include questions keys.')


    instruction += ' See the following example as a guide but give a different example and write a narrative: '

    return instruction


def instructions_from_display_json(display_json: list[dict]) -> pd.DataFrame:
    """
    Extracts the instructions from the display_json format.

    Args:
        display_json (list): The display_json format.

    Returns:
        pd.DataFrame: A DataFrame containing columns instructions, domain, difficulty_level, type, and tags.
    """

    instruction_set = []
    for json_object in display_json:
        instructions = json_object['instructions']
        metadata = json_object['metadata']
        instruction_set.append({
            'instructions': instructions,
            'domain': metadata['domain'],
            'difficulty_level': metadata['difficulty_level'],
            'type': metadata['type'],
            'tags': metadata['tags']
        })
    return pd.DataFrame(instruction_set)




def get_template_text(seed_template: dict[str, str] | list[dict[str, str]]) -> str:
    # order the keys in example_seed_template
    def order_keys(template):
        """
        Orders the keys in the template.
        The order is as follows:
        - the template metadata
        - the template instructions
        - the first question
        - the first set of options
        - the first answer
        - and so on for the rest of the questions, options, and answers
        """
        starting_keys = ['metadata', 'instructions']
        sorted_question_keys = sorted([key for key in template.keys() if 'question' in key])
        sorted_option_keys = sorted([key for key in template.keys() if 'options' in key])
        sorted_answer_keys = sorted([key for key in template.keys() if 'answer' in key])
        sorted_keys = starting_keys
        for question_key, option_key, answer_key in zip(sorted_question_keys, sorted_option_keys, sorted_answer_keys):
            sorted_keys.extend([question_key, option_key, answer_key])
        return {key: template[key] for key in sorted_keys}

    if isinstance(seed_template, dict):
        ordered_seed_template = order_keys(seed_template)
        seed_template_text = json.dumps(ordered_seed_template)
    elif isinstance(seed_template, list):
        ordered_seed_template = [order_keys(template) for template in seed_template]
        seed_template_text = '\n'.join([json.dumps(template) for template in ordered_seed_template])
    else:
        raise ValueError("Invalid example_seed_template type. Must be dict or list of dicts")
    return seed_template_text


def build_new_template_prompt(seed_template: dict[str, str] | list[dict[str, str]], metadata: dict[str, str]) -> str:

    def get_information_from_template(template):
        """
        Gets the number of options and whether the answer is included in the options from the template.
        """
        num_options = len(template['options_1'])
        include_answer = template['answer_1'] in template['options_1']
        text_variables = ';'.join(re.findall(r'{.*?}', template['question_1'] + '\n'.join(template['options_1']) + template['answer_1']))
        task_description = template['instructions'].split('. ')[0] + '. '
        return num_options, include_answer, text_variables, task_description
    
    if isinstance(seed_template, dict):
        num_options, include_answer, text_variables, task_description = get_information_from_template(seed_template)
    elif isinstance(seed_template, list) and len(seed_template) > 0:
        num_options, include_answer, text_variables, task_description = get_information_from_template(seed_template[0])
    else:
        raise ValueError(f"Invalid example_seed_template type. Must be dict or list of dicts.  \n{seed_template}")
    


    new_instructions = build_instructions(metadata, num_options=num_options, include_answer=include_answer, variables=text_variables)

    new_instructions = task_description + '. '.join(new_instructions.split('. ')[1:])
    new_instructions = re.sub(r'<.*?>', '', new_instructions)
    
    starting_content = {'metadata': metadata, 'instructions': new_instructions}

    return json.dumps(starting_content)

def build_generate_template_prompt(seed_templates: dict[str, str] | list[dict[str, str]], metadata: dict[str, str]) -> tuple[str, str]:
    
    user_prompt = deepcopy(generation_prompt)

    
    seed_template_text = get_template_text(seed_templates)
    user_prompt = user_prompt.replace('{seed_templates}', seed_template_text)
    starting_content = build_new_template_prompt(seed_templates, metadata)
    user_prompt = user_prompt.replace('{starting_content}', starting_content)
    instructions = json.loads(starting_content)['instructions']
    return user_prompt, instructions

# TODO: do this.
def is_display_json(json_object: dict) -> bool:
    """
    Checks whether the JSON object is formatted correctly

    ## Args:
    **json_object** (dict): the json object
    """
    raise NotImplementedError


def generate_template(client: LanguageModelClient, seed_templates: dict[str, str] | list[dict[str, str]], metadata: dict[str, str]) -> tuple[Dict, Dict, Dict]:
    """
    Generates a single template using the provided GPTClient, template, instructions, and metadata.

    ## Args:
    **client**: (LanguageModelClient) The LanguageModelClient instance used for template generation.
    **seed_template**: (dict[str, str] | list[dict[str, str]]) The template(s) to be used as seeds for generation. The template(s) should be in display_json format.
    **metadata**: (dict[str, str]) The metadata to be used for generation.

    ## Returns:
    **tuple[dict, dict]**: A tuple containing the response JSON object and the generation object
    """
    user_prompt, instructions = build_generate_template_prompt(seed_templates, metadata)
    while True:
        # client = LanguageModelClient(provider_name='dummy')
        request_object = client.build_request(user_prompts = user_prompt, request_type='generate_seed')
        generation = client.make_request(user_prompts = user_prompt, request_type='generate_seed')
        generation_string = client.get_content(generation)

        try:
            response_json = json.loads(generation_string)
            response_json.update({'metadata': metadata, 'instructions': instructions})
            break
        except json.JSONDecodeError:
            print(f"Invalid JSON response: {generation_string}")
            continue
        except TypeError:
            print(f"Invalid response: {generation_string}")
            continue
    return response_json, generation, request_object


async def call_language_model_async(**kwargs) -> str:
    client = LanguageModelClient(provider_name='openai')
    loop = asyncio.get_running_loop()
    with futures.ThreadPoolExecutor() as executor:
        x = await loop.run_in_executor(executor, lambda: client.make_request(model = 'gpt-4o', **kwargs))
        return x

# async def batch_call_language_model_async(params: list[dict]) -> list[str]:
#     tasks = [call_language_model_async(**param) for param in params]
#     responses = await asyncio.gather(*tasks)
#     return responses





async def batch_call_language_model_async(params: list[dict]) -> list[str]:
    tasks = [call_language_model_async(**param) for param in params]
    responses = []

    for task in asyncio.as_completed(tasks):
        
        while True:
            try:
                result = await task
                responses.append(result)
                break
            except openai.RateLimitError as e:
                # message = str(e)
                # wait_time = re.search(r'(?:Please retry after )(\d+)', message).group(1)
                print(f"Rate limit exceeded, retrying task: {e}")
                # wait_time = int(wait_time) + 1
                await asyncio.sleep(10)  # Wait before retrying
            except Exception as e:
                print(f"Task failed with exception: {e}")
                break

    return responses

async def retry_task(param, i):
    while True:
        try:
            result = await call_language_model_async(**param)
            return result
        except openai.RateLimitError as e:
            message = str(e)
            wait_time = re.search(r'(?:Please retry after )(\d+)', message)
            if wait_time is not None:
                wait_time = wait_time.group(1)
                print(f"Rate limit exceeded, retrying task after {wait_time} seconds.")
                wait_time = int(wait_time) + 1
            else:
                wait_time = 10  # Default wait time if not specified
            await asyncio.sleep(wait_time)  # Wait before retrying
        except Exception as e:
            print(f"Task {i} failed with exception: {e}")
            # print stack trace
            traceback.print_exc()
            return None
        

def fatal_code(error_message):
    return error_message['error']['type'] == 'insufficient_quota'

# use exponential backoff
async def retry_task_with_backoff(param, i):
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, giveup=fatal_code)
    async def call_language_model_async_retry(param):
        return await call_language_model_async(**param)
    try:
        result = await call_language_model_async_retry(param)
        return result
    except Exception as e:
        print(f"Task {i} failed with exception: {e}")
        # print stack trace
        traceback.print_exc()
        return None


async def batch_call_language_model_async_retry(params: list[dict]) -> list[str]:
    tasks = [asyncio.create_task(retry_task_with_backoff(param, i)) for i, param in enumerate(params)]
    responses = await asyncio.gather(*tasks)
    return [response for response in responses if response is not None]