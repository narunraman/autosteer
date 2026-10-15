# src/steer_generation/template_generation.py

# Built-in packages
import typing
# External packages
import pandas as pd
# Local packages
from . import steer_dataset as sd

template_generation_metadata_keys = [
    'input_messages',
    "generation_object",
    "response_format",
    "temperature"
]


def build_seed_template_metadata(request_object: list | dict, generation_object: list | typing.Any) -> pd.DataFrame:
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

# TODO: need to save the generation metadata to the dataset i.e., a chat completions object
# seed_generation_metadata = {
#   "id": "chatcmpl-123",
#   "object": "chat.czompletion",
#   "created": 1677652288,
#   "model": "gpt-4o-mini",
#   "system_fingerprint": "fp_44709d6fcb",
#   "choices": [{
#     "index": 0,
#     "message": {
#       "role": "assistant",
#       "content": "\n\nHello there, how may I assist you today?",
#     },
#     "logprobs": null,
#     "finish_reason": "stop"
#   }],
#   "usage": {
#     "prompt_tokens": 9,
#     "completion_tokens": 12,
#     "total_tokens": 21
#   }
# }
# Probably also want response_format and the system and user prompts in the metadata



# def metadata_combinations(domains: list[str], difficulty_levels: list[str], types: list[str]) -> list[dict]:
#     """
#     Generate metadata combinations based on the given lists of domains, difficulty levels, types, and tags.

#     Args:
#         domains (List[str]): A list of domain names.
#         difficulty_levels (List[str]): A list of difficulty levels.
#         types (List[str]): A list of types.

#     Returns:
#         List[dict]: A list of dictionaries representing the metadata combinations. Each dictionary contains the following keys:
#             - 'domain': The domain name.
#             - 'difficulty_level': The difficulty level.
#             - 'type': The type.
#             - 'tags': The tags. Will always be first_person
#     """
#     metadata_list = [
#         {
#             'domain': elm[0],
#             'difficulty_level': elm[1],
#             'type': elm[2],
#             'tags': 'first_person',
#         }
#         for elm in product(domains, difficulty_levels, types)
#     ]
#     return metadata_list



def get_valid_combinations(metadata_combinations: list[dict[str, str]], example_questions: pd.DataFrame) -> list[dict[str, str]]:
    """
    Gets the valid metadata combinations based on the given example questions.
    - Checks if the type and difficulty level pair is present in the example questions.

    Args:
    - metadata_combinations (List[dict]): A list of metadata combinations.
    - example_questions (pd.DataFrame): A DataFrame containing the example questions.

    Returns:
    - List[dict]: A list of valid metadata combinations.
    """
    valid_combinations = []
    valid_metadata = sd.get_metadata_df(example_questions)
    for metadata in metadata_combinations:
        relevant_metadata = {key: value for key, value in metadata.items() if key in ['type', 'difficulty_level']}
        command = sd.build_command(relevant_metadata)
        if valid_metadata.query(command).empty:
            continue
        valid_combinations.append(metadata)
    return valid_combinations

def clean_response_json(json_object):
    """
    Cleans the response JSON object by removing the 'metadata' key.

    Args:
        json_object (dict): The JSON object to be cleaned.

    Returns:
        dict: The cleaned JSON object.
    """
    metadata_keys = ['domain', 'difficulty_level', 'type', 'tags']
    for key in metadata_keys:
        json_object.pop(key, None)
    return json_object
    