# src/steer_generation/webapp_utils.py

# Built-in packages
import asyncio
from concurrent import futures
from copy import deepcopy
from itertools import compress
import json
import re
import time
# External packages
import streamlit as st
# Local packages
from autosteer.llm_client import LanguageModelClient
from autosteer.dataset_filler import format_fill_values

### Write Templates Utils ###

def get_json_from_prompt(prompt):
    # find index of the first { character in the prompt string
    start_index = prompt.find('{')
    return json.loads(prompt[start_index:])

def get_prompt(messages, role = 'user'):
    user_prompt = []
    for message in messages:
        if message['role'] == role:
            json_object = json.loads(message['content'])
            user_prompt.append(json_to_mcq(json_object))
    return user_prompt

def clean_string(string):
    # Remove leading and trailing whitespaces
    string = string.rstrip()
    # Remove extra whitespaces
    # string = " ".join(string.split())
    # remove trailing newlines
    # string = string.rstrip('\n')
    # add a period at the end of the string if it doesn't have one
    # if string and string[-1] not in ['.', '?', '!']:
        # string += '. '
    # replace $ with \$ if it is not already escaped
    string = re.sub(r'(?<!\\)\$', r'\\$', string)
    # replace \n with  \n
    string = string.replace('\n', '  \n')
    # format any latex for markdown
    string = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', string, flags=re.DOTALL)
    string = re.sub(r'\\\((.*?)\\\)', r'$\1$', string, flags=re.DOTALL)
    
    return string

def build_template_row(question_json, domain, question_type, difficulty_level, tags_list):
    # append metadata to the question json
    tags = ';'.join(tags_list)
    template_row = deepcopy(question_json)
    template_row.update(
        {
            "domain": domain,
            "type": question_type,
            "difficulty_level": difficulty_level,
            "tags": tags
        }
    )
    return template_row

# TODO: This is old.
def build_instructions_row(instructions, domain, difficulty_level, tags, question_type=''):
    instruction_json = {
        "domain": domain,
        "type": question_type,
        "difficulty_level": difficulty_level,
        "tags": tags,
        "instructions": instructions
    }
    return instruction_json

def json_to_mcq(json_object):
    """
    Convert a JSON object in the dataset_json format to a multiple choice question string
    """
    mcq_string = f"Question 0-{json_object['sub_id']}:  \n"
    mcq_string += f" {json_object['question_text']}  \n"
    correct_answer = ''
    for i, option in enumerate(json_object['options']):
        mcq_string += f"{chr(65+i)}. {option['option_text']}  \n"
        if option['correct_answer']:
            correct_answer = option['option_text']
    mcq_string += f"Correct Answer: {correct_answer}  \n"
    return mcq_string


def verify_template(template):
    client = LanguageModelClient()
    response = client.make_request(user_prompts = template, request_type = 'verify_template', return_type='string')
    return response

### Generate Seed Templates Utils ###


### View Templates Utils ###

def closest_match(options, answer):
    # Find the closest match to the answer in the options
    return min(options, key=lambda x: abs(len(x) - len(answer)))

def clean_option(option):
    # Remove leading and trailing whitespaces
    option = option.strip()
    # Remove extra whitespaces
    option = " ".join(option.split())
    # Remove last period
    if option and option[-1] == '.':
        option = option[:-1]
    return option

def display_json_to_mcq(display_json: dict, include_answer = False) -> str:
    """
    Convert a display JSON object to a multiple choice question string
    ## Args:
    **display_json** (dict): A JSON object in the display_json format
    ## Returns:
    **mcq_string** (str): A string representation of the multiple choice question
    """
    mcq_string = ""
    num_sub_questions = len([key for key in display_json.keys() if key.startswith('question')])
    for sub_id in range(num_sub_questions):
        question_text = display_json[f'question_{sub_id+1}']
        mcq_string += f" {question_text}  \n  \n"
        for option_id, option in enumerate(display_json[f'options_{sub_id+1}']):
            option = clean_option(option)
            if include_answer and option == display_json[f'answer_{sub_id+1}']:
                # print(f"Option {option_id} is the correct answer: {option} for display_json {display_json}")
                mcq_string += f":green[{chr(65+option_id)}. {option}]  \n"
            else:
                mcq_string += f"{chr(65+option_id)}. {option}  \n"
    mcq_string = clean_string(mcq_string)
    return mcq_string

def format_display_json(json_object, fill_values):
    fill_values = format_fill_values(fill_values)
    display_json = deepcopy(json_object)
    if 'options_list' in fill_values:
        for i, options in enumerate(fill_values['options_list']):
            # Ensure that each element in options are strings if they are not already
            display_json[f'options_{i+1}'] = [str(option) for option in options]
            # display_json[f'options_{i+1}'] = options
        if 'correct_answer' in fill_values:
        # check if correct_answer is a list of list of booleans
            if isinstance(fill_values['correct_answer'][0], list):
                for i, boolean_mask in enumerate(fill_values['correct_answer']):
                    display_json[f'answer_{i+1}'] = list(compress(display_json[f'options_{i+1}'], boolean_mask))[0]
            else:
                for i, answer in enumerate(fill_values['correct_answer']):
                    display_json[f'answer_{i+1}'] = answer
    return display_json

def parse_answer(response):
    # Parse the option letter from the model response. It will follow 'Correct Answer: ' in the response
    if response is not None:
        return tuple(re.findall(r'Correct Answer: ([A-Z])', response))
    return ('Z',)

def get_correct_option_letters(display_json):
    option_letters = []
    num_answers = len([key for key in display_json.keys() if key.startswith('answer')])
    for answer_num in range(1, num_answers + 1):
        if answer_num > 1:
            option_offset = len(display_json[f'options_{answer_num-1}'])
        else:
            option_offset = 0
        option_index = display_json[f'options_{answer_num}'].index(display_json[f'answer_{answer_num}'])
        option_letters.append(chr(65 + option_index + option_offset))
    return tuple(option_letters)

def determine_correct(predicted_answers, correct_answers):
    # return a boolean list of whether the predicted answer is correct
    return [predicted == correct for predicted, correct in zip(predicted_answers, correct_answers)]

### Async Utils ###

class TestModelClient:
    def make_request(self, kwargs):
        # Simulate a synchronous request
        def combine_args(**args):
            # combine all the arguments into a single string separated by commas
            return ', '.join([f"{k}={v}, {type(v)}" for k, v in args.items()])
        return combine_args(**kwargs)

    def get_content(self, json_object):
        return json_object


### General Utils ###
class StatusManager:
    def __init__(self):
        self.statuses = {
            'ready': 'ready',
            'preparing': 'preparing',
            'not_ready': 'not_ready',
            'running': 'running',
            'view': 'view'
        }
        self.current_status = self.statuses['not_ready']

    def set_status(self, status):
        if status in self.statuses.values():
            self.current_status = status
        else:
            raise ValueError(f"Status '{status}' is not a valid status.")

    def is_status(self, status):
        if status in self.statuses.values():
            return self.current_status == status
        else:
            raise ValueError(f"Status '{status}' is not a valid status.")

    def add_status(self, name, value):
        if name not in self.statuses:
            self.statuses[name] = value
        else:
            raise ValueError(f"Status '{name}' already exists.")

    def remove_status(self, name):
        if name in self.statuses:
            del self.statuses[name]
        else:
            raise ValueError(f"Status '{name}' does not exist.")

def show_message(message, type = 'success', duration=2):
    success_placeholder = st.empty()
    with success_placeholder:
        if type == 'error':
            st.error(message)
        elif type == 'warning':
            st.warning(message)
        elif type == 'info':
            st.info(message)
        elif type == 'success':
            st.success(message)
        else:
            st.write(message)
        time.sleep(duration)
        success_placeholder.empty()

### Code Editor Config ###

css_string = '''
background-color: #bee1e5;

body > #root .ace-streamlit-dark~& {
   background-color: #262830;
}

.ace-streamlit-dark~& span {
   color: #fff;
   opacity: 0.6;
}

span {
   color: #000;
   opacity: 0.5;
}

.code_editor-info.message {
   width: inherit;
   margin-right: 75px;
   order: 2;
   text-align: center;
   opacity: 0;
   transition: opacity 0.7s ease-out;
}

.code_editor-info.message.show {
   opacity: 0.6;
}

.ace-streamlit-dark~& .code_editor-info.message.show {
   opacity: 0.5;
}
'''


info_bar = {
    "name": "language info",
    "css": css_string,
    "style": {
                "order": "1",
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "center",
                "width": "100%",
                "height": "2.5rem",
                "padding": "0rem 0.75rem",
                "borderRadius": "8px 8px 0px 0px",
                "zIndex": "9993"
            },
    "info": [{
                "name": "Template",
                "style": {"width": "100px"}
            }]
}

custom_btns = [{
      "name": "Save",
      "feather": "Save",
      "hasText": True,
      "alwaysOn": True,
      "commands": ["save-state", 
                   ["response","saved"],
                   ["infoMessage", 
                    {
                     "text":"Saved edits!",
                     "timeout": 2500, 
                     "classToggle": "show"
                    }
                   ]
                  ],
      "style": {"right": "0.4rem"}
    }]