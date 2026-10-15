# src/steer_generation/llm_client.py

# Built-in packages
import os
import re
# External packages
from abc import ABC, abstractmethod
import time
from openai import OpenAI, AzureOpenAI
import anthropic
import openai
# Local packages

# Constants
instruction_prompts = {
    
    'verify_template': 'Return the output as JSON in the format [{"question_id": 0, "sub_id": 0, "options": [{"option_id": 0, "option_text": option_0_text, "correct_answer": is_correct}, {"option_id": 1, "option_text": option_1_text, "correct_answer": is_correct}]}]. If there are more parts to the question then continue in the same format, updating the sub_id.',

    'generate_seed': 'Return the output as JSON in the format {"question_1": question_text, "options_1": [option_1_text, option_2_text], "answer_1": correct_option_text}. If more than one question then continue in the same format.',

    'generate_question': 'Return the output as JSON in the format {"question_1": question_text, "options_1": [option_1_text, option_2_text], "answer_1": correct_option_text}. If more than one question then continue in the same format.'
    # 'Return the output as JSON in the format [{"question_id": 0, "sub_id": 0, "options": [{"option_id": 0, "option_text": option_0_text, "correct_answer": is_correct}, {"option_id": 1, "option_text": option_1_text, "correct_answer": is_correct}]}]. If there are more parts to the question then continue in the same format, updating the sub_id.',
}

response_prompts = {
    'explanation': "Let's think step by step. Explain why the answer is correct without referencing any option letters. For instance, do not say 'Option A is correct because...' or anything of that form.",

    'validation': "Let's think step by step. Explain your reasoning and then answer the question with the correct option letter. Format the response as 'Explanation: [explanation] Correct Answer: [option letter]'. BEGIN YOUR RESPONSE WITH 'Explanation:'.",
}

# max_completion_tokens needs 2024-09-01-preview or later; reasoning models need 2025-03-01-preview or later.
latest_azure_api_version = "2025-04-01-preview"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Do NOT override user environment; read default once and reuse
DEFAULT_GENERATION_MODEL = os.environ.get('GENERATION_MODEL', 'gpt-5.4')

# Parameters that some models (OpenAI/Azure reasoning models, newer Claude models) reject.
OPTIONAL_PARAMS = ("temperature", "top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "logprobs", "response_format")


def _create_with_retry(create, request_kwargs, error_cls):
    """Call create(**request_kwargs); on a 400 naming an optional parameter, retry once without it.

    Users can type any model name, so instead of listing which models accept which parameters we
    send them and drop the one the API says it does not support.
    """
    try:
        return create(**request_kwargs)
    except error_cls as exc:
        message = str(exc)
        if 'max_completion_tokens' in request_kwargs and 'max_completion_tokens' in message:
            # Older OpenAI-compatible servers only know max_tokens.
            retry = dict(request_kwargs)
            retry['max_tokens'] = retry.pop('max_completion_tokens')
            return create(**retry)
        rejected = [p for p in OPTIONAL_PARAMS if p in request_kwargs and re.search(rf"\b{p}\b", message)]
        if not rejected:
            raise
        return create(**{k: v for k, v in request_kwargs.items() if k not in rejected})


# Generation object class

# TODO: need to complete this.
class GenerationObject:
    def __init__(self, input_messages, output_messages, **kwargs):
        self.input_messages = input_messages
        self.output_messages = output_messages
        for key in kwargs:
            self[key] = kwargs[key]

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"'{key}' not found in Dataset")
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value



# Base class for all providers
class BaseProvider(ABC):
    @abstractmethod
    def __init__(self, api_key, **kwargs):
        pass

    def build_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', return_type = 'completion', **kwargs):
        messages = apply_prompt_template(user_prompts, system_prompts, assistant_prompts)
        if request_type in instruction_prompts:
            messages = insert_system_prompt(messages, instruction_prompts[request_type])
        elif request_type in response_prompts:
            messages = insert_user_prompt(messages, response_prompts[request_type])
        elif request_type == 'completion':
            pass
        else:
            raise ValueError(f"Unknown request type: {request_type}. Please use 'completion', 'explanation', 'generate_seed', 'generate_question', 'validation', or 'verify_template'.")
        

        if 'response_format' not in kwargs and request_type in instruction_prompts:
            kwargs['response_format'] = {'type': 'json_object'}

        return {
            'messages': messages,
            'request_type': request_type,
            'return_type': return_type,
            **kwargs
        }

    def make_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', return_type = 'generation_object', **kwargs):

        kwargs = self.build_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type = request_type, return_type = return_type, **kwargs)

        kwargs.pop('request_type')
        kwargs.pop('return_type')
        
        if return_type == 'generation_object':
            return self.get_completion(**kwargs)
        elif return_type == 'string':
            return self.get_completion_content(**kwargs)
        else:
            raise ValueError(f"Unknown return type: {return_type}. Please use 'generation_object' or 'string'.")
    
    @abstractmethod
    def get_completion_content(self):
        pass

    @abstractmethod
    def get_completion(self, messages, **kwargs):
        pass

    @abstractmethod
    def get_content(self, generation_object):
        pass




class OpenAIProvider(BaseProvider):
    def __init__(self, api_key, **kwargs):
        self.handler = OpenAI(api_key=api_key)
    

    def make_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', **kwargs):
        return super().make_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type=request_type, **kwargs)

    def _create(self, messages, model, **kwargs):
        # max_tokens is deprecated on chat completions and rejected by reasoning models;
        # max_completion_tokens is accepted by OpenAI, Azure, Gemini and vLLM.
        if 'max_tokens' in kwargs:
            kwargs.setdefault('max_completion_tokens', kwargs.pop('max_tokens'))
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return _create_with_retry(self.handler.chat.completions.create, {"model": model, "messages": messages, **kwargs}, openai.BadRequestError)

    def get_completion_content(
        self,
        messages,
        model = DEFAULT_GENERATION_MODEL,
        temperature = None,
        seed = None,
        **kwargs,
    ):
        response = self._create(messages, model, temperature=temperature, seed=seed, **kwargs)
        return response.choices[0].message.content

    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_GENERATION_MODEL,
        **kwargs,
    ):
        return self._create(messages, model, **kwargs)

    def get_content(self, generation_object):
        return generation_object.choices[0].message.content
    

# Concrete provider classes
class AzureProvider(OpenAIProvider):
    def __init__(self, api_key, endpoint, api_version=os.environ.get('api_version', latest_azure_api_version)):
        self.handler = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    def make_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', **kwargs):
        return super().make_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type=request_type, **kwargs)

    def get_content(self, generation_object):
        return generation_object.choices[0].message.content

def final_answer(text):
    """The answer part of a reply from a local server that leaves the model's reasoning in the text: after gpt-oss's
    final channel (mlx_lm.server returns the raw harmony format), or after a closing </think>."""
    if not text:
        return text
    marker = '<|channel|>final<|message|>'
    if marker in text:
        text = text.rsplit(marker, 1)[1]
        for end in ('<|return|>', '<|end|>'):
            text = text.split(end, 1)[0]
        return text.strip()
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    return text


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAI chat API at another base URL (Gemini's OpenAI endpoint, vLLM, Together, mlx_lm.server, ...)."""
    def __init__(self, api_key, endpoint, **kwargs):
        self.handler = OpenAI(api_key=api_key, base_url=endpoint)

    def get_completion_content(self, messages, model = DEFAULT_GENERATION_MODEL, temperature = None, seed = None, **kwargs):
        return final_answer(super().get_completion_content(messages, model, temperature=temperature, seed=seed, **kwargs))

    def get_content(self, generation_object):
        return final_answer(generation_object.choices[0].message.content)


def _anthropic_messages(messages, response_format=None):
    """Split OpenAI-style messages into Anthropic's system text and user/assistant turns."""
    system = [m['content'] for m in messages if m.get('role') == 'system']
    turns = [{'role': m['role'], 'content': m['content']} for m in messages if m.get('role') in ('user', 'assistant')]
    if isinstance(response_format, dict) and response_format.get('type') in ('json_object', 'json_schema'):
        system.append('Return only a JSON object, with no text before or after it.')
    return '\n\n'.join(str(s) for s in system if s), turns


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key, **kwargs):
        self.handler = anthropic.Anthropic(api_key=api_key)

    def make_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', **kwargs):
        return super().make_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type=request_type, **kwargs)
    
    def get_completion_content(
        self, 
        messages,
        model = DEFAULT_GENERATION_MODEL,
        temperature = None,
        seed = None,
        **kwargs,
        # max_tokens = None,
        # stop = None,
        # tools = None,
    ):
        # The Messages API takes the system prompt separately and has no response_format or seed.
        system, turns = _anthropic_messages(messages, kwargs.pop('response_format', None))
        request_kwargs = {"model": model, "messages": turns, "max_tokens": kwargs.pop('max_tokens', 16000), **kwargs}
        if system:
            request_kwargs["system"] = system
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        # Newer Claude models reject non-default temperature/top_p/top_k; retry once without the rejected one.
        response = _create_with_retry(self.handler.messages.create, request_kwargs, anthropic.BadRequestError)
        return ''.join(block.text for block in response.content if getattr(block, 'type', '') == 'text')
    

    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_GENERATION_MODEL,
        **kwargs,
    ):
        system, turns = _anthropic_messages(messages, kwargs.pop('response_format', None))
        if system:
            kwargs['system'] = system
        kwargs.setdefault('max_tokens', 16000)
        kwargs.pop('seed', None)
        return _create_with_retry(self.handler.messages.create, {"model": model, "messages": turns, **kwargs}, anthropic.BadRequestError)

    def get_content(self, generation_object):
        return ''.join(block.text for block in generation_object.content if getattr(block, 'type', '') == 'text')

class DummyProvider(BaseProvider):
    def __init__(self, **kwargs):
        self.dummy = None
    
    def make_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', **kwargs):
        return super().make_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type=request_type, **kwargs)
    
    def get_completion(self, **kwargs):
        time.sleep(1)
        return {'messages': [{'role': 'assistant', 'content': 'This is a dummy completion'}], }
    
    def get_completion_content(self):
        return 'This is a dummy completion'
    
    def get_content(self, generation_object):
        return generation_object['messages'][0]['content']
    




# Provider factory
class ProviderFactory:
    @staticmethod
    def create_provider(provider_name, **kwargs):
        if provider_name.lower() == 'azure':
            return AzureProvider(**kwargs)
        elif provider_name.lower() == 'openai':
            return OpenAIProvider(**kwargs)
        elif provider_name.lower() == 'anthropic':
            return AnthropicProvider(**kwargs)
        elif provider_name.lower() == 'openai_compatible':
            return OpenAICompatibleProvider(**kwargs)
        elif provider_name.lower() == 'dummy':
            return DummyProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

# Client class
class LanguageModelClient:
    def __init__(self, provider_name='OPENAI', api_key=None, endpoint=None): # TODO: change this back
        name = provider_name.lower()
        if name == 'gemini':
            # Gemini through its OpenAI-compatible endpoint.
            api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ['GOOGLE_API_KEY']
            endpoint = endpoint or GEMINI_OPENAI_BASE_URL
            provider_name = 'openai_compatible'
        elif name == 'openai_compatible':
            # Any server that speaks the OpenAI chat API (vLLM, Together, ...); a key is optional.
            api_key = api_key or os.environ.get('OPENAI_COMPATIBLE_API_KEY') or 'EMPTY'
            endpoint = endpoint or os.environ['OPENAI_COMPATIBLE_BASE_URL']
        elif not api_key and name != 'dummy':
            provider_api_name = f"{provider_name.upper()}_OPENAI" if name == 'azure' else provider_name.upper()
            api_key = os.environ[f"{provider_api_name}_API_KEY"]
        if name == 'azure':
            if not endpoint:
                endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

        self.provider = ProviderFactory.create_provider(
            provider_name,
            api_key=api_key,
            endpoint=endpoint
        )

    def build_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', return_type = 'generation_object', **kwargs):
        return self.provider.build_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type = request_type, return_type = return_type, **kwargs)

    def make_request(self, user_prompts, system_prompts = None, assistant_prompts = None, request_type = 'completion', return_type = 'generation_object', **kwargs) -> dict:
        return self.provider.make_request(user_prompts, system_prompts = system_prompts, assistant_prompts = assistant_prompts, request_type = request_type, return_type = return_type, **kwargs)
    
    def get_content(self, generation_object) -> str:
        return self.provider.get_content(generation_object)
    
def apply_prompt_template(user_prompts, system_prompts, assistant_prompts):
    """
    Generates a list of messages based on the given user prompts, system prompts, and assistant prompts.

    Args:
        user_prompts (list): A list of user prompts.
        system_prompts (list): A list of system prompts.
        assistant_prompts (list): A list of assistant prompts.

    Returns:
        list: A list of messages containing the prompts from all roles (system, user, assistant).
    """
    if isinstance(user_prompts, str):
        user_prompts = [user_prompts]
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts]
    if isinstance(assistant_prompts, str):
        assistant_prompts = [assistant_prompts]    
    if assistant_prompts and len(user_prompts) > len(assistant_prompts) + 1:
        raise ValueError("Two consecutive user prompts without an assistant prompt.")
    messages = []
    for i, user_prompt in enumerate(user_prompts):
        if system_prompts:
            messages.append({'role': 'system', 'content': system_prompts[i]})
        messages.append({'role': 'user', 'content': user_prompt})
        if assistant_prompts and i < len(assistant_prompts):
            messages.append({'role': 'assistant', 'content': assistant_prompts[i]})
    return messages



def insert_system_prompt(messages, system_prompt, prompt_type='openai'):
    if messages[0]['role'] != 'system' and prompt_type != 'anthropic':
        messages.insert(0, {
            'role': 'system',
            'content': system_prompt
        })
    elif messages[0]['role'] == 'system':
        messages[0]['content'] += '\n\n' + system_prompt
    else:
        if messages[0]['role'] != 'user':
            raise ValueError("If there is no system prompt, then the first message should be with the role user.")
        messages[0]['content'] += '\n\n' + system_prompt
    
    return messages

def insert_user_prompt(messages, user_prompt):
    if messages[-1]['role'] != 'user':
        messages.append({
            'role': 'user',
            'content': user_prompt
        })
    else:
        messages[-1]['content'] += '\n\n' + user_prompt
    return messages
