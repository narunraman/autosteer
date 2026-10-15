"""Provider request construction and forgiving JSON parsing (no network)."""
from types import SimpleNamespace
from unittest import mock

import anthropic
import httpx
import openai
import pytest

from autosteer import llm_client
from autosteer.json_utils import parse_model_json


# ---- parse_model_json ----------------------------------------------------------------------------
def test_plain_json():
    assert parse_model_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert parse_model_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_model_json('Here:\n```\n{"a": 1}\n```\nDone.') == {"a": 1}


def test_prose_wrapped_json():
    assert parse_model_json('Sure! {"a": {"b": "}"}} Hope that helps {"c": 2}') == {"a": {"b": "}"}}


def test_trailing_commas():
    assert parse_model_json('{"a": [1, 2,], "b": "x,]",}') == {"a": [1, 2], "b": "x,]"}
    assert parse_model_json('```json\n{"a": 1,}\n```') == {"a": 1}


def test_array_output():
    assert parse_model_json('[{"a": 1}]') == [{"a": 1}]


def test_invalid_json_raises_with_excerpt():
    with pytest.raises(ValueError, match="no json here at all"):
        parse_model_json("no json here at all")
    with pytest.raises(ValueError):
        parse_model_json("")


# ---- OpenAI-style providers ----------------------------------------------------------------------
def _bad_request(cls, message):
    request = httpx.Request("POST", "https://example.invalid/v1")
    return cls(message, response=httpx.Response(400, request=request), body=None)


class FakeCreate:
    def __init__(self, error_cls=None, reject=None):
        self.calls, self.error_cls, self.reject = [], error_cls, reject

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject and self.reject in kwargs:
            raise _bad_request(self.error_cls, f"Unsupported parameter: '{self.reject}' is not supported with this model.")
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _openai_provider(create, cls=llm_client.OpenAIProvider):
    provider = cls.__new__(cls)
    provider.handler = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return provider


def test_openai_request_uses_max_completion_tokens_and_json_mode():
    create = FakeCreate()
    provider = _openai_provider(create)
    out = provider.make_request("hi", request_type="generate_seed", return_type="string", model="gpt-x", max_tokens=100)
    assert out == "ok"
    sent = create.calls[0]
    assert sent["model"] == "gpt-x"
    assert sent["max_completion_tokens"] == 100 and "max_tokens" not in sent
    assert sent["response_format"] == {"type": "json_object"}
    assert "temperature" not in sent and "seed" not in sent
    assert sent["messages"][0]["role"] == "system"


def test_openai_drops_temperature_rejected_by_reasoning_model():
    create = FakeCreate(openai.BadRequestError, reject="temperature")
    provider = _openai_provider(create)
    provider.make_request("hi", return_type="string", model="o-reasoner", temperature=0.2, seed=1)
    assert "temperature" in create.calls[0]
    assert len(create.calls) == 2
    assert "temperature" not in create.calls[1] and create.calls[1]["seed"] == 1


def test_openai_unrelated_400_is_raised():
    def create(**kwargs):
        raise _bad_request(openai.BadRequestError, "context length exceeded")

    with pytest.raises(openai.BadRequestError):
        _openai_provider(create).make_request("hi", return_type="string", model="m", temperature=0.2)


def test_max_completion_tokens_falls_back_to_max_tokens_on_old_servers():
    create = FakeCreate(openai.BadRequestError, reject="max_completion_tokens")
    _openai_provider(create).make_request("hi", return_type="string", model="m", max_tokens=5)
    assert create.calls[1]["max_tokens"] == 5 and "max_completion_tokens" not in create.calls[1]


def test_get_completion_passes_kwargs_through():
    create = FakeCreate()
    provider = _openai_provider(create, llm_client.AzureProvider)
    provider.make_request("hi", model="my-deployment", system_prompts="sys")
    assert create.calls[0]["model"] == "my-deployment"
    assert create.calls[0]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_azure_default_api_version_supports_max_completion_tokens():
    assert llm_client.latest_azure_api_version >= "2024-09-01-preview"
    with mock.patch.object(llm_client, "AzureOpenAI") as azure:
        llm_client.AzureProvider(api_key="k", endpoint="https://x.openai.azure.com")
    assert azure.call_args.kwargs["azure_endpoint"] == "https://x.openai.azure.com"
    assert azure.call_args.kwargs["api_version"] >= "2024-09-01-preview"


def test_gemini_uses_openai_compatible_endpoint():
    with mock.patch.object(llm_client, "OpenAI") as client:
        llm_client.LanguageModelClient(provider_name="gemini", api_key="k")
    assert client.call_args.kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_openai_compatible_uses_given_endpoint():
    with mock.patch.object(llm_client, "OpenAI") as client:
        llm_client.LanguageModelClient(provider_name="openai_compatible", api_key="k", endpoint="http://vllm:8000/v1")
    assert client.call_args.kwargs["base_url"] == "http://vllm:8000/v1"


# ---- Anthropic -----------------------------------------------------------------------------------
def _anthropic_provider(create):
    provider = llm_client.AnthropicProvider.__new__(llm_client.AnthropicProvider)
    provider.handler = SimpleNamespace(messages=SimpleNamespace(create=create))
    return provider


def _anthropic_response(**kwargs):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])


def test_anthropic_request_shape():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _anthropic_response()

    out = _anthropic_provider(create).make_request(
        "hi", system_prompts="be brief", request_type="generate_seed", return_type="string", model="claude-opus-5-5", seed=3)
    assert out == "ok"
    sent = calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["max_tokens"] > 0
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert "be brief" in sent["system"] and "JSON" in sent["system"]
    for absent in ("temperature", "seed", "response_format"):
        assert absent not in sent


def test_anthropic_get_completion_has_max_tokens_and_no_seed():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _anthropic_response()

    _anthropic_provider(create).make_request("hi", model="claude-sonnet-5-5", seed=1, response_format={"type": "json_object"})
    assert calls[0]["max_tokens"] > 0 and "seed" not in calls[0] and "response_format" not in calls[0]


def test_anthropic_drops_rejected_temperature():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise _bad_request(anthropic.BadRequestError, "`temperature` may only be set to 1 for this model")
        return _anthropic_response()

    out = _anthropic_provider(create).make_request("hi", return_type="string", model="claude-opus-5-5", temperature=0.0)
    assert out == "ok" and len(calls) == 2 and "temperature" not in calls[1]
