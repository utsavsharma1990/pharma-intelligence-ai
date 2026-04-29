"""
Tests for the LLM provider layer.

We mock the underlying SDKs so tests don't burn API credits or require
network access. Real end-to-end tests happen in the demo step below.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.llm import (
    AnthropicProvider,
    EchoProvider,
    LLMMessage,
    LLMProvider,
    OpenAIProvider,
)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _mock_anthropic_text_response(text: str):
    """Build a fake Anthropic Messages API response."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _mock_anthropic_tool_response(tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "respond"
    block.input = tool_input
    resp = MagicMock()
    resp.content = [block]
    return resp


@patch("anthropic.Anthropic")
def test_anthropic_complete_returns_text(MockAnthropic):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_text_response("hi there")
    MockAnthropic.return_value = mock_client

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-5")
    result = provider.complete([LLMMessage(role="user", content="hello")])
    assert result == "hi there"


@patch("anthropic.Anthropic")
def test_anthropic_complete_passes_system_prompt(MockAnthropic):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_text_response("ok")
    MockAnthropic.return_value = mock_client

    provider = AnthropicProvider(api_key="sk-test", model="m")
    provider.complete([LLMMessage(role="user", content="x")], system="be helpful")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "be helpful"


@patch("anthropic.Anthropic")
def test_anthropic_structured_output(MockAnthropic):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_tool_response(
        {"agent": "safety", "reason": "AE question"}
    )
    MockAnthropic.return_value = mock_client

    provider = AnthropicProvider(api_key="sk-test", model="m")
    result = provider.complete_structured(
        messages=[LLMMessage(role="user", content="q")],
        schema={"type": "object", "properties": {"agent": {"type": "string"}}},
    )
    assert result == {"agent": "safety", "reason": "AE question"}


def test_anthropic_empty_api_key_raises():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is empty"):
        AnthropicProvider(api_key="", model="m")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _mock_openai_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("openai.OpenAI")
def test_openai_complete_returns_text(MockOpenAI):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response("hello!")
    MockOpenAI.return_value = mock_client

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
    result = provider.complete([LLMMessage(role="user", content="hi")])
    assert result == "hello!"


@patch("openai.OpenAI")
def test_openai_structured_output_parses_json(MockOpenAI):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        '{"agent": "search", "reason": "general"}'
    )
    MockOpenAI.return_value = mock_client

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
    result = provider.complete_structured(
        messages=[LLMMessage(role="user", content="q")],
        schema={"type": "object", "properties": {"agent": {"type": "string"}}},
    )
    assert result == {"agent": "search", "reason": "general"}


def test_openai_empty_api_key_raises():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is empty"):
        OpenAIProvider(api_key="", model="m")


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

def test_llm_provider_is_abstract():
    """Subclassing without implementing abstract methods should fail."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_both_providers_have_model_name():
    with patch("anthropic.Anthropic"):
        a = AnthropicProvider(api_key="x", model="claude-test")
        assert a.model_name == "claude-test"
    with patch("openai.OpenAI"):
        o = OpenAIProvider(api_key="x", model="gpt-test")
        assert o.model_name == "gpt-test"


# ---------------------------------------------------------------------------
# EchoProvider (offline dev mode)
# ---------------------------------------------------------------------------

def test_echo_provider_complete_keyword_match():
    p = EchoProvider()
    out = p.complete([LLMMessage(role="user",
                                  content="What were the adverse events?")])
    assert "pneumonitis" in out.lower()


def test_echo_provider_complete_default_fallback():
    p = EchoProvider()
    out = p.complete([LLMMessage(role="user", content="random unmatched query")])
    assert "EchoProvider" in out


def test_echo_provider_structured_routes_safety_query():
    p = EchoProvider()
    schema = {
        "type": "object",
        "properties": {
            "agent": {"type": "string",
                       "enum": ["search", "comparative", "safety"]},
            "reason": {"type": "string"},
        },
        "required": ["agent"],
    }
    result = p.complete_structured(
        messages=[LLMMessage(role="user",
                              content="What adverse events were reported?")],
        schema=schema,
    )
    assert result["agent"] == "safety"
    assert "reason" in result


def test_echo_provider_structured_routes_compare_query():
    p = EchoProvider()
    schema = {
        "type": "object",
        "properties": {
            "agent": {"type": "string",
                       "enum": ["search", "comparative", "safety"]},
        },
    }
    result = p.complete_structured(
        messages=[LLMMessage(role="user",
                              content="Compare drug A vs drug B")],
        schema=schema,
    )
    assert result["agent"] == "comparative"


def test_echo_provider_custom_responses():
    p = EchoProvider(responses={"hello": "world"})
    out = p.complete([LLMMessage(role="user", content="hello there")])
    assert out == "world"