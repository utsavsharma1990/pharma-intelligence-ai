"""
LLM provider abstraction.

Same Strategy pattern as vector_store.py and embeddings.py:
ABC defines the contract; implementations swap via config.

Why direct SDKs (not LangChain ChatAnthropic / ChatOpenAI wrappers)?
  - LangChain's wrappers add an abstraction layer we don't control
  - Errors get swallowed/transformed in unhelpful ways
  - Direct SDKs are simpler to debug, reason about, and version-pin
  - We'll still use LangGraph for ORCHESTRATION; each node calls our
    LLMProvider directly. Best of both worlds.

Why two methods (complete + complete_structured)?
  - `complete`: free-form generation. Used by Synthesizer agent.
  - `complete_structured`: enforces JSON output matching a schema. Used by
    Supervisor for routing decisions and by agents that return structured
    answers. Anthropic does this with tool_use; OpenAI does it with
    response_format / function calling. We hide that asymmetry behind
    a single method.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """
    A single message in a conversation.
    role: 'user' | 'assistant' | (system messages handled separately)
    """
    role: str
    content: str


# ---------------------------------------------------------------------------
# The ABC
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract LLM. All providers implement this contract."""

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Free-form text generation."""
        ...

    @abstractmethod
    def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        """
        Generate output matching a JSON schema.

        `schema` is a JSON Schema-style dict, e.g.:
            {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": ["safety", "search"]},
                    "reason": {"type": "string"}
                },
                "required": ["agent"]
            }

        Returns the parsed dict. Raises if the LLM produces non-conforming output
        after retries (this is rare in practice with modern models).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """For logging / debugging — which model is actually being called."""
        ...


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude via the official SDK.

    Structured output uses tool_use under the hood: we declare a single
    "respond" tool with the user's schema as its input_schema, then force
    the model to call that tool. The tool's input is our parsed JSON.
    This is more reliable than asking the model to "just output JSON" —
    Anthropic's tool-use mode has hard schema validation server-side.
    """

    def __init__(self, api_key: str, model: str):
        # Lazy import: keeps `from src.core.llm import LLMProvider` cheap
        from anthropic import Anthropic
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is empty. "
                "Set it in .env (LLM_PROVIDER=anthropic requires this)."
            )
        self._client = Anthropic(api_key=api_key)
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model":       self._model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        # resp.content is a list of content blocks; we take all text blocks
        return "".join(
            block.text for block in resp.content if block.type == "text"
        )

    def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        # Wrap user's schema in a synthetic tool. Forcing tool_choice to this
        # specific tool guarantees the model's output goes through Anthropic's
        # schema validator — more reliable than free-form JSON.
        tool = {
            "name":         "respond",
            "description":  "Respond with structured output matching the schema.",
            "input_schema": schema,
        }
        kwargs: dict[str, Any] = {
            "model":       self._model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    [{"role": m.role, "content": m.content} for m in messages],
            "tools":       [tool],
            "tool_choice": {"type": "tool", "name": "respond"},
        }
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        # Find the tool_use block and return its parsed input
        for block in resp.content:
            if block.type == "tool_use" and block.name == "respond":
                return block.input  # already a dict, not a JSON string
        raise RuntimeError(
            f"Anthropic did not return tool_use for the 'respond' tool. "
            f"Got: {resp.content}"
        )


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    OpenAI via the official SDK.

    Structured output uses response_format={"type": "json_schema", ...},
    which OpenAI's API enforces server-side just like Anthropic's tool_use.
    """

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty. "
                "Set it in .env (LLM_PROVIDER=openai requires this)."
            )
        self._client = OpenAI(api_key=api_key)
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        # OpenAI takes system as a message, not a separate parameter
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend({"role": m.role, "content": m.content} for m in messages)

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend({"role": m.role, "content": m.content} for m in messages)

        # OpenAI requires the schema to declare additionalProperties:false at the top
        # for strict JSON schema mode. We wrap defensively.
        strict_schema = dict(schema)
        if strict_schema.get("type") == "object":
            strict_schema.setdefault("additionalProperties", False)

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name":   "response",
                    "schema": strict_schema,
                    "strict": True,
                },
            },
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

# ---------------------------------------------------------------------------
# Echo / Fake provider for offline development
# ---------------------------------------------------------------------------

class EchoProvider(LLMProvider):
    """
    Fake LLM provider for offline development and testing.

    Returns deterministic, scripted responses based on simple keyword matching
    on the input. Lets you build and run the full agent system end-to-end
    without spending API credits.

    Usage:
        # In .env:
        LLM_PROVIDER=echo

        # Or programmatically:
        provider = EchoProvider()

    Two response modes:
      - complete(): keyword-matches the last user message to pick a canned reply
      - complete_structured(): inspects the schema and synthesizes a valid response

    This is NOT an LLM. It cannot reason, summarize, or generalize. It exists
    purely so the AGENT WIRING (LangGraph, routing, retrieval -> prompt -> output)
    can be developed and tested without network access or credits.
    """

    def __init__(self, responses: Optional[dict[str, str]] = None):
        # Keyword -> canned response. First key found in the user message wins.
        # The agents we build can override these by passing custom responses.
        self._responses: dict[str, str] = responses or {
            "adverse":     "Based on retrieved AE data, common adverse events include pneumonitis (3.0%), colitis (2.0%), and fatigue (40.0%). Serious events are rare but include immune-related toxicities.",
            "safety":      "The safety profile shows expected immune-related events. Most common: fatigue, nausea. Serious: pneumonitis, colitis.",
            "eligib":      "Eligibility requires adults 18+, ECOG 0-1, no prior anti-PD-1 therapy, no active brain metastases.",
            "compare":     "Comparing the trials: Trial A (Pembrolizumab) had 3% pneumonitis vs Trial B (Nivolumab) at 3.1%. Both showed similar OS benefit.",
            "phase":       "This is a Phase 3 study with planned enrollment of 800 participants.",
            "primary":     "The primary outcome is Overall Survival (OS), measured up to 3 years from randomization.",
            "endpoint":    "Primary endpoint: OS. Secondary endpoints include PFS and ORR.",
            "sponsor":     "The trial is sponsored by Merck Sharp & Dohme LLC (industry sponsor).",
            "trial":       "Found 3 matching clinical trials in the index. The most relevant is NCT05123456 (Phase 3, RECRUITING).",
        }
        self._default = (
            "[EchoProvider] I am a deterministic fake LLM for offline dev. "
            "Real responses require a funded ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )

    @property
    def model_name(self) -> str:
        return "echo-fake-llm"

    def complete(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        # Find the last user message and keyword-match against canned replies
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        ).lower()
        for keyword, reply in self._responses.items():
            if keyword in last_user:
                return reply
        return self._default

    def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        """
        Synthesize a schema-valid dict by inspecting the schema's properties.
        For routing decisions (the most common use case), we keyword-match
        the user query against likely agent names in the enum.
        """
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        ).lower()

        result: dict[str, Any] = {}
        properties = schema.get("properties", {})

        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")

            # If the property has an enum, try to keyword-match the user query
            # against the enum values. This is how routing decisions work.
            if "enum" in prop_def:
                enum_values = prop_def["enum"]
                # First-pass: substring match
                matched = next(
                    (v for v in enum_values
                     if isinstance(v, str) and v.lower() in last_user),
                    None,
                )
                # Second-pass: keyword routing for common agent names
                if matched is None:
                    if any(k in last_user for k in ("adverse", "safety", "side effect")):
                        matched = next((v for v in enum_values
                                        if "safe" in v.lower()), enum_values[0])
                    elif any(k in last_user for k in ("compare", "vs", "versus")):
                        matched = next((v for v in enum_values
                                        if "compar" in v.lower()), enum_values[0])
                    elif any(k in last_user for k in ("eligib", "enroll", "criteria")):
                        matched = next((v for v in enum_values
                                        if "eligib" in v.lower()
                                        or "search" in v.lower()), enum_values[0])
                    else:
                        matched = enum_values[0]
                result[prop_name] = matched

            elif prop_type == "string":
                result[prop_name] = f"[echo] {prop_name} for: {last_user[:60]}"
            elif prop_type == "integer":
                result[prop_name] = 0
            elif prop_type == "number":
                result[prop_name] = 0.0
            elif prop_type == "boolean":
                result[prop_name] = False
            elif prop_type == "array":
                result[prop_name] = []
            elif prop_type == "object":
                result[prop_name] = {}

        return result
# ---------------------------------------------------------------------------
# Factory — config-driven provider selection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    from src.core.config import get_settings
    settings = get_settings().llm

    if settings.provider == "echo":
        return EchoProvider()
    if settings.provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    if settings.provider == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    raise ValueError(f"Unknown LLM provider: {settings.provider}")