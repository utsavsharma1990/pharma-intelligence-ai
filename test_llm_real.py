"""
Real end-to-end LLM call. Uses your actual ANTHROPIC_API_KEY from .env.
This is the moment we confirm the LLM layer works against a live API.
"""

from src.core.llm import LLMMessage, get_llm_provider

provider = get_llm_provider()
print(f"Using provider: {type(provider).__name__} ({provider.model_name})\n")

# 1. Plain completion
print("=" * 60)
print("1. Free-form completion")
print("=" * 60)
text = provider.complete(
    messages=[LLMMessage(role="user",
                         content="In one sentence, what is a Phase III clinical trial?")],
    system="You are a clinical research expert. Be concise.",
)
print(text)

# 2. Structured output — same pattern the Supervisor agent will use
print("\n" + "=" * 60)
print("2. Structured output (routing decision simulation)")
print("=" * 60)
result = provider.complete_structured(
    messages=[LLMMessage(role="user",
                         content="What were the adverse events in the Pembrolizumab trial?")],
    system=("Classify the user's question and decide which specialist agent "
            "should answer. Available agents: 'search', 'comparative', 'safety'."),
    schema={
        "type": "object",
        "properties": {
            "agent":  {"type": "string", "enum": ["search", "comparative", "safety"]},
            "reason": {"type": "string"},
        },
        "required": ["agent", "reason"],
    },
)
print(f"Routing decision: {result}")
print(f"\n✅ LLM layer works.")