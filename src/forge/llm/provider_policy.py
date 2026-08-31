"""Explicit provider-specific transport rules.

The local agent loop deliberately does not know which model provider it uses.
This module owns only API-envelope and reasoning-continuation differences.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    """Rules needed at the model-transport boundary, never by AgentLoop."""

    provider: str
    api_mode: str
    replay_chat_reasoning_content: bool
    controls_chat_thinking_per_turn: bool
    requires_assistant_content_for_tool_calls: bool
    label: str
    supports_reasoning_with_tools: bool = True
    stability: str = "stable"

    @property
    def tool_turn_policy(self) -> str:
        if self.controls_chat_thinking_per_turn:
            return "thinking disabled for the full local agent task"
        return "provider-default reasoning behavior"

    @property
    def capability_summary(self) -> str:
        """Return a user-facing capability summary without exposing credentials."""

        reasoning = "reasoning with tools" if self.supports_reasoning_with_tools else "tool reasoning disabled"
        return f"{self.label}; {reasoning}; {self.stability}"


OPENAI_RESPONSES_POLICY = ProviderPolicy(
    provider="configured",
    api_mode="responses",
    replay_chat_reasoning_content=False,
    controls_chat_thinking_per_turn=False,
    requires_assistant_content_for_tool_calls=False,
    label="Responses API",
)
COMPATIBLE_CHAT_POLICY = ProviderPolicy(
    provider="configured",
    api_mode="chat_completions",
    replay_chat_reasoning_content=False,
    controls_chat_thinking_per_turn=False,
    requires_assistant_content_for_tool_calls=False,
    label="Chat Completions compatibility API",
)
DEEPSEEK_CHAT_POLICY = ProviderPolicy(
    provider="deepseek",
    api_mode="chat_completions",
    # Do not make local context depend on unbounded, provider-private
    # reasoning_content replay. The transport keeps thinking disabled for the
    # full task instead.
    replay_chat_reasoning_content=False,
    controls_chat_thinking_per_turn=True,
    requires_assistant_content_for_tool_calls=True,
    label="DeepSeek Chat Completions API",
    supports_reasoning_with_tools=False,
    stability="experimental compatibility",
)


def provider_policy(*, provider: str, api_mode: str) -> ProviderPolicy:
    """Select a reviewed transport policy for one configured provider."""

    if provider == "deepseek" and api_mode == "chat_completions":
        return DEEPSEEK_CHAT_POLICY
    if provider == "configured" and api_mode == "responses":
        return OPENAI_RESPONSES_POLICY
    if provider == "configured" and api_mode == "chat_completions":
        return COMPATIBLE_CHAT_POLICY
    raise ValueError(
        f"unsupported provider/API combination: provider={provider!r}, "
        f"api_mode={api_mode!r}"
    )
