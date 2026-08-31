import pytest

from forge.llm.provider_policy import (
    COMPATIBLE_CHAT_POLICY,
    DEEPSEEK_CHAT_POLICY,
    OPENAI_RESPONSES_POLICY,
    provider_policy,
)


def test_provider_policies_keep_reasoning_rules_at_the_transport_boundary() -> None:
    assert OPENAI_RESPONSES_POLICY.replay_chat_reasoning_content is False
    assert COMPATIBLE_CHAT_POLICY.controls_chat_thinking_per_turn is False
    assert DEEPSEEK_CHAT_POLICY.replay_chat_reasoning_content is True
    assert DEEPSEEK_CHAT_POLICY.controls_chat_thinking_per_turn is True


def test_provider_policy_rejects_an_unreviewed_provider_api_pair() -> None:
    with pytest.raises(ValueError, match="unsupported provider/API combination"):
        provider_policy(provider="deepseek", api_mode="responses")
