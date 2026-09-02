import pytest

from dbagent.config import ConfigurationError, DBAgentConfig


def test_defaults_do_not_require_an_api_key() -> None:
    config = DBAgentConfig.from_env({})

    assert config.openai_api_key is None
    assert config.model == "gpt-5.6-sol"
    assert config.reasoning_effort == "medium"
    assert config.base_url is None
    assert config.api_mode == "responses"
    assert config.provider == "configured"


def test_values_are_loaded_from_the_environment_mapping() -> None:
    config = DBAgentConfig.from_env(
        {
            "OPENAI_API_KEY": "from-environment",
            "DBAGENT_BASE_URL": "https://provider.example/v1",
            "DBAGENT_API_MODE": "CHAT_COMPLETIONS",
            "DBAGENT_PROVIDER": "configured",
            "DBAGENT_MODEL": "custom-model",
            "DBAGENT_REASONING_EFFORT": "HIGH",
        }
    )

    assert config.openai_api_key == "from-environment"
    assert config.model == "custom-model"
    assert config.reasoning_effort == "high"
    assert config.base_url == "https://provider.example/v1"
    assert config.api_mode == "chat_completions"
    assert config.provider == "configured"
    assert "from-environment" not in repr(config)


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="DBAGENT_MODEL must not be empty"):
        DBAgentConfig.from_env({"DBAGENT_MODEL": "  "})


def test_unknown_reasoning_effort_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="DBAGENT_REASONING_EFFORT"):
        DBAgentConfig.from_env({"DBAGENT_REASONING_EFFORT": "extreme"})


def test_invalid_base_url_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="DBAGENT_BASE_URL"):
        DBAgentConfig.from_env({"DBAGENT_BASE_URL": "provider.example/v1"})


def test_unknown_api_mode_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="DBAGENT_API_MODE"):
        DBAgentConfig.from_env({"DBAGENT_API_MODE": "legacy"})


def test_deepseek_provider_requires_chat_completions() -> None:
    with pytest.raises(ConfigurationError, match="requires DBAGENT_API_MODE"):
        DBAgentConfig.from_env({"DBAGENT_PROVIDER": "deepseek"})
