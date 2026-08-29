import pytest

from forge.config import ConfigurationError, ForgeConfig


def test_defaults_do_not_require_an_api_key() -> None:
    config = ForgeConfig.from_env({})

    assert config.openai_api_key is None
    assert config.model == "gpt-5.6-sol"
    assert config.reasoning_effort == "medium"
    assert config.base_url is None
    assert config.api_mode == "responses"


def test_values_are_loaded_from_the_environment_mapping() -> None:
    config = ForgeConfig.from_env(
        {
            "OPENAI_API_KEY": "from-environment",
            "FORGE_BASE_URL": "https://provider.example/v1",
            "FORGE_API_MODE": "CHAT_COMPLETIONS",
            "FORGE_MODEL": "custom-model",
            "FORGE_REASONING_EFFORT": "HIGH",
        }
    )

    assert config.openai_api_key == "from-environment"
    assert config.model == "custom-model"
    assert config.reasoning_effort == "high"
    assert config.base_url == "https://provider.example/v1"
    assert config.api_mode == "chat_completions"
    assert "from-environment" not in repr(config)


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="FORGE_MODEL must not be empty"):
        ForgeConfig.from_env({"FORGE_MODEL": "  "})


def test_unknown_reasoning_effort_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="FORGE_REASONING_EFFORT"):
        ForgeConfig.from_env({"FORGE_REASONING_EFFORT": "extreme"})


def test_invalid_base_url_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="FORGE_BASE_URL"):
        ForgeConfig.from_env({"FORGE_BASE_URL": "provider.example/v1"})


def test_unknown_api_mode_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="FORGE_API_MODE"):
        ForgeConfig.from_env({"FORGE_API_MODE": "legacy"})
