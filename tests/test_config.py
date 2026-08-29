import pytest

from forge.config import ConfigurationError, ForgeConfig


def test_defaults_do_not_require_an_api_key() -> None:
    config = ForgeConfig.from_env({})

    assert config.openai_api_key is None
    assert config.model == "gpt-5.6-sol"
    assert config.reasoning_effort == "medium"


def test_values_are_loaded_from_the_environment_mapping() -> None:
    config = ForgeConfig.from_env(
        {
            "OPENAI_API_KEY": "from-environment",
            "FORGE_MODEL": "custom-model",
            "FORGE_REASONING_EFFORT": "HIGH",
        }
    )

    assert config.openai_api_key == "from-environment"
    assert config.model == "custom-model"
    assert config.reasoning_effort == "high"
    assert "from-environment" not in repr(config)


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="FORGE_MODEL must not be empty"):
        ForgeConfig.from_env({"FORGE_MODEL": "  "})


def test_unknown_reasoning_effort_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="FORGE_REASONING_EFFORT"):
        ForgeConfig.from_env({"FORGE_REASONING_EFFORT": "extreme"})
