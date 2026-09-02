import pytest
from pathlib import Path

from dbagent.config import ConfigurationError, DBAgentConfig
from dbagent.model_presets import (
    DEEPSEEK_BASE_URL,
    default_deepseek_key_file,
    load_deepseek_api_key,
    model_presets,
    resolve_model_selection,
)


def _configured() -> DBAgentConfig:
    return DBAgentConfig(
        openai_api_key="configured-provider-secret",
        model="gpt-5.6-luna",
        reasoning_effort="high",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )


def test_deepseek_preset_reads_environment_key_only_for_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBAGENT_DEEPSEEK_API_KEY", "deepseek-test-secret")
    configured = _configured()

    result = resolve_model_selection(
        "deepseek-flash",
        active_config=configured,
        startup_config=configured,
    )

    assert result.model == "deepseek-v4-flash"
    assert result.openai_api_key == "deepseek-test-secret"
    assert result.base_url == DEEPSEEK_BASE_URL
    assert result.api_mode == "chat_completions"
    assert result.reasoning_effort == "high"
    assert result.provider == "deepseek"
    assert "deepseek-test-secret" not in repr(result)


def test_configured_preset_restores_startup_provider_after_deepseek_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBAGENT_DEEPSEEK_API_KEY", "deepseek-test-secret")
    configured = _configured()
    deepseek = resolve_model_selection(
        "deepseek-pro",
        active_config=configured,
        startup_config=configured,
    )

    restored = resolve_model_selection(
        "terra",
        active_config=deepseek,
        startup_config=configured,
    )

    assert restored.model == "gpt-5.6-terra"
    assert restored.openai_api_key == "configured-provider-secret"
    assert restored.base_url == "https://provider.example/v1"
    assert restored.provider == "configured"


def test_deepseek_selection_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DBAGENT_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        "dbagent.model_presets.default_deepseek_key_file",
        lambda: Path("missing-deepseek-key.txt"),
    )

    with pytest.raises(ConfigurationError, match="DBAGENT_DEEPSEEK_API_KEY"):
        resolve_model_selection(
            "deepseek-flash",
            active_config=_configured(),
            startup_config=_configured(),
        )


def test_deepseek_key_file_is_used_when_no_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "deepseek-key.txt"
    path.write_text("local-deepseek-key\n", encoding="utf-8")
    monkeypatch.delenv("DBAGENT_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("dbagent.model_presets.default_deepseek_key_file", lambda: path)

    assert load_deepseek_api_key() == "local-deepseek-key"


def test_default_deepseek_key_file_is_the_repository_local_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DBAGENT_DEEPSEEK_KEY_FILE", raising=False)
    monkeypatch.setattr(
        "dbagent.model_presets.project_deepseek_key_path",
        lambda: tmp_path / "api_key.txt",
    )

    assert default_deepseek_key_file() == tmp_path / "api_key.txt"


def test_model_presets_show_current_supported_deepseek_choices() -> None:
    aliases = {item.alias for item in model_presets()}

    assert {"luna", "terra", "sol", "deepseek-flash", "deepseek-pro"} <= aliases
