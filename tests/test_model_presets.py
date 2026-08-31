from pathlib import Path

import pytest

from forge.config import ConfigurationError, ForgeConfig
from forge.model_presets import (
    DEEPSEEK_BASE_URL,
    load_deepseek_api_key,
    model_presets,
    resolve_model_selection,
)


def _configured() -> ForgeConfig:
    return ForgeConfig(
        openai_api_key="configured-provider-secret",
        model="gpt-5.6-luna",
        reasoning_effort="high",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )


def test_deepseek_preset_reads_external_key_only_for_selected_model(tmp_path: Path) -> None:
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("deepseek-test-secret\n", encoding="utf-8")
    configured = _configured()

    result = resolve_model_selection(
        "deepseek-flash",
        active_config=configured,
        startup_config=configured,
        deepseek_key_file=key_file,
    )

    assert result.model == "deepseek-v4-flash"
    assert result.openai_api_key == "deepseek-test-secret"
    assert result.base_url == DEEPSEEK_BASE_URL
    assert result.api_mode == "chat_completions"
    assert result.reasoning_effort == "high"
    assert result.provider == "deepseek"
    assert "deepseek-test-secret" not in repr(result)


def test_configured_preset_restores_startup_provider_after_deepseek_switch(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("deepseek-test-secret\n", encoding="utf-8")
    configured = _configured()
    deepseek = resolve_model_selection(
        "deepseek-pro",
        active_config=configured,
        startup_config=configured,
        deepseek_key_file=key_file,
    )

    restored = resolve_model_selection(
        "terra",
        active_config=deepseek,
        startup_config=configured,
        deepseek_key_file=key_file,
    )

    assert restored.model == "gpt-5.6-terra"
    assert restored.openai_api_key == "configured-provider-secret"
    assert restored.base_url == "https://provider.example/v1"
    assert restored.provider == "configured"


@pytest.mark.parametrize("contents", ["", "\n\n", "first\nsecond\n"])
def test_deepseek_key_file_requires_exactly_one_nonempty_line(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "api_key.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="exactly one"):
        load_deepseek_api_key(path)


def test_missing_deepseek_key_file_reports_path_not_secret(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(ConfigurationError, match="not found") as error:
        load_deepseek_api_key(path)

    assert "secret" not in str(error.value).lower()


def test_model_presets_show_current_supported_deepseek_choices() -> None:
    aliases = {item.alias for item in model_presets()}

    assert {"luna", "terra", "sol", "deepseek-flash", "deepseek-pro"} <= aliases
