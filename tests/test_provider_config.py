import os
from pathlib import Path

import pytest

from forge.config import ConfigurationError
from forge.provider_config import load_backup_config, load_repl_config


def _write_backup(path: Path) -> None:
    path.write_text(
        """
model_provider = "demo"
model = "gpt-5.6-luna"
model_reasoning_effort = "max"

[model_providers.demo]
name = "Demo provider"
base_url = "https://provider.example/v1"
experimental_bearer_token = "backup-token-for-test"
""".strip(),
        encoding="utf-8",
    )


def test_load_backup_config_reads_provider_without_mutating_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_backup(path)
    monkeypatch.setenv("FORGE_MODEL", "environment-model")

    config = load_backup_config(path)

    assert config.openai_api_key == "backup-token-for-test"
    assert config.base_url == "https://provider.example/v1"
    assert config.api_mode == "chat_completions"
    assert config.model == "gpt-5.6-luna"
    assert config.reasoning_effort == "max"
    assert config.openai_api_key not in repr(config)
    assert os.environ["FORGE_MODEL"] == "environment-model"


def test_load_repl_config_uses_environment_when_backup_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_BACKUP_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("OPENAI_API_KEY", "environment-token")

    config = load_repl_config()

    assert config.openai_api_key == "environment-token"
    assert config.api_mode == "responses"


def test_explicit_backup_path_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_repl_config(tmp_path / "missing.toml")
