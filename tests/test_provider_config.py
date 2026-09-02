from pathlib import Path

import pytest

from forge.config import ConfigurationError
from forge.provider_config import load_repl_config


def test_load_repl_config_reads_only_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("forge.provider_config.project_config_path", lambda: tmp_path / "missing.toml")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-token")

    config = load_repl_config()

    assert config.openai_api_key == "environment-token"
    assert config.api_mode == "responses"


def test_explicit_external_credential_file_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_repl_config(tmp_path / "provider.toml")


def test_load_repl_config_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("forge.provider_config.project_config_path", lambda: tmp_path / "missing.toml")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="project-local config.toml"):
        load_repl_config()


def test_load_repl_config_prefers_repository_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
model_provider = "demo"
model = "gpt-5.6-luna"
model_reasoning_effort = "max"

[model_providers.demo]
base_url = "https://provider.example/v1"
experimental_bearer_token = "backup-test-token"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("forge.provider_config.project_config_path", lambda: path)

    config = load_repl_config()

    assert config.openai_api_key == "backup-test-token"
    assert config.model == "gpt-5.6-luna"
    assert config.api_mode == "chat_completions"


def test_explicit_config_path_takes_precedence_over_repository_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "config.toml"
    local.write_text(
        """
model_provider = "local"
[model_providers.local]
base_url = "https://local.example/v1"
experimental_bearer_token = "local-token"
""".strip(),
        encoding="utf-8",
    )
    selected = tmp_path / "selected.toml"
    selected.write_text(
        """
model_provider = "selected"
[model_providers.selected]
base_url = "https://selected.example/v1"
experimental_bearer_token = "selected-token"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("forge.provider_config.project_config_path", lambda: local)

    config = load_repl_config(selected)

    assert config.openai_api_key == "selected-token"
    assert config.base_url == "https://selected.example/v1"
