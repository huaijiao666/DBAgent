"""Local provider configuration for interactive DBA sessions.

The optional provider TOML lives beside the source checkout and is ignored by
Git.  This keeps the DBA launcher self-contained without searching a user's
home directory or persisting credentials into session data.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dbagent.config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    ConfigurationError,
    DBAgentConfig,
)


def project_config_path() -> Path:
    """Return the ignored provider file located at the repository root."""

    return Path(__file__).resolve().parents[2] / "config.toml"


def configured_config_path() -> Path | None:
    """Return an explicit override without inspecting personal directories."""

    configured = (
        os.environ.get("DBAGENT_CONFIG_PATH", "").strip()
        or os.environ.get("DBAGENT_BACKUP_CONFIG", "").strip()
    )
    return Path(configured).expanduser() if configured else None


def load_repl_config(path: Path | None = None) -> DBAgentConfig:
    """Load an explicit or repository-local TOML, then process environment."""

    if path is not None:
        return load_backup_config(path)
    explicit_path = configured_config_path()
    if explicit_path is not None:
        return load_backup_config(explicit_path)
    local_path = project_config_path()
    if local_path.is_file():
        return load_backup_config(local_path)
    config = DBAgentConfig.from_env()
    if config.openai_api_key is None:
        raise ConfigurationError(
            "No project-local config.toml was found and OPENAI_API_KEY is not set. "
            "Create config.toml at the DBAgent repository root, set DBAGENT_CONFIG_PATH, "
            "pass --config-path, or set OPENAI_API_KEY."
        )
    return config


def load_backup_config(path: Path) -> DBAgentConfig:
    """Read a selected compatible provider configuration only into memory."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Provider config was not found: {path}") from error
    except OSError as error:
        raise ConfigurationError(f"Unable to read provider config: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Provider config is invalid TOML: {path}") from error

    provider_name = raw.get("model_provider")
    providers = raw.get("model_providers")
    if not isinstance(providers, Mapping):
        raise ConfigurationError("Provider config has no model_providers table")
    provider = providers.get(provider_name) if isinstance(provider_name, str) else None
    if not isinstance(provider, Mapping):
        if len(providers) == 1:
            provider = next(iter(providers.values()))
        else:
            raise ConfigurationError("Provider config does not identify a usable model provider")

    base_url = _required_string(provider, "base_url")
    token = _required_string(provider, "experimental_bearer_token")
    if token == "<redacted>":
        raise ConfigurationError("Provider bearer token is redacted")
    environment = dict(os.environ)
    environment.update(
        {
            "OPENAI_API_KEY": token,
            "DBAGENT_BASE_URL": base_url,
            "DBAGENT_API_MODE": "chat_completions",
            "DBAGENT_MODEL": _optional_string(raw, "model", DEFAULT_MODEL),
            "DBAGENT_REASONING_EFFORT": _optional_string(
                raw, "model_reasoning_effort", DEFAULT_REASONING_EFFORT
            ),
        }
    )
    return DBAgentConfig.from_env(environment)


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Backup provider config is missing '{name}'")
    return value.strip()


def _optional_string(values: Mapping[str, Any], name: str, default: str) -> str:
    value = values.get(name, default)
    return value.strip() if isinstance(value, str) and value.strip() else default
