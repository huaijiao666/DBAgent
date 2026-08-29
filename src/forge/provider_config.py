"""Load a temporary provider configuration for the DBA interactive client."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    ConfigurationError,
    ForgeConfig,
)

_DEFAULT_BACKUP_RELATIVE_PATH = Path(
    "Downloads"
) / "WeChat Files" / "wxid_8lhimj8hmlcv22" / "FileStorage" / "File" / "2026-03" / "sxdt" / "config.toml"


def default_backup_config_path() -> Path:
    """Return the user-specific backup location without embedding a username."""

    configured = os.environ.get("FORGE_BACKUP_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / _DEFAULT_BACKUP_RELATIVE_PATH


def load_repl_config(path: Path | None = None) -> ForgeConfig:
    """Load the backup provider when available, otherwise use process env.

    The returned value owns the API key in memory only. This function never
    mutates ``os.environ`` and never writes the configuration back to disk.
    An explicit ``path`` is treated as required so a typo is not silently hidden.
    """

    if path is not None:
        return load_backup_config(path)

    backup_path = default_backup_config_path()
    if backup_path.is_file():
        return load_backup_config(backup_path)

    config = ForgeConfig.from_env()
    if config.openai_api_key is None:
        raise ConfigurationError(
            "Backup provider config was not found at "
            f"{backup_path}. Set FORGE_BACKUP_CONFIG or OPENAI_API_KEY."
        )
    return config


def load_backup_config(path: Path) -> ForgeConfig:
    """Read the selected provider from a Codex-style TOML backup."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Backup provider config was not found: {path}") from error
    except OSError as error:
        raise ConfigurationError(f"Unable to read backup provider config: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Backup provider config is invalid TOML: {path}") from error

    provider_name = raw.get("model_provider")
    providers = raw.get("model_providers")
    if not isinstance(providers, Mapping):
        raise ConfigurationError("Backup provider config has no model_providers table")
    provider = providers.get(provider_name) if isinstance(provider_name, str) else None
    if not isinstance(provider, Mapping):
        if len(providers) == 1:
            provider = next(iter(providers.values()))
        else:
            raise ConfigurationError(
                "Backup provider config does not identify a usable model provider"
            )

    base_url = _required_string(provider, "base_url")
    token = _required_string(provider, "experimental_bearer_token")
    if token == "<redacted>":
        raise ConfigurationError("Backup provider bearer token is redacted")

    model = _optional_string(raw, "model", DEFAULT_MODEL)
    reasoning_effort = _optional_string(
        raw,
        "model_reasoning_effort",
        DEFAULT_REASONING_EFFORT,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "OPENAI_API_KEY": token,
            "FORGE_BASE_URL": base_url,
            # The backup provider used by this project exposes Chat Completions.
            "FORGE_API_MODE": "chat_completions",
            "FORGE_MODEL": model,
            "FORGE_REASONING_EFFORT": reasoning_effort,
        }
    )
    return ForgeConfig.from_env(environment)


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Backup provider config is missing '{name}'")
    return value.strip()


def _optional_string(values: Mapping[str, Any], name: str, default: str) -> str:
    value = values.get(name, default)
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()
