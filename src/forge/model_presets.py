"""Explicit, local model/provider choices for the DBA interactive shell.

The module keeps provider routing and credential loading outside the agent loop.
Secrets are read only when a selected preset needs them and are returned solely
inside an in-memory :class:`ForgeConfig`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from forge.config import ConfigurationError, ForgeConfig


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def project_deepseek_key_path() -> Path:
    """Return the ignored DeepSeek key file at the repository root."""

    return Path(__file__).resolve().parents[2] / "api_key.txt"


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """A short alias shown by ``/models`` and accepted by ``/model``."""

    alias: str
    model: str
    label: str
    provider: str
    uses_deepseek_key: bool = False


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        alias="luna",
        model="gpt-5.6-luna",
        label="GPT-5.6 Luna (configured provider)",
        provider="configured",
    ),
    ModelPreset(
        alias="terra",
        model="gpt-5.6-terra",
        label="GPT-5.6 Terra (configured provider)",
        provider="configured",
    ),
    ModelPreset(
        alias="sol",
        model="gpt-5.6-sol",
        label="GPT-5.6 Sol (configured provider)",
        provider="configured",
    ),
    ModelPreset(
        alias="deepseek-flash",
        model="deepseek-v4-flash",
        label="DeepSeek V4 Flash (experimental compatibility)",
        provider="deepseek",
        uses_deepseek_key=True,
    ),
    ModelPreset(
        alias="deepseek-pro",
        model="deepseek-v4-pro",
        label="DeepSeek V4 Pro (experimental compatibility)",
        provider="deepseek",
        uses_deepseek_key=True,
    ),
)


def model_presets() -> tuple[ModelPreset, ...]:
    """Return the stable, user-facing model selection list."""

    return MODEL_PRESETS


def resolve_model_selection(
    selection: str,
    *,
    active_config: ForgeConfig,
    startup_config: ForgeConfig,
) -> ForgeConfig:
    """Build the next in-memory configuration for a model selection.

    Bare model names remain supported for advanced users. Named configured
    presets intentionally restore the startup provider URL and credential after
    a previous DeepSeek selection.
    """

    normalized = selection.strip().lower()
    if not normalized:
        raise ConfigurationError("model selection must not be empty")
    preset = next((item for item in MODEL_PRESETS if item.alias == normalized), None)
    if preset is None:
        return ForgeConfig(
            openai_api_key=active_config.openai_api_key,
            model=selection.strip(),
            reasoning_effort=active_config.reasoning_effort,
            base_url=active_config.base_url,
            api_mode=active_config.api_mode,
            provider=active_config.provider,
        )
    if not preset.uses_deepseek_key:
        return ForgeConfig(
            openai_api_key=startup_config.openai_api_key,
            model=preset.model,
            reasoning_effort=active_config.reasoning_effort,
            base_url=startup_config.base_url,
            api_mode=startup_config.api_mode,
            provider=startup_config.provider,
        )
    return ForgeConfig(
        openai_api_key=load_deepseek_api_key(),
        model=preset.model,
        reasoning_effort=active_config.reasoning_effort,
        base_url=DEEPSEEK_BASE_URL,
        api_mode="chat_completions",
        provider="deepseek",
    )


def default_deepseek_key_file() -> Path:
    """Resolve an explicit key path or the repository-local ignored file."""

    configured = os.environ.get("FORGE_DEEPSEEK_KEY_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return project_deepseek_key_path()


def load_deepseek_api_key(path: Path | None = None) -> str:
    """Read one local DeepSeek key, preferring an explicit environment override."""

    environment_key = os.environ.get("FORGE_DEEPSEEK_API_KEY", "").strip()
    if environment_key:
        return environment_key
    resolved_path = (path or default_deepseek_key_file()).expanduser()
    try:
        raw = resolved_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"DeepSeek key is not available. Set FORGE_DEEPSEEK_API_KEY or create "
            f"one local key file at: {resolved_path}"
        ) from error
    except OSError as error:
        raise ConfigurationError(f"Unable to read DeepSeek key file: {resolved_path}") from error
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ConfigurationError("DeepSeek key file must contain exactly one non-empty key line")
    return lines[0]
