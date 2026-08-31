"""Explicit, local model/provider choices for the DBA interactive shell.

The module keeps provider routing and credential loading outside the agent loop.
Secrets are read only when a selected preset needs them and are returned solely
inside an in-memory :class:`ForgeConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge.config import ConfigurationError, ForgeConfig


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_KEY_FILE = Path("C:/AAA/DBAgent/api_key.txt")


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
        label="DeepSeek V4 Flash (tool-safe reasoning)",
        provider="deepseek",
        uses_deepseek_key=True,
    ),
    ModelPreset(
        alias="deepseek-pro",
        model="deepseek-v4-pro",
        label="DeepSeek V4 Pro (tool-safe reasoning)",
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
    deepseek_key_file: Path = DEFAULT_DEEPSEEK_KEY_FILE,
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
        openai_api_key=load_deepseek_api_key(deepseek_key_file),
        model=preset.model,
        reasoning_effort=active_config.reasoning_effort,
        base_url=DEEPSEEK_BASE_URL,
        api_mode="chat_completions",
        provider="deepseek",
    )


def load_deepseek_api_key(path: Path = DEFAULT_DEEPSEEK_KEY_FILE) -> str:
    """Read exactly one API key from the user-authorized external file.

    The key never enters an environment variable, session checkpoint, trace, or
    rendered UI. Errors name only the file path, never its contents.
    """

    try:
        raw = path.expanduser().read_text(encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"DeepSeek key file was not found: {path}. "
            "Create that external file or choose another model."
        ) from error
    except OSError as error:
        raise ConfigurationError(f"Unable to read DeepSeek key file: {path}") from error

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ConfigurationError(
            "DeepSeek key file must contain exactly one non-empty key line"
        )
    return lines[0]
