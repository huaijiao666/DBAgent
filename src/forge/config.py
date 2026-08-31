"""Environment-backed configuration for Forge."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_API_MODE = "responses"
SUPPORTED_API_MODES = frozenset({"responses", "chat_completions"})
SUPPORTED_PROVIDERS = frozenset({"configured", "deepseek"})
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)


class ConfigurationError(ValueError):
    """Raised when an environment setting is missing or invalid."""


@dataclass(frozen=True, slots=True)
class ForgeConfig:
    """Immutable settings loaded from the process environment."""

    openai_api_key: str | None = field(repr=False)
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    base_url: str | None = None
    api_mode: str = DEFAULT_API_MODE
    provider: str = "configured"

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> ForgeConfig:
        """Build configuration from an environment mapping.

        ``environment`` is injectable so tests do not depend on or mutate the
        real process environment. Production callers should omit it.
        """

        source = os.environ if environment is None else environment

        api_key = source.get("OPENAI_API_KEY", "").strip() or None
        model = source.get("FORGE_MODEL", DEFAULT_MODEL).strip()
        reasoning_effort = source.get(
            "FORGE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ).strip().lower()
        base_url = source.get("FORGE_BASE_URL", "").strip() or None
        api_mode = source.get("FORGE_API_MODE", DEFAULT_API_MODE).strip().lower()
        provider = source.get("FORGE_PROVIDER", "configured").strip().lower()

        if not model:
            raise ConfigurationError("FORGE_MODEL must not be empty")
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            allowed = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            raise ConfigurationError(
                "FORGE_REASONING_EFFORT must be one of: " + allowed
            )
        if api_mode not in SUPPORTED_API_MODES:
            allowed = ", ".join(sorted(SUPPORTED_API_MODES))
            raise ConfigurationError(
                "FORGE_API_MODE must be one of: " + allowed
            )
        if provider not in SUPPORTED_PROVIDERS:
            allowed = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ConfigurationError("FORGE_PROVIDER must be one of: " + allowed)
        if provider == "deepseek" and api_mode != "chat_completions":
            raise ConfigurationError(
                "DeepSeek requires FORGE_API_MODE=chat_completions"
            )
        if base_url is not None:
            parsed_base_url = urlparse(base_url)
            if (
                parsed_base_url.scheme not in {"http", "https"}
                or not parsed_base_url.netloc
            ):
                raise ConfigurationError(
                    "FORGE_BASE_URL must be an absolute http(s) URL"
                )

        return cls(
            openai_api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            api_mode=api_mode,
            provider=provider,
        )
