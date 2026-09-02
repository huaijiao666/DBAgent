"""Environment-backed configuration for DBAgent."""

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
class DBAgentConfig:
    """Immutable settings loaded from the process environment."""

    openai_api_key: str | None = field(repr=False)
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    base_url: str | None = None
    api_mode: str = DEFAULT_API_MODE
    provider: str = "configured"

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> DBAgentConfig:
        """Build configuration from an environment mapping.

        ``environment`` is injectable so tests do not depend on or mutate the
        real process environment. Production callers should omit it.
        """

        source = os.environ if environment is None else environment

        api_key = source.get("OPENAI_API_KEY", "").strip() or None
        model = source.get("DBAGENT_MODEL", DEFAULT_MODEL).strip()
        reasoning_effort = source.get(
            "DBAGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ).strip().lower()
        base_url = source.get("DBAGENT_BASE_URL", "").strip() or None
        api_mode = source.get("DBAGENT_API_MODE", DEFAULT_API_MODE).strip().lower()
        provider = source.get("DBAGENT_PROVIDER", "configured").strip().lower()

        if not model:
            raise ConfigurationError("DBAGENT_MODEL must not be empty")
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            allowed = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            raise ConfigurationError(
                "DBAGENT_REASONING_EFFORT must be one of: " + allowed
            )
        if api_mode not in SUPPORTED_API_MODES:
            allowed = ", ".join(sorted(SUPPORTED_API_MODES))
            raise ConfigurationError(
                "DBAGENT_API_MODE must be one of: " + allowed
            )
        if provider not in SUPPORTED_PROVIDERS:
            allowed = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ConfigurationError("DBAGENT_PROVIDER must be one of: " + allowed)
        if provider == "deepseek" and api_mode != "chat_completions":
            raise ConfigurationError(
                "DeepSeek requires DBAGENT_API_MODE=chat_completions"
            )
        if base_url is not None:
            parsed_base_url = urlparse(base_url)
            if (
                parsed_base_url.scheme not in {"http", "https"}
                or not parsed_base_url.netloc
            ):
                raise ConfigurationError(
                    "DBAGENT_BASE_URL must be an absolute http(s) URL"
                )

        return cls(
            openai_api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            api_mode=api_mode,
            provider=provider,
        )
