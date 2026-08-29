"""Environment-backed configuration for Forge."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
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

        if not model:
            raise ConfigurationError("FORGE_MODEL must not be empty")
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            allowed = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            raise ConfigurationError(
                "FORGE_REASONING_EFFORT must be one of: " + allowed
            )

        return cls(
            openai_api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
        )
