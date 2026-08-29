"""Command-line entry point for the minimal read-only Forge agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from forge.agent import AgentLoop, AgentStatus
from forge.config import ConfigurationError, ForgeConfig
from forge.llm import ModelCommunicationError, OpenAIResponsesClient
from forge.tools import create_coding_registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the minimal Forge coding agent."
    )
    parser.add_argument("task", help="Programming task for the agent.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root (default: current directory).",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=12,
        help="Maximum model turns before hard termination (default: 12).",
    )
    arguments = parser.parse_args(argv)

    try:
        config = ForgeConfig.from_env()
        model_client = OpenAIResponsesClient(config)
        registry = create_coding_registry(arguments.workspace)
        state = AgentLoop(
            model_client,
            registry,
            max_steps=arguments.max_steps,
        ).run(arguments.task, workspace=arguments.workspace)
    except (ConfigurationError, ModelCommunicationError, OSError, ValueError) as error:
        print(f"Forge failed: {error}", file=sys.stderr)
        return 1

    if state.status is AgentStatus.MAX_STEPS:
        print(
            f"Forge stopped after reaching max_steps={state.max_steps}.",
            file=sys.stderr,
        )
        return 2

    print(state.final_answer or "")
    return 0


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
