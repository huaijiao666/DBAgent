"""Command-line entry point for the minimal read-only Forge agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from forge.agent import AgentLoop, AgentStatus, TaskPlan
from forge.agent.verification import VerificationStatus
from forge.config import ConfigurationError, ForgeConfig
from forge.llm import (
    ModelCommunicationError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from forge.trace import TraceRecorder
from forge.tools import create_coding_registry
from forge.workspace import Workspace


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
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="JSONL trace path relative to the workspace (default: .forge/trace.jsonl).",
    )
    arguments = parser.parse_args(argv)

    trace = None
    try:
        config = ForgeConfig.from_env()
        workspace_root = Workspace(arguments.workspace).root
        trace_path = _resolve_trace_path(workspace_root, arguments.trace_file)
        trace = TraceRecorder(trace_path, workspace=workspace_root, console=True)
        model_client = _create_model_client(config)
        registry = create_coding_registry(arguments.workspace)
        state = AgentLoop(
            model_client,
            registry,
            max_steps=arguments.max_steps,
            trace=trace,
        ).run(arguments.task, workspace=arguments.workspace)
    except (ConfigurationError, ModelCommunicationError, OSError, ValueError) as error:
        print(f"Forge failed: {error}", file=sys.stderr)
        return 1
    finally:
        if trace is not None:
            trace.close()

    _print_plan_history(getattr(state, "plan_history", ()))
    if state.status is AgentStatus.MAX_STEPS:
        print(
            f"INCOMPLETE: Forge stopped after reaching max_steps={state.max_steps}.",
            file=sys.stderr,
        )
        return 2

    if getattr(state, "verification_status", None) is VerificationStatus.PASSED:
        print("VERIFIED", file=sys.stderr)
    print(state.final_answer or "")
    return 0


def _print_plan_history(plan_history: Sequence[TaskPlan]) -> None:
    """Display every accepted plan snapshot so status changes are observable."""

    if not plan_history:
        return
    print("Plan status updates:", file=sys.stderr)
    for number, plan in enumerate(plan_history, start=1):
        print(f"  update {number}: {plan.goal}", file=sys.stderr)
        for step in plan.steps:
            print(
                f"    [{step.status.value}] {step.step_id}: {step.description}",
                file=sys.stderr,
            )


def _create_model_client(config: ForgeConfig):
    if config.api_mode == "chat_completions":
        return OpenAIChatCompletionsClient(config)
    return OpenAIResponsesClient(config)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolve_trace_path(workspace: Path, user_path: Path | None) -> Path:
    candidate = user_path or Path(".forge") / "trace.jsonl"
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("trace file must be inside the workspace") from error
    if resolved == workspace or resolved.exists() and resolved.is_dir():
        raise ValueError("trace file must be a file path")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
