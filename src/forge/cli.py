"""Command-line entry point for the Forge coding agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from forge.agent import AgentLoop, AgentStatus, TaskMode, TaskPlan, resolve_task_mode
from forge.agent.verification import VerificationStatus
from forge.config import ConfigurationError, ForgeConfig
from forge.console import safe_print
from forge.discovery import select_workspace
from forge.llm import (
    ModelCommunicationError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from forge.trace import TraceRecorder
from forge.tools import create_coding_registry
from forge.ui import TerminalUI
from forge.workspace import Workspace


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Forge local coding agent."
    )
    parser.add_argument("task", help="Programming task for the agent.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root (default: the exact current directory).",
    )
    parser.add_argument(
        "--discover-workspace",
        action="store_true",
        help="Opt in to searching parent directories for a project root.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=60,
        help="Maximum model turns before hard termination (default: 60).",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TaskMode],
        default=TaskMode.AUTO.value,
        help="Task mode: auto, ask, or code (default: auto).",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="JSONL trace path relative to the workspace (default: .forge/trace.jsonl).",
    )
    arguments = parser.parse_args(argv)

    trace = None
    ui = None
    try:
        config = ForgeConfig.from_env()
        discovery = select_workspace(
            arguments.workspace or Path.cwd(),
            discover_parent=arguments.discover_workspace,
        )
        workspace_root = Workspace(discovery.root).root
        trace_path = _resolve_trace_path(workspace_root, arguments.trace_file)
        ui = TerminalUI()
        ui.start(
            task=arguments.task,
            workspace=workspace_root,
            model=config.model,
            max_steps=arguments.max_steps,
            mode=resolve_task_mode(arguments.task, arguments.mode).value,
        )
        trace = TraceRecorder(
            trace_path,
            workspace=workspace_root,
            console=True,
            renderer=ui,
        )
        model_client = _create_model_client(config)
        registry = create_coding_registry(workspace_root)
        state = AgentLoop(
            model_client,
            registry,
            max_steps=arguments.max_steps,
            mode=arguments.mode,
            trace=trace,
        ).run(
            arguments.task,
            workspace=workspace_root,
            launch_directory=(
                discovery.start
            ),
        )
    except (ConfigurationError, ModelCommunicationError, OSError, ValueError) as error:
        if ui is not None:
            ui.error(str(error))
        safe_print(f"Forge failed: {error}", stream=sys.stderr)
        return 1
    finally:
        if trace is not None:
            trace.close()

    if ui is not None:
        ui.render_plan_history(getattr(state, "plan_history", ()))
        ui.finish(state)
    else:
        _print_plan_history(getattr(state, "plan_history", ()))
    if state.status is AgentStatus.MAX_STEPS:
        safe_print(
            f"INCOMPLETE: Forge stopped after reaching max_steps={state.max_steps}.",
            stream=sys.stderr,
        )
        return 2

    if getattr(state, "verification_status", None) is VerificationStatus.PASSED:
        safe_print("VERIFIED", stream=sys.stderr)
    safe_print(state.final_answer or "")
    return 0


def _print_plan_history(plan_history: Sequence[TaskPlan]) -> None:
    """Display only the latest plan; transitions were already shown live."""

    if not plan_history:
        return
    plan = plan_history[-1]
    safe_print(f"Current plan: {plan.goal}", stream=sys.stderr)
    for step in plan.steps:
        safe_print(
            f"  [{step.status.value}] {step.step_id}: {step.description}",
            stream=sys.stderr,
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
