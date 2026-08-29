"""The first minimal Coding Agent loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from forge.agent.context import ContextBudget, ContextManager
from forge.agent.plan import PlanStore, update_plan_tool
from forge.agent.state import AgentState, AgentStatus
from forge.agent.verification import VerificationStatus, VerificationTracker
from forge.llm import ModelRequest, ModelResponse
from forge.tools import ToolRegistry

DEFAULT_AGENT_INSTRUCTIONS = """You are Forge, a local coding agent.
Inspect the workspace, reproduce problems, make the smallest necessary edits, and
verify changes with deterministic commands. Tool paths and command working
directories are relative to the workspace root. For Python repositories, prefer
get_repo_map, search_symbol, and read_symbol before broad file reads. Prefer
apply_patch for changes to existing files, use create_file for new files, and
inspect git_diff after editing. Use write_file only when a patch cannot safely
express a small whole-file change. Before ordinary inspection, create a complete
plan with update_plan: keep its goal and success criteria stable, and update step
statuses only when a meaningful milestone changes. Never claim a command passed
unless its returned status proves it. After every code change, run a targeted
deterministic test or compiler check. Before the final answer, run an appropriate
verification command after the latest change. A final answer is not evidence of
verification; answer only after the runtime has observed a passing check.
"""

_RECOVERY_AFTER_ROUNDS = 2
_READ_PROGRESS_TOOLS = {
    "get_repo_map",
    "search_symbol",
    "read_symbol",
    "list_files",
    "read_file",
    "search_text",
}


class ModelClient(Protocol):
    def create_response(self, request: ModelRequest) -> ModelResponse: ...


class TraceSink(Protocol):
    def record(
        self,
        event: str,
        *,
        step: int,
        payload: dict[str, object] | None = None,
    ) -> None: ...


class AgentLoop:
    """Alternate between model responses and local tool observations."""

    def __init__(
        self,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        *,
        max_steps: int = 12,
        instructions: str = DEFAULT_AGENT_INSTRUCTIONS,
        context_budget: ContextBudget | None = None,
        trace: TraceSink | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._instructions = instructions
        self._context_budget = context_budget
        self._trace = trace

    def run(self, task: str, *, workspace: Path) -> AgentState:
        state = AgentState.start(
            task=task,
            workspace=workspace,
            max_steps=self._max_steps,
        )
        context_manager = ContextManager(task, budget=self._context_budget)
        plan_store = PlanStore()
        verification = VerificationTracker()
        no_progress_rounds = 0
        run_registry = self._tool_registry.clone()
        run_registry.register(update_plan_tool(plan_store))
        tool_schemas = run_registry.schemas()

        while state.step < state.max_steps:
            state.step += 1
            snapshot = context_manager.build_context(step=state.step)
            state.context = list(snapshot.input_items)
            state.context_usage.append(snapshot.usage)
            self._record_trace(
                "model_request",
                state.step,
                {
                    "input_items": len(snapshot.input_items),
                    "tools": [tool.name for tool in tool_schemas],
                    "context_usage": _context_usage_payload(snapshot.usage),
                },
            )
            response = self._model_client.create_response(
                ModelRequest(
                    input=snapshot.input_items,
                    instructions=self._instructions,
                    tools=tool_schemas,
                    parallel_tool_calls=True,
                )
            )
            state.response_ids.append(response.response_id)
            self._record_trace(
                "model_response",
                state.step,
                _model_response_payload(response),
            )

            if not response.function_calls:
                if verification.requires_verification and not verification.is_verified:
                    state.final_answer = None
                    hint = _final_verification_hint(verification)
                    _append_hint(
                        state,
                        context_manager,
                        hint,
                        trace=self._trace,
                        step=state.step,
                    )
                    _sync_verification_state(state, verification, no_progress_rounds)
                    continue
                state.final_answer = response.output_text
                state.status = AgentStatus.COMPLETED
                _sync_verification_state(state, verification, no_progress_rounds)
                self._record_trace(
                    "final",
                    state.step,
                    {
                        "status": (
                            "VERIFIED"
                            if verification.is_verified
                            else "COMPLETED"
                        ),
                        "verification_status": verification.status.value,
                        "answer_characters": len(response.output_text),
                    },
                )
                return state

            executed_calls = []
            turn_progress = False
            for call in response.function_calls:
                state.tool_calls.append(call)
                self._record_trace(
                    "tool_start",
                    state.step,
                    {
                        "call_id": call.call_id,
                        "tool_name": call.name,
                        "argument_keys": _argument_keys(call.arguments_json),
                    },
                )
                observation = run_registry.dispatch(call)
                state.observations.append(observation)
                executed_calls.append((call, observation))
                self._record_trace(
                    "tool_result",
                    state.step,
                    _tool_result_payload(call, observation),
                )
                event = verification.observe(call, observation)
                turn_progress = turn_progress or event.mutation
                if event.mutation:
                    _append_hint(
                        state,
                        context_manager,
                        (
                            "Code changed successfully. Run a targeted deterministic "
                            "test, compiler, or linter check against the changed area "
                            "before moving to the final answer."
                        ),
                        trace=self._trace,
                        step=state.step,
                    )
                    _record_patch_event(self._trace, state.step, observation)
                turn_progress = turn_progress or (
                    observation.success and call.name in _READ_PROGRESS_TOOLS
                )
                if event.record is not None:
                    if event.record.passed:
                        turn_progress = True
                    else:
                        _append_hint(
                            state,
                            context_manager,
                            _verification_failure_hint(event.record),
                            trace=self._trace,
                            step=state.step,
                        )
                        _record_verification_event(
                            self._trace,
                            state.step,
                            event.record,
                            verification.status,
                        )
                        if event.repeated_failure_count >= 2:
                            _append_hint(
                                state,
                                context_manager,
                                _repeated_failure_hint(
                                    event.repeated_failure_count
                                ),
                                trace=self._trace,
                                step=state.step,
                            )
                else:
                    _record_verification_event(
                        self._trace,
                        state.step,
                        event.record,
                        verification.status,
                    )
                if call.name == "update_plan" and observation.success:
                    _record_plan_event(
                        self._trace,
                        state.step,
                        plan_store.plan,
                    )
            context_manager.record_turn(response, executed_calls)
            if plan_store.plan is not None:
                state.plan = plan_store.plan
                state.plan_history = list(plan_store.history)
                context_manager.set_plan(plan_store.plan.to_prompt())
            no_progress_rounds = (
                0 if turn_progress else no_progress_rounds + 1
            )
            if no_progress_rounds >= _RECOVERY_AFTER_ROUNDS:
                _append_hint(
                    state,
                    context_manager,
                    (
                        f"No meaningful progress for {no_progress_rounds} tool "
                        "rounds. Re-check your assumptions and inspect the "
                        "relevant files before trying another change."
                    ),
                    trace=self._trace,
                    step=state.step,
                )
            context_manager.set_verification_status(verification.latest_summary())
            _sync_verification_state(state, verification, no_progress_rounds)

        state.status = AgentStatus.MAX_STEPS
        state.final_answer = None
        _sync_verification_state(state, verification, no_progress_rounds)
        self._record_trace(
            "final",
            state.step,
            {
                "status": "INCOMPLETE",
                "verification_status": verification.status.value,
                "answer_characters": 0,
            },
        )
        return state

    def _record_trace(
        self,
        event: str,
        step: int,
        payload: dict[str, object],
    ) -> None:
        if self._trace is not None:
            self._trace.record(event, step=step, payload=payload)


def _sync_verification_state(
    state: AgentState,
    tracker: VerificationTracker,
    no_progress_rounds: int,
) -> None:
    state.latest_verification = tracker.latest
    state.verification_history = list(tracker.history)
    state.verification_status = tracker.status
    state.repeated_failure_count = tracker.repeated_failure_count
    state.no_progress_rounds = no_progress_rounds


def _append_hint(
    state: AgentState,
    context_manager: ContextManager,
    hint: str,
    *,
    trace: TraceSink | None = None,
    step: int = 0,
) -> None:
    if state.recovery_hints and state.recovery_hints[-1] == hint:
        return
    state.recovery_hints.append(hint)
    context_manager.add_runtime_guidance(hint)
    if trace is not None:
        trace.record(
            "recovery",
            step=step,
            payload={"reason": hint},
        )


def _verification_failure_hint(record) -> str:
    return (
        f"Deterministic {record.kind} verification failed with "
        f"return_code={record.return_code}. Treat its stdout/stderr as feedback, "
        "diagnose the failure, and continue fixing instead of claiming completion."
    )


def _repeated_failure_hint(count: int) -> str:
    return (
        f"The same deterministic verification failure repeated {count} times. "
        "Re-check your assumptions and the relevant files before retrying."
    )


def _final_verification_hint(tracker: VerificationTracker) -> str:
    if tracker.status is VerificationStatus.STALE:
        return (
            "The latest passing verification predates a code change. Run an "
            "appropriate targeted test, compiler, or linter command now before "
            "claiming completion."
        )
    if tracker.status is VerificationStatus.FAILED:
        return (
            "The latest deterministic verification failed. Inspect its feedback, "
            "make a corrective change if needed, and rerun the verification command "
            "before claiming completion."
        )
    return (
        "Code was changed but no current deterministic verification has passed. "
        "Run a targeted test, compiler, or linter command before claiming completion."
    )


def _context_usage_payload(usage) -> dict[str, object]:
    return {
        "input_characters": usage.input_characters,
        "approximate_tokens": usage.approximate_tokens,
        "budget_characters": usage.budget_characters,
        "recent_observations": usage.recent_observations,
        "compacted_observations": usage.compacted_observations,
    }


def _model_response_payload(response: ModelResponse) -> dict[str, object]:
    usage = None
    if response.usage is not None:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {
        "response_id": response.response_id,
        "model": response.model,
        "status": response.status,
        "function_call_count": len(response.function_calls),
        "function_names": [call.name for call in response.function_calls],
        "output_text_characters": len(response.output_text),
        "usage": usage,
    }


def _argument_keys(arguments_json: str) -> list[str]:
    try:
        arguments = json.loads(arguments_json)
    except (TypeError, json.JSONDecodeError):
        return []
    return sorted(str(key) for key in arguments) if isinstance(arguments, dict) else []


def _tool_result_payload(call, observation) -> dict[str, object]:
    content = observation.content
    payload: dict[str, object] = {
        "call_id": call.call_id,
        "tool_name": call.name,
        "success": observation.success,
        "content_type": type(content).__name__,
    }
    if isinstance(content, str):
        payload["content_characters"] = len(content)
    elif isinstance(content, Mapping):
        payload["result_keys"] = sorted(str(key) for key in content)
        if call.name == "run_command":
            payload.update(
                {
                    "return_code": content.get("return_code"),
                    "timed_out": content.get("timed_out"),
                    "stdout_characters": len(str(content.get("stdout", ""))),
                    "stderr_characters": len(str(content.get("stderr", ""))),
                    "stdout_truncated": bool(content.get("stdout_truncated")),
                    "stderr_truncated": bool(content.get("stderr_truncated")),
                }
            )
        if call.name == "apply_patch":
            payload.update(
                {
                    "applied": content.get("applied"),
                    "changed_files": _changed_file_names(content.get("changed_files")),
                    "hunks_applied": content.get("hunks_applied"),
                    "failure_reason": content.get("failure_reason"),
                }
            )
        if call.name == "update_plan":
            payload["update_number"] = content.get("update_number")
    if call.name in {"create_file", "write_file"}:
        arguments = _parse_arguments(call.arguments_json)
        if isinstance(arguments.get("path"), str):
            payload["path"] = arguments["path"]
    return payload


def _changed_file_names(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            result.append(item["path"])
        elif isinstance(item, str):
            result.append(item)
    return result


def _record_patch_event(
    trace: TraceSink | None,
    step: int,
    observation,
) -> None:
    if trace is None or not isinstance(observation.content, Mapping):
        return
    content = observation.content
    if not observation.success or not content.get("applied"):
        return
    trace.record(
        "patch_applied",
        step=step,
        payload={
            "changed_files": _changed_file_names(content.get("changed_files")),
            "hunks_applied": content.get("hunks_applied"),
        },
    )


def _record_plan_event(
    trace: TraceSink | None,
    step: int,
    plan,
) -> None:
    if trace is None or plan is None:
        return
    statuses = {item.step_id: item.status.value for item in plan.steps}
    current_step = next(
        (
            item.step_id
            for item in plan.steps
            if item.status.value == "in_progress"
        ),
        None,
    )
    trace.record(
        "plan_updated",
        step=step,
        payload={
            "goal": plan.goal,
            "step_statuses": statuses,
            "current_step": current_step,
            "success_criteria_count": len(plan.success_criteria),
        },
    )


def _record_verification_event(
    trace: TraceSink | None,
    step: int,
    record,
    status: VerificationStatus,
) -> None:
    if trace is None:
        return
    payload: dict[str, object] = {"status": status.value}
    if record is not None:
        payload.update(
            {
                "kind": record.kind,
                "command": _safe_command(record.command),
                "return_code": record.return_code,
                "timed_out": record.timed_out,
                "stdout_characters": len(record.stdout),
                "stderr_characters": len(record.stderr),
            }
        )
    trace.record("verification", step=step, payload=payload)


def _safe_command(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    for index, part in enumerate(command):
        path = Path(part)
        if index == 0 and path.is_absolute():
            result.append(path.name)
        else:
            result.append(part)
    return result


def _parse_arguments(arguments_json: str) -> dict[str, object]:
    try:
        value = json.loads(arguments_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
