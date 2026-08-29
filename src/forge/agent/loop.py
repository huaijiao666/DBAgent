"""The first minimal Coding Agent loop."""

from __future__ import annotations

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
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._instructions = instructions
        self._context_budget = context_budget

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
            response = self._model_client.create_response(
                ModelRequest(
                    input=snapshot.input_items,
                    instructions=self._instructions,
                    tools=tool_schemas,
                    parallel_tool_calls=True,
                )
            )
            state.response_ids.append(response.response_id)

            if not response.function_calls:
                if verification.requires_verification and not verification.is_verified:
                    state.final_answer = None
                    hint = _final_verification_hint(verification)
                    _append_hint(state, context_manager, hint)
                    _sync_verification_state(state, verification, no_progress_rounds)
                    continue
                state.final_answer = response.output_text
                state.status = AgentStatus.COMPLETED
                _sync_verification_state(state, verification, no_progress_rounds)
                return state

            executed_calls = []
            turn_progress = False
            for call in response.function_calls:
                state.tool_calls.append(call)
                observation = run_registry.dispatch(call)
                state.observations.append(observation)
                executed_calls.append((call, observation))
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
                    )
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
                        )
                        if event.repeated_failure_count >= 2:
                            _append_hint(
                                state,
                                context_manager,
                                _repeated_failure_hint(
                                    event.repeated_failure_count
                                ),
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
                )
            context_manager.set_verification_status(verification.latest_summary())
            _sync_verification_state(state, verification, no_progress_rounds)

        state.status = AgentStatus.MAX_STEPS
        state.final_answer = None
        _sync_verification_state(state, verification, no_progress_rounds)
        return state


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
) -> None:
    if state.recovery_hints and state.recovery_hints[-1] == hint:
        return
    state.recovery_hints.append(hint)
    context_manager.add_runtime_guidance(hint)


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
