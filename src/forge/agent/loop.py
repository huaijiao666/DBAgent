"""The first minimal Coding Agent loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, Protocol

from forge.agent.context import ContextBudget, ContextManager
from forge.agent.mode import TaskMode, instructions_for_mode, resolve_task_mode
from forge.agent.plan import PlanStore, TaskPlan, update_plan_tool
from forge.agent.state import AgentState, AgentStatus
from forge.agent.verification import (
    VerificationStatus,
    VerificationTracker,
    suggested_verification_commands,
)
from forge.llm import (
    ModelConnectionError,
    ModelAPIError,
    ModelRateLimitError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    ModelCommunicationError,
)
from forge.tools import ToolObservation, ToolRegistry

_RECOVERY_AFTER_ROUNDS = 2
_ASK_MAX_TOOL_ROUNDS = 3
_DEFAULT_MODEL_RETRIES = 5
_RETRYABLE_MODEL_ERRORS = (
    ModelConnectionError,
    ModelTimeoutError,
    ModelRateLimitError,
)
_RETRYABLE_API_STATUS_CODES = frozenset({408, 409, 425, 500, 502, 503, 504})
_READ_PROGRESS_TOOLS = {
    "get_repo_map",
    "search_symbol",
    "read_symbol",
    "list_files",
    "read_file",
    "search_text",
}
_EVIDENCE_PROGRESS_TOOLS = {*_READ_PROGRESS_TOOLS, "git_diff"}
_ASK_TOOL_NAMES = {
    *_READ_PROGRESS_TOOLS,
    "git_diff",
    "run_command",
}
_PATCH_FRESHNESS_TOOLS = {"read_file", "git_diff", "run_command"}


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
        max_model_retries: int = _DEFAULT_MODEL_RETRIES,
        mode: TaskMode | str = TaskMode.AUTO,
        instructions: str | None = None,
        context_budget: ContextBudget | None = None,
        initial_plan: TaskPlan | None = None,
        verification_required: bool = False,
        trace: TraceSink | None = None,
        state_checkpoint: Callable[[AgentState], None] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_model_retries < 0:
            raise ValueError("max_model_retries must not be negative")
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._max_model_retries = max_model_retries
        self._mode = mode if isinstance(mode, TaskMode) else TaskMode(mode)
        self._instructions = instructions
        self._context_budget = context_budget
        self._initial_plan = initial_plan
        self._verification_required = verification_required
        self._trace = trace
        self._state_checkpoint = state_checkpoint

    def run(
        self,
        task: str,
        *,
        workspace: Path,
        launch_directory: Path | None = None,
    ) -> AgentState:
        resolved_mode = resolve_task_mode(task, self._mode)
        state = AgentState.start(
            task=task,
            workspace=workspace,
            launch_directory=launch_directory,
            max_steps=self._max_steps,
            mode=resolved_mode,
        )
        context_manager = ContextManager(
            task,
            execution_context=_execution_context(state),
            budget=self._context_budget,
        )
        plan_store = PlanStore.resume(self._initial_plan)
        verification = VerificationTracker(
            mutation_generation=1 if self._verification_required else 0
        )
        no_progress_rounds = 0
        malformed_patch_failures = 0
        patch_reinspection_required = False
        write_fallback_available = False
        repair_required = False
        repair_readonly_rounds = 0
        last_reported_compaction = (0, 0)
        ask_tool_rounds = 0
        seen_evidence: set[tuple[tuple[str, str], int]] = set()
        run_registry = (
            self._tool_registry.select(_ASK_TOOL_NAMES)
            if resolved_mode is TaskMode.ASK
            else self._tool_registry.clone()
        )
        if resolved_mode is TaskMode.CODE:
            run_registry.register(update_plan_tool(plan_store))
        if plan_store.plan is not None:
            state.plan = plan_store.plan
            state.plan_history = list(plan_store.history)
            context_manager.set_plan(plan_store.plan.to_prompt())
        if self._verification_required:
            context_manager.set_verification_status(
                "status=stale; a prior DBA turn left failed or stale deterministic "
                "evidence, so this continuation must verify the current files"
            )
        tool_schemas = run_registry.schemas()
        self._record_trace(
            "run_started",
            0,
            {
                "mode": resolved_mode.value,
                "workspace": ".",
                "max_steps": state.max_steps,
                "tool_count": len(tool_schemas),
            },
        )

        while state.step < state.max_steps:
            state.step += 1
            request_tool_schemas = (
                tuple(tool for tool in tool_schemas if tool.name != "apply_patch")
                if patch_reinspection_required
                else tool_schemas
            )
            if resolved_mode is TaskMode.CODE and not write_fallback_available:
                request_tool_schemas = tuple(
                    tool for tool in request_tool_schemas if tool.name != "write_file"
                )
            repair_action_required = repair_required and repair_readonly_rounds >= 2
            if repair_action_required:
                request_tool_schemas = tuple(
                    tool
                    for tool in request_tool_schemas
                    if tool.name not in _READ_PROGRESS_TOOLS
                )
            snapshot = context_manager.build_context(step=state.step)
            state.context = list(snapshot.input_items)
            state.context_usage.append(snapshot.usage)
            self._record_trace(
                "model_request",
                state.step,
                {
                    "input_items": len(snapshot.input_items),
                    "tools": [tool.name for tool in request_tool_schemas],
                    "context_usage": _context_usage_payload(snapshot.usage),
                },
            )
            compaction = (
                snapshot.usage.compacted_observations,
                snapshot.usage.truncated_items,
            )
            if compaction != last_reported_compaction and any(compaction):
                self._record_trace(
                    "context_compacted",
                    state.step,
                    {
                        "compacted_observations": snapshot.usage.compacted_observations,
                        "recent_observations": snapshot.usage.recent_observations,
                        "truncated_items": snapshot.usage.truncated_items,
                        "input_characters": snapshot.usage.input_characters,
                        "budget_characters": snapshot.usage.budget_characters,
                        "approximate_tokens": snapshot.usage.approximate_tokens,
                    },
                )
                last_reported_compaction = compaction
            force_final = (
                (
                    state.step == state.max_steps
                    or (
                        resolved_mode is TaskMode.ASK
                        and (
                            ask_tool_rounds >= _ASK_MAX_TOOL_ROUNDS
                            or no_progress_rounds >= 1
                        )
                    )
                )
                and not (
                    verification.requires_verification
                    and not verification.is_verified
                )
            )
            request_instructions = self._instructions or instructions_for_mode(
                resolved_mode
            )
            if repair_action_required:
                request_instructions += (
                    "\n\nA deterministic check has failed and two diagnosis rounds "
                    "have already completed. Reading/discovery tools are temporarily "
                    "withheld. Use the evidence you have: apply a focused patch, run "
                    "a targeted verification command, or update the plan honestly."
                )
            remaining_after_this_turn = state.max_steps - state.step
            if (
                resolved_mode is TaskMode.CODE
                and not force_final
                and remaining_after_this_turn <= 2
            ):
                request_instructions += (
                    "\n\nThe hard step budget is nearly exhausted: only "
                    f"{remaining_after_this_turn} later model turn(s) remain. "
                    "Do not explore or reread unchanged files. Execute the most "
                    "important missing deterministic verification or corrective "
                    "action now, then update the existing plan accurately."
                )
            if force_final:
                request_instructions += (
                    "\n\nThis is the final model turn. Do not call tools. Give the "
                    "user a concise, honest result now. If work remains, state "
                    "exactly what is incomplete instead of claiming success."
                )
            response = self._create_model_response(
                ModelRequest(
                    input=snapshot.input_items,
                    instructions=request_instructions,
                    tools=request_tool_schemas,
                    # A coding turn can mutate files or verify a mutation.
                    # Ask providers for one decision at a time so a second
                    # call is not planned against a file state that the first
                    # call is about to change. Read-only ASK mode keeps safe
                    # parallel discovery available.
                    parallel_tool_calls=resolved_mode is TaskMode.ASK,
                    tool_choice="none" if force_final else "auto",
                ),
                step=state.step,
            )
            state.response_ids.append(response.response_id)
            self._record_trace(
                "model_response",
                state.step,
                _model_response_payload(response),
            )
            if response.function_calls and response.output_text.strip():
                self._publish_live(
                    "assistant_update",
                    state.step,
                    {"text": response.output_text.strip()},
                )

            if not response.function_calls:
                answer = response.output_text.strip()
                if not answer:
                    if state.step == state.max_steps:
                        state.status = AgentStatus.MAX_STEPS
                        state.final_answer = None
                        _sync_verification_state(
                            state, verification, no_progress_rounds
                        )
                        self._record_incomplete(state, verification)
                        return state
                    _append_hint(
                        state,
                        context_manager,
                        "The model returned no text and no tool call. Provide a "
                        "useful final answer or take one concrete next action.",
                        trace=self._trace,
                        step=state.step,
                    )
                    continue
                if verification.requires_verification and not verification.is_verified:
                    # Do not surface an unverified completion claim. The trace
                    # records that text existed, while user-visible state remains
                    # explicitly incomplete until deterministic evidence passes.
                    state.final_answer = None
                    hint = _final_verification_hint(verification)
                    if state.step == state.max_steps:
                        state.status = AgentStatus.MAX_STEPS
                        _sync_verification_state(
                            state, verification, no_progress_rounds
                        )
                        self._record_incomplete(state, verification)
                        return state
                    _append_hint(
                        state,
                        context_manager,
                        hint,
                        trace=self._trace,
                        step=state.step,
                    )
                    _sync_verification_state(state, verification, no_progress_rounds)
                    continue
                if (
                    resolved_mode is TaskMode.CODE
                    and plan_store.plan is not None
                    and not _plan_is_complete(plan_store.plan)
                ):
                    state.final_answer = answer if state.step == state.max_steps else None
                    if state.step == state.max_steps:
                        state.status = AgentStatus.MAX_STEPS
                        _sync_verification_state(
                            state, verification, no_progress_rounds
                        )
                        self._record_incomplete(state, verification)
                        return state
                    _append_hint(
                        state,
                        context_manager,
                        "The current plan still has unfinished steps. Complete or "
                        "explicitly block them before the final answer.",
                        trace=self._trace,
                        step=state.step,
                    )
                    continue
                state.final_answer = answer
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
                        "answer_characters": len(answer),
                        "mode": resolved_mode.value,
                    },
                )
                return state

            executed_calls = []
            turn_progress = False
            turn_repeated_evidence = False
            round_had_mutation = False
            round_had_failed_verification = False
            round_had_successful_verification = False
            truncated_tool_calls = _tool_calls_may_be_truncated(response)
            for call in response.function_calls:
                state.tool_calls.append(call)
                tool_start_payload = _tool_start_payload(call)
                current_plan_step = _current_plan_step(plan_store.plan)
                if current_plan_step:
                    tool_start_payload["plan_step"] = current_plan_step
                self._record_trace(
                    "tool_start",
                    state.step,
                    tool_start_payload,
                )
                if truncated_tool_calls:
                    observation = ToolObservation(
                        call_id=call.call_id,
                        tool_name=call.name,
                        success=False,
                        content=(
                            "Tool call was not executed because the model response "
                            "hit its output-length limit and its arguments may be "
                            "incomplete. Re-issue one complete native function call."
                        ),
                    )
                elif call.name == "write_file" and not write_fallback_available:
                    observation = ToolObservation(
                        call_id=call.call_id,
                        tool_name=call.name,
                        success=False,
                        content=(
                            "write_file is withheld by default because it replaces an "
                            "entire existing file. Use apply_patch for a focused edit. "
                            "It becomes available only after an apply_patch failure as "
                            "a small-file fallback."
                        ),
                    )
                elif call.name in _READ_PROGRESS_TOOLS and repair_action_required:
                    observation = ToolObservation(
                        call_id=call.call_id,
                        tool_name=call.name,
                        success=False,
                        content=(
                            "Diagnosis read budget is exhausted after a deterministic "
                            "verification failure. Use apply_patch or run_command with "
                            "the existing failure evidence."
                        ),
                    )
                elif call.name == "apply_patch" and patch_reinspection_required:
                    observation = ToolObservation(
                        call_id=call.call_id,
                        tool_name=call.name,
                        success=False,
                        content=(
                            "apply_patch is temporarily withheld after a successful "
                            "patch. First inspect the changed state with read_file, "
                            "git_diff, or run_command; then request the next patch."
                        ),
                    )
                else:
                    observation = run_registry.dispatch(call)
                state.observations.append(observation)
                executed_calls.append((call, observation))
                self._record_trace(
                    "tool_result",
                    state.step,
                    _tool_result_payload(call, observation),
                )
                if (
                    call.name == "apply_patch"
                    and not observation.success
                    and not patch_reinspection_required
                ):
                    write_fallback_available = True
                    if _has_malformed_arguments(observation):
                        malformed_patch_failures += 1
                    _append_hint(
                        state,
                        context_manager,
                        _patch_failure_hint(observation),
                        trace=self._trace,
                        step=state.step,
                    )
                    if malformed_patch_failures == 2:
                        tool_schemas = tuple(
                            tool
                            for tool in tool_schemas
                            if tool.name != "apply_patch"
                        )
                        fallback = (
                            "apply_patch produced malformed function arguments twice, "
                            "so it is temporarily unavailable for the rest of this "
                            "run. Do not retry it. For a small existing file you have "
                            "already read, use write_file as the local fallback; then "
                            "run a targeted deterministic verification."
                        )
                        _append_hint(
                            state,
                            context_manager,
                            fallback,
                            trace=self._trace,
                            step=state.step,
                        )
                        self._record_trace(
                            "tool_fallback",
                            state.step,
                            {
                                "disabled_tool": "apply_patch",
                                "reason": "two malformed argument payloads",
                                "fallback_tool": "write_file",
                            },
                        )
                event = verification.observe(call, observation)
                if call.name == "write_file" and observation.success:
                    write_fallback_available = False
                if call.name == "apply_patch" and event.mutation:
                    write_fallback_available = False
                if observation.success and call.name in _PATCH_FRESHNESS_TOOLS:
                    patch_reinspection_required = False
                turn_progress = turn_progress or event.mutation
                if event.mutation:
                    round_had_mutation = True
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
                    suggestions = suggested_verification_commands(call, observation)
                    if suggestions:
                        rendered = " or ".join(
                            " ".join(command) for command in suggestions
                        )
                        _append_hint(
                            state,
                            context_manager,
                            (
                                "Local deterministic verification suggestion for "
                                f"the changed file(s): {rendered}. Run it now unless "
                                "a more targeted repository test is available."
                            ),
                            trace=self._trace,
                            step=state.step,
                        )
                    if call.name == "apply_patch":
                        patch_reinspection_required = True
                        _append_hint(
                            state,
                            context_manager,
                            (
                                "Before requesting another apply_patch, inspect or "
                                "verify the changed state with read_file, git_diff, or "
                                "run_command. apply_patch is temporarily withheld to "
                                "prevent stale-context edits."
                            ),
                            trace=self._trace,
                            step=state.step,
                        )
                    _record_patch_event(self._trace, state.step, observation)
                if observation.success and call.name in _EVIDENCE_PROGRESS_TOOLS:
                    signature = (
                        _call_signature(call),
                        verification.mutation_generation,
                    )
                    if signature not in seen_evidence:
                        seen_evidence.add(signature)
                        turn_progress = True
                    else:
                        turn_repeated_evidence = True
                if event.record is not None:
                    _record_verification_event(
                        self._trace,
                        state.step,
                        event.record,
                        verification.status,
                    )
                    if event.record.passed:
                        round_had_successful_verification = True
                        turn_progress = True
                    else:
                        round_had_failed_verification = True
                        _append_hint(
                            state,
                            context_manager,
                            _verification_failure_hint(event.record),
                            trace=self._trace,
                            step=state.step,
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
                if call.name == "update_plan" and observation.success:
                    # A real plan transition is meaningful progress. A repeated
                    # identical snapshot is intentionally not progress, so it
                    # cannot mask a stalled tool loop.
                    content = observation.content
                    if isinstance(content, Mapping):
                        turn_progress = turn_progress or bool(
                            content.get("changed", False)
                        )
                if (
                    call.name == "update_plan"
                    and observation.success
                    and isinstance(observation.content, Mapping)
                    and observation.content.get("changed", False)
                ):
                    _record_plan_event(
                        self._trace,
                        state.step,
                        plan_store.plan,
                    )
            context_manager.record_turn(response, executed_calls)
            if resolved_mode is TaskMode.ASK:
                ask_tool_rounds += 1
            if plan_store.plan is not None:
                state.plan = plan_store.plan
                state.plan_history = list(plan_store.history)
                context_manager.set_plan(plan_store.plan.to_prompt())
            if round_had_mutation or round_had_successful_verification:
                repair_required = False
                repair_readonly_rounds = 0
            elif round_had_failed_verification:
                repair_required = True
                repair_readonly_rounds = 0
            elif repair_required:
                repair_readonly_rounds += 1
                if repair_readonly_rounds == 2:
                    _append_hint(
                        state,
                        context_manager,
                        (
                            "Two diagnosis rounds followed a deterministic test failure. "
                            "The next round withholds read/discovery tools so you must "
                            "apply a focused fix, rerun verification, or update the plan."
                        ),
                        trace=self._trace,
                        step=state.step,
                    )
            no_progress_rounds = (
                0 if turn_progress else no_progress_rounds + 1
            )
            recovery_threshold = (
                1
                if resolved_mode is TaskMode.CODE and turn_repeated_evidence
                else _RECOVERY_AFTER_ROUNDS
            )
            if no_progress_rounds >= recovery_threshold:
                recovery = (
                    "The read-only investigation is repeating evidence already "
                    "collected. Stop exploring and answer the user now unless "
                    "one specific unresolved fact is essential."
                    if resolved_mode is TaskMode.ASK
                    else (
                        f"No meaningful progress for {no_progress_rounds} tool "
                        "round(s). Stop rereading unchanged files. Consult the "
                        "current plan and take a different concrete action now. "
                        "If implementation exists and verification is pending, "
                        "run the appropriate targeted or full deterministic "
                        "command; otherwise update the plan honestly and finish."
                    )
                )
                _append_hint(
                    state,
                    context_manager,
                    recovery,
                    trace=self._trace,
                    step=state.step,
                )
            context_manager.set_verification_status(verification.latest_summary())
            _sync_verification_state(state, verification, no_progress_rounds)
            self._record_trace(
                "step_summary",
                state.step,
                {
                    "tools": len(executed_calls),
                    "succeeded": sum(
                        observation.success for _call, observation in executed_calls
                    ),
                    "failed": sum(
                        not observation.success
                        for _call, observation in executed_calls
                    ),
                    "current_plan_step": _current_plan_step(plan_store.plan),
                    "verification": verification.status.value,
                    "no_progress_rounds": no_progress_rounds,
                },
            )
            self._checkpoint(state)

        state.status = AgentStatus.MAX_STEPS
        state.final_answer = None
        _sync_verification_state(state, verification, no_progress_rounds)
        self._record_incomplete(state, verification)
        return state

    def _checkpoint(self, state: AgentState) -> None:
        """Persist a completed local step without making checkpointing mandatory."""

        if self._state_checkpoint is not None:
            self._state_checkpoint(state)

    def _record_incomplete(
        self,
        state: AgentState,
        verification: VerificationTracker,
    ) -> None:
        self._record_trace(
            "final",
            state.step,
            {
                "status": "INCOMPLETE",
                "verification_status": verification.status.value,
                "answer_characters": len(state.final_answer or ""),
                "mode": state.mode.value,
            },
        )

    def _create_model_response(
        self,
        request: ModelRequest,
        *,
        step: int,
    ) -> ModelResponse:
        """Retry only transient provider failures without consuming agent steps."""

        max_attempts = self._max_model_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return self._model_client.create_response(request)
            except ModelCommunicationError as error:
                retryable = _is_retryable_model_error(error)
                will_retry = retryable and attempt < max_attempts
                self._record_trace(
                    "model_error",
                    step,
                    {
                        "error_type": type(error).__name__,
                        "retryable": retryable,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "will_retry": will_retry,
                        "status_code": getattr(error, "status_code", None),
                    },
                )
                if will_retry:
                    self._record_trace(
                        "recovery",
                        step,
                        {
                            "reason": (
                                "Transient model communication failure; retrying "
                                f"request ({attempt}/{max_attempts})."
                            )
                        },
                    )
                    continue
                self._record_trace(
                    "final",
                    step,
                    {
                        "status": "ERROR",
                        "error_type": type(error).__name__,
                        "retryable": retryable,
                    },
                )
                raise


    def _record_trace(
        self,
        event: str,
        step: int,
        payload: dict[str, object],
    ) -> None:
        if self._trace is not None:
            self._trace.record(event, step=step, payload=payload)

    def _publish_live(
        self,
        event: str,
        step: int,
        payload: dict[str, object],
    ) -> None:
        publisher = getattr(self._trace, "publish", None)
        if callable(publisher):
            publisher(event, step=step, payload=payload)


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


def _plan_is_complete(plan) -> bool:
    return bool(plan.steps) and all(
        step.status.value == "completed" for step in plan.steps
    )


def _current_plan_step(plan) -> str | None:
    if plan is None:
        return None
    active = next(
        (step.step_id for step in plan.steps if step.status.value == "in_progress"),
        None,
    )
    if active is not None:
        return active
    return next(
        (step.step_id for step in plan.steps if step.status.value == "pending"),
        None,
    )


def _call_signature(call) -> tuple[str, str]:
    try:
        arguments = json.loads(call.arguments_json)
    except (TypeError, json.JSONDecodeError):
        return call.name, call.arguments_json
    return call.name, json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def _patch_failure_hint(observation) -> str:
    if _has_malformed_arguments(observation):
        return (
            "The apply_patch function arguments were malformed JSON, so the patch "
            "engine did not run and no files changed. Do not repeat the same tool "
            "payload. For a small file that has already been read in full, use "
            "write_file as the explicit fallback, then verify it."
        )
    reason = "unknown patch validation failure"
    if isinstance(observation.content, Mapping):
        value = observation.content.get("failure_reason")
        if isinstance(value, str) and value.strip():
            reason = value.strip()
    return (
        f"The patch was rejected atomically and no files changed: {reason}. "
        "Do not repeat the identical patch. If context did not match or was "
        "ambiguous, read the target once, then use smaller exact old_lines that "
        "identify one location. For a small file only, write_file is an explicit "
        "fallback after patching has genuinely failed."
    )


def _has_malformed_arguments(observation) -> bool:
    return isinstance(observation.content, str) and observation.content.startswith(
        "Invalid JSON arguments:"
    )


def _context_usage_payload(usage) -> dict[str, object]:
    return {
        "input_characters": usage.input_characters,
        "approximate_tokens": usage.approximate_tokens,
        "budget_characters": usage.budget_characters,
        "recent_observations": usage.recent_observations,
        "compacted_observations": usage.compacted_observations,
        "truncated_items": usage.truncated_items,
    }


def _execution_context(state: AgentState) -> str:
    """Render immutable local path facts that summaries must not guess."""

    workspace = str(state.workspace)
    launch_directory = str(state.launch_directory)
    location_guidance = (
        "The user launched DBA from inside the workspace."
        if state.launch_directory == state.workspace
        else (
            "The user launched DBA from a nested directory. Commands executed by "
            "tools still use the workspace root unless their validated cwd says "
            "otherwise. If giving manual commands, explain any required cd step."
        )
    )
    return (
        f"Absolute workspace root: {workspace}\n"
        f"User launch directory: {launch_directory}\n"
        "All relative repository tool paths are rooted at the absolute workspace "
        "above. Do not replace it with a parent repository or infer another root.\n"
        f"{location_guidance}"
    )


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


def _tool_start_payload(call) -> dict[str, object]:
    arguments = _parse_arguments(call.arguments_json)
    payload: dict[str, object] = {
        "call_id": call.call_id,
        "tool_name": call.name,
        "argument_keys": sorted(str(key) for key in arguments),
    }
    for key in ("path", "query", "symbol_id", "cwd"):
        value = arguments.get(key)
        if isinstance(value, str):
            payload[key] = value
            payload.setdefault("target", value)
    command = arguments.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        payload["command"] = _safe_command(
            [str(item) for item in command if isinstance(item, str)]
        )
    files = arguments.get("files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        payload["files"] = [
            str(item.get("path"))
            for item in files
            if isinstance(item, Mapping) and isinstance(item.get("path"), str)
        ]
    return payload


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
        if not observation.success:
            payload["failure_reason"] = content[:500]
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
    descriptions = {item.step_id: item.description for item in plan.steps}
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
            "current_step_description": descriptions.get(current_step),
            "completed_steps": sum(
                status == "completed" for status in statuses.values()
            ),
            "total_steps": len(plan.steps),
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


def _is_retryable_model_error(error: ModelCommunicationError) -> bool:
    """Return whether retrying the exact request is safe and useful.

    Client-side transport failures and rate limits are transient by definition.
    A provider ``4xx`` normally means the request itself is invalid and must be
    fixed locally, so repeating it would only consume time and quota. A small
    allow-list covers common transient HTTP failures reported by compatible
    Chat Completions providers.
    """

    if isinstance(error, _RETRYABLE_MODEL_ERRORS):
        return True
    return (
        isinstance(error, ModelAPIError)
        and error.status_code in _RETRYABLE_API_STATUS_CODES
    )


def _tool_calls_may_be_truncated(response: ModelResponse) -> bool:
    """Refuse to execute tools emitted by an output-limited model response.

    Providers can return JSON that happens to parse after reaching an output
    limit, while silently omitting the end of a command or patch. Treating that
    call as executable would turn an output-token limit into a local side
    effect. Both Chat Completions' ``length`` finish reason and Responses'
    ``max_output_tokens`` incomplete detail normalize into this guard.
    """

    if not response.function_calls:
        return False
    if response.status.casefold() in {"length", "incomplete"}:
        return True
    details = response.incomplete_details or {}
    reason = details.get("reason")
    return reason in {"max_output_tokens", "length"}
