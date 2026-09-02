# DBAgent architecture

DBAgent follows one rule: the model proposes actions, while the local Harness
owns state, execution, safety and completion evidence.

```text
User / CLI / TUI / Browser UI
              |
              v
DBAgentRepl or BrowserAgentController
              |
              v
AgentLoop ---- ContextManager ---- AgentState / TaskPlan
   |              |                       |
   |              +-- local bounded prompt +-- verification and recovery facts
   v
Model transport (Responses or Chat Completions compatibility adapter)
   |
   v
ToolRegistry -> Workspace / PatchApplier / CommandExecutor / RepositoryIndex
   |
   v
TraceRecorder -> JSONL, CLI, TUI and browser event stream
```

## State ownership

`AgentState` explicitly records the task, plan, step budget, observations,
verification result, recovery hints and final status. `ContextManager` builds
every model request from local data; it does not use `previous_response_id` or
provider-side conversation storage. Recent native tool observations are kept in
full within a budget, while older evidence is compacted deterministically.

Relevant source snippets are tagged with their workspace-relative path. A
successful patch, create or write operation invalidates retained snippets for
the files it changed, so old source is not presented as current code.

## Execution and verification

The model can only request declared native function tools. Tool arguments are
untrusted until validated by the local registry. File paths are canonicalized
against the selected workspace and symlink escapes are rejected. Patches are
validated before writes and handled multi-file write failures are rolled back.

`run_command` uses argv without a shell, a timeout, bounded output and a
scrubbed environment that excludes API keys. It is policy constrained but is
not an operating-system sandbox; do not run untrusted repositories with high
privileges.

After code changes, deterministic command evidence drives the verification
state. A final textual model response is insufficient by itself: the loop uses
the saved verification state, plan completion and the step budget to decide
between verified completion, incomplete work, blocked work and errors.

## Observability and recovery

Every important transition becomes a redacted JSONL trace event. The same event
can be rendered by the line CLI, ANSI TUI and loopback browser dashboard. The
browser UI never receives provider credentials; it only talks to the local
Python process through a random-token loopback endpoint. The interactive DBA
launcher may load a user-authorized, Git-ignored provider TOML or key file
from the editable checkout root (or an explicit local path); those values
remain in memory and never enter session files or traces.

Runs can accept steering at safe boundaries. Session checkpoints persist the
bounded conversation, structured plan and key observations after completed
steps. A resumed session restores task context, but it does not pretend to
resume an interrupted HTTP request or a command that was already executing.

## Deliberate limits

The system has one model-driven agent loop rather than a hidden planner or
reviewer fleet. Python repository intelligence is AST-based and shallow;
dynamic runtime behavior and visual quality still need deterministic tests or
human inspection. Provider compatibility, especially non-OpenAI tool calling,
is capability-dependent and should be tested before an important demo.
