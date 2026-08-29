# Forge end-to-end demo

The demo repository is `tests/fixtures/taskboard_repo`. It is a small but
realistic Python project with separate `models`, `repository`, `service`, and
`cli` modules, a `pyproject.toml`, and six baseline pytest tests. Each demo run
copies this template into a fresh workspace, initializes a local Git repository,
and stages the baseline files without making a commit. This keeps `git_diff`
scoped to the demo workspace and keeps the three tasks independent.

## Tasks

### A. Locate and fix a bug

`TaskService.search` is documented to match a task title or one of its tags, but
the initial implementation only checks titles. The Agent must add a tag-only
regression test, fix the service, run focused tests, and run the full suite.

### B. Add a cross-file feature

Add completed-task archiving: `InMemoryTaskRepository.remove_completed()` removes
completed tasks and returns a count, `TaskService.archive_completed()` delegates
to it, and a `purge` CLI command prints `archived N tasks`. The Agent must add
repository, service, and CLI tests while preserving existing APIs.

### C. Change behavior and add regression coverage

Change `TaskService.list_open` from insertion order to case-insensitive title
order while continuing to exclude completed tasks. The Agent must add a
deliberately out-of-order regression test and run focused plus full pytest.

## Observed runs

The final isolated runs used `gpt-5.6-luna` with reasoning effort `max` through
the temporary launcher. All three final runs ended `VERIFIED`:

| task | final workspace | result | changed files | verification |
| --- | --- | --- | --- | --- |
| A | `.forge/e2e_runs_20260829/task_a_safe` | VERIFIED (12 steps) | `taskboard/service.py`, `tests/test_service.py` | focused and full pytest passed |
| B | `.forge/e2e_runs_20260829/task_b_clean` | VERIFIED (14 steps) | repository, service, CLI, and their tests | focused 8 passed; full 8 passed |
| C | `.forge/e2e_runs_20260829/task_c_clean` | VERIFIED (11 steps) | `taskboard/service.py`, `tests/test_service.py` | focused 5 passed; full 7 passed |

The traces are JSONL files under each workspace's `.forge/trace.jsonl`; they
contain no API key or bearer token. The traces also record plan updates, tool
calls, patch results, verification status, and final status.

## Failure review

* An initial A run hit `max_steps=12` after a context-mismatch patch; the local
  patch tool rejected it atomically, and a later run succeeded with a larger
  generic budget. This was model efficiency, not a harness defect.
* A broad priority-feature prompt caused the model to repeatedly inspect files
  and then hit an external provider error without editing. A narrower but still
  cross-file archive task made the demo stable; no runtime special case was
  added.
* One archive attempt produced a patch that removed existing methods and later
  ran `compileall` instead of pytest. The verification trace exposed the test
  failure; this was a model patch/verification mistake, not a dispatch failure.
* Those attempts also revealed a generic safety issue: Git could discover the
  parent Forge checkout when a workspace lacked its own `.git`. The command
  executor now sets `GIT_CEILING_DIRECTORIES` to the workspace parent, with a
  regression test. The fixture runs initialize their own local Git repository.

## Known limitations

* Model behavior and provider latency are nondeterministic; a good task may need
  a larger `max_steps` budget, and the harness intentionally reports
  `INCOMPLETE` rather than guessing completion.
* The fixture is intentionally small and in-memory. It demonstrates local
  repository exploration, patching, tests, and Git diff, not persistence,
  packaging, or a production CLI deployment.
* `git_diff` is useful when the selected workspace is a Git checkout. A plain
  non-Git directory returns Git's deterministic error observation; the harness
  does not create commits automatically.
* Static repository intelligence is Python-AST based and deliberately shallow;
  it does not resolve dynamic imports or prove semantic correctness.
* Deterministic verification recognizes common pytest/compiler/linter commands;
  unfamiliar project-specific commands are still executable but do not update
  the verification tracker.
