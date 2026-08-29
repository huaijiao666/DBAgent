# Forge development instructions

These instructions apply to the entire repository.

## Project objective

Build a local, repository-aware, self-verifying Coding Agent Harness from first principles. A future completed system should explore repositories, understand relevant code, maintain an explicit plan, edit through patches, run local verification, recover from failures, manage growing context, and emit an observable execution trace.

## Milestone discipline

- Work on exactly the milestone requested by the user.
- Inspect the repository and current implementation before proposing changes.
- Explain the milestone's design and module boundaries briefly before editing.
- Do not implement or scaffold behavior belonging to later milestones.
- Preserve stable existing architecture. Report conflicts before considering a broad refactor.
- Run focused tests for the milestone and check for regressions in existing tests.
- Stop after reporting the completed milestone.

## Hard implementation constraints

- Support Python 3.11 and newer.
- Do not use an Agent framework or Agent SDK. Prohibited examples include LangChain, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, AutoGen, and CrewAI.
- Do not invoke Codex CLI, Claude Code, or another ready-made Coding Agent as the project's runtime.
- The OpenAI official Python client may be used only for model API communication.
- Prefer the OpenAI Responses API.
- Use `gpt-5.6-sol` as the default model unless the user explicitly changes the requirement.
- The model may request tools only through native function calling.
- Execute filesystem, search, patch, shell, and verification tools locally. Do not use hosted Code Interpreter, hosted shell, Files API, server-side apply patch, or equivalent hosted execution.
- Implement conversation and context management, the agent loop, tool definitions and dispatch, tool-call parsing, local execution, retries, termination, safety, editing, and verification inside this project.

## Configuration and secrets

- Read `OPENAI_API_KEY` only from the process environment.
- Never place a real API key in source code, tests, Git history, documentation, logs, fixtures, or example configuration.
- Keep local `.env` files ignored. `.env.example` may list variable names and safe defaults but must never contain credentials.
- Keep model behavior settings explicit, including `FORGE_MODEL` and `FORGE_REASONING_EFFORT`.

## Architecture principles

- Keep clear module boundaries; do not accumulate unrelated behavior in an entry-point module.
- Prefer the Python standard library when it provides a clear implementation.
- Favor direct, explainable code over metaprogramming, hidden global state, and premature abstraction.
- Represent task state, plan, step number, observations, tool calls, and verification results with explicit data structures.
- Make key components independently testable, especially the tool registry, path sandbox, command executor, patch application, context manager, repository index, and termination logic.
- Prefer patch-oriented editing over whole-file rewrites in the eventual Agent runtime.
- Use deterministic compiler, test, and lint results for verification whenever available; do not ask another model to guess whether work is correct.
- Produce structured events for significant steps so a CLI and trace recorder can observe execution.

## Local execution and safety

- Restrict all project tool access to the user-selected workspace.
- Resolve every filesystem path before access and reject paths outside the workspace.
- Apply timeouts to shell commands and limit captured stdout and stderr.
- Define an explicit policy for dangerous commands and operations.
- Treat model-generated tool input as untrusted data and validate it at the local execution boundary.
- Keep local edits reversible and avoid destructive operations unless the user explicitly requests them.

## Testing expectations

- Add focused unit tests with each new component.
- Test success paths, validation failures, safety boundaries, timeouts, truncation, and termination behavior where relevant.
- Run the focused tests first, then the existing test suite before finishing a milestone.
- Do not claim verification that was not actually run.

## Git workflow

- Do not run `git commit`.
- Do not amend, rebase, squash, reset history, or force-push.
- Do not create unrelated branches.
- Preserve unrelated user changes in a dirty worktree.
- At milestone completion, report the Git diff summary, changed files, tests run and results, architecture added, known limitations, and at least three points the user should understand before committing.
