# Taskboard fixture

Taskboard is a small local task-management package used for DBAgent end-to-end
demos. It has separate model, repository, service, and CLI modules plus pytest
coverage. The fixture intentionally keeps the domain small enough that a coding
agent can inspect, edit, and verify it in a short run.

Run the tests from this directory with:

```text
python -m pytest -q
```

The public search contract is to match a task title or one of its tags. The
initial tests cover title search; the tag-only case is left for the bug-fix
task.
