# Terminal Snake

Dependency-free Snake for Python 3.11+, played one turn at a time in a terminal.

## Run

From the repository root:

```text
python -m snake_game
```

Enter `W`, `A`, `S`, or `D`, followed by Enter, to turn and advance one turn.
Enter `Q` followed by Enter to quit. After a win or game over, enter `R` to
restart or `Q` to quit.

## Test

The tests require `pytest` in the environment:

```text
python -m pytest -q
```

The game itself has no third-party runtime dependencies and does not use pygame.
