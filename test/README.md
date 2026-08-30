# Snake

A small, deterministic terminal Snake game implemented with a pure Python game
model and a minimal line-oriented runner. It uses only the Python standard
library at runtime.

## Run

From the repository root:

```text
python3 -m snake_game
```

The runner prints the board and waits for one command at a time. Press Enter
to continue in the current direction, or use:

- **W** — up
- **A** — left
- **S** — down
- **D** — right
- **Q** — quit

Food is shown as `*`, the head as `@`, and the body as `o`.

## Design notes

- `snake_game.game` contains immutable `GameState` values and side-effect-free
  rules, making movement straightforward to test.
- Immediate 180-degree turns are ignored, while a move into the old tail is
  allowed when the snake is not growing.
- Food placement is deterministic: a free preferred cell is used when given;
  otherwise the first free cell in row-major order is selected. No random
  generator or external dependency is needed.
- `snake_game.runner` is the I/O adapter. Importing it is safe; the input loop
  starts only through `python3 -m snake_game` or an explicit call to `main()`.
