# Tic-Tac-Toe

A complete, playable, zero-dependency browser Tic-Tac-Toe game built with
vanilla HTML, CSS, and JavaScript. No packages, no build step, and no server —
just open the file in a browser.

## Features

- **Responsive, polished board** — a 3×3 grid that adapts to any screen size.
- **Player-turn status** — a live readout announces whose turn it is, who won,
  or a draw.
- **Win and draw detection** — all eight winning lines are checked.
- **Highlighted winning cells** — the winning row/column/diagonal is
  highlighted and animated.
- **Restart** — reset the board with one click.
- **Keyboard accessibility** — arrow keys navigate between cells and
  Enter/Space activate a focused cell (native button behavior); a visible
  focus ring is provided.

## Files

| File           | Purpose                                                    |
| -------------- | ---------------------------------------------------------- |
| `index.html`   | Page markup and mounts the game shell.                     |
| `styles.css`   | Styling, layout, animations, focus states.                 |
| `game.js`      | DOM-free rules engine **and** browser UI controller.       |
| `tests.js`     | Deterministic Node unit tests for the rules engine.        |
| `README.md`    | This documentation.                                        |

## Running the game

Open `index.html` in any modern browser. There is no build step.

## Rules engine design

`game.js` separates game logic from the DOM so it can be tested
deterministically in Node.

- The board is an array of length 9, indexed `0..8`:

  ```
  0 1 2
  3 4 5
  6 7 8
  ```

- Each cell is `''` (empty), `'X'`, or `'O'`.
- When loaded as a plain `<script>`, the rules engine is exposed as
  `window.TTT`.
- When loaded as a CommonJS module (Node), it `module.exports` the same API.

### Rules API

| Constant      | Description                                    |
| ------------- | ---------------------------------------------- |
| `PLAYERS`     | `['X', 'O']`                                   |
| `LINES`       | The 8 winning index triples.                   |

| Function                 | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
| `createBoard()`          | Returns a fresh empty 9-cell board.                  |
| `nextPlayer(board)`      | Whose turn it is (`'X'` or `'O'`).                   |
| `isValidMove(b,i,p)`     | Whether playing `p` at index `i` is legal.           |
| `play(b,i,p)`            | Returns a new board with the move applied; illegal moves return an unchanged copy. |
| `winningLine(b,p)`       | The winning line for `p`, or `null`.                 |
| `statusOf(b)`            | Descriptor: `{ winner, line, draw, over, current }`. |

All rules functions are pure — they never mutate the input board.

## Running the tests

Requires [Node.js](https://nodejs.org/) (any recent version).

```bash
node --check game.js
node tests.js
```

`node --check` validates `game.js` parses correctly (including the browser-only
UI block). `node tests.js` runs a deterministic suite covering board creation,
turn alternation, move legality, win and draw detection, and API shape, then
exits `0` on success.

Both commands must exit successfully for the suite to pass.
