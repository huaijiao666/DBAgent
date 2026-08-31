# Snake

A complete, playable Snake game built with zero dependencies: plain HTML, CSS,
and vanilla JavaScript drawn on an HTML5 Canvas. No build step, no npm
packages, and no web server required — just open `index.html` in a browser.

## Features

- Start / Restart via the on-screen button or the keyboard.
- Move the snake with **Arrow Keys** or **WASD**.
- Live **score** display.
- **Wall & self collision** end the game with a game-over overlay.
- **Responsive** canvas that resizes to fit the viewport for any screen size.
- Board full **win** condition.
- Seedable, deterministic game logic kept fully DOM-free and unit-testable in
  Node.

## Files

| File          | Purpose                                                        |
| ------------- | -------------------------------------------------------------- |
| `index.html`  | Markup, HUD, canvas and keyboard wiring.                        |
| `styles.css`  | Layout, theming, responsive canvas container, game-over overlay.|
| `game.js`     | `Snake` module: pure game rules + browser-only bindings.         |
| `tests.js`    | Deterministic Node unit tests for the game logic.               |
| `README.md`   | This documentation.                                             |

## Run the game

Open `index.html` in any modern web browser (Chrome, Firefox, Edge, Safari).
The file can be opened directly from disk (`file://`) — no server needed.

## Run the tests

Requires only Node.js (no packages):

```sh
node --check game.js   # syntax check
node tests.js          # run the logic tests
```

Both commands should exit cleanly (tests print `passed: N, failed: 0`).

## Controls

| Action                  | Key(s)                       |
| ----------------------- | ---------------------------- |
| Move up                 | `ArrowUp` or `W`             |
| Move down               | `ArrowDown` or `S`           |
| Move left               | `ArrowLeft` or `A`           |
| Move right              | `ArrowRight` or `D`          |
| Restart after game over | `Space` or the Restart button|

## Architecture

The core rules live in the `Snake` module exposed by `game.js`:

- `Snake.DIRECTIONS` — movement deltas.
- `Snake.createState(opts)` — build an initial board with a seedable RNG.
- `Snake.step(state)` — advance one tick (handles motion, eating, collisions,
  and the win condition).
- `Snake.changeDirection(state, dir)` — set the next heading, rejecting an
  immediate 180-degree reverse.
- `Snake.placeFood(state)` / `Snake.occupiesCell(...)` — board helpers.

This module never touches `document` or `window`, so `tests.js` can `require`
it directly under Node. All browser wiring (canvas rendering, input, the game
loop, responsive resize) is gated behind a `typeof document !== 'undefined'`
check so Node never executes it.

## Determinism

The game uses a small, seeded PRNG (mulberry32). Provided you supply the same
`seed`, `createState`, food placement, and movement are fully reproducible,
which keeps the automated tests stable across runs.
