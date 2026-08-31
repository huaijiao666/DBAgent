# Snake

A small, dependency-free Snake game. The game loop, rules, and rendering live
in a single file, `game.js`; the browser wiring in the same file is skipped
when it is loaded under Node.js, so the core logic is unit-testable.

## Play

Open `index.html` in a browser. No build step or server is required (the game
runs fine from `file://`, though your browser may disable local storage for
high scores on that scheme — the game degrades gracefully).

## Controls

- Arrows or **WASD** — move
- **Space** — start / pause / resume
- **Enter** — start / restart when idle, game over, or won
- **R** — restart
- On-screen D-pad and buttons work on touch devices

## How it works

- A 24×24 grid. Apples appear one at a time on a free cell.
- Eating an apple grows the snake, adds to the score, and (when the board is
  cleared) wins the game.
- Hitting the wall or your own body ends the game.
- The snake cannot reverse into itself.
- The game runs on `requestAnimationFrame` with a fixed logical step interval,
  so movement speed is independent of the display refresh rate.
- High score is persisted to `localStorage`.

## Development / testing

`game.js` exposes `SnakeGame` and `mulberry32` via `module.exports` in CommonJS
environments (Node), and attaches them to `window`/`globalThis` in browsers.
The seeded PRNG makes tests deterministic.

Run the test suite:

```sh
node tests.js
```

Validates:

- Full rule set: movement, growth, apples, score, walls, self-collision.
- Win state when the board is cleared.
- Deterministic RNG.
- Symmetric API (paused state, direction clamping, direction changes).

Sanity-check the source:

```sh
node --check game.js
```

## Files

| File         | Purpose                                  |
| ------------ | ---------------------------------------- |
| `game.js`    | Core game logic plus browser wiring      |
| `index.html` | Page markup shell                        |
| `styles.css` | Styling for the static layout and HUD    |
| `tests.js`   | Deterministic Node.js test suite         |
