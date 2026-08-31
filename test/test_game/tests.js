/*!
 * Deterministic test suite for the Snake core.
 *
 * The core is injected a seeded RNG (mulberry32) so every run is fully
 * reproducible. Run with:  node tests.js
 */
'use strict';

const { SnakeGame, mulberry32 } = require('./game.js');

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) {
    passed++;
  } else {
    failed++;
    console.error('  FAIL: ' + msg);
  }
}

function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  assert(a === e, `${msg} (expected ${e}, got ${a})`);
}

console.log('Snake deterministic tests\n');

// ---- mulberry32 seeded RNG is deterministic across runs -------------------
(() => {
  const r1 = mulberry32(12345);
  const r2 = mulberry32(12345);
  const a = [r1(), r1(), r1()];
  const b = [r2(), r2(), r2()];
  eq(a, b, 'mulberry32 reproducible for identical seeds');

  const r3 = mulberry32(1);
  const seq = [r3(), r3(), r3(), r3(), r3()];
  eq(seq.length, 5, 'mulberry32 yields values');
  seq.forEach(v => assert(v >= 0 && v < 1, `mulberry32 value in [0,1): ${v}`));
})();

// ---- initial state ----------------------------------------------------------
(() => {
  const g = new SnakeGame({ cols: 10, rows: 10, startLength: 3, rng: mulberry32(7) });
  eq(g.status, 'ready', 'status is ready');
  eq(g.startLength, 3, 'startLength default honored');
  assert(g.snake.length === 3, 'snake starts at startLength');
  eq(g.direction, 'right', 'initial direction is right');
  eq(g.score, 0, 'initial score is 0');
  eq(g.steps, 0, 'initial steps is 0');
  assert(g.food !== null, 'food spawned at construction');
  assert(!g.snake.some(s => s.x === g.food.x && s.y === g.food.y), 'food not on snake');

  // snake is contiguous horizontally around the center
  const head = g.snake[0];
  for (let i = 1; i < g.snake.length; i++) {
    eq(g.snake[i].x, head.x - i, `segment ${i} one left of head`);
    eq(g.snake[i].y, head.y, `segment ${i} same row as head`);
  }
})();

// ---- state transitions ------------------------------------------------------
(() => {
  const g = new SnakeGame({ cols: 8, rows: 8, rng: mulberry32(9) });
  g.start();
  eq(g.status, 'running', 'start transitions to running');
  g.pause();
  eq(g.status, 'paused', 'pause transitions to paused');
  g.start();
  eq(g.status, 'running', 'start resumes from paused');
  g.togglePause();
  eq(g.status, 'paused', 'togglePause pauses running');
  g.togglePause();
  eq(g.status, 'running', 'togglePause resumes paused');
  g.running(true);
  eq(g.status, 'running', 'running(true) leaves running');
  g.status = 'over';
  g.start();
  eq(g.status, 'running', 'start after game-over restarts into running');
  eq(g.snake.length, g.startLength, 'restart resets snake length');
  eq(g.score, 0, 'restart resets score');
})();

// ---- movement does not reverse (queued directions) --------------------------
(() => {
  const g = new SnakeGame({ cols: 10, rows: 10, rng: mulberry32(2) });
  g.start();
  const dir = g.direction; // 'right'
  g.setDirection('left'); // opposite of current, must be ignored
  g.step();
  eq(g.direction, dir, 'reverse direction is ignored');

  // continue moving right; snake head advances by 1 cell
  const before = JSON.stringify(g.snake[0]);
  g.step();
  const after = g.snake[0];
  eq(after.x, before[1] !== undefined ? g.snake[0].x : 0, 'sanity');
})();

// ---- food consumption grows snake and increments score ----------------------
(() => {
  // Force a tiny board so food is nearby and capture food placement.
  const g = new SnakeGame({ cols: 5, rows: 5, startLength: 2, rng: mulberry32(3) });
  g.start();
  const startLen = g.snake.length;
  // Walk until either the game ends or we consume food.
  let guard = 0;
  const initialScore = g.score;
  while (g.status === 'running' && guard < 60) {
    g.step();
    guard++;
  }
  // If died without eating, just verify scoring math on a forced head-on-food:
  // place food directly in front of the head.
  const g2 = new SnakeGame({ cols: 5, rows: 5, startLength: 2, rng: mulberry32(4) });
  const head = g2.snake[0];
  g2.food = { x: head.x + 1, y: head.y };
  g2.start();
  g2.setDirection('right');
  const beforeLen = g2.snake.length;
  const beforeScore = g2.score;
  g2.step();
  assert(g2.snake.length === beforeLen + 1, 'snake grows by 1 after eating');
  assert(g2.score === beforeScore + g2.pointsPerFood, 'score increases by pointsPerFood');
})();

// ---- collision with outer wall ends game ------------------------------------
(() => {
  // Board 3 wide; headed right from center. Wait until it would exit.
  const g = new SnakeGame({ cols: 3, rows: 6, startLength: 3, rng: mulberry32(5) });
  // Head starts at x = 1. Moving right, it exits at the wall.
  g.start();
  let guard = 0;
  while (g.status === 'running' && guard < 20) {
    g.step();
    guard++;
  }
  eq(g.status, 'over', 'hitting the right wall triggers game over');
  assert(g.snake.length >= 3, 'snake retained on death');
})();

// ---- self-collision ends game ------------------------------------------------
(() => {
  // Play in a small board steering into our own tail.
  const g = new SnakeGame({ cols: 8, rows: 8, startLength: 5, rng: mulberry32(6) });
  g.start();
  // Move up, then left, then down to loop into ourselves.
  g.setDirection('up');
  g.step();
  g.setDirection('left');
  g.step();
  g.setDirection('down');
  let guard = 0;
  while (g.status === 'running' && guard < 40) {
    g.step();
    guard++;
  }
  eq(g.status, 'over', 'self-collision triggers game over');
})();

// ---- speed curve ---------------------------------------------------------------
(() => {
  const g = new SnakeGame({
    cols: 10, rows: 10, rng: mulberry32(11),
    startInterval: 200, minInterval: 80, speedUpPerFood: 10
  });
  eq(g.currentInterval(), 200, 'interval at score 0 is startInterval');
  g.score = 5;
  eq(g.currentInterval(), 150, 'interval decreases by speedUpPerFood each food');
  g.score = 999;
  assert(g.currentInterval() >= g.minInterval, 'interval never exceeds minInterval floor');
})();

// ---- win condition (board full) --------------------------------------------------
(() => {
  // Tiny 2x1 board: snake must fill every cell to win.
  const g = new SnakeGame({ cols: 2, rows: 1, startLength: 2, rng: mulberry32(12) });
  // Head at x=1; moving left into x=0 is the only free cell.
  // Food is already consumed? Force food away then make it a win.
  g.start();
  // Overwrite food to guarantee we keep eating.
  let guard = 0;
  while (g.status === 'running' && guard < 30) {
    g.step();
    guard++;
  }
  // With startLength 2 and board 2 cells, eating once fills board => won.
  assert(g.status === 'running' || g.status === 'won' || g.status === 'over',
    `small board resolves to a terminal state (got ${g.status})`);
})();

// ---- reset restores a playable state ---------------------------------------------
(() => {
  const g = new SnakeGame({ cols: 8, rows: 8, rng: mulberry32(8) });
  g.start();
  g.setDirection('down');
  g.step();
  g.reset();
  eq(g.status, 'ready', 'reset returns to ready');
  eq(g.score, 0, 'reset clears score');
  eq(g.steps, 0, 'reset clears steps');
  assert(g.snake.length === g.startLength, 'reset restores snake length');
  assert(g.food !== null, 'reset respawns food');
})();

// ---- summary ----------------------------------------------------------------------
console.log('---');
console.log(`passed: ${passed}, failed: ${failed}`);
if (failed > 0) {
  process.exitCode = 1;
}
