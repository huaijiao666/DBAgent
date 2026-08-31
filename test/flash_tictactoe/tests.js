'use strict';

/**
 * Deterministic Node unit tests for the DOM-free Tic-Tac-Toe rules in game.js.
 * Run with: node tests.js
 */

const assert = require('assert');
const TTT = require('./game.js');

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log('  \u2713 ' + name);
  } catch (err) {
    failed++;
    failures.push({ name: name, err: err });
    console.error('  \u2717 ' + name);
    console.error('      ' + (err && err.message));
  }
}

function eq(actual, expected, label) {
  assert.deepStrictEqual(actual, expected, label);
}

console.log('Tic-Tac-Toe rules tests');
console.log('-----------------------');

// --- Board creation ---
test('createBoard returns 9 empty cells', function () {
  const b = TTT.createBoard();
  assert.strictEqual(b.length, 9);
  assert.ok(b.every(function (c) { return c === ''; }));
});

// --- Initial turn ---
test('nextPlayer starts as X on empty board', function () {
  assert.strictEqual(TTT.nextPlayer(TTT.createBoard()), 'X');
});

test('nextPlayer alternates after a move', function () {
  const b = TTT.play(TTT.createBoard(), 0, 'X');
  assert.strictEqual(TTT.nextPlayer(b), 'O');
  const b2 = TTT.play(b, 4, 'O');
  assert.strictEqual(TTT.nextPlayer(b2), 'X');
});

// --- Move legality ---
test('isValidMove accepts an empty cell and correct player', function () {
  assert.strictEqual(TTT.isValidMove(TTT.createBoard(), 0, 'X'), true);
});

test('isValidMove rejects occupied cell', function () {
  const b = TTT.createBoard();
  b[2] = 'X';
  assert.strictEqual(TTT.isValidMove(b, 2, 'O'), false);
});

test('isValidMove rejects wrong player turn', function () {
  assert.strictEqual(TTT.isValidMove(TTT.createBoard(), 0, 'O'), false);
});

test('isValidMove rejects out-of-range index', function () {
  assert.strictEqual(TTT.isValidMove(TTT.createBoard(), 9, 'X'), false);
  assert.strictEqual(TTT.isValidMove(TTT.createBoard(), -1, 'X'), false);
});

test('isValidMove rejects non-integer index', function () {
  assert.strictEqual(TTT.isValidMove(TTT.createBoard(), 1.5, 'X'), false);
});

test('isValidMove rejects invalid player symbol', function () {
  assert.strictEqual(TTT.isValidMove(TTT.createBoard(), 0, 'Z'), false);
});

test('isValidMove rejects move after game is won', function () {
  const b = TTT.createBoard();
  b[0] = 'X'; b[1] = 'X'; b[2] = 'X';
  b[3] = 'O'; b[4] = 'O';
  assert.strictEqual(TTT.statusOf(b).winner, 'X');
  assert.strictEqual(TTT.isValidMove(b, 5, 'O'), false);
});

// --- play ---
test('play applies a legal move', function () {
  const b = TTT.createBoard();
  const next = TTT.play(b, 4, 'X');
  assert.strictEqual(next[4], 'X');
  assert.strictEqual(b[4], ''); // input unchanged
});

test('play rejects an illegal move and returns copy', function () {
  const b = TTT.createBoard();
  const next = TTT.play(b, 0, 'O'); // wrong turn
  assert.strictEqual(next[0], '');
});

// --- Winning detection ---
test('detects horizontal win and line', function () {
  const b = TTT.createBoard();
  b[0] = 'X'; b[1] = 'X'; b[2] = 'X';
  b[3] = 'O'; b[4] = 'O';
  const s = TTT.statusOf(b);
  assert.strictEqual(s.winner, 'X');
  eq(s.line, [0, 1, 2]);
  assert.strictEqual(s.over, true);
  assert.strictEqual(s.draw, false);
});

test('detects vertical win', function () {
  const b = TTT.createBoard();
  b[1] = 'O'; b[4] = 'O'; b[7] = 'O';
  const s = TTT.statusOf(b);
  assert.strictEqual(s.winner, 'O');
  eq(s.line, [1, 4, 7]);
});

test('detects diagonal win (top-left to bottom-right)', function () {
  const b = TTT.createBoard();
  b[0] = 'X'; b[4] = 'X'; b[8] = 'X';
  const s = TTT.statusOf(b);
  assert.strictEqual(s.winner, 'X');
  eq(s.line, [0, 4, 8]);
});

test('detects anti-diagonal win', function () {
  const b = TTT.createBoard();
  b[2] = 'O'; b[4] = 'O'; b[6] = 'O';
  const s = TTT.statusOf(b);
  assert.strictEqual(s.winner, 'O');
  eq(s.line, [2, 4, 6]);
});

test('winningLine returns null when no win', function () {
  assert.strictEqual(TTT.winningLine(TTT.createBoard(), 'X'), null);
});

test('playing to a win sets over=true', function () {
  let b = TTT.createBoard();
  const moves = [[0, 'X'], [3, 'O'], [1, 'X'], [4, 'O'], [2, 'X']];
  for (const m of moves) {
    b = TTT.play(b, m[0], m[1]);
  }
  const s = TTT.statusOf(b);
  assert.strictEqual(s.winner, 'X');
  assert.strictEqual(s.over, true);
});

// --- Draw detection ---
test('detects a draw (no winner, board full)', function () {
  const b = ['X', 'O', 'X', 'O', 'X', 'O', 'O', 'X', 'O'];
  const s = TTT.statusOf(b);
  assert.strictEqual(s.winner, null);
  assert.strictEqual(s.draw, true);
  assert.strictEqual(s.over, true);
  assert.strictEqual(s.line, null);
});

test('not a draw while cells remain', function () {
  const b = ['X', 'O', '', '', '', '', '', '', ''];
  const s = TTT.statusOf(b);
  assert.strictEqual(s.draw, false);
  assert.strictEqual(s.over, false);
});

test('in-progress game reports next player', function () {
  const b = ['X', 'O', '', '', '', '', '', '', ''];
  const s = TTT.statusOf(b);
  assert.strictEqual(s.current, 'X');
});

test('API exposes expected functions', function () {
  for (const k of ['createBoard', 'nextPlayer', 'isValidMove', 'play',
    'winningLine', 'statusOf', 'PLAYERS', 'LINES']) {
    assert.ok(k in TTT, 'missing export ' + k);
  }
});

console.log('-----------------------');
console.log('Passed: ' + passed + '  Failed: ' + failed);

if (failed > 0) {
  for (const f of failures) {
    console.error('\nFAIL: ' + f.name);
    if (f.err && f.err.stack) console.error(f.err.stack);
  }
  process.exit(1);
}

process.exit(0);
