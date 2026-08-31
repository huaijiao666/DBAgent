/* Snake - shared game logic and browser bindings.
 *
 * The core game logic (SnakeEngine, DIRECTIONS, etc.) is intentionally
 * free of any DOM / window references so it can be unit-tested under Node.
 * The browser-only bindings are defined in a separate function that is only
 * invoked when a `document` global is present.
 */

'use strict';

/* ---- Constant directions (as [x, y] deltas) ---- */
var Snake = (function () {
  var DIRECTIONS = {
    up: { x: 0, y: -1 },
    down: { x: 0, y: 1 },
    left: { x: -1, y: 0 },
    right: { x: 1, y: 0 }
  };

  var OPPOSITE = {
    up: 'down',
    down: 'up',
    left: 'right',
    right: 'left'
  };

  /* Seedable PRNG (mulberry32) so tests are deterministic. */
  function createRng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function copyCell(c) { return { x: c.x, y: c.y }; }

  /**
   * Create an initial game state.
   * opts: { width, height, startLength, seed }
   */
  function createState(opts) {
    var options = opts || {};
    var width = options.width || 20;
    var height = options.height || 20;
    var startLength = options.startLength || 3;
    var seed = options.seed != null ? options.seed : Date.now();

    var startX = Math.floor(width / 2);
    var startY = Math.floor(height / 2);

    var snake = [];
    for (var i = 0; i < startLength; i++) {
      snake.push({ x: startX - i, y: startY });
    }

    var state = {
      width: width,
      height: height,
      snake: snake,
      direction: 'right',
      nextDirection: 'right',
      score: 0,
      running: true,
      over: false,
      won: false,
      rng: createRng(seed),
      _rngSeed: seed
    };

    state.food = placeFood(state);
    return state;
  }

  function randomEmptyCell(state, rng) {
    var empty = [];
    for (var y = 0; y < state.height; y++) {
      for (var x = 0; x < state.width; x++) {
        if (!occupiesCell(state.snake, x, y)) {
          empty.push({ x: x, y: y });
        }
      }
    }
    if (empty.length === 0) return null;
    var rand = state.rng();
    var idx = Math.floor(rand * empty.length);
    if (idx >= empty.length) idx = empty.length - 1;
    return empty[idx];
  }

  function placeFood(state) {
    var cell = randomEmptyCell(state, state.rng);
    return cell; // null when board is full (win condition)
  }

  function occupiesCell(snake, x, y) {
    for (var i = 0; i < snake.length; i++) {
      if (snake[i].x === x && snake[i].y === y) return true;
    }
    return false;
  }

  /**
   * Set the intended next direction, disallowing an immediate 180-degree turn
   * that would reverse into the snake's own body.
   */
  function changeDirection(state, dir) {
    if (!DIRECTIONS[dir]) return;
    // Cannot reverse relative to the current heading.
    if (OPPOSITE[dir] === state.direction) return;
    state.nextDirection = dir;
  }

  /**
   * Advance the game state by one step. Returns the (possibly updated)
   * state. Mutates and returns the same object.
   */
  function step(state) {
    if (state.over || !state.running) return state;

    var dir = state.nextDirection || state.direction;
    state.direction = dir;
    var delta = DIRECTIONS[dir];

    var head = state.snake[0];
    var newHead = { x: head.x + delta.x, y: head.y + delta.y };

    // Wall collision -> game over.
    if (
      newHead.x < 0 || newHead.x >= state.width ||
      newHead.y < 0 || newHead.y >= state.height
    ) {
      state.over = true;
      state.running = false;
      return state;
    }

    var ate = (
      state.food !== null &&
      newHead.x === state.food.x && newHead.y === state.food.y
    );

    state.snake.unshift(newHead);
    if (ate) {
      state.score++;
      state.food = placeFood(state);
      if (state.food === null) {
        // Board fully filled -> win.
        state.won = true;
        state.running = false;
        return state;
      }
    } else {
      state.snake.pop();
      // Self collision (head vs body) -> game over.
      // Defer tail: tail has already popped, so compare against rest.
      for (var i = 1; i < state.snake.length; i++) {
        if (state.snake[i].x === newHead.x && state.snake[i].y === newHead.y) {
          state.over = true;
          state.running = false;
          return state;
        }
      }
    }

    return state;
  }

  return {
    DIRECTIONS: DIRECTIONS,
    OPPOSITE: OPPOSITE,
    createRng: createRng,
    createState: createState,
    placeFood: placeFood,
    occupiesCell: occupiesCell,
    changeDirection: changeDirection,
    step: step
  };
})();

/* ---- Browser-only bindings (not executed under Node) ---- */
if (typeof document !== 'undefined' && document.getElementById) {
  (function () {
    var canvas = document.getElementById('game');
    var ctx = canvas.getContext('2d');
    var scoreEl = document.getElementById('score');
    var statusEl = document.getElementById('status');
    var runButton = document.getElementById('run');

    var TICKS_PER_SECOND = 8;
    var startLength = 3;
    var CELL = 24;

    var state = null;
    var timer = null;

    function resizeCanvas() {
      var cols = state ? state.width : 20;
      var rows = state ? state.height : 20;
      var size = Math.max(16, Math.floor(window.innerWidth * 0.9 / cols) - 0);
      var avail = (window.innerHeight - 140);
      if (size > avail / rows) size = Math.floor(avail / rows);
      size = Math.max(16, size);
      canvas.width = cols * size;
      canvas.height = rows * size;
      return size;
    }

    function draw() {
      var cell = resizeCanvas();

      // Clear board.
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid lines.
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      for (var gx = 0; gx <= state.width; gx++) {
        ctx.beginPath();
        ctx.moveTo(gx * cell, 0);
        ctx.lineTo(gx * cell, canvas.height);
        ctx.stroke();
      }
      for (var gy = 0; gy <= state.height; gy++) {
        ctx.beginPath();
        ctx.moveTo(0, gy * cell);
        ctx.lineTo(canvas.width, gy * cell);
        ctx.stroke();
      }

      // Food.
      if (state.food) {
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(
          (state.food.x + 0.5) * cell,
          (state.food.y + 0.5) * cell,
          cell * 0.32, 0, Math.PI * 2
        );
        ctx.fill();
      }

      // Snake.
      for (var i = 0; i < state.snake.length; i++) {
        var seg = state.snake[i];
        ctx.fillStyle = i === 0 ? '#22c55e' : '#16a34a';
        ctx.fillRect(seg.x * cell + 1, seg.y * cell + 1, cell - 2, cell - 2);
      }

      // Status text.
      if (state.over) {
        statusEl.textContent = 'Game Over - press Restart';
      } else if (state.won) {
        statusEl.textContent = 'You win! Press Restart';
      } else {
        statusEl.textContent = 'Running';
      }
    }

    function newGame() {
      state = Snake.createState({
        width: 24,
        height: 24,
        startLength: startLength,
        seed: Date.now()
      });
      stopTimer();
      startTimer();
      updateScoreUI();
      draw();
    }

    function stopTimer() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    }

    function startTimer() {
      timer = setInterval(function () {
        Snake.step(state);
        updateScoreUI();
        draw();
      }, 1000 / TICKS_PER_SECOND);
    }

    function updateScoreUI() {
      scoreEl.textContent = String(state.score);
    }

    document.addEventListener('keydown', function (e) {
      var key = e.key.toLowerCase();
      var dirMap = {
        arrowup: 'up', w: 'up',
        arrowdown: 'down', s: 'down',
        arrowleft: 'left', a: 'left',
        arrowright: 'right', d: 'right'
      };
      var dir = dirMap[key];
      if (dir) {
        e.preventDefault();
        if (state) Snake.changeDirection(state, dir);
      }
      if (key === ' ' && state && state.over) {
        e.preventDefault();
        newGame();
      }
    });

    runButton.addEventListener('click', function () {
      if (state && !state.over && !state.won) {
        stopTimer();
        state.running = false;
        runButton.textContent = 'Run';
      } else {
        newGame();
      }
    });

    window.addEventListener('resize', function () {
      if (state) draw();
    });

    newGame();
  })();
}

/* Node-friendly export so tests.js can require this file. */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Snake;
}
