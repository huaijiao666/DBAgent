/*!
 * Snake — a zero-dependency Snake game.
 *
 * Two layers live in this one file:
 *   1. A pure, framework-free core (`SnakeGame` class + `mulberry32` seeded
 *      RNG) that runs identically in the browser and in Node.js. The core
 *      never touches the DOM, so it is fully unit-testable.
 *   2. Browser-only wiring (canvas rendering, keyboard/touch input, HUD,
 *      overlays, high score persistence). Skipped automatically under Node.
 *
 * Exposed when loaded in Node: module.exports = { SnakeGame, mulberry32 }.
 */
(function (global) {
  'use strict';

  // ----------------------------------------------------------------------
  // Deterministic seeded PRNG (mulberry32). The browser normally uses
  // Math.random; tests inject a seeded generator for reproducible runs.
  // ----------------------------------------------------------------------
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  var DIRS = {
    up: { x: 0, y: -1 },
    down: { x: 0, y: 1 },
    left: { x: -1, y: 0 },
    right: { x: 1, y: 0 }
  };
  var OPPOSITE = { up: 'down', down: 'up', left: 'right', right: 'left' };

  // ----------------------------------------------------------------------
  // Core game logic — pure simulation, no DOM access.
  // ----------------------------------------------------------------------
  class SnakeGame {
    /**
     * @param {Object} [options]
     * @param {number} [options.cols=24]          Board width in cells.
     * @param {number} [options.rows=24]          Board height in cells.
     * @param {number} [options.startLength=3]    Initial snake length.
     * @param {number} [options.pointsPerFood=10] Points gained per food.
     * @param {number} [options.startInterval=150] ms per step at score 0.
     * @param {number} [options.minInterval=70]   Fastest allowed step (ms).
     * @param {number} [options.speedUpPerFood=4] ms removed per food eaten.
     * @param {Function} [options.rng=Math.random] () => [0, 1) generator.
     */
    constructor(options) {
      options = options || {};
      this.cols = options.cols || 24;
      this.rows = options.rows || 24;
      this.startLength = options.startLength || 3;
      this.pointsPerFood = options.pointsPerFood || 10;
      this.startInterval = options.startInterval || 150;
      this.minInterval = options.minInterval || 70;
      this.speedUpPerFood = options.speedUpPerFood || 4;
      this.rng = typeof options.rng === 'function' ? options.rng : Math.random;
      // status: 'ready' | 'running' | 'paused' | 'over' | 'won'
      this.status = 'ready';
      this.reset();
    }

    /** Rebuild a fresh game in the 'ready' state. */
    reset() {
      var cx = Math.floor(this.cols / 2);
      var cy = Math.floor(this.rows / 2);
      this.snake = [];
      for (var i = 0; i < this.startLength; i++) {
        this.snake.push({ x: cx - i, y: cy });
      }
      this.direction = 'right';
      this.queue = [];
      this.score = 0;
      this.steps = 0;
      this.food = null;
      this.status = 'ready';
      this.spawnFood();
      return this;
    }

    // ------------------------- state transitions -------------------------

    /** Begin (or, after game over/win, restart) play. */
    start() {
      if (this.status === 'over' || this.status === 'won') {
        this.reset();
      }
      this.status = 'running';
      return this;
    }

    /** Pause a running game. No-op otherwise. */
    pause() {
      if (this.status === 'running') this.status = 'paused';
      return this;
    }

    /** Resume a paused game. No-op otherwise. */
    resume() {
      if (this.status === 'paused') this.status = 'running';
      return this;
    }

    /** Pause <-> resume toggle. */
    togglePause() {
      return this.status === 'paused' ? this.resume() : this.pause();
    }

    running(active) {
      if (active) this.status = 'running';
      else if (this.status === 'running') this.status = 'paused';
      return this;
    }

    // ------------------------------ controls ------------------------------

    /**
     * Queue a direction change. Rejects 180-degree reversals and input that
     * arrives while the game is not running. Up to 3 turns may be buffered so
     * fast successive presses register fairly.
     * @returns {boolean} true if the direction was accepted.
     */
    setDirection(name) {
      if (!DIRS[name] || this.status !== 'running') return false;
      var last = this.queue.length ? this.queue[this.queue.length - 1] : this.direction;
      if (name === last || OPPOSITE[name] === last) return false;
      if (this.queue.length < 3) this.queue.push(name);
      return true;
    }

    /** Current ms-per-step, based on score and clamped to [minInterval, startInterval]. */
    currentInterval() {
      var reduced = this.startInterval - (this.score / this.pointsPerFood) * this.speedUpPerFood;
      return Math.max(this.minInterval, Math.round(reduced));
    }

    // ------------------------------ simulation ------------------------------

    /** true if cell (x, y) is covered by a snake segment. */
    _occupied(x, y, excludeTail) {
      var end = excludeTail ? this.snake.length - 1 : this.snake.length;
      for (var i = 0; i < end; i++) {
        if (this.snake[i].x === x && this.snake[i].y === y) return true;
      }
      return false;
    }

    /** Place food on a random free cell. Returns false when the board is full. */
    spawnFood() {
      var free = [];
      for (var y = 0; y < this.rows; y++) {
        for (var x = 0; x < this.cols; x++) {
          if (!this._occupied(x, y, false)) free.push({ x: x, y: y });
        }
      }
      if (free.length === 0) {
        this.food = null;
        return false;
      }
      this.food = free[Math.floor(this.rng() * free.length)];
      return true;
    }

    /**
     * Advance the simulation by exactly one tick. No-op unless running.
     * @returns {SnakeGame} this, for chaining.
     */
    step() {
      if (this.status !== 'running') return this;

      if (this.queue.length) this.direction = this.queue.shift();

      var d = DIRS[this.direction];
      var head = this.snake[0];
      var nx = head.x + d.x;
      var ny = head.y + d.y;

      // Wall collision.
      if (nx < 0 || ny < 0 || nx >= this.cols || ny >= this.rows) {
        this.status = 'over';
        return this;
      }

      var eating = !!this.food && this.food.x === nx && this.food.y === ny;
      // The tail cell only becomes free if we are NOT eating this tick.
      if (this._occupied(nx, ny, !eating)) {
        this.status = 'over';
        return this;
      }

      this.snake.unshift({ x: nx, y: ny });
      if (eating) {
        this.score += this.pointsPerFood;
        if (!this.spawnFood()) this.status = 'won'; // board cleared
      } else {
        this.snake.pop();
      }
      this.steps += 1;
      return this;
    }
  }

  global.SnakeGame = SnakeGame;
  global.mulberry32 = mulberry32;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SnakeGame: SnakeGame, mulberry32: mulberry32 };
  }
})(typeof window !== 'undefined' ? window : globalThis);

// ======================================================================
// Browser wiring — skipped when this file is loaded under Node.js.
// ======================================================================
(function () {
  'use strict';

  if (typeof document === 'undefined') return;

  var COLS = 24;
  var ROWS = 24;
  var CELL = 20; // logical pixels per cell

  function bootBrowser() {
    var canvas = document.getElementById('game-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    if (!ctx) return;

    var game = new SnakeGame({ cols: COLS, rows: ROWS });

    var els = {
      score: document.getElementById('score'),
      high: document.getElementById('high-score'),
      length: document.getElementById('length'),
      statusPill: document.getElementById('status-pill'),
      status: document.getElementById('status'),
      primary: document.getElementById('btn-primary'),
      restart: document.getElementById('btn-restart'),
      overlayStart: document.getElementById('overlay-start'),
      overlayPause: document.getElementById('overlay-pause'),
      overlayOver: document.getElementById('overlay-over'),
      overlayTitle: document.getElementById('overlay-title'),
      overlayStartBtn: document.getElementById('overlay-start-btn'),
      overlayResumeBtn: document.getElementById('overlay-resume-btn'),
      overlayRestartBtn: document.getElementById('overlay-restart-btn'),
      finalScore: document.getElementById('final-score'),
      highScoreBadge: document.getElementById('high-score-badge')
    };

    // ------------------------- high score storage -------------------------
    var STORAGE_KEY = 'snake-high-score';
    var highScore = 0;
    var announced = null;

    function loadHighScore() {
      try {
        var v = parseInt(localStorage.getItem(STORAGE_KEY), 10);
        if (!isNaN(v) && v > 0) highScore = v;
      } catch (e) { /* file:// sandboxes may block storage; ignore */ }
    }
    function saveHighScore() {
      try { localStorage.setItem(STORAGE_KEY, String(highScore)); } catch (e) {}
    }

    // ------------------------------- actions -------------------------------
    function primaryAction() {
      if (game.status === 'running' || game.status === 'paused') {
        game.togglePause();
      } else {
        game.start(); // resets the board when over/won
        announced = null;
      }
      syncUI();
    }

    function restartAction() {
      game.reset();
      game.start();
      announced = null;
      syncUI();
    }

    function onGameEnd() {
      if (announced === game.status) return;
      announced = game.status;
      var isRecord = game.score > highScore;
      if (isRecord) {
        highScore = game.score;
        saveHighScore();
      }
      els.overlayTitle.textContent = game.status === 'won' ? 'You win! Board cleared.' : 'Game over';
      els.finalScore.textContent = String(game.score);
      els.highScoreBadge.classList.toggle('hidden', !isRecord);
      syncUI();
    }

    // -------------------------------- HUD --------------------------------
    var STATUS_LABELS = {
      ready: 'Ready',
      running: 'Playing',
      paused: 'Paused',
      over: 'Game over',
      won: 'You win!'
    };

    function updateHUD() {
      els.score.textContent = String(game.score);
      els.high.textContent = String(highScore);
      els.length.textContent = String(game.snake.length);
      els.status.textContent = STATUS_LABELS[game.status] || game.status;
    }

    function syncUI() {
      els.primary.textContent =
        game.status === 'running' ? 'Pause' :
        game.status === 'paused' ? 'Resume' :
        (game.status === 'over' || game.status === 'won') ? 'Play again' : 'Start';
      els.statusPill.className = 'status-pill status-' + game.status;
      els.overlayStart.classList.toggle('hidden', game.status !== 'ready');
      els.overlayPause.classList.toggle('hidden', game.status !== 'paused');
      els.overlayOver.classList.toggle('hidden', game.status !== 'over' && game.status !== 'won');
      updateHUD();
    }

    // ------------------------------- input -------------------------------
    var KEY_DIRS = {
      ArrowUp: 'up', KeyW: 'up',
      ArrowDown: 'down', KeyS: 'down',
      ArrowLeft: 'left', KeyA: 'left',
      ArrowRight: 'right', KeyD: 'right'
    };

    document.addEventListener('keydown', function (e) {
      var onButton = e.target && e.target.tagName === 'BUTTON';
      var dir = KEY_DIRS[e.code];
      if (dir) {
        e.preventDefault();
        game.setDirection(dir);
        return;
      }
      if (e.code === 'Space') {
        if (!onButton) {
          e.preventDefault();
          primaryAction();
        }
        return;
      }
      if (e.code === 'Enter') {
        if (!onButton && (game.status === 'ready' || game.status === 'over' || game.status === 'won')) {
          e.preventDefault();
          primaryAction();
        }
        return;
      }
      if (e.code === 'KeyR') {
        e.preventDefault();
        restartAction();
      }
    });

    function bind(el, handler) {
      if (el) el.addEventListener('click', handler);
    }
    bind(els.primary, primaryAction);
    bind(els.restart, restartAction);
    bind(els.overlayStartBtn, primaryAction);
    bind(els.overlayResumeBtn, primaryAction);
    bind(els.overlayRestartBtn, primaryAction);

    // Touch / pointer d-pad. Both pointerdown (snappy) and click (fallback)
    // fire setDirection, which de-duplicates identical presses.
    document.querySelectorAll('[data-dir]').forEach(function (btn) {
      btn.addEventListener('pointerdown', function (e) {
        e.preventDefault();
        game.setDirection(btn.getAttribute('data-dir'));
      });
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        game.setDirection(btn.getAttribute('data-dir'));
      });
    });

    // ------------------------------ rendering ------------------------------
    var logicalW = COLS * CELL;
    var logicalH = ROWS * CELL;

    function fitCanvas() {
      var dpr = window.devicePixelRatio || 1;
      var rect = canvas.getBoundingClientRect();
      var w = Math.max(1, Math.round(rect.width * dpr));
      var h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.setTransform(w / logicalW, 0, 0, h / logicalH, 0, 0);
    }

    function roundRectPath(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function render(now) {
      fitCanvas();

      // Board backdrop.
      var bg = ctx.createLinearGradient(0, 0, 0, logicalH);
      bg.addColorStop(0, '#111c31');
      bg.addColorStop(1, '#0b1322');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, logicalW, logicalH);

      // Checkerboard.
      ctx.fillStyle = 'rgba(148, 163, 184, 0.045)';
      for (var y = 0; y < ROWS; y++) {
        for (var x = 0; x < COLS; x++) {
          if ((x + y) % 2 === 0) ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
        }
      }

      // Food.
      if (game.food) {
        var fx = game.food.x * CELL + CELL / 2;
        var fy = game.food.y * CELL + CELL / 2;
        var pulse = 1 + 0.1 * Math.sin(now / 160);
        var fr = CELL * 0.33 * pulse;
        ctx.beginPath();
        ctx.arc(fx, fy, fr * 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.16)';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(fx, fy, fr, 0, Math.PI * 2);
        var fg = ctx.createRadialGradient(fx - fr * 0.3, fy - fr * 0.3, fr * 0.1, fx, fy, fr);
        fg.addColorStop(0, '#fb7185');
        fg.addColorStop(1, '#e11d48');
        ctx.fillStyle = fg;
        ctx.fill();
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(fx, fy - fr * 0.9);
        ctx.quadraticCurveTo(fx + fr * 0.3, fy - fr * 1.35, fx + fr * 0.55, fy - fr * 1.0);
        ctx.stroke();
      }

      // Snake body (tail first so the head draws on top).
      var segs = game.snake;
      for (var i = segs.length - 1; i >= 0; i--) {
        var s = segs[i];
        var t = i / Math.max(1, segs.length - 1);
        var pad = CELL * 0.09;
        roundRectPath(s.x * CELL + pad, s.y * CELL + pad, CELL - pad * 2, CELL - pad * 2, CELL * 0.3);
        var hue = 148 - t * 24;
        var sat = 74 - t * 14;
        var light = 56 - t * 18;
        ctx.fillStyle = 'hsl(' + hue + ', ' + sat + '%, ' + light + '%)';
        ctx.fill();
      }

      // Eyes on the head.
      if (segs.length > 0) {
        var head = segs[0];
        var d = DIRS[game.direction];
        var cx = head.x * CELL + CELL / 2;
        var cy = head.y * CELL + CELL / 2;
        var fwd = CELL * 0.12;
        var off = CELL * 0.17;
        var e1x = cx + d.x * fwd + -d.y * off;
        var e1y = cy + d.y * fwd + d.x * off;
        var e2x = cx + d.x * fwd - -d.y * off;
        var e2y = cy + d.y * fwd - d.x * off;
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath();
        ctx.arc(e1x, e1y, CELL * 0.09, 0, Math.PI * 2);
        ctx.arc(e2x, e2y, CELL * 0.09, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(e1x + d.x * CELL * 0.025, e1y + d.y * CELL * 0.025, CELL * 0.045, 0, Math.PI * 2);
        ctx.arc(e2x + d.x * CELL * 0.025, e2y + d.y * CELL * 0.025, CELL * 0.045, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // -------------------------------- loop --------------------------------
    var last = 0;
    var acc = 0;

    function frame(now) {
      requestAnimationFrame(frame);
      if (!last) last = now;
      var dt = now - last;
      last = now;

      if (game.status === 'running') {
        var interval = game.currentInterval();
        acc = Math.min(acc + dt, interval * 4); // cap after tab switches
        var guard = 0;
        while (acc >= interval && game.status === 'running' && guard < 4) {
          game.step();
          acc -= interval;
          guard++;
        }
        if (game.status === 'over' || game.status === 'won') onGameEnd();
        updateHUD();
      }
      render(now);
    }

    window.addEventListener('resize', fitCanvas);
    loadHighScore();
    syncUI();
    requestAnimationFrame(frame);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootBrowser);
  } else {
    bootBrowser();
  }
})();
