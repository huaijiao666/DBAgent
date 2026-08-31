'use strict';

/**
 * DOM-free Tic-Tac-Toe rules engine.
 *
 * This module holds ALL game logic (state, moves, win/draw detection) so it can
 * be imported and unit-tested deterministically in Node without a DOM.
 *
 * The board is an array of length 9, sequential indices 0..8:
 *
 *   0 1 2
 *   3 4 5
 *   6 7 8
 *
 * Each cell is '' (empty), 'X', or 'O'.
 *
 * When loaded as a plain script in the browser, TTT exposes itself as a global
 * `window.TTT`. When loaded as a CommonJS module in Node, it exports the same
 * API.
 */
const TTT = (function () {
  const PLAYERS = ['X', 'O'];

  // The eight winning lines represented as index triples.
  const LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];

  /**
   * Create an empty board.
   * @returns {string[]}
   */
  function createBoard() {
    return Array(9).fill('');
  }

  /**
   * Determine whose turn it is: X if the counts are equal, else O.
   * @param {string[]} board
   * @returns {'X'|'O'}
   */
  function nextPlayer(board) {
    let x = 0;
    let o = 0;
    for (let i = 0; i < board.length; i++) {
      if (board[i] === 'X') x++;
      else if (board[i] === 'O') o++;
    }
    return x === o ? 'X' : 'O';
  }

  /**
   * Check whether a completed move is legal.
   * @param {string[]} board
   * @param {number} index 0..8
   * @param {string} player 'X'|'O'
   * @returns {boolean}
   */
  function isValidMove(board, index, player) {
    if (!Number.isInteger(index) || index < 0 || index >= board.length) {
      return false;
    }
    if (board[index] !== '') return false;
    if (player !== 'X' && player !== 'O') return false;
    if (statusOf(board).winner !== null) return false;
    if (nextPlayer(board) !== player) return false;
    return true;
  }

  /**
   * Compute a new board with the given move applied. Illegal moves return the
   * input board unchanged.
   * @param {string[]} board
   * @param {number} index
   * @param {string} player
   * @returns {string[]}
   */
  function play(board, index, player) {
    if (!isValidMove(board, index, player)) {
      return board.slice();
    }
    const next = board.slice();
    next[index] = player;
    return next;
  }

  /**
   * Identify the winning line for a player, or null.
   * @returns {number[]|null}
   */
  function winningLine(board, player) {
    for (const line of LINES) {
      if (board[line[0]] === player && board[line[1]] === player &&
          board[line[2]] === player) {
        return line;
      }
    }
    return null;
  }

  /**
   * Full status descriptor for a board.
   * @returns {{winner: ('X'|'O'|null), line: (number[]|null), draw: boolean, over: boolean, current: ('X'|'O')}}
   */
  function statusOf(board) {
    for (const player of PLAYERS) {
      const line = winningLine(board, player);
      if (line) {
        return {
          winner: player,
          line: line,
          draw: false,
          over: true,
          current: player,
        };
      }
    }
    const draw = board.every(function (cell) { return cell !== ''; });
    return {
      winner: null,
      line: null,
      draw: draw,
      over: draw,
      current: nextPlayer(board),
    };
  }

  return {
    PLAYERS: PLAYERS,
    LINES: LINES,
    createBoard: createBoard,
    nextPlayer: nextPlayer,
    isValidMove: isValidMove,
    play: play,
    winningLine: winningLine,
    statusOf: statusOf,
  };
})();

// ---------------------------------------------------------------------------
// UI controller. This section only runs in a browser (when window/document
// exist). It is guarded so that Node's `node --check` and the tests can parse
// and import the module without a DOM.
// ---------------------------------------------------------------------------
if (typeof document !== 'undefined' && typeof window !== 'undefined') {
  (function initUI(TTT) {
    let board = TTT.createBoard();
    let current = 'X';
    const cells = [];
    const boardEl = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const restartBtn = document.getElementById('restart');

    function update() {
      const status = TTT.statusOf(board);
      if (status.winner) {
        statusEl.textContent = 'Player ' + status.winner + ' wins!';
        statusEl.className = 'status winner';
      } else if (status.draw) {
        statusEl.textContent = "It's a draw!";
        statusEl.className = 'status draw';
      } else {
        statusEl.textContent = 'Player ' + current + "'s turn";
        statusEl.className = 'status';
      }
    }

    function highlight(line) {
      if (!line) return;
      for (const i of line) {
        cells[i].classList.add('win');
      }
    }

    function clearHighlight() {
      for (const cell of cells) {
        cell.classList.remove('win');
      }
    }

    function render() {
      for (let i = 0; i < 9; i++) {
        cells[i].textContent = board[i];
        cells[i].setAttribute('aria-label', 'Cell ' + (i + 1) + ': ' +
          (board[i] || 'empty'));
      }
      clearHighlight();
      highlight(TTT.statusOf(board).line);
      update();
    }

    function makeMove(index) {
      const status = TTT.statusOf(board);
      if (status.over) return;
      const next = TTT.play(board, index, current);
      if (next === board) return; // illegal move, unchanged
      board = next;
      current = TTT.nextPlayer(board);
      render();
    }

    function restart() {
      board = TTT.createBoard();
      current = 'X';
      render();
    }

    function setupBoard() {
      for (let i = 0; i < 9; i++) {
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'cell';
        cell.dataset.index = i;
        cell.addEventListener('click', function () {
          makeMove(Number(cell.dataset.index));
        });
        boardEl.appendChild(cell);
        cells.push(cell);
      }
    }

    restartBtn.addEventListener('click', restart);

    // Keyboard accessibility: arrow keys move focus among cells
    // (browser tab focus already handles Enter/Space to activate buttons).
    boardEl.addEventListener('keydown', function (event) {
      const active = document.activeElement;
      if (!active || active.dataset.index === undefined) return;
      const idx = Number(active.dataset.index);
      let target = -1;
      const row = Math.floor(idx / 3);
      const col = idx % 3;
      switch (event.key) {
        case 'ArrowUp': target = row > 0 ? idx - 3 : -1; break;
        case 'ArrowDown': target = row < 2 ? idx + 3 : -1; break;
        case 'ArrowLeft': target = col > 0 ? idx - 1 : -1; break;
        case 'ArrowRight': target = col < 2 ? idx + 1 : -1; break;
      }
      if (target >= 0) {
        event.preventDefault();
        cells[target].focus();
      }
    });

    setupBoard();
    render();
  })(TTT);
}

// CommonJS export for Node tests.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TTT;
}
// Browser global for direct <script> inclusion.
if (typeof window !== 'undefined') {
  window.TTT = TTT;
}
