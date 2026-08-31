# -*- coding: utf-8 -*-
"""Pure game logic for Tetris.

This module has *no* dependencies beyond the standard library so it can be
unit-tested deterministically (piece generation is driven by a seeded RNG).
The pygame UI in ``tetris.py`` only talks to this module.
"""

import random

# --------------------------------------------------------------------------- #
# Board constants
# --------------------------------------------------------------------------- #
COLS = 10
ROWS = 20
# Standard weights used to score line clears (single..tetris).
LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}


# --------------------------------------------------------------------------- #
# Tetromino definitions.
#
# Each shape is a list of rotation states.  Each state is a list of
# (row, col) cell offsets, normalised so the top-left of the shape's
# bounding "well" region is (0, 0).  Cells are described in a 4x4 grid for
# compactness; after rotation we recentre the shape to make rotation feel
# natural around a pivot.
# --------------------------------------------------------------------------- #
class Tetromino:
    __slots__ = ("shape", "rotation", "row", "col")

    # shape -> list of 4 rotation states; each state is a list of (row, col)
    _defs = {
        # I
        "I": [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(1, 0), (0, 0), (-1, 0), (-2, 0)],
            [(0, -2), (0, -1), (0, 0), (0, 1)],
            [(-1, -1), (0, -1), (1, -1), (2, -1)],
        ],
        # J
        "J": [
            [(0, 0), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (0, 1), (-1, 0), (-2, 0)],
            [(0, 0), (0, 1), (0, 2), (-1, 2)],
            [(0, 0), (0, 1), (1, 2), (2, 2)],
        ],
        # L
        "L": [
            [(0, 0), (1, 0), (1, -1), (1, -2)],
            [(0, 0), (-1, 0), (-2, 0), (-1, 1)],
            [(0, 0), (0, 1), (1, 2), (0, 2)],
            [(0, 0), (1, 0), (2, 0), (1, -1)],
        ],
        # O
        "O": [
            [(0, 0), (1, 0), (0, -1), (1, -1)],
            [(0, 0), (1, 0), (0, -1), (1, -1)],
            [(0, 0), (1, 0), (0, -1), (1, -1)],
            [(0, 0), (1, 0), (0, -1), (1, -1)],
        ],
        # S
        "S": [
            [(0, 0), (0, 1), (1, 0), (-1, 1)],
            [(0, 0), (-1, 0), (-1, -1), (0, 1)],
            [(0, 0), (0, 1), (1, 0), (-1, 1)],
            [(0, 0), (-1, 0), (-1, -1), (0, 1)],
        ],
        # T
        "T": [
            [(0, 0), (0, 1), (-1, 0), (0, -1)],
            [(0, 0), (0, 1), (-1, 0), (1, 0)],
            [(0, 0), (0, 1), (1, 0), (0, -1)],
            [(0, 0), (0, 1), (-1, 0), (1, 0)],
        ],
        # Z
        "Z": [
            [(0, 0), (0, 1), (-1, 0), (1, 1)],
            [(0, 0), (-1, 0), (-1, 1), (0, -1)],
            [(0, 0), (0, 1), (-1, 0), (1, 1)],
            [(0, 0), (-1, 0), (-1, 1), (0, -1)],
        ],
    }

    # Fixed bag order: each of the seven shapes appears exactly once
    # before the bag is shuffled again (the "7-bag" randomiser).
    BAG = ["I", "J", "L", "O", "S", "T", "Z"]

    def __init__(self, shape, rotation=0, row=-2, col=3):
        self.shape = shape
        self.rotation = rotation
        self.row = row
        self.col = col

    # -- helpers ----------------------------------------------------------- #
    def cells(self):
        """Return the absolute (row, col) cells occupied by this piece."""
        base = self._defs[self.shape][self.rotation % 4]
        return [(self.row + dr, self.col + dc) for dr, dc in base]

    def rotate(self, direction):
        """Return a *new* Tetromino rotated by ``direction`` (+1/-1 steps)."""
        return Tetromino(
            self.shape,
            self.rotation + direction,
            self.row,
            self.col,
        )


# --------------------------------------------------------------------------- #
# 7-bag piece generator
# --------------------------------------------------------------------------- #
class SevenBag:
    """Deterministic-per-seed, evenly-distributed piece picker."""

    def __init__(self, rng=None):
        self._rng = rng if rng is not None else random.Random()
        self._bag = []

    def _refill(self):
        bag = list(Tetromino.BAG)
        self._rng.shuffle(bag)
        self._bag = bag

    def next_piece(self):
        """Return the next shape name from the bag."""
        if not self._bag:
            self._refill()
        return self._bag.pop()


# --------------------------------------------------------------------------- #
# The game state
# --------------------------------------------------------------------------- #
class GameState:
    """Owns the board, the piece generator, scoring and game-over state.

    All public mutating methods return a result dict describing what happened
    so the UI can react (e.g. play a sound, show a "line clear" flash).
    """

    def __init__(self, rng=None):
        self.bag = SevenBag(rng)
        # grid[0][0] is the top-left cell (visible cell).
        # grid[0] is the top row of the 10x20 play area.
        self.grid = [[None] * COLS for _ in range(ROWS)]
        self.current = None
        self.hold = None
        self.can_hold = True
        self.next_queue = []
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self._level_lines = 0  # lines cleared since last level-up

        self.ghost_row = None  # for the falling ghost preview

        # Start filling the preview and give ourselves the first piece.
        for _ in range(3):
            self.next_queue.append(self.bag.next_piece())
        self._spawn()

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    def _col_height(self):
        """Height (in cells) of the 10x20 visible grid, used for spawning."""
        return ROWS

    def _spawn(self):
        """Pop the next piece, set it as current, or end the game."""
        if self.game_over:
            return None

        shape = self.next_queue.pop(0)
        # Keep the preview flushed so the player always has 3 future pieces.
        self.next_queue.append(self.bag.next_piece())

        piece = Tetromino(shape)
        self.current = piece
        self.can_hold = True
        self._update_ghost()

        # If it collides immediately, the game is over.
        if self._collides(piece):
            self.game_over = True
            self.current = None
            self._clear_ghost()
        return piece

    def _collides(self, piece):
        """True if any cell is out of bounds or overlaps a filled cell.

        Cells above the visible board (negative rows) are allowed until they
        enter the played area, which mirrors classic behaviour where pieces
        can hang above the top edge.
        """
        for r, c in piece.cells():
            if r < 0:
                continue
            if r >= ROWS or c < 0 or c >= COLS:
                return True
            if self.grid[r][c] is not None:
                return True
        return False

    def _update_ghost(self):
        ghost = self.current
        if ghost is None:
            self._clear_ghost()
            return
        # Drop a copy straight down until it would collide.
        probe = Tetromino(ghost.shape, ghost.rotation, ghost.row, ghost.col)
        while not self._collides(probe):
            probe = Tetromino(probe.shape, probe.rotation, probe.row + 1, probe.col)
        self.ghost_row = probe.row - 1

    def _clear_ghost(self):
        self.ghost_row = None

    # ------------------------------------------------------------------ #
    # Public moves
    # ------------------------------------------------------------------ #
    def move(self, dcol):
        """Shift the current piece horizontally by ``dcol`` cells."""
        if self.current is None:
            return {"moved": False}
        probe = Tetromino(self.current.shape, self.current.rotation,
                          self.current.row, self.current.col + dcol)
        if not self._collides(probe):
            self.current.col += dcol
            self._update_ghost()
            return {"moved": True}
        return {"moved": False}

    def rotate(self, direction=1):
        """Rotate the current piece, with simple wall-kick attempts."""
        if self.current is None:
            return {"rotated": False}
        base = self.current
        # Try the plain rotation first, then small horizontal nudges.
        for kick in (0, -1, 1, -2, 2):
            probe = Tetromino(base.shape, base.rotation + direction,
                              base.row, base.col + kick)
            if not self._collides(probe):
                self.current = probe
                self._update_ghost()
                return {"rotated": True, "kicked": kick}
        return {"rotated": False}

    def hard_drop(self):
        """Instantly drop the piece to the bottom and lock it."""
        if self.current is None:
            return {"dropped": False}
        dist = self._drop_distance()
        self.current.row += dist
        ghost = self.current
        return self._lock_and_spawn()

    def _drop_distance(self):
        base = self.current
        d = 0
        probe = Tetromino(base.shape, base.rotation, base.row, base.col)
        while not self._collides(probe):
            probe = Tetromino(probe.shape, probe.rotation, probe.row + 1, probe.col)
            d += 1
        return d

    def soft_drop(self):
        """Move down by one cell; locks if it can't move."""
        if self.current is None:
            return {"dropped": False}
        probe = Tetromino(self.current.shape, self.current.rotation,
                          self.current.row + 1, self.current.col)
        if not self._collides(probe):
            self.current.row += 1
            self._update_ghost()
            return {"dropped": True}
        return self._lock_and_spawn()

    def hold_piece(self):
        """Swap current and held pieces (once per spawn)."""
        if self.current is None or not self.can_hold or self.game_over:
            return {"holed": False}
        swapped = self.hold
        self.hold = self.current.shape
        self.can_hold = False
        if swapped is None:
            self._spawn()
        else:
            self.current = Tetromino(swapped)
            self.can_hold = True  # swapping back in the held piece resets hold
            self._update_ghost()
        return {"holed": True}

    # ------------------------------------------------------------------ #
    # Internal lock / clear
    # ------------------------------------------------------------------ #
    def _lock_and_spawn(self):
        """Lock the current piece, clear full lines, score, then spawn."""
        if self.current is None:
            return {"locked": False, "cleared": 0, "spawned": False}

        piece = self.current
        for r, c in piece.cells():
            if 0 <= r < ROWS and 0 <= c < COLS:
                self.grid[r][c] = piece.shape

        # Find full rows.
        full = [r for r in range(ROWS) if all(self.grid[r][c] is not None
                                              for c in range(COLS))]
        cleared = len(full)
        if cleared:
            for r in full:
                del self.grid[r]
                self.grid.insert(0, [None] * COLS)

        # Scoring & levelling.
        self.score += LINE_SCORES[cleared] * self.level
        self.lines += cleared
        self._level_lines += cleared
        if self._level_lines >= 10:
            self.level += 1
            self._level_lines -= 10

        self.current = None
        self._clear_ghost()

        result = {"locked": True, "cleared": cleared, "spawned": False}
        self._spawn()
        result["spawned"] = self.current is not None and not self.game_over
        return result

    # ------------------------------------------------------------------ #
    # Introspection for the UI / rendering
    # ------------------------------------------------------------------ #
    def next_shapes(self):
        """The next 3 shapes, oldest first."""
        return list(self.next_queue)

    def current_cells(self):
        """Absolute cells of the current piece (or [])."""
        if self.current is None:
            return []
        return self.current.cells()

    def ghost_cells(self):
        """Absolute cells where the ghost would land (or [])."""
        if self.current is None or self.ghost_row is None:
            return []
        return [(r - (self.current.row - self.ghost_row), c)
                for r, c in self.current.cells()]

    def ghost_offsets(self):
        """The ghost cells represented as offsets below the current piece."""
        if self.current is None or self.ghost_row is None:
            return []
        offset = self.ghost_row - self.current.row
        return [(r + offset, c) for r, c in self.current.cells()]
