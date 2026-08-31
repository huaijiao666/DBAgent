#!/usr/bin/env python3
"""
Tetris — a cleanly playable terminal implementation with a polished UI.

Core game logic (board, pieces, 7-bag RNG, collision, rotation with wall-kick,
locking, line clearing, scoring, level/game-over) is separated from rendering so
it can be unit-tested deterministically. Rendering uses the standard library
`curses` and ANSI colour sequences; no third-party dependencies are required.

Controls:
    left/right arrows (or A/D)  move the piece
    up / X (or W)               rotate clockwise
    Z                           rotate counter-clockwise
    down (or S)                 soft drop
    space                       hard drop
    P                           pause / resume
    Q                           quit
    R                           restart after game over
    N                           start a new game (also before game over)
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
VISIBLE_ROWS = BOARD_HEIGHT  # all rows are visible in this version
_BOARD_BOTTOM_ROW = 2 + BOARD_HEIGHT + 1

# Left how-to-play / controls panel (drawn under the CONTROLS header).
LEFT_PANEL = [
    "←/→      : move",
    "↑ / D    : soft drop",
    "Space    : hard drop",
    "X / Z    : rotate",
    "P        : pause",
    "Q        : quit",
    "R        : restart",
]

# SRS-like piece definitions. Each piece is a list of 4x4 rotation states,
# every state a list of (x, y) cell offsets relative to its bounding box.
# The classic 7 tetrominoes.
_SHAPES = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
    "O": [
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "T": [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "Z": [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (0, 2)],
    ],
}

# Display colour for each piece.
COLOURS = {
    "I": "\u001b[96m",  # cyan
    "J": "\u001b[94m",  # blue
    "L": "\u001b[93m",  # yellow
    "O": "\u001b[33m",  # yellow-orange
    "S": "\u001b[92m",  # green
    "T": "\u001b[95m",  # magenta
    "Z": "\u001b[91m",  # red
}

RESET = "\u001b[0m"

# Score values per line count at the current level multiplier.
LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
# Drag & drop score per cell.
SOFT_DROP_SCORE = 1
HARD_DROP_SCORE = 2

# Fall delay in seconds at level 1.
BASE_FALL_DELAY = 0.85
# How much the delay shrinks per level step (never faster than MIN_FALL_DELAY).
LEVEL_DROP_FACTOR = 0.80
MIN_FALL_DELAY = 0.065
# Number of lines required to advance a level.
LINES_PER_LEVEL = 10


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

class Bag:
    """7-bag randomizer: draws all 7 pieces once before reshuffling."""

    def __init__(self, rng=None):
        self._rng = rng if rng is not None else random.Random()
        self._bag: list[str] = []
        self._refill()

    def _refill(self) -> None:
        self._bag = list(_SHAPES.keys())
        self._rng.shuffle(self._bag)

    def next(self) -> str:
        if not self._bag:
            self._refill()
        return self._bag.pop()


class Piece:
    """A single tetromino instance positioned on the board."""

    __slots__ = ("shape", "rotation", "x", "y")

    def __init__(self, shape: str, rotation: int = 0, x: int = 0, y: int = 0):
        self.shape = shape
        self.rotation = rotation % 4
        self.x = x
        self.y = y

    def cells(self) -> list[tuple[int, int]]:
        """Absolute board cells occupied by this piece."""
        return [
            (self.x + cx, self.y + cy)
            for cx, cy in _SHAPES[self.shape][self.rotation]
        ]

    def rotated(self, delta: int) -> "Piece":
        return Piece(self.shape, (self.rotation + delta) % 4, self.x, self.y)

    def moved(self, dx: int, dy: int) -> "Piece":
        return Piece(self.shape, self.rotation, self.x + dx, self.y + dy)


class GameState:
    """Immutable-ish snapshot of the game, convenient for rendering and tests."""

    def __init__(
        self,
        board,
        current: Piece | None,
        next_piece: str | None,
        score: int,
        level: int,
        lines: int,
        over: bool,
        paused: bool,
        combo: int,
    ):
        self.board = board
        self.current = current
        self.next_piece = next_piece
        self.score = score
        self.level = level
        self.lines = lines
        self.over = over
        self.paused = paused
        self.combo = combo


class Game:
    """Contains all Tetris rules and runs independent of any UI."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.bag = Bag(self.rng)
        self.board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.level = 1
        self.lines = 0
        self.combo = 0
        self.over = False
        self.paused = False
        self.next_piece = self.bag.next()
        self.current: Piece | None = None
        self._spawn()

    # --- spawning ---------------------------------------------------------

    def _spawn(self) -> None:
        shape = self.next_piece
        self.next_piece = self.bag.next()
        start_x = (BOARD_WIDTH - 4) // 2
        self.current = Piece(shape, 0, start_x, 0)
        # If spawn collides immediately the game is over.
        if self._collides(self.current):
            self.current = None
            self.over = True

    # --- collision ---------------------------------------------------------

    def _collides(self, piece: Piece) -> bool:
        for x, y in piece.cells():
            if x < 0 or x >= BOARD_WIDTH or y >= BOARD_HEIGHT:
                return True
            if y >= 0 and self.board[y][x] is not None:
                return True
        return False

    # --- actions ----------------------------------------------------------

    def move(self, dx: int, dy: int = 0) -> bool:
        if self.over or self.paused or self.current is None:
            return False
        moved = self.current.moved(dx, dy)
        if self._collides(moved):
            return False
        self.current = moved
        return True

    def rotate(self, delta: int) -> bool:
        """Rotate with simple wall-kick: try the rotation, then nudges."""
        if self.over or self.paused or self.current is None:
            return False
        rotated = self.current.rotated(delta)
        if not self._collides(rotated):
            self.current = rotated
            return True
        # Try left/right kicks.
        for kick in (1, -1, 2, -2):
            trial = rotated.moved(kick, 0)
            if not self._collides(trial):
                self.current = trial
                return True
        return False

    def soft_drop(self) -> bool:
        """Move the piece down one row, awarding soft-drop points."""
        if self.over or self.paused or self.current is None:
            return False
        if not self.move(0, 1):
            # Cannot move down; locking is the gravity/tick path's job.
            return False
        self.score += SOFT_DROP_SCORE
        return True

    def hard_drop(self) -> int:
        """Drop the piece to its lowest valid position, lock it, return lines cleared."""
        if self.over or self.paused or self.current is None:
            return 0
        dist = 0
        while not self._collides(self.current.moved(0, 1)):
            self.current = self.current.moved(0, 1)
            dist += 1
        self.score += dist * HARD_DROP_SCORE
        cleared = self._lock()
        return cleared

    # --- locking & clearing ------------------------------------------------

    def _lock(self) -> int:
        assert self.current is not None
        for x, y in self.current.cells():
            if y < 0:
                # Locking above the board means game over.
                self.over = True
                self.current = None
                return 0
            self.board[y][x] = self.current.shape
        cleared = self._clear_lines()
        if cleared == BOARD_HEIGHT:
            # Locking filled the entire visible board: that is a top-out.
            self.over = True
        if self.over:
            self.current = None
        else:
            self._spawn()
        return cleared

    def _clear_lines(self) -> int:
        remaining = [row for row in self.board if any(cell is None for cell in row)]
        cleared_count = BOARD_HEIGHT - len(remaining)
        if cleared_count:
            self.board = (
                [[None] * BOARD_WIDTH for _ in range(cleared_count)] + remaining
            )
            self.lines += cleared_count
            if cleared_count >= 4:
                self.combo += 1
            else:
                self.combo = 0
            # Each consecutive tetra (4+ lines) raises the multiplier by 0.5.
            multiplier = 1 + self.combo * 0.5
            # Clamp so a pathological full-board clear never raises KeyError.
            score_key = min(cleared_count, max(LINE_SCORES))
            base = LINE_SCORES[score_key]
            self.score += int(base * multiplier * self.level)
            self._update_level()
        else:
            self.combo = 0
        return cleared_count

    def _update_level(self) -> None:
        self.level = 1 + self.lines // LINES_PER_LEVEL

    def fall_delay(self) -> float:
        delay = BASE_FALL_DELAY * (LEVEL_DROP_FACTOR ** (self.level - 1))
        return max(delay, MIN_FALL_DELAY)

    def tick(self) -> None:
        """Gravity: attempt to move down each 'tick'; lock if unable."""
        if self.over or self.paused or self.current is None:
            return
        if self._collides(self.current.moved(0, 1)):
            self._lock()
        else:
            self.current = self.current.moved(0, 1)

    def toggle_pause(self) -> None:
        if not self.over:
            self.paused = not self.paused

    def restart(self) -> None:
        self.__init__(self.rng)

    def snapshot(self) -> GameState:
        return GameState(
            board=[row[:] for row in self.board],
            current=self.current,
            next_piece=self.next_piece,
            score=self.score,
            level=self.level,
            lines=self.lines,
            over=self.over,
            paused=self.paused,
            combo=self.combo,
        )


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------

try:  # curses is not available on every platform (e.g. plain Windows Python)
    import curses
except ImportError:  # pragma: no cover - only hit when curses is missing
    curses = None


_BLOCK = "\u2588\u2588"  # two full blocks to approximate square cells
_SPACE_CELL = "  "


def _empty_board_lines(current: Piece | None, board):
    """Overlay the active piece on the board (ghost not used for simple UI)."""
    grid = [row[:] for row in board]
    if current is not None:
        for x, y in current.cells():
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                grid[y][x] = current.shape
    return grid


def _cell_text(kind: str | None) -> str:
    if kind is None:
        return _SPACE_CELL
    return f"{COLOURS.get(kind, '')}{_BLOCK}{RESET}"


def _draw_mini_piece(stdscr, piece_type: str, row: int, col: int) -> None:
    cells = _SHAPES[piece_type][0]
    # Normalise so the preview is centred in a 4x2 space.
    for cx, cy in cells:
        try:
            stdscr.addstr(
                row + cy, col + cx * 2, f"{COLOURS[piece_type]}{_BLOCK}{RESET}"
            )
        except curses.error:
            pass


def run_cli(game: Game) -> None:
    """Run the game loop with a full terminal UI."""
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    curses.curs_set(0)
    try:
        if curses.has_colors():
            curses.start_color()
        stdscr.nodelay(True)
        _loop(stdscr, game)
    finally:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


def ui_backend() -> str:
    """Choose the UI backend from the module-level ``curses`` flag.

    On platforms without the C-level ``_curses`` extension (e.g. stock Windows
    Python) the module-level ``curses`` is None, so we fall back to the
    Tkinter GUI.  Tests and platform shims set that flag directly, which makes
    this choice deterministic without probing/importing curses here.
    """
    return "tkinter" if curses is None else "curses"


def run_tk(game: Game) -> None:
    """Run a keyboard-controlled Tkinter fallback when curses is unavailable.

    ``curses`` ships with Unix-like Python installations but not the standard
    Windows distribution. Tkinter is bundled with the same Windows Python, so
    this keeps the game playable without asking the user to install a package.
    """

    try:
        import tkinter as tk
    except ImportError as error:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "Neither curses nor Tkinter is available. Install a terminal UI "
            "package or a Python distribution with Tk support."
        ) from error

    import time

    cell_size = 28
    board_width = BOARD_WIDTH * cell_size
    board_height = BOARD_HEIGHT * cell_size
    piece_colours = {
        "I": "#20c4d9",
        "J": "#3b82f6",
        "L": "#f59e0b",
        "O": "#facc15",
        "S": "#34d399",
        "T": "#c084fc",
        "Z": "#fb7185",
    }
    root = tk.Tk()
    root.title("Tetris — DBAgent edition")
    root.configure(background="#111827")
    root.resizable(False, False)

    container = tk.Frame(root, background="#111827", padx=16, pady=16)
    container.pack()
    canvas = tk.Canvas(
        container,
        width=board_width,
        height=board_height,
        background="#0b1220",
        highlightthickness=2,
        highlightbackground="#334155",
    )
    canvas.grid(row=0, column=0)
    sidebar = tk.Frame(container, background="#111827", padx=18)
    sidebar.grid(row=0, column=1, rowspan=2, sticky="n")
    title = tk.Label(
        sidebar,
        text="TETRIS",
        font=("Segoe UI", 22, "bold"),
        fg="#f8fafc",
        bg="#111827",
    )
    title.pack(anchor="w")
    status = tk.StringVar(value="Playing")
    score = tk.StringVar(value="Score: 0")
    level = tk.StringVar(value="Level: 1")
    lines = tk.StringVar(value="Lines: 0")
    next_piece = tk.StringVar(value="Next: ?")
    for value in (status, score, level, lines, next_piece):
        tk.Label(
            sidebar,
            textvariable=value,
            font=("Segoe UI", 11, "bold"),
            fg="#cbd5e1",
            bg="#111827",
        ).pack(anchor="w", pady=(10, 0))

    last_tick = time.monotonic()

    def draw() -> None:
        state = game.snapshot()
        canvas.delete("all")
        grid = _empty_board_lines(state.current, state.board)
        for y, row in enumerate(grid):
            for x, kind in enumerate(row):
                left = x * cell_size
                top = y * cell_size
                fill = piece_colours.get(kind, "#162033") if kind else "#0f172a"
                canvas.create_rectangle(
                    left,
                    top,
                    left + cell_size,
                    top + cell_size,
                    fill=fill,
                    outline="#1e293b",
                )
        score.set(f"Score: {state.score}")
        level.set(f"Level: {state.level}")
        lines.set(f"Lines: {state.lines}")
        next_piece.set(f"Next: {state.next_piece or '-'}")
        if state.over:
            status.set("GAME OVER — press R")
            canvas.create_text(
                board_width // 2,
                board_height // 2,
                text="GAME OVER\nPress R to restart",
                fill="#f8fafc",
                font=("Segoe UI", 18, "bold"),
                justify="center",
            )
        elif state.paused:
            status.set("PAUSED — press P")
            canvas.create_text(
                board_width // 2,
                board_height // 2,
                text="PAUSED",
                fill="#f8fafc",
                font=("Segoe UI", 20, "bold"),
            )
        else:
            status.set("Playing")

    def on_key(event) -> None:
        key = event.keysym.lower()
        if key == "q":
            root.destroy()
            return
        if key == "p":
            game.toggle_pause()
        elif key == "r":
            game.restart()
        elif key == "left" or key == "a":
            game.move(-1)
        elif key == "right" or key == "d":
            game.move(1)
        elif key in {"up", "x", "w"}:
            game.rotate(1)
        elif key == "z":
            game.rotate(-1)
        elif key in {"down", "s"}:
            game.soft_drop()
        elif key == "space":
            game.hard_drop()
        draw()

    def frame() -> None:
        nonlocal last_tick
        now = time.monotonic()
        if not game.over and not game.paused and now - last_tick >= game.fall_delay():
            game.tick()
            last_tick = now
        draw()
        root.after(16, frame)

    root.bind("<KeyPress>", on_key)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    draw()
    root.after(16, frame)
    root.mainloop()


def _loop(stdscr, game: Game) -> None:
    import time
    last_tick = time.monotonic()
    while True:
        drawn = False
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            drawn = _handle_key(stdscr, game, key) or drawn
        now = time.monotonic()
        if not game.over and not game.paused and now - last_tick >= game.fall_delay():
            game.tick()
            last_tick = now
        _render(stdscr, game)
        stdscr.refresh()
        time.sleep(0.016)


def _handle_key(stdscr, game: Game, key: int) -> bool:
    action = _map_key(key)
    if action == "quit":
        raise _QuitGame
    if action == "pause":
        game.toggle_pause()
    elif action == "restart":
        game.restart()
    elif game.over:
        return False
    elif action == "left":
        game.move(-1)
    elif action == "right":
        game.move(1)
    elif action == "rotate_cw":
        game.rotate(1)
    elif action == "rotate_ccw":
        game.rotate(-1)
    elif action == "soft":
        game.soft_drop()
    elif action == "hard":
        game.hard_drop()
    return True


class _QuitGame(Exception):
    pass


def _map_key(key: int) -> str | None:
    if key in (ord("q"), ord("Q")):
        return "quit"
    if key in (ord("p"), ord("P")):
        return "pause"
    if key in (ord("r"), ord("R")):
        return "restart"
    if key in (curses.KEY_LEFT, ord("a"), ord("A")):
        return "left"
    if key in (curses.KEY_RIGHT, ord("d"), ord("D")):
        return "right"
    if key in (curses.KEY_UP, ord("x"), ord("X"), ord("w"), ord("W")):
        return "rotate_cw"
    if key == ord("z") or key == ord("Z"):
        return "rotate_ccw"
    if key in (curses.KEY_DOWN, ord("s"), ord("S")):
        return "soft"
    if key == ord(" "):
        return "hard"
    return None


def _render(stdscr, game: Game) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    board_col = max(2, (width - (BOARD_WIDTH * 2) - 22) // 6)
    # Reserve a fixed left gutter for the CONTROLS / how-to-play panel and
    # always push the board clear of it, so the panel is never hidden and
    # never overlaps the board's left border.
    panel_width = max([len("CONTROLS")] + [len(line) for line in LEFT_PANEL])
    panel_left = 1
    gutter_width = panel_width + 3
    board_col = max(2, width - (BOARD_WIDTH * 2) - gutter_width - 22)
    if board_col < gutter_width:
        board_col = gutter_width
    hud_col = board_col + BOARD_WIDTH * 2 + 4

    state = game.snapshot()
    grid = _empty_board_lines(state.current, state.board)

    # Border / board.
    _box(stdscr, board_col - 1, 1, BOARD_WIDTH * 2 + 1, BOARD_HEIGHT)

    for y in range(min(BOARD_HEIGHT, height - 3)):
        for x in range(BOARD_WIDTH):
            try:
                stdscr.addstr(2 + y, board_col + x * 2, _cell_text(grid[y][x]))
            except curses.error:
                pass

    # Left controls / how-to-play panel. The board is always pushed clear of
    # the gutter, so the panel can be drawn unconditionally and will never
    # overlap the board's left border.
    try:
        stdscr.addstr(1, panel_left, "CONTROLS")
        for i, line in enumerate(LEFT_PANEL):
            stdscr.addstr(3 + i, panel_left, line)
    except curses.error:
        pass

    # HUD.
    y = 2
    _hud_line(stdscr, hud_col, y, "SCORE", f"{state.score}")
    _hud_line(stdscr, hud_col, y + 2, "LEVEL", f"{state.level}")
    _hud_line(stdscr, hud_col, y + 4, "LINES", f"{state.lines}")
    _hud_line(stdscr, hud_col, y + 6, "COMBO x" + f"{state.combo}", "")

    # Next piece preview.
    _hud_line(stdscr, hud_col, y + 9, "NEXT", "")
    if state.next_piece:
        _draw_mini_piece(stdscr, state.next_piece, y + 10, hud_col)

    # Bottom controls hint.
    hint = "Arrows move \u00b7 X/Z rotate \u00b7 Space drop \u00b7 P pause \u00b7 Q quit"
    try:
        stdscr.addstr(height - 1, 1, hint[: width - 2])
    except curses.error:
        pass

    # Overlays.
    if state.paused:
        _overlay(stdscr, "PAUSED", "Press P to resume", width // 2)
    if state.over:
        _overlay(stdscr, "GAME OVER", f"Final score: {state.score}   Press R to restart", width // 2)
    stdscr.noutrefresh()


def _box(stdscr, x: int, y: int, w: int, h: int) -> None:
    try:
        stdscr.addstr(y, x, "+" + "-" * (w - 2) + "+")
        for r in range(1, h - 1):
            stdscr.addstr(y + r, x, "|")
            stdscr.addstr(y + r, x + w - 1, "|")
        stdscr.addstr(y + h - 1, x, "+" + "-" * (w - 2) + "+")
    except curses.error:
        pass


def _hud_line(stdscr, col: int, row: int, label: str, value: str) -> None:
    try:
        stdscr.addstr(row, col, f"{label}: {value}")
    except curses.error:
        pass


def _overlay(stdscr, title: str, subtitle: str, center_x: int) -> None:
    h, _ = stdscr.getmaxyx()
    cy = h // 2
    box_w = max(len(title), len(subtitle)) + 6
    left = center_x - box_w // 2
    _box(stdscr, left, cy - 2, box_w + 2, 5)
    try:
        stdscr.addstr(cy - 1, left + 3, title)
        stdscr.addstr(cy + 1, left + 3, subtitle)
    except curses.error:
        pass


def main() -> None:
    game = Game()
    try:
        if ui_backend() == "curses":
            run_cli(game)
        else:
            run_tk(game)
    except (_QuitGame, KeyboardInterrupt):
        pass
    finally:
        print("Thanks for playing Tetris!")


if __name__ == "__main__":
    main()
