# -*- coding: utf-8 -*-
"""A playable, polish-focused console (terminal) Tetris.

This is the interactive UI layer.  All game logic - piece generation (7-bag),
movement, rotation with wall-kicks, hold, scoring, ghost preview, game-over
detection - lives in :mod:`tetris_core`, which has zero third-party dependencies
so it can be unit-tested deterministically.  This module only *renders* that
state and translates keystrokes into move calls.

Because ``pygame`` is not available in this environment, the game is delivered
as a crisp ANSI colour terminal application that runs anywhere a terminal and a
``TERM``/colour-capable console exist.

Controls
--------
Left / Right arrow  or ``A`` / ``D``   move the piece sideways
Up / ``W`` / ``X``                     rotate clockwise
``Z``                                  rotate counter-clockwise
Down / ``S``                           soft-drop (faster, +1 point/cell)
Space                                  hard-drop (lock immediately)
``C``                                  hold / swap piece
``P`` / ``Esc``                        pause / resume
``R``                                  restart (with a brief countdown)
``Q``                                  quit
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import tetris_core as core

# --------------------------------------------------------------------------- #
# Colour palette (ANSI true-colour / 24-bit RGB).
# --------------------------------------------------------------------------- #
ANSI_RESET = "\x1b[0m"
ANSI_CLEAR = "\x1b[2J\x1b[H"

# Each tetromino maps to a vivid fill background.
SHAPE_BG = {
    "I": "\x1b[48;2;0;220;220m",     # cyan
    "J": "\x1b[48;2;40;100;240m",    # blue
    "L": "\x1b[48;2;240;150;20m",    # orange
    "O": "\x1b[48;2;250;210;40m",    # yellow
    "S": "\x1b[48;2;90;210;60m",     # green
    "T": "\x1b[48;2;180;80;200m",    # purple
    "Z": "\x1b[48;2;230;60;60m",     # red
    "G": "\x1b[48;2;70;70;80m",      # ghost (dimmer)
    "W": "\x1b[48;2;255;255;255m",   # line-clear flash (white)
    "-": "\x1b[48;2;28;32;42m",      # empty cell
}
TEXT_DARK = "\x1b[38;2;10;12;16m"
TEXT_DIM = "\x1b[38;2;120;128;140m"
TEXT_BRIGHT = "\x1b[38;2;230;238;248m"
TEXT_ACCENT = "\x1b[38;2;90;220;200m"

# Each visible board cell is rendered as a 2-wide block of solid fill so the
# board looks pleasantly rectangular even on DOS/Windows pseudo-terminals.
FILL = "\u2588\u2588"          # u+2588 full block, x2


def _empty_cell() -> str:
    return SHAPE_BG["-"] + "  " + ANSI_RESET


def _cell(shape: str) -> str:
    """Render one 2-char cell for a shape (ghost/white-flash handled by key)."""
    return SHAPE_BG[shape] + FILL + ANSI_RESET


# --------------------------------------------------------------------------- #
# Drop speed per level (seconds per row).  Classic accelerating curve.
# --------------------------------------------------------------------------- #
def drop_interval(level: int) -> float:
    """Row-to-row gravity interval in seconds; speeds up as level rises."""
    level = max(1, level)
    base = 1.0 / (0.8 + (level - 1) * 0.15)
    return max(0.06, base)


# --------------------------------------------------------------------------- #
# Game : a thin controller tying a core.GameState to input + tick.
# --------------------------------------------------------------------------- #
class Game:
    """High-level controller: owns a :class:`core.GameState`.

    Exposes the two entry points the UI (and the unit tests) drive:
    ``handle_key`` for discrete inputs and ``step`` for an autonomous gravity
    tick.  Pause / game-over are tracked here and surfaced as flags.
    """

    def __init__(self, rng=None):
        self.state = core.GameState(rng=rng)
        self.paused = False
        self.over = False

    # -- discrete input -------------------------------------------------- #
    def handle_key(self, key: str):
        """Map a canonical key name to a gameplay move.

        Returns a result dict from :mod:`tetris_core` or ``{"handled": False}``
        when nothing meaningful happened (including while paused/over).
        """
        if self.paused or self.state.game_over:
            return {"handled": False}

        s = self.state
        if key in ("left", "a"):
            return s.move(-1)
        if key in ("right", "d"):
            return s.move(1)
        if key in ("up", "w", "x"):
            return s.rotate(1)
        if key in ("z",):
            return s.rotate(-1)
        if key in ("down", "s"):
            return s.soft_drop()
        if key == "space":
            return s.hard_drop()
        if key in ("c",):
            return s.hold_piece()
        return {"handled": False}

    # -- autonomous tick ------------------------------------------------ #
    def step(self):
        """Advance one gravity tick while running (no-op if paused/over)."""
        if self.paused or self.state.game_over:
            return None
        res = self.state.soft_drop()
        if res.get("locked"):
            self.over = self.state.game_over
        return res


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_mini(shape: Optional[str], width: int) -> List[str]:
    """Render a single tetromino into a tiny block map for hold/next previews.

    Uses the shape's canonical definition cells so previews are accurate.
    """
    if shape is None:
        return ["  " * width]
    t = core.Tetromino(shape)
    cells = t.cells()
    grid = [[" "] * 4 for _ in range(4)]
    for r, c in cells:
        if 0 <= r < 4 and 0 <= c < 4:
            grid[r][c] = shape
    rows = ["".join(_cell(g) if g != " " else _empty_cell() for g in row)
            for row in grid]
    return rows


def build_frame(g: Game, flash_rows) -> List[str]:
    """Compose the full UI frame as a list of pre-coloured text lines.

    ``flash_rows`` is ``None`` normally, or a set of row indices that are being
    cleared and should render as bright white (the flash effect).
    """
    s = g.state
    cells = [[" " for _ in range(core.COLS)] for _ in range(core.ROWS)]

    # 1. Locked blocks (hidden when flashing).
    for r in range(core.ROWS):
        for c in range(core.COLS):
            sh = s.grid[r][c]
            if sh and (flash_rows is None or r not in flash_rows):
                cells[r][c] = sh

    # 2. Ghost projection beneath the current piece.
    if not g.paused and not s.game_over:
        for (r, c) in s.ghost_cells():
            if 0 <= r < core.ROWS and 0 <= c < core.COLS:
                if cells[r][c] == " ":
                    cells[r][c] = "G"

    # 3. Current piece on top.
    if not g.paused and not s.game_over:
        for (r, c) in s.current_cells():
            if 0 <= r < core.ROWS and 0 <= c < core.COLS:
                cells[r][c] = s.current.shape

    # Board rows (each cell is a 2-wide fill coloured by shape).
    char_of = {" ": _empty_cell(),
               "G": _cell("G"),
               "W": _cell("W")}
    board_lines = []
    for r in range(core.ROWS):
        row = ""
        for c in range(core.COLS):
            key = cells[r][c]
            if key == " ":
                row += _empty_cell()
            elif key == "G":
                row += _cell("G")
            elif flash_rows and r in flash_rows:
                row += _cell("W")
            else:
                row += _cell(key)
        board_lines.append(row)

    # ---- Preview / hold helpers ----------------------------------------- #
    def panel_block(title: str, shape: Optional[str], locked: bool) -> List[str]:
        w = 4
        dim = TEXT_DIM if locked else TEXT_BRIGHT
        lines = [f"  {dim}{title}{ANSI_RESET}"]
        for ln in _render_mini(shape, w)[:3]:
            lines.append("  " + ln + "  ")
        return lines

    hold_shape = s.hold
    nxt = s.next_shapes()

    hold_panel = panel_block("HOLD", hold_shape, locked=not s.can_hold)

    next_lines = panel_block("NEXT", (nxt[0] if nxt else None), locked=False)
    next_lines += [TEXT_DIM + "  plus 2 more  " + ANSI_RESET]

    # ---- HUD ------------------------------------------------------------- #
    hud = [
        "",
        f"  {TEXT_BRIGHT}SCORE{ANSI_RESET}   {TEXT_ACCENT}{s.score:>6}{ANSI_RESET}",
        f"  {TEXT_BRIGHT}LEVEL{ANSI_RESET}   {TEXT_ACCENT}{s.level:>6}{ANSI_RESET}",
        f"  {TEXT_BRIGHT}LINES{ANSI_RESET}   {TEXT_ACCENT}{s.lines:>6}{ANSI_RESET}",
        f"  {TEXT_BRIGHT}NEXT{ANSI_RESET}    {TEXT_ACCENT}{' '.join(nxt)}{ANSI_RESET}",
        "",
    ]

    # ---- Combine panels ------------------------------------------------ #
    left = hold_panel
    right = hud
    mid = board_lines

    height = max(len(left), len(mid), len(right))
    row_content: List[str] = []
    sep = "   "
    for i in range(height):
        l = left[i] if i < len(left) else " " * 12
        m = mid[i] if i < len(mid) else ""
        r = right[i] if i < len(right) else ""
        row_content.append(l + sep + m + sep + r)

    # ---- Status line ----------------------------------------------------- #
    status = status_line(g)
    row_content.append("")
    row_content.append(status)
    return row_content


def status_line(g: Game) -> str:
    """The footer line describing the current mode / next restart."""
    if g.over:
        return TEXT_BRIGHT + "  GAME OVER  " + TEXT_DIM + \
               "[R] restart   [Q] quit" + ANSI_RESET
    if g.paused:
        return TEXT_BRIGHT + "  PAUSED     " + TEXT_DIM + \
               "[P]/[Esc] resume   [Q] quit" + ANSI_RESET
    return TEXT_DIM + ("  Left/Right:move  Up:rotate  Z:ccw  "
                       "Down:soft  Space:hard  C:hold  P:pause  R:restart  "
                       "Q:quit") + ANSI_RESET


# --------------------------------------------------------------------------- #
# Terminal plumbing (raw key input + full-repaint rendering).
# --------------------------------------------------------------------------- #
class TerminalIO:
    """Cross-platform raw-key reader and screen writer.

    Uses ``termios``/``select`` on POSIX and ``msvcrt`` on Windows, so the game
    runs identically in cmd, PowerShell, the Windows Terminal, and any Unix
    terminal.
    """

    def __init__(self):
        self._raw = False
        self._posix = sys.platform not in ("win32", "win64")

    def enter_raw(self):
        if self._posix:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            self._raw = True
        else:
            import msvcrt  # noqa: F401  (Windows raw-read path)
            self._raw = True

    def exit_raw(self):
        if self._posix and self._raw:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._raw = False

    def read_key(self) -> Optional[str]:
        """Return a canonical key name, or None when nothing pressed.

        Non-blocking in all cases.
        """
        if self._posix:
            import select
            if not select.select([sys.stdin], [], [], 0)[0]:
                return None
            ch = sys.stdin.read(1)
        else:
            import msvcrt
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch == "\x00" or ch == "\xe0":    # arrow / function prefix
                ch2 = msvcrt.getwch()
                return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(ch2)
        return self._map_ansi(ch)

    @staticmethod
    def _map_ansi(ch: str) -> Optional[str]:
        mapping = {
            "a": "a", "d": "d", "s": "s", "w": "w",
            "A": "a", "D": "d", "S": "s", "W": "w",
            "x": "x", "X": "x", "z": "z", "Z": "z",
            "c": "c", "C": "c",
            "p": "pause", "P": "pause", "\x03": "quit",   # Ctrl-C
            "r": "restart", "R": "restart",
            "q": "quit", "Q": "quit",
            " ": "space",
            "\r": "enter",
        }
        if ch == "\x1b":
            return "esc"
        return mapping.get(ch)

    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()

    def clear_screen(self):
        self.write("\x1b[0m" + ANSI_CLEAR)


# --------------------------------------------------------------------------- #
# The interactive main loop.
# --------------------------------------------------------------------------- #
RESTART_DELAY = 1.0      # seconds showing the restart countdown
FLASH_DURATION = 0.28    # seconds of the line-clear white flash


def run_game(seed=None) -> int:
    """Run the interactive game until the player quits.

    ``seed`` is optional and, when given, forces a deterministic piece order
    (handy for demos / tests driving the loop in a headless way).
    """
    io = TerminalIO()
    try:
        io.enter_raw()
    except (ImportError, OSError):
        # Not a real terminal; we cannot play interactively.  Provide a clear
        # diagnostic instead of crashing silently.
        print("This game needs an interactive terminal with raw-input support.")
        return 42

    game = Game(rng=seed)
    acc = 0.0
    last = time.perf_counter()
    flash_t = 0.0
    flash_rows: Optional[set] = None
    restart_at = None
    paused_at = False

    try:
        io.clear_screen()
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now

            # ---- handle a key if pressed -------------------------------- #
            key = io.read_key()
            if key is not None:
                if key == "quit":
                    break
                if key == "pause":
                    if restart_at is None:
                        game.paused = not game.paused
                    continue
                if key == "start":        # placeholder, not used
                    pass
                if key == "restart":
                    # Restart after a short countdown for good UX.
                    restart_at = now + RESTART_DELAY
                    continue
                # Otherwise it is a gameplay move.
                if not game.paused and not game.over and restart_at is None:
                    game.handle_key(key)

            # ---- pending restart ---------------------------------------- #
            if restart_at is not None:
                if now >= restart_at:
                    return run_game(seed)   # fresh game, fresh RNG / state
                continue                    # hold the field until it fires

            # ---- flash expiry ------------------------------------------- #
            if flash_rows is not None and now >= flash_t:
                flash_rows = None

            # ---- gravity & line clears ----------------------------------- #
            if not game.paused and not game.over:
                interval = drop_interval(game.state.level)
                acc += dt
                while acc >= interval:
                    acc -= interval
                    res = game.state.soft_drop()
                    game.over = game.state.game_over
                    if res.get("cleared"):
                        n = res["cleared"]
                        flash_rows = _full_rows(game)
                        flash_t = now + FLASH_DURATION

            # ---- render --------------------------------------------------- #
            frame = build_frame(game, flash_rows)
            io.write("\x1b[0m\x1b[H" + "\n".join(frame) + ANSI_RESET + "\r\n")

            # ---- keep CPU sensible -------------------------------------- #
            time.sleep(0.004)
    except KeyboardInterrupt:
        pass
    finally:
        io.exit_raw()
        io.write("\x1b[0m" + ANSI_CLEAR)
    return 0


def _full_rows(game: Game) -> set:
    """Return the set of currently-full row indices (for the flash effect)."""
    return {r for r in range(core.ROWS)
            if all(game.state.grid[r][c] is not None
                   for c in range(core.COLS))}


if __name__ == "__main__":
    sys.exit(run_game())
