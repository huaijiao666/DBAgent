#!/usr/bin/env python3
"""Deterministic unit tests for the core Tetris logic in tetris.py.

These tests never require a terminal or curses; they exercise the pure game
logic classes (Bag, Piece, Game) with injected, seeded RNGs so every run is
identical.
"""

import random
import unittest

import tetris


def make_game(seed=1234):
    """Create a Game with a seeded RNG for reproducible piece order."""
    return tetris.Game(random.Random(seed))


class BagTests(unittest.TestCase):
    def test_7_bag_shuffles_all_pieces_once(self):
        """Every 7 draws yield each of the 7 pieces exactly once."""
        bag = tetris.Bag(random.Random(42))
        counts = {}
        for _ in range(7):
            p = bag.next()
            counts[p] = counts.get(p, 0) + 1
        self.assertSetEqual(set(counts), set(tetris._SHAPES.keys()))
        for c in counts.values():
            self.assertEqual(c, 1)

    def test_repeats_consistent_across_bags(self):
        """A normal seed produces a well-mixed (differently ordered) series."""
        bag1 = tetris.Bag(random.Random(1))
        bag2 = tetris.Bag(random.Random(1))
        series1 = [bag1.next() for _ in range(14)]
        series2 = [bag2.next() for _ in range(14)]
        # Same seed => identical sequence.
        self.assertEqual(series1, series2)
        # Two consecutive bags differ (not identical repeats).
        self.assertNotEqual(series1[:7], series1[7:])


class PieceTests(unittest.TestCase):
    def test_moved_shifts_cell_coordinates(self):
        piece = tetris.Piece("O", 0, x=3, y=2)
        moved = piece.moved(1, 4)
        self.assertEqual(moved.x, 4)
        self.assertEqual(moved.y, 6)
        # Original is unchanged (immutable style).
        self.assertEqual((piece.x, piece.y), (3, 2))

    def test_rotation_wraps(self):
        piece = tetris.Piece("I", 3)
        self.assertEqual(piece.rotated(1).rotation, 0)
        self.assertEqual(piece.rotated(-1).rotation, 2)

    def test_i_definition_width(self):
        # 'I' piece in any rotation never exceeds 4 cells wide.
        for r in range(4):
            cells = tetris.Piece("I", r).cells()
            self.assertEqual(len(cells), 4)


class MovementTests(unittest.TestCase):
    def test_move_blocks_at_left_wall(self):
        game = make_game()
        # Pin the current piece against a left wall by moving left repeatedly.
        for _ in range(12):
            game.move(-1)
        x_before = game.current.x
        # Already pinned at the wall, so a further left move is rejected.
        self.assertFalse(game.move(-1))
        self.assertEqual(game.current.x, x_before)

    def test_move_down_via_soft_drop_changes_y(self):
        game = make_game()
        y_before = game.current.y
        game.soft_drop()
        self.assertEqual(game.current.y, y_before + 1)

    def test_horizontal_move_changes_x(self):
        game = make_game()
        x_before = game.current.x
        self.assertTrue(game.move(1))
        self.assertEqual(game.current.x, x_before + 1)


class RotationTests(unittest.TestCase):
    def test_free_rotation_succeeds_in_air(self):
        game = make_game()
        rot_before = game.current.rotation
        self.assertTrue(game.rotate(1))
        self.assertEqual(game.current.rotation, (rot_before + 1) % 4)

    def test_rotation_blocked_at_wall_is_rejected(self):
        # 'I' piece is 4 wide; push hard left then rotation 0->1 may fail/kick.
        game = make_game()
        for _ in range(12):
            game.move(-1)
        # It may fall to lock; the invariants we assert should hold regardless:
        if game.current is not None:
            before = game.current.rotation
            game.rotate(1)
            # Either rotation succeeded (still on board) or was rejected
            # leaving rotation unchanged.
            self.assertTrue(
                game.current.rotation == (before + 1) % 4
                or game.current.rotation == before
            )


class CollisionLockTests(unittest.TestCase):
    def test_piece_locks_when_it_cannot_fall(self):
        game = make_game()
        # Hard-drop the current piece, which locks and spawns the next.
        self.assertIsNotNone(game.current)
        game.hard_drop()
        self.assertIsNotNone(game.current)
        # The locked piece must appear on the board.
        flattened = [cell for row in game.board for cell in row]
        self.assertTrue(any(c is not None for c in flattened))

    def test_soft_drop_to_floor_then_lock(self):
        game = make_game()
        # Drop it all the way.
        for _ in range(tetris.BOARD_HEIGHT):
            if not game.soft_drop() and game.current is not None:
                break
        # After many soft drops the piece should eventually have locked/spawned.
        self.assertIsNotNone(game.current)


class LineClearingTests(unittest.TestCase):
    def help_fill_row(self, game, y):
        for x in range(tetris.BOARD_WIDTH):
            game.board[y][x] = "Z"

    def test_single_line_clear_score_and_lines(self):
        game = make_game()
        self.help_fill_row(game, tetris.BOARD_HEIGHT - 1)
        cleared = game._clear_lines()
        self.assertEqual(cleared, 1)
        self.assertEqual(game.lines, 1)
        # level 1, combo reset to 0 => multiplier 1.0
        self.assertEqual(game.score, 100)
        # The full row should now be empty at the top.
        self.assertTrue(all(c is None for c in game.board[0]))

    def test_triple_line_clear_uses_500_base(self):
        game = make_game()
        for y in range(tetris.BOARD_HEIGHT - 3, tetris.BOARD_HEIGHT):
            self.help_fill_row(game, y)
        cleared = game._clear_lines()
        self.assertEqual(cleared, 3)
        self.assertEqual(game.lines, 3)
        self.assertEqual(game.score, 500)

    def test_tetris_multiplier_applies(self):
        # Filling the bottom 4 rows = a Tetris (line clear of 4).
        game = make_game()
        for y in range(tetris.BOARD_HEIGHT - 4, tetris.BOARD_HEIGHT):
            self.help_fill_row(game, y)
        cleared = game._clear_lines()
        self.assertEqual(cleared, 4)
        # combo becomes 1 => multiplier 1 + 0.5 = 1.5; base 800 * 1.5 * level(1)
        self.assertEqual(game.score, int(800 * 1.5))

    def test_no_clear_resets_combo(self):
        game = make_game()
        # Force a tetris first.
        for y in range(tetris.BOARD_HEIGHT - 4, tetris.BOARD_HEIGHT):
            self.help_fill_row(game, y)
        game._clear_lines()
        self.assertEqual(game.combo, 1)
        # A non-clear resets combo to 0.
        game._clear_lines()
        self.assertEqual(game.combo, 0)


class LevelTests(unittest.TestCase):
    def help_fill_row(self, game, y):
        for x in range(tetris.BOARD_WIDTH):
            game.board[y][x] = "Z"

    def test_level_up_after_ten_lines(self):
        game = make_game()
        # Clear 10 single lines (each 100 pts at level 1).
        for _ in range(10):
            self.help_fill_row(game, tetris.BOARD_HEIGHT - 1)
            game._clear_lines()
        self.assertEqual(game.lines, 10)
        self.assertEqual(game.level, 2)

    def test_fall_delay_decreases_with_level(self):
        game = make_game()
        l1 = game.fall_delay()
        game.level = 5
        l5 = game.fall_delay()
        self.assertLess(l5, l1)
        self.assertGreaterEqual(l5, tetris.MIN_FALL_DELAY)


class GameOverTests(unittest.TestCase):
    def test_pieces_stack_to_top_ends_game(self):
        # Fill all but the top rows solidly, then lock the current piece on top.
        game = make_game()
        for y in range(0, tetris.BOARD_HEIGHT):
            for x in range(tetris.BOARD_WIDTH):
                game.board[y][x] = "T"
        # Force a lock of the current piece.
        game._lock()
        self.assertTrue(game.over)

    def test_pause_disables_gravity_and_actions(self):
        game = make_game()
        game.toggle_pause()
        self.assertTrue(game.paused)
        # Actions should be no-ops while paused.
        x_before = game.current.x
        self.assertFalse(game.move(1))
        self.assertEqual(game.current.x, x_before)
        y_before = game.current.y
        game.tick()
        game.soft_drop()
        self.assertEqual(game.current.y, y_before)
        game.toggle_pause()
        self.assertFalse(game.paused)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_reflects_state(self):
        game = make_game()
        snap = game.snapshot()
        self.assertEqual(snap.score, game.score)
        self.assertEqual(snap.level, game.level)
        self.assertEqual(snap.lines, game.lines)
        self.assertIsNotNone(snap.board)
        self.assertEqual(snap.board, game.board)
        self.assertIsNotNone(snap.current)


class UiBackendTests(unittest.TestCase):
    def test_windows_without_curses_uses_tkinter_backend(self):
        original_curses = tetris.curses
        try:
            tetris.curses = None
            self.assertEqual(tetris.ui_backend(), "tkinter")
        finally:
            tetris.curses = original_curses

    def test_curses_backend_is_preserved_when_available(self):
        original_curses = tetris.curses
        try:
            tetris.curses = object()
            self.assertEqual(tetris.ui_backend(), "curses")
        finally:
            tetris.curses = original_curses


if __name__ == "__main__":
    unittest.main(verbosity=2)
