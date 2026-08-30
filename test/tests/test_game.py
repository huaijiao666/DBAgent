"""Focused tests for the pure Snake model and terminal adapter."""

import subprocess
import sys

import pytest

from snake_game import (
    Direction,
    Point,
    change_direction,
    create_game,
    hits_self,
    hits_wall,
    move_snake,
    place_food,
)
from snake_game.runner import KEYS, render, requested_direction


def test_point_movement_is_normal_and_non_mutating():
    point = Point(2, 3)

    assert point.moved(Direction.RIGHT) == Point(3, 3)
    assert point == Point(2, 3)


def test_move_advances_head_and_drops_tail():
    state = create_game(
        5,
        3,
        snake=(Point(1, 1), Point(0, 1)),
        direction=Direction.RIGHT,
        food=Point(4, 2),
    )

    moved = move_snake(state)

    assert moved.snake == (Point(2, 1), Point(1, 1))
    assert moved.food == Point(4, 2)
    assert moved.score == 0
    assert state.snake == (Point(1, 1), Point(0, 1))


def test_legal_turn_is_accepted():
    assert change_direction(Direction.RIGHT, Direction.UP) is Direction.UP


def test_immediate_reverse_turn_is_rejected():
    assert change_direction(Direction.RIGHT, Direction.LEFT) is Direction.RIGHT


def test_eating_grows_snake_and_increments_score():
    state = create_game(
        4,
        2,
        snake=(Point(1, 0),),
        direction=Direction.RIGHT,
        food=Point(2, 0),
    )

    moved = move_snake(state)

    assert moved.snake == (Point(2, 0), Point(1, 0))
    assert moved.score == 1
    assert moved.food == Point(0, 0)


def test_food_preference_is_used_when_free():
    preferred = Point(3, 2)

    assert place_food((Point(0, 0),), 4, 3, preferred) == preferred


def test_food_fallback_is_deterministic_row_major():
    snake = (Point(0, 0), Point(1, 0), Point(0, 1))

    assert place_food(snake, 3, 2, Point(1, 0)) == Point(2, 0)
    assert place_food(snake, 3, 2, Point(1, 0)) == Point(2, 0)


def test_food_is_none_when_board_is_full():
    assert place_food((Point(0, 0), Point(1, 0)), 2, 1) is None


def test_wall_helper_detects_outside_points():
    assert hits_wall(Point(-1, 0), 3, 2)
    assert hits_wall(Point(3, 1), 3, 2)
    assert not hits_wall(Point(2, 1), 3, 2)


def test_wall_collision_ends_game():
    state = create_game(2, 1, snake=(Point(1, 0),), direction=Direction.RIGHT)

    moved = move_snake(state)

    assert moved.game_over
    assert moved.snake == state.snake


def test_self_collision_ends_game():
    state = create_game(
        4,
        4,
        snake=(Point(1, 1), Point(2, 1), Point(2, 2), Point(1, 2), Point(0, 2), Point(0, 1)),
        direction=Direction.RIGHT,
        food=Point(3, 3),
    )

    moved = move_snake(state)

    assert moved.game_over


def test_move_into_old_tail_is_allowed_when_not_growing():
    state = create_game(
        3,
        3,
        snake=(Point(1, 1), Point(0, 1), Point(0, 0), Point(1, 0)),
        direction=Direction.UP,
        food=Point(2, 2),
    )

    moved = move_snake(state)

    assert moved.snake == (Point(1, 0), Point(1, 1), Point(0, 1), Point(0, 0))
    assert moved.status == "playing"
    assert not hits_self(Point(1, 0), state.snake, growing=False)


def test_last_free_cell_wins_and_removes_food():
    state = create_game(
        2,
        2,
        snake=(Point(0, 1), Point(0, 0), Point(1, 0)),
        direction=Direction.RIGHT,
        food=Point(1, 1),
    )

    moved = move_snake(state)

    assert moved.won
    assert moved.snake == (Point(1, 1), Point(0, 1), Point(0, 0), Point(1, 0))
    assert moved.food is None
    assert moved.score == 1


def test_finished_game_does_not_advance():
    state = create_game(1, 1, snake=(Point(0, 0),), food=None)
    state = state.__class__(state.width, state.height, state.snake, state.direction, state.food, state.score, "won")

    assert move_snake(state) is state


def test_runner_ignores_immediate_reverse_input():
    assert requested_direction(Direction.RIGHT, "a") is Direction.RIGHT



def test_runner_key_map_and_render_are_import_safe():
    state = create_game(3, 2, snake=(Point(1, 0),), direction=Direction.RIGHT, food=Point(2, 1))

    assert KEYS["w"] is Direction.UP
    output = render(state)
    assert "@" in output and "*" in output


def test_runner_module_import_does_not_start_input_loop():
    result = subprocess.run(
        [sys.executable, "-c", "import snake_game.runner; print('imported safely')"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "imported safely"
    assert result.stderr == ""


def test_invalid_board_dimensions_are_rejected():
    with pytest.raises(ValueError):
        create_game(0, 2)
