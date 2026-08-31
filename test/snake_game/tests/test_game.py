from snake_game import Direction, GameStatus, Point, SnakeGame


def test_movement_advances_head_and_removes_tail() -> None:
    game = SnakeGame(width=8, height=6, initial_snake=[Point(2, 3), Point(1, 3)])
    game.food = Point(7, 5)

    game.step()

    assert game.snake == [Point(3, 3), Point(2, 3)]
    assert game.status is GameStatus.PLAYING


def test_eating_food_grows_snake_and_increases_score() -> None:
    game = SnakeGame(width=8, height=6, initial_snake=[Point(2, 3), Point(1, 3)])
    game.food = Point(3, 3)

    game.step()

    assert game.snake == [Point(3, 3), Point(2, 3), Point(1, 3)]
    assert game.score == 1
    assert game.food not in game.snake


def test_wall_collision_ends_game() -> None:
    game = SnakeGame(width=3, height=3, initial_snake=[Point(1, 1)])
    game.food = Point(2, 2)
    game.turn(Direction.UP)

    game.step()
    game.step()

    assert game.status is GameStatus.GAME_OVER
    assert game.snake == [Point(1, 0)]


def test_self_collision_ends_game() -> None:
    game = SnakeGame(
        width=5,
        height=5,
        initial_snake=[Point(2, 2), Point(2, 1), Point(1, 1), Point(1, 2)],
    )
    game.turn(Direction.UP)

    game.step()

    assert game.status is GameStatus.GAME_OVER
    assert game.snake[0] == Point(2, 2)


def test_food_is_spawned_on_an_empty_cell() -> None:
    snake = [Point(0, 0), Point(1, 0), Point(2, 0)]
    game = SnakeGame(width=4, height=3, initial_snake=snake)

    assert game.food is not None
    assert game.food not in game.snake
    assert 0 <= game.food.x < game.width
    assert 0 <= game.food.y < game.height


def test_full_board_is_a_win() -> None:
    points = [Point(x, y) for y in range(3) for x in range(3)]

    game = SnakeGame(width=3, height=3, initial_snake=points)

    assert game.status is GameStatus.WON
    assert game.food is None


def test_restart_restores_initial_state() -> None:
    game = SnakeGame(width=8, height=6, initial_snake=[Point(2, 3), Point(1, 3)])
    game.food = Point(3, 3)
    game.step()
    game.restart()

    assert game.snake == [Point(2, 3), Point(1, 3)]
    assert game.score == 0
    assert game.status is GameStatus.PLAYING
    assert game.food not in game.snake
