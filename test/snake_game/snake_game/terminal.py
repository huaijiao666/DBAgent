"""Cross-platform line-oriented terminal interface for Snake."""

from __future__ import annotations

from .game import Direction, GameStatus, SnakeGame

_COMMANDS = {
    "w": Direction.UP,
    "a": Direction.LEFT,
    "s": Direction.DOWN,
    "d": Direction.RIGHT,
}


def render(game: SnakeGame) -> str:
    snake = set(game.snake)
    head = game.snake[0]
    rows = []
    for y in range(game.height):
        row = []
        for x in range(game.width):
            point = (x, y)
            if point == (head.x, head.y):
                row.append("@"); continue
            if any(segment.x == x and segment.y == y for segment in snake):
                row.append("o"); continue
            if game.food is not None and (game.food.x, game.food.y) == point:
                row.append("*"); continue
            row.append(" ")
        rows.append("|" + "".join(row) + "|")
    border = "+" + "-" * game.width + "+"
    state = game.status.value.replace("_", " ")
    return "\n".join([border, *rows, border, f"Score: {game.score} | Status: {state}"])


def run() -> None:
    game = SnakeGame()
    print("Snake — use W/A/S/D then Enter; Q quits; R restarts.")
    while True:
        print(render(game))
        if game.status is not GameStatus.PLAYING:
            print("Press R then Enter to restart, or Q then Enter to quit.")
        command = input("> ").strip().lower()
        if command == "q":
            print("Goodbye!")
            return
        if command == "r":
            game.restart()
            continue
        if game.status is GameStatus.PLAYING and command in _COMMANDS:
            game.turn(_COMMANDS[command])
            game.step()
        else:
            print("Enter W, A, S, D, R, or Q.")
