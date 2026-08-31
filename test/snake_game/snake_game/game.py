"""Deterministic, terminal-independent Snake game logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class GameStatus(Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"
    WON = "won"


class SnakeGame:
    """State and rules for a turn-based Snake board."""

    def __init__(
        self,
        width: int = 20,
        height: int = 10,
        *,
        rng: random.Random | None = None,
        initial_snake: Iterable[Point] | None = None,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("board dimensions must be at least 3x3")
        self.width = width
        self.height = height
        self._rng = rng or random.Random()
        self._initial_snake = tuple(initial_snake) if initial_snake else None
        self.restart()

    def restart(self) -> None:
        center = Point(self.width // 2, self.height // 2)
        self.snake = list(self._initial_snake or (center, Point(center.x - 1, center.y)))
        if not self._valid_points(self.snake) or len(set(self.snake)) != len(self.snake):
            raise ValueError("initial snake must contain unique points on the board")
        self.direction = Direction.RIGHT
        self.status = GameStatus.PLAYING
        self.score = 0
        self.food = self._spawn_food()
        if self.food is None:
            self.status = GameStatus.WON

    def turn(self, direction: Direction) -> None:
        """Change direction, ignoring an immediate reversal."""
        if len(self.snake) > 1 and direction.value == tuple(-n for n in self.direction.value):
            return
        self.direction = direction

    def step(self) -> GameStatus:
        """Advance one turn and return the resulting status."""
        if self.status is not GameStatus.PLAYING:
            return self.status
        dx, dy = self.direction.value
        head = self.snake[0]
        new_head = Point(head.x + dx, head.y + dy)
        growing = new_head == self.food
        body_to_check = self.snake if growing else self.snake[:-1]
        if not self._inside(new_head) or new_head in body_to_check:
            self.status = GameStatus.GAME_OVER
            return self.status
        self.snake.insert(0, new_head)
        if growing:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                self.status = GameStatus.WON
        else:
            self.snake.pop()
        return self.status

    def _inside(self, point: Point) -> bool:
        return 0 <= point.x < self.width and 0 <= point.y < self.height

    def _valid_points(self, points: Iterable[Point]) -> bool:
        return all(self._inside(point) for point in points)

    def _spawn_food(self) -> Point | None:
        occupied = set(self.snake)
        available = [
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if Point(x, y) not in occupied
        ]
        return self._rng.choice(available) if available else None
