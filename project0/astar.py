from __future__ import annotations

import heapq
from itertools import count
from typing import Protocol, TypeAlias

from pacman_module.game import Agent, Directions


Position = tuple[int, int]


class FoodGrid(Protocol):
    """Subset of the food-grid API used by the search agent."""

    def asList(self) -> list[Position]:
        """Return the positions containing food."""
        ...


class SearchState(Protocol):
    """Subset of the game-state API used by the search agent."""

    def getPacmanPosition(self) -> Position:
        """Return Pacman's current position."""
        ...

    def getFood(self) -> FoodGrid:
        """Return the remaining food grid."""
        ...

    def getCapsules(self) -> list[Position]:
        """Return the remaining capsule positions."""
        ...

    def getLegalActions(self) -> list[str]:
        """Return Pacman's legal actions."""
        ...

    def generatePacmanSuccessors(self) -> list[tuple[SearchState, str]]:
        """Return successor states and actions."""
        ...

    def isWin(self) -> bool:
        """Return whether this state is a winning state."""
        ...


SearchEntry: TypeAlias = tuple[
    int,
    int,
    SearchState,
    list[str],
    int,
]

CAPSULE_PENALTY = 5


def state_key(
    state: SearchState,
) -> tuple[Position, tuple[Position, ...], tuple[Position, ...]]:
    """Return the hashable search identity of a Pacman state."""

    return (
        state.getPacmanPosition(),
        tuple(sorted(state.getFood().asList())),
        tuple(sorted(state.getCapsules())),
    )


def manhattan_distance(first: Position, second: Position) -> int:
    """Return the Manhattan distance between two grid positions."""

    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def heuristic(state: SearchState) -> int:
    """Return an admissible lower bound on the cost of eating all food.

    The bound combines the Manhattan distance from Pacman to the closest
    remaining food with a Manhattan minimum spanning tree over the food.
    Capsule penalties are deliberately omitted because they are non-negative.
    """

    food = state.getFood().asList()
    if not food:
        return 0

    position = state.getPacmanPosition()
    connection = min(manhattan_distance(position, dot) for dot in food)
    tree_cost = 0
    remaining = set(food)
    tree = {remaining.pop()}

    while remaining:
        distance, next_dot = min(
            (
                manhattan_distance(tree_dot, dot),
                dot,
            )
            for tree_dot in tree
            for dot in remaining
        )
        tree_cost += distance
        remaining.remove(next_dot)
        tree.add(next_dot)

    return connection + tree_cost


class PacmanAgent(Agent):
    """Pacman agent based on A* search."""

    def __init__(self):
        super().__init__()
        self.moves: list[str] | None = None

    def get_action(self, state: SearchState) -> str:  # type: ignore[override]
        """Return the next legal move on an optimal winning path."""

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            action = self.moves.pop(0)
            legal_actions = state.getLegalActions()
            if action in legal_actions:
                return action

        legal_actions = state.getLegalActions()
        if Directions.STOP in legal_actions:
            return Directions.STOP
        return legal_actions[0] if legal_actions else Directions.STOP

    def astar(self, state: SearchState) -> list[str]:
        """Return a shortest sequence of moves that wins from ``state``."""

        entries: list[SearchEntry] = []
        sequence = count()
        heapq.heappush(
            entries,
            (heuristic(state), next(sequence), state, [], 0),
        )
        best_cost = {state_key(state): 0}

        while entries:
            priority, order, current, path, cost = heapq.heappop(entries)
            del priority, order
            current_key = state_key(current)
            if cost != best_cost.get(current_key):
                continue

            if current.isWin():
                return path

            for successor, action in current.generatePacmanSuccessors():
                successor_key = state_key(successor)
                capsule_was_eaten = len(successor.getCapsules()) < len(
                    current.getCapsules()
                )
                step_cost = 1 + (
                    CAPSULE_PENALTY if capsule_was_eaten else 0
                )
                successor_cost = cost + step_cost
                if successor_cost >= best_cost.get(
                    successor_key, float('inf')
                ):
                    continue

                best_cost[successor_key] = successor_cost
                priority = successor_cost + heuristic(successor)
                heapq.heappush(
                    entries,
                    (
                        priority,
                        next(sequence),
                        successor,
                        path + [action],
                        successor_cost,
                    ),
                )

        return []
