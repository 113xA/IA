from __future__ import annotations

from collections import deque
from typing import Protocol

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


def state_key(
    state: SearchState,
) -> tuple[Position, tuple[Position, ...], tuple[Position, ...]]:
    """Return the hashable search identity of a Pacman state."""

    return (
        state.getPacmanPosition(),
        tuple(sorted(state.getFood().asList())),
        tuple(sorted(state.getCapsules())),
    )


class PacmanAgent(Agent):
    """Pacman agent based on breadth-first search."""

    def __init__(self):
        super().__init__()
        self.moves: list[str] | None = None

    def get_action(self, state: SearchState) -> str:  # type: ignore[override]
        """Return the next legal move on a shortest winning path."""

        if self.moves is None:
            self.moves = self.bfs(state)

        if self.moves:
            action = self.moves.pop(0)
            legal_actions = state.getLegalActions()
            if action in legal_actions:
                return action

        legal_actions = state.getLegalActions()
        if Directions.STOP in legal_actions:
            return Directions.STOP
        return legal_actions[0] if legal_actions else Directions.STOP

    def bfs(self, state: SearchState) -> list[str]:
        """Return a shortest sequence of moves that wins from ``state``."""

        fringe: deque[tuple[SearchState, list[str]]] = deque()
        fringe.append((state, []))
        visited: set[
            tuple[Position, tuple[Position, ...], tuple[Position, ...]]
        ] = {state_key(state)}

        while fringe:
            current, path = fringe.popleft()

            if current.isWin():
                return path

            for successor, action in current.generatePacmanSuccessors():
                successor_key = state_key(successor)
                if successor_key in visited:
                    continue
                visited.add(successor_key)
                fringe.append((successor, path + [action]))

        return []
