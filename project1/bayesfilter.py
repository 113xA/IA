from collections import deque
from math import comb

import numpy as np

from pacman_module.game import Actions, Agent, Directions, manhattanDistance


# Fear parameter of each ghost policy, see `pacman_module/ghostAgents.py`.
GHOST_FEAR = {'afraid': 1.0, 'fearless': 0.0, 'terrified': 3.0}

# Parameters of the binomial noise of the distance sensor.
NOISE_N = 4
NOISE_P = 0.5


def _neighbors(walls, cell):
    """Lists the cells that are one legal move away from a cell.

    Arguments:
        walls: The W x H grid of walls.
        cell: A (x, y) position.

    Returns:
        The list of neighboring (x, y) positions that are not walls.
    """

    return [
        (cell[0] + dx, cell[1] + dy)
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
        if not walls[cell[0] + dx][cell[1] + dy]
    ]


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """

        W, H = walls.width, walls.height
        T = np.zeros((W, H, W, H))
        favored = 2 ** GHOST_FEAR[self.ghost]

        for i in range(W):
            for j in range(H):
                if walls[i][j]:
                    continue

                moves = _neighbors(walls, (i, j))

                if not moves:  # A ghost without legal action stays in place.
                    T[i, j, i, j] = 1.0
                    continue

                distance = manhattanDistance((i, j), position)

                for move in moves:
                    away = manhattanDistance(move, position) >= distance
                    T[(i, j) + move] = favored if away else 1.0

                T[i, j] /= T[i, j].sum()

        return T

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        W, H = walls.width, walls.height
        O_t = np.zeros((W, H))
        noise = [
            comb(NOISE_N, z) * NOISE_P ** z * (1 - NOISE_P) ** (NOISE_N - z)
            for z in range(NOISE_N + 1)
        ]

        for i in range(W):
            for j in range(H):
                if walls[i][j]:
                    continue

                distance = manhattanDistance((i, j), position)
                z = evidence - distance + NOISE_N * NOISE_P

                if z in range(NOISE_N + 1):
                    O_t[i, j] = noise[int(z)]

        return O_t

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O_t = self.observation_matrix(walls, evidence, position)

        predicted = np.tensordot(belief, T, axes=2)
        posterior = O_t * predicted
        total = posterior.sum()

        return posterior / total if total > 0 else predicted

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

        self.walls = None
        self.distances = None

    def _distance_maps(self, walls):
        """Computes the maze distance between every pair of cells.

        Arguments:
            walls: The W x H grid of walls.

        Returns:
            The W x H x W x H matrix of maze distances, which are 0 when there
            is no path between two cells.
        """

        W, H = walls.width, walls.height
        distances = np.zeros((W, H, W, H))

        for i in range(W):
            for j in range(H):
                if walls[i][j]:
                    continue

                queue = deque([(i, j)])

                while queue:
                    cell = queue.popleft()

                    for move in _neighbors(walls, cell):
                        if move != (i, j) and not distances[(i, j) + move]:
                            distances[(i, j) + move] = (
                                distances[(i, j) + cell] + 1
                            )
                            queue.append(move)

        return distances

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        if self.walls != walls:
            self.walls = walls
            self.distances = self._distance_maps(walls)

        position = tuple(int(x) for x in position)
        moves = _neighbors(walls, position)
        targets = [b for b, gone in zip(beliefs, eaten) if not gone]

        if not moves or not targets:
            return Directions.STOP

        # Move towards the ghost expected to be the closest in the maze.
        best = min(moves, key=lambda move: min(
            (belief * self.distances[move]).sum() for belief in targets
        ))

        return Actions.vectorToDirection(
            (best[0] - position[0], best[1] - position[1])
        )

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
