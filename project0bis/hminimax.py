"""Bounded heuristic adversarial search agent for Project 0bis."""

from pacman_module.game import Agent, Directions


class PacmanAgent(Agent):
    """Pacman agent using depth-limited heuristic Minimax."""

    def __init__(self):
        super().__init__()
        self.depth = 7

    def get_action(self, state):
        """Return the best legal action found within the search horizon."""
        successors = state.generatePacmanSuccessors()
        if not successors:
            return Directions.STOP

        depth = self._max_depth(state)

        cache = {}
        values = []
        for successor, action in successors:
            value = self._min_value(
                successor,
                depth - 1,
                cache,
                {self._signature(state)},
            )
            values.append((value, action))

        return max(values, key=lambda item: item[0])[1]

    def _max_depth(self, state):
        """Choose a search horizon from the current board complexity."""
        food_depth = 7 if state.getNumFood() <= 5 else 6
        board_width = state.getWalls().width
        board_depth = max(5, 8 - (board_width // 12))
        return min(self.depth, food_depth, board_depth)

    def _max_value(self, state, depth, cache, path):
        if state.isWin() or state.isLose() or depth <= 0:
            return self._evaluate(state)

        key = (state, depth, 0)
        if key in cache:
            return cache[key]
        signature = self._signature(state)
        if signature in path:
            return self._evaluate(state)

        next_path = path | {signature}
        values = [
            self._min_value(successor, depth - 1, cache, next_path)
            for successor, _ in state.generatePacmanSuccessors()
        ]
        value = max(values) if values else self._evaluate(state)
        cache[key] = value
        return value

    def _min_value(self, state, depth, cache, path):
        if state.isWin() or state.isLose() or depth <= 0:
            return self._evaluate(state)

        key = (state, depth, 1)
        if key in cache:
            return cache[key]
        signature = self._signature(state)
        if signature in path:
            return self._evaluate(state)

        next_path = path | {signature}
        values = [
            self._max_value(successor, depth - 1, cache, next_path)
            for successor, _ in state.generateGhostSuccessors(1)
        ]
        value = min(values) if values else self._evaluate(state)
        cache[key] = value
        return value

    @staticmethod
    def _evaluate(state):
        """Estimate the desirability of a non-terminal state."""
        score = state.getScore()
        if state.isWin():
            return score + 100000.0
        if state.isLose():
            return score - 100000.0

        pacman = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        ghost_distance = min(
            (abs(pacman[0] - ghost[0]) + abs(pacman[1] - ghost[1])
             for ghost in ghosts),
            default=100.0,
        )
        if ghost_distance <= 1.0:
            score -= 1000.0
        else:
            score -= 30.0 / ghost_distance

        food = state.getFood()
        food_positions = [
            (x, y)
            for x in range(food.width)
            for y in range(food.height)
            if food[x][y]
        ]
        if food_positions:
            food_distance = min(
                abs(pacman[0] - x) + abs(pacman[1] - y)
                for x, y in food_positions
            )
            score -= 1.5 * food_distance
            score -= 4.0 * len(food_positions)
        return score

    @staticmethod
    def _signature(state):
        """Describe the observable position without the accumulating score."""
        pacman = state.getPacmanState()
        ghosts = tuple(
            (ghost.getPosition(), ghost.getDirection())
            for ghost in state.getGhostStates()
        )
        food = state.getFood()
        food_positions = tuple(
            (x, y)
            for x in range(food.width)
            for y in range(food.height)
            if food[x][y]
        )
        return (
            pacman.getPosition(),
            pacman.getDirection(),
            ghosts,
            food_positions,
            tuple(state.getCapsules()),
        )
