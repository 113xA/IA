"""Exact adversarial search agent for Project 0bis."""

from pacman_module.game import Agent, Directions


class PacmanAgent(Agent):
    """Pacman agent that assumes an adversarial ghost."""

    def get_action(self, state):
        """Return the Pacman action with the best minimax value."""
        successors = state.generatePacmanSuccessors()
        if not successors:
            return Directions.STOP

        cache = {}
        values = []
        for successor, action in successors:
            value = self._min_value(
                successor, cache, {self._signature(state)}
            )
            values.append((value, action))

        return max(values, key=lambda item: item[0])[1]

    def _max_value(self, state, cache, path):
        if state.isWin() or state.isLose():
            return state.getScore()

        key = (state, 0)
        if key in cache:
            return cache[key]
        signature = self._signature(state)
        if signature in path:
            return state.getScore()

        next_path = path | {signature}
        values = [
            self._min_value(successor, cache, next_path)
            for successor, _ in state.generatePacmanSuccessors()
        ]
        value = max(values) if values else state.getScore()
        cache[key] = value
        return value

    def _min_value(self, state, cache, path):
        if state.isWin() or state.isLose():
            return state.getScore()

        key = (state, 1)
        if key in cache:
            return cache[key]
        signature = self._signature(state)
        if signature in path:
            return state.getScore()

        next_path = path | {signature}
        values = [
            self._max_value(successor, cache, next_path)
            for successor, _ in state.generateGhostSuccessors(1)
        ]
        value = min(values) if values else state.getScore()
        cache[key] = value
        return value

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
