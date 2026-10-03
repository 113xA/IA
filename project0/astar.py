from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def state_key(state):
    """Return the hashable search identity of a Pacman state."""

    return (
        state.getPacmanPosition(),
        tuple(sorted(state.getFood().asList())),
        tuple(sorted(state.getCapsules())),
    )


def heuristic(state):
    """Return an admissible lower bound on moves needed to eat all food.

    The bound combines the Manhattan distance from Pacman to the closest
    remaining food with a Manhattan minimum spanning tree over the food.
    """

    food = state.getFood().asList()
    if not food:
        return 0

    position = state.getPacmanPosition()
    connection = min(manhattanDistance(position, dot) for dot in food)
    tree_cost = 0
    remaining = set(food)
    tree = {remaining.pop()}

    while remaining:
        distance, next_dot = min(
            (
                manhattanDistance(tree_dot, dot),
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
        self.moves = None

    def get_action(self, state):
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

    def astar(self, state):
        """Return a shortest sequence of moves that wins from ``state``."""

        fringe = PriorityQueue()
        fringe.push((state, [], 0), heuristic(state))
        best_cost = {state_key(state): 0}

        while not fringe.isEmpty():
            _, (current, path, cost) = fringe.pop()
            current_key = state_key(current)
            if cost != best_cost.get(current_key):
                continue

            if current.isWin():
                return path

            for successor, action in current.generatePacmanSuccessors():
                successor_key = state_key(successor)
                successor_cost = cost + 1
                if successor_cost >= best_cost.get(
                    successor_key, float('inf')
                ):
                    continue

                best_cost[successor_key] = successor_cost
                priority = successor_cost + heuristic(successor)
                fringe.push(
                    (successor, path + [action], successor_cost),
                    priority,
                )

        return []
