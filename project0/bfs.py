from pacman_module.game import Agent, Directions
from pacman_module.util import Queue


def state_key(state):
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
        self.moves = None

    def get_action(self, state):
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

    def bfs(self, state):
        """Return a shortest sequence of moves that wins from ``state``."""

        fringe = Queue()
        fringe.push((state, []))
        closed = set()

        while not fringe.isEmpty():
            current, path = fringe.pop()

            if current.isWin():
                return path

            current_key = state_key(current)
            if current_key in closed:
                continue
            closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                fringe.push((successor, path + [action]))

        return []
