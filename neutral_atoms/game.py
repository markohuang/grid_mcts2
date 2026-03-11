from typing import NamedTuple, Sequence

from .env import NeutralAtomsEnv
from .types import EnvConfig, Tasks
from .mcts import Node, ActionHistory
from .network import make_features


class Target(NamedTuple):
    correctness_value: float
    latency_value: float
    policy: Sequence[float]
    bootstrap_discount: float


class Game:
    """A single episode of interaction with the environment."""

    def __init__(
        self,
        tasks: Tasks,
        initial_positions: list[tuple[int, int]],
        env_config: EnvConfig,
        action_space_size: int,
        discount: float,
    ):
        self.tasks = tasks
        self.initial_positions = initial_positions
        self.env_config = env_config
        self.environment = NeutralAtomsEnv(tasks, initial_positions, env_config)
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = False
        self.last_info = {}
        self.latency_reward = 0.0

    def terminal(self) -> bool:
        return self.done or not self.environment.legal_actions()

    def legal_actions(self) -> list[int]:
        return self.environment.legal_actions()

    def apply(self, action: int):
        result = self.environment.step(action)
        self.rewards.append(result.reward)
        self.history.append(action)
        self.done = result.done
        self.last_info = result.info
        if self.done and result.info.get('tasks_done', 0) >= len(self.tasks):
            self.latency_reward = -result.info.get('cost', 0.0)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.action_space_size)
        ])
        self.root_values.append(root.value())

    def make_observation(self, state_index: int) -> dict:
        """Reconstruct observation at a given state index.

        Returns dict with 'features' key ready for network inference.
        state_index == -1 means current state.
        """
        if state_index == -1:
            obs = self.environment._get_observation()
        else:
            env = NeutralAtomsEnv(self.tasks, self.initial_positions, self.env_config)
            obs = env.reset()
            for action in self.history[:state_index]:
                obs = env.step(action).observation
        return {'features': make_features(obs, self.tasks)}

    def make_target(self, state_index: int, td_steps: int, to_play: int) -> Target:
        """Creates the value target for training."""
        bootstrap_index = state_index + td_steps
        value = 0.0
        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount ** i

        if bootstrap_index < len(self.root_values):
            bootstrap_discount = self.discount ** td_steps
        else:
            bootstrap_discount = 0.0

        return Target(
            value,
            self.latency_reward,
            self.child_visits[state_index],
            bootstrap_discount,
        )

    def to_play(self) -> int:
        return -1

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)
