import math
from typing import NamedTuple, Sequence

from .env import NeutralAtomsEnv
from .network import make_features


class Target(NamedTuple):
    correctness_value: float
    latency_value: float
    policy: Sequence[float]
    bootstrap_discount: float


class Game:

    def __init__(self, config, tasks, initial_positions):
        self.tasks = tasks
        self.initial_positions = initial_positions
        self.env_config = config.env
        self.environment = NeutralAtomsEnv(tasks, initial_positions, config.env)
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = config.network.num_actions  # board_size
        self.discount = config.mcts.discount
        self.policy_target_temperature = config.training.policy_target_temperature
        self.done = False
        self.last_info = {}
        self.latency_reward = 0.0
        self.observation_cache = []
        self.current_qubit_cache = []  # track which qubit was active at each step

    def terminal(self):
        return self.done or not self.environment.legal_actions()

    def legal_actions(self):
        return self.environment.legal_actions()

    def apply(self, action):
        result = self.environment.step(action)
        self.rewards.append(result.reward)
        self.history.append(action)
        self.done = result.done
        self.last_info = result.info
        if self.done and result.info.get('tasks_done', 0) >= len(self.tasks):
            move_dist = result.info.get('total_move_distance', 0.0)
            # Normalize by episode length so latency target fits in value bins
            self.latency_reward = -move_dist / max(len(self.history), 1)

    def store_search_statistics(self, root):
        tau = self.policy_target_temperature
        if tau == 1.0:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append([
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in range(self.action_space_size)
            ])
        else:
            log_visits = {a: math.log(child.visit_count + 1e-8) / tau
                          for a, child in root.children.items()}
            max_log = max(log_visits.values())
            exp_visits = {a: math.exp(v - max_log) for a, v in log_visits.items()}
            exp_sum = sum(exp_visits.values())
            self.child_visits.append([
                exp_visits.get(a, 0) / exp_sum for a in range(self.action_space_size)
            ])
        self.root_values.append(root.value())

    def cache_observation(self):
        obs = self.environment._get_observation()
        self.observation_cache.append(make_features(obs, self.tasks))
        self.current_qubit_cache.append(obs.get('current_qubit', -1))

    def make_observation(self, state_index):
        if state_index == -1:
            obs = self.environment._get_observation()
            return {
                'features': make_features(obs, self.tasks),
                'current_qubit': obs.get('current_qubit', -1),
            }
        if state_index < len(self.observation_cache):
            return {
                'features': self.observation_cache[state_index],
                'current_qubit': self.current_qubit_cache[state_index] if state_index < len(self.current_qubit_cache) else -1,
            }
        env = NeutralAtomsEnv(self.tasks, self.initial_positions, self.env_config)
        env.reset()
        for action in self.history[:state_index]:
            env.step(action)
        obs = env._get_observation()
        return {
            'features': make_features(obs, self.tasks),
            'current_qubit': obs.get('current_qubit', -1),
        }

    def make_target(self, state_index, td_steps):
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
