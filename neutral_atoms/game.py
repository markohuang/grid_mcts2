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
        self.current_qubit_cache = []
        self.mcts_depths = []
        self.mcts_reward_fracs = []
        self.mcts_reward_sum_means = []
        self.mcts_reward_sum_stds = []
        self.mcts_reward_abs_sum_means = []
        self.mcts_boundary_reach_fracs = []
        self.mcts_reached_terminal_fracs = []
        self.mcts_sign_changes_means = []

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
        self.mcts_depths.append(getattr(root, '_mcts_avg_depth', 0))
        self.mcts_reward_fracs.append(getattr(root, '_mcts_reward_frac', 0))
        self.mcts_reward_sum_means.append(getattr(root, '_mcts_reward_sum_mean', 0))
        self.mcts_reward_sum_stds.append(getattr(root, '_mcts_reward_sum_std', 0))
        self.mcts_reward_abs_sum_means.append(getattr(root, '_mcts_reward_abs_sum_mean', 0))
        self.mcts_boundary_reach_fracs.append(getattr(root, '_mcts_boundary_reach_frac', 0))
        self.mcts_reached_terminal_fracs.append(getattr(root, '_mcts_reached_terminal_frac', 0))
        self.mcts_sign_changes_means.append(getattr(root, '_mcts_sign_changes_mean', 0))

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

    def to_dict(self):
        return {
            'tasks': self.tasks,
            'initial_positions': list(self.initial_positions),
            'history': self.history,
            'rewards': self.rewards,
            'child_visits': self.child_visits,
            'root_values': self.root_values,
            'latency_reward': self.latency_reward,
            'action_space_size': self.action_space_size,
            'discount': self.discount,
            'policy_target_temperature': self.policy_target_temperature,
            'last_info': self.last_info,
            'done': self.done,
            # MCTS diagnostics — useful for evaluating search quality
            'mcts_depths': self.mcts_depths,
            'mcts_reward_fracs': self.mcts_reward_fracs,
            'mcts_reward_sum_means': self.mcts_reward_sum_means,
            'mcts_reward_sum_stds': self.mcts_reward_sum_stds,
            'mcts_reward_abs_sum_means': self.mcts_reward_abs_sum_means,
            'mcts_boundary_reach_fracs': self.mcts_boundary_reach_fracs,
            'mcts_reached_terminal_fracs': self.mcts_reached_terminal_fracs,
            'mcts_sign_changes_means': self.mcts_sign_changes_means,
        }

    @classmethod
    def from_dict(cls, d, env_config):
        game = object.__new__(cls)
        game.tasks = d['tasks']
        game.initial_positions = [tuple(p) for p in d['initial_positions']]
        game.env_config = env_config
        game.environment = NeutralAtomsEnv(game.tasks, game.initial_positions, env_config)
        game.environment.reset()
        game.observation_cache = []
        game.current_qubit_cache = []
        for action in d['history']:
            game.cache_observation()
            game.environment.step(action)
        game.cache_observation()
        game.history = d['history']
        game.rewards = d['rewards']
        game.child_visits = d['child_visits']
        game.root_values = d['root_values']
        game.latency_reward = d['latency_reward']
        game.action_space_size = d['action_space_size']
        game.discount = d['discount']
        game.policy_target_temperature = d['policy_target_temperature']
        game.last_info = d['last_info']
        game.done = d['done']
        game.mcts_depths = d.get('mcts_depths', [])
        game.mcts_reward_fracs = d.get('mcts_reward_fracs', [])
        game.mcts_reward_sum_means = d.get('mcts_reward_sum_means', [])
        game.mcts_reward_sum_stds = d.get('mcts_reward_sum_stds', [])
        game.mcts_reward_abs_sum_means = d.get('mcts_reward_abs_sum_means', [])
        game.mcts_boundary_reach_fracs = d.get('mcts_boundary_reach_fracs', [])
        game.mcts_reached_terminal_fracs = d.get('mcts_reached_terminal_fracs', [])
        game.mcts_sign_changes_means = d.get('mcts_sign_changes_means', [])
        return game
