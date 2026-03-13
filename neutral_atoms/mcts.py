from __future__ import annotations

import math
import numpy as np
import torch
from typing import TYPE_CHECKING

from .network import Network, NetworkOutput, make_features

if TYPE_CHECKING:
    from .game import Game

MAXIMUM_FLOAT_VALUE = float('inf')


def get_temperature(steps, config):
    t_init = config.temperature_init
    t_final = config.temperature_final
    decay_steps = config.temperature_decay_steps
    if steps >= decay_steps:
        return t_final
    alpha = steps / decay_steps
    return t_init + alpha * (t_final - t_init)


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.reward = 0

    def expanded(self):
        return bool(self.children)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MinMaxStats:
    def __init__(self, known_bounds):
        self.maximum = known_bounds.max
        self.minimum = known_bounds.min

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


# ---- MCTS Algorithm ----

def play_game(game: Game, config, network: Network) -> Game:
    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)
        root = Node(0)
        current_observation = game.make_observation(-1)
        with torch.no_grad():
            network_output = network.inference(current_observation, aslist=True)
        _expand_node(root, game.legal_actions(), network_output, reward=0)
        _backpropagate(
            [root], network_output.value,
            config.discount, min_max_stats,
        )
        _add_exploration_noise(config, root)

        run_mcts(config, root, game.history, network, min_max_stats, game.environment)
        action = _select_action(network.training_steps(), root, config, game.action_space_size)
        game.cache_observation()
        game.apply(action)
        game.store_search_statistics(root)
    game.cache_observation()  # terminal state for bootstrap targets
    return game


def run_mcts(config, root, history, network, min_max_stats, env):
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        sim_env = env.clone()

        while node.expanded():
            action, node = _select_child(config, node, min_max_stats)
            result = sim_env.step(action)
            search_path.append(node)

        obs = sim_env._get_observation()
        obs_features = {
            'features': make_features(obs, sim_env.tasks),
            'current_qubit': obs.get('current_qubit', -1),
        }
        with torch.no_grad():
            network_output = network.inference(obs_features, aslist=True)
        _expand_node(node, sim_env.legal_actions(), network_output, result.reward)
        _backpropagate(
            search_path, network_output.value,
            config.discount, min_max_stats,
        )


def _select_action(training_steps, node, config, action_space_size):
    visit_counts = [
        (child.visit_count, action)
        for action, child in node.children.items()
    ]
    t = get_temperature(training_steps, config)
    return _softmax_sample(visit_counts, t, action_space_size)


def _select_child(config, node, min_max_stats):
    _, action, child = max(
        (_ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
    return action, child


def _ucb_score(config, parent, child, min_max_stats):
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount * child.value()
        )
    else:
        value_score = 0
    return prior_score + value_score


def _expand_node(node, actions, network_output, reward):
    node.reward = reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def _backpropagate(search_path, value, discount, min_max_stats):
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value


def _add_exploration_noise(config, node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def _softmax_sample(visit_counts, temperature, action_space_size):
    def softmax_stable(x):
        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
    actions_prob = np.full(action_space_size, -np.inf)
    for count, action_idx in visit_counts:
        actions_prob[action_idx] = count
    actions_prob = softmax_stable(actions_prob / temperature)
    return np.random.choice(action_space_size, p=actions_prob)
