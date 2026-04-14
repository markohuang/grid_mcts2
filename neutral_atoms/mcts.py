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

def play_game(game: Game, config, network: Network,
              add_exploration_noise: bool = True,
              deterministic: bool = False,
              temperature_override: float | None = None) -> Game:
    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)
        root = Node(0)
        current_observation = game.make_observation(-1)
        with torch.no_grad():
            network_output = network.inference(current_observation, aslist=True)
        _expand_node(root, game.legal_actions(), network_output, reward=0,
                 sim_env=game.environment, config=config)
        _backpropagate(
            [root], network_output.value,
            config.discount, min_max_stats,
        )
        if add_exploration_noise:
            _add_exploration_noise(config, root)

        run_mcts(config, root, game.history, network, min_max_stats, game.environment)
        action = _select_action(
            network.training_steps(), root, config, game.action_space_size,
            deterministic=deterministic, temperature_override=temperature_override,
        )
        game.cache_observation()
        game.apply(action)
        game.store_search_statistics(root)
    game.cache_observation()  # terminal state for bootstrap targets
    return game


def run_mcts(config, root, history, network, min_max_stats, env):
    total_depth = 0
    total_nonzero_reward_sims = 0
    total_reward_sum = 0.0
    total_reward_sq_sum = 0.0
    total_abs_reward_sum = 0.0
    total_boundary_sims = 0
    total_sign_changes = 0
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        sim_env = env.clone()
        sim_reward_sum = 0.0
        sim_abs_reward_sum = 0.0
        sim_boundary_reached = False
        sim_sign_changes = 0
        prev_nonzero_sign = 0

        while node.expanded():
            layer_before = sim_env.tasks_done
            action, node = _select_child(config, node, min_max_stats)
            result = sim_env.step(action, skip_obs=True)
            search_path.append(node)
            sim_reward_sum += result.reward
            sim_abs_reward_sum += abs(result.reward)
            if result.info.get('tasks_done', 0) > layer_before:
                sim_boundary_reached = True
            reward_sign = 1 if result.reward > 1e-6 else -1 if result.reward < -1e-6 else 0
            if reward_sign != 0:
                if prev_nonzero_sign != 0 and reward_sign != prev_nonzero_sign:
                    sim_sign_changes += 1
                prev_nonzero_sign = reward_sign

        total_depth += len(search_path) - 1
        if abs(sim_reward_sum) > 1e-6:
            total_nonzero_reward_sims += 1
        total_reward_sum += sim_reward_sum
        total_reward_sq_sum += sim_reward_sum ** 2
        total_abs_reward_sum += sim_abs_reward_sum
        total_sign_changes += sim_sign_changes
        if sim_boundary_reached:
            total_boundary_sims += 1

        obs_features = {
            'features': sim_env.get_features(),
            'current_qubit': sim_env.current_qubit,
        }
        with torch.no_grad():
            network_output = network.inference(obs_features, aslist=True)
        _expand_node(node, sim_env.legal_actions(), network_output, result.reward,
                     sim_env=sim_env, config=config)
        _backpropagate(
            search_path, network_output.value,
            config.discount, min_max_stats,
        )
    num_sims = config.num_simulations
    reward_mean = total_reward_sum / num_sims
    reward_var = max(total_reward_sq_sum / num_sims - reward_mean ** 2, 0.0)
    root._mcts_avg_depth = total_depth / num_sims
    root._mcts_reward_frac = total_nonzero_reward_sims / num_sims
    root._mcts_reward_sum_mean = reward_mean
    root._mcts_reward_sum_std = math.sqrt(reward_var)
    root._mcts_reward_abs_sum_mean = total_abs_reward_sum / num_sims
    root._mcts_boundary_reach_frac = total_boundary_sims / num_sims
    root._mcts_sign_changes_mean = total_sign_changes / num_sims


def _select_action(training_steps, node, config, action_space_size,
                   deterministic=False, temperature_override=None):
    visit_counts = [
        (child.visit_count, action)
        for action, child in node.children.items()
    ]
    if deterministic:
        _, action = max(visit_counts)
        return action
    t = temperature_override if temperature_override is not None else get_temperature(training_steps, config)
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


def _expand_node(node, actions, network_output, reward, sim_env=None, config=None):
    node.reward = reward
    if not actions:
        return
    logits = [network_output.policy_logits[a] for a in actions]
    # Prior mixture: add β · plan_cost_delta / cost_ub to each action's log-prior via
    # one-step lookahead on a cloned env. Pure heuristic injection — a learned policy
    # gets warm-started toward cross-layer plan-cost-descending moves, then distills
    # the mixture into π_θ through visit counts.
    if config is not None and config.prior_mix_weight > 0 and sim_env is not None:
        beta = config.prior_mix_weight
        cub = max(sim_env.cost_ub, 1)
        for i, a in enumerate(actions):
            probe = sim_env.clone()
            info = probe.step(a, skip_obs=True).info
            logits[i] = logits[i] + beta * (info['plan_cost_delta'] / cub)
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    Z = sum(exps)
    for action, e in zip(actions, exps):
        node.children[action] = Node(e / Z)


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
    if temperature == 0.0:
        best = max(visit_counts, key=lambda x: x[0])[1]
        return best
    for count, action_idx in visit_counts:
        actions_prob[action_idx] = math.log(count + 1e-8) / temperature
    actions_prob = softmax_stable(actions_prob)
    return np.random.choice(action_space_size, p=actions_prob)
