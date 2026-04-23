from __future__ import annotations

import math
import numpy as np
import torch
from typing import TYPE_CHECKING

from .network import Network, NetworkOutput, make_features

if TYPE_CHECKING:
    from .game import Game

MAXIMUM_FLOAT_VALUE = float('inf')


def get_temperature(move_in_episode, config):
    # lc0-style per-move linear decay (see docs/selfplay/lc0_temperature_design.md).
    # Every episode runs the same envelope from move 0, independent of training state.
    # Setting temperature_decay_moves=0 degenerates to a constant T = temperature_init.
    t_init = config.temperature_init
    t_final = config.temperature_final
    decay_moves = config.temperature_decay_moves
    if decay_moves <= 0 or move_in_episode >= decay_moves:
        return t_final if decay_moves > 0 else t_init
    alpha = move_in_episode / decay_moves
    return t_init + alpha * (t_final - t_init)


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        self.network_value = 0.0  # V(s) from network at expansion; fallback for unvisited children in Gumbel completedQ

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
              *,
              add_exploration_noise: bool = True,
              deterministic: bool = False,
              temperature_override: float | None = None) -> Game:
    use_gumbel = getattr(config, 'gumbel', None) is not None and config.gumbel.enabled
    while not game.terminal() and len(game.history) < config.max_moves:
        root = Node(0)
        current_observation = game.make_observation(-1)
        with torch.no_grad():
            network_output = network.inference(current_observation, aslist=True)
        _expand_node(root, game.legal_actions(), network_output, reward=0,
                 sim_env=game.environment, config=config)
        if use_gumbel:
            action = _gumbel_plan(config, root, game.environment, network)
        else:
            min_max_stats = MinMaxStats(config.known_bounds)
            _backpropagate(
                [root], network_output.value,
                config.discount, min_max_stats,
            )
            if add_exploration_noise:
                _add_exploration_noise(config, root)
            run_mcts(config, root, game.history, network, min_max_stats, game.environment)
            action = _select_action(
                len(game.history), root, config, game.action_space_size,
                deterministic=deterministic, temperature_override=temperature_override,
            )
        game.cache_observation()
        game.apply(action)
        game.store_search_statistics(root)
    game.cache_observation()  # terminal state for bootstrap targets
    return game


def run_mcts(config, root, history, network, min_max_stats, env):
    assert config.num_simulations > 0, "num_simulations must be > 0"
    total_depth = 0
    total_nonzero_reward_sims = 0
    total_reward_sum = 0.0
    total_reward_sq_sum = 0.0
    total_abs_reward_sum = 0.0
    total_boundary_sims = 0
    total_reached_terminal_sims = 0
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

        legal = sim_env.legal_actions()
        if legal:
            obs_features = {
                'features': sim_env.get_features(),
                'current_qubit': sim_env.current_qubit,
            }
            with torch.no_grad():
                network_output = network.inference(obs_features, aslist=True)
            leaf_value = network_output.value
        else:
            # Terminal leaf: bootstrap with 0 instead of feeding an OOD terminal obs
            # to the network. _expand_node early-returns on empty actions, so the
            # node stays childless; subsequent sims reaching here re-bootstrap with 0.
            network_output = None
            leaf_value = 0.0
            total_reached_terminal_sims += 1
        _expand_node(node, legal, network_output, result.reward,
                     sim_env=sim_env, config=config)
        _backpropagate(
            search_path, leaf_value,
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
    root._mcts_reached_terminal_frac = total_reached_terminal_sims / num_sims
    root._mcts_sign_changes_mean = total_sign_changes / num_sims


def _select_action(move_in_episode, node, config, action_space_size,
                   deterministic=False, temperature_override=None):
    assert node.children, "_select_action called on node with no children"
    visit_counts = [
        (child.visit_count, action)
        for action, child in node.children.items()
    ]
    if deterministic:
        _, action = max(visit_counts)
        return action
    t = temperature_override if temperature_override is not None else get_temperature(move_in_episode, config)
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
    if network_output is not None:
        node.network_value = network_output.value
    if not actions:
        return
    logits = [network_output.policy_logits[a] for a in actions]
    # Prior mixture: add β · plan_cost_delta / cost_ub to each action's log-prior via
    # one-step lookahead on a cloned env. Pure heuristic injection — a learned policy
    # gets warm-started toward cross-layer plan-cost-descending moves, then distills
    # the mixture into π_θ through visit counts.
    if config is not None and config.prior_mix_weight > 0 and sim_env is not None:
        assert sim_env.track_plan_delta or sim_env.reward_mode in ('plan_cost', 'plan_cost_unbiased'), \
            "prior_mix_weight > 0 requires env.track_plan_delta=True or a plan_cost reward_mode; " \
            "otherwise plan_cost_delta is silently 0 and the mixture is a no-op"
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
        if min_max_stats is not None:
            min_max_stats.update(node.value())
        value = node.reward + discount * value


def _add_exploration_noise(config, node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# ---- Gumbel AlphaZero (docs/gumbel_pczero_plan.md) ----

def _gumbel_plan(config, root, env, network):
    """Gumbel AlphaZero planning at root. Returns the action to play.

    Root uses Gumbel-Top-m + sequential halving (replaces Dirichlet + pUCT + softmax-visit sample).
    Non-root uses deterministic argmax of [π'(a) - N(a)/(1+ΣN)] (replaces pUCT).
    Policy target attached to root._gumbel_policy (guaranteed-improvement π')."""
    # Seed root with its own V estimate so root.visit_count=1 / value_sum=V (same as pUCT path).
    root.visit_count = 1
    root.value_sum = root.network_value
    actions = list(root.children.keys())
    assert actions, "_gumbel_plan called with no legal actions"

    gumbel_noise = {a: float(np.random.gumbel()) for a in actions}
    priors_logit = {a: math.log(max(root.children[a].prior, 1e-10)) for a in actions}

    # Initial Gumbel-Top-m: pick m candidates by g(a) + logit(a) (sampling-without-replacement)
    m = max(1, min(config.gumbel.num_samples_m, len(actions)))
    candidates = sorted(actions, key=lambda a: gumbel_noise[a] + priors_logit[a], reverse=True)[:m]

    total_sims = config.num_simulations
    num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1

    # Stats buckets (root._mcts_* — check_wave_health reads these)
    total_depth = 0
    sims_run = 0
    boundary_sims = 0
    terminal_sims = 0
    reward_sum_accum = 0.0
    reward_sq_accum = 0.0

    for _ in range(num_phases):
        m_phase = len(candidates)
        n_per_cand = max(1, total_sims // (num_phases * m_phase))
        for cand in candidates:
            for _ in range(n_per_cand):
                d, b, t, rs, rsq = _gumbel_simulate(config, root, cand, env, network)
                total_depth += d
                boundary_sims += b
                terminal_sims += t
                reward_sum_accum += rs
                reward_sq_accum += rsq
                sims_run += 1
        if m_phase <= 1:
            break
        scores = _gumbel_halving_scores(root, candidates, config, gumbel_noise, priors_logit)
        candidates = sorted(candidates, key=lambda a: scores[a], reverse=True)[:max(1, m_phase // 2)]

    # Final winner: highest halving score over survivors
    final_scores = _gumbel_halving_scores(root, candidates, config, gumbel_noise, priors_logit)
    winner = max(candidates, key=lambda a: final_scores[a])

    # Improved-policy target over ALL root legal actions (not just survivors)
    root._gumbel_policy = _gumbel_improved_policy(root, config)

    # Populate stats (parity with run_mcts; reward_frac / sign_changes not tracked in gumbel yet)
    n = max(sims_run, 1)
    reward_mean = reward_sum_accum / n
    reward_var = max(reward_sq_accum / n - reward_mean ** 2, 0.0)
    root._mcts_avg_depth = total_depth / n
    root._mcts_reward_frac = 0.0
    root._mcts_reward_sum_mean = reward_mean
    root._mcts_reward_sum_std = math.sqrt(reward_var)
    root._mcts_reward_abs_sum_mean = 0.0
    root._mcts_boundary_reach_frac = boundary_sims / n
    root._mcts_reached_terminal_frac = terminal_sims / n
    root._mcts_sign_changes_mean = 0.0
    return winner


def _gumbel_simulate(config, root, root_action, env, network):
    """One simulation: forced first action at root, then deterministic non-root traversal to leaf."""
    sim_env = env.clone()
    search_path = [root]
    depth = 0
    boundary = 0
    reward_sum = 0.0
    reward_sq = 0.0

    layer_before = sim_env.tasks_done
    result = sim_env.step(root_action, skip_obs=True)
    reward_sum += result.reward
    reward_sq += result.reward ** 2
    if result.info.get('tasks_done', 0) > layer_before:
        boundary = 1
    node = root.children[root_action]
    search_path.append(node)
    depth += 1

    while node.expanded():
        layer_before = sim_env.tasks_done
        action, node = _gumbel_non_root_select(node, config)
        result = sim_env.step(action, skip_obs=True)
        reward_sum += result.reward
        reward_sq += result.reward ** 2
        if result.info.get('tasks_done', 0) > layer_before:
            boundary = 1
        search_path.append(node)
        depth += 1

    legal = sim_env.legal_actions()
    terminal = 0
    if legal:
        obs_features = {
            'features': sim_env.get_features(),
            'current_qubit': sim_env.current_qubit,
        }
        with torch.no_grad():
            network_output = network.inference(obs_features, aslist=True)
        leaf_value = network_output.value
    else:
        network_output = None
        leaf_value = 0.0
        terminal = 1
    _expand_node(node, legal, network_output, result.reward, sim_env=sim_env, config=config)
    _backpropagate(search_path, leaf_value, config.discount, None)
    return depth, boundary, terminal, reward_sum, reward_sq


def _gumbel_non_root_select(node, config):
    """Deterministic non-root: argmax_a [π'(a) - N(a)/(1+ΣN)]."""
    total_N = sum(c.visit_count for c in node.children.values())
    pi_prime = _gumbel_improved_policy(node, config)
    best_a, best_child, best_score = None, None, -float('inf')
    for a, c in node.children.items():
        s = pi_prime[a] - c.visit_count / (1 + total_N)
        if s > best_score:
            best_score, best_a, best_child = s, a, c
    return best_a, best_child


def _gumbel_completed_q(node, config):
    """completedQ(a) = r(a) + γ·V(child) if visited else node.network_value (parent V fallback)."""
    return {a: (c.reward + config.discount * c.value()) if c.visit_count > 0 else node.network_value
            for a, c in node.children.items()}


def _gumbel_improved_policy(node, config):
    """π'(a) ∝ softmax(logit(a) + σ(completedQ(a))) over all children at this node."""
    if not node.children:
        return {}
    q_vals = _gumbel_completed_q(node, config)
    actions = list(node.children.keys())
    logits = np.array([math.log(max(node.children[a].prior, 1e-10)) for a in actions])
    q_arr = np.array([q_vals[a] for a in actions])
    q_min, q_max = q_arr.min(), q_arr.max()
    q_range = q_max - q_min
    q_norm = (q_arr - q_min) / q_range if q_range > 1e-8 else np.zeros_like(q_arr)
    sigma = config.gumbel.c_visit * config.gumbel.c_scale * q_norm
    scores = logits + sigma
    scores -= scores.max()
    exps = np.exp(scores)
    pi_prime = exps / exps.sum()
    return {a: float(pi_prime[i]) for i, a in enumerate(actions)}


def _gumbel_halving_scores(root, candidates, config, gumbel_noise, priors_logit):
    """Score for sequential halving: g(a) + logit(a) + σ(completedQ(a)). Gumbel noise persists."""
    q_vals = _gumbel_completed_q(root, config)
    q_subset = np.array([q_vals[a] for a in candidates])
    q_min, q_max = q_subset.min(), q_subset.max()
    q_range = q_max - q_min
    max_N = max((root.children[a].visit_count for a in candidates), default=0)
    out = {}
    for a in candidates:
        q_n = (q_vals[a] - q_min) / q_range if q_range > 1e-8 else 0.0
        sig = (config.gumbel.c_visit + max_N) * config.gumbel.c_scale * q_n
        out[a] = gumbel_noise[a] + priors_logit[a] + sig
    return out


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
