"""Batched Gumbel AlphaZero (fast_gumbel backend).

Batches NN leaf evaluations within each sequential-halving phase instead of
firing one NN call per simulation.  Uses visit-count-only virtual loss (no
value_sum adjustment) to discourage concurrent paths from colliding in the
tree — the Gumbel non-root selector uses N(a)/(1+ΣN) for discouragement, so
only incrementing visit_count is enough.

Parity contract:
    nn_batch_size=1, virtual_loss=0.0 → byte-identical to classic _gumbel_plan
    at the same seed.
"""

from __future__ import annotations
import math

import numpy as np
import torch

from neutral_atoms.game import Game
from neutral_atoms.mcts import (
    Node,
    _expand_node, _backpropagate,
    _gumbel_non_root_select, _gumbel_improved_policy, _gumbel_halving_scores,
)

from .nn_backend import NNBackend


def _apply_visit_loss(path: list) -> None:
    for node in path:
        node.visit_count += 1


def _revert_visit_loss(path: list) -> None:
    for node in path:
        node.visit_count -= 1


def _gumbel_descend(root: Node, root_action: int, sim_env,
                    c_visit: float, c_scale: float, discount: float):
    """Forced root action + deterministic non-root descent to an unexpanded leaf.

    Returns (path, leaf, leaf_reward, depth, boundary, reward_sum, reward_sq).
    leaf_reward is the reward on the edge entering the leaf node.
    """
    path = [root]
    layer_before = sim_env.tasks_done
    result = sim_env.step(root_action, skip_obs=True)
    leaf_reward = result.reward
    reward_sum = result.reward
    reward_sq = result.reward ** 2
    boundary = int(result.info.get('tasks_done', 0) > layer_before)
    node = root.children[root_action]
    path.append(node)
    depth = 1

    while node.expanded():
        layer_before = sim_env.tasks_done
        action, node = _gumbel_non_root_select(node, None, c_visit, c_scale, discount)
        result = sim_env.step(action, skip_obs=True)
        leaf_reward = result.reward
        reward_sum += result.reward
        reward_sq += result.reward ** 2
        if result.info.get('tasks_done', 0) > layer_before:
            boundary = 1
        path.append(node)
        depth += 1

    return path, node, leaf_reward, depth, boundary, reward_sum, reward_sq


class _GLeafReq:
    __slots__ = ('path', 'leaf', 'leaf_reward', 'sim_env', 'legal',
                 'slot_id', 'is_collision', 'terminal',
                 'depth', 'boundary', 'reward_sum', 'reward_sq')

    def __init__(self, path, leaf, leaf_reward, sim_env, legal,
                 slot_id, is_collision, terminal, depth, boundary, reward_sum, reward_sq):
        self.path = path
        self.leaf = leaf
        self.leaf_reward = leaf_reward
        self.sim_env = sim_env
        self.legal = legal
        self.slot_id = slot_id
        self.is_collision = is_collision
        self.terminal = terminal
        self.depth = depth
        self.boundary = boundary
        self.reward_sum = reward_sum
        self.reward_sq = reward_sq


def run_gumbel_batched(mcts_cfg, root: Node, env, network, *,
                       nn_batch_size: int, virtual_loss: float):
    """Batched Gumbel planning. Mutates root tree; returns winning action."""
    c_visit = mcts_cfg.gumbel.c_visit
    c_scale = mcts_cfg.gumbel.c_scale
    discount = mcts_cfg.discount
    total_sims = mcts_cfg.num_simulations
    use_vl = virtual_loss > 0.0

    actions = list(root.children.keys())
    assert actions, "run_gumbel_batched: no legal actions at root"

    env._shared_cost_cache = {}
    gumbel_noise = {a: float(np.random.gumbel()) for a in actions}
    priors_logit = {a: math.log(max(root.children[a].prior, 1e-10)) for a in actions}

    m = max(1, min(mcts_cfg.gumbel.num_samples_m, len(actions)))
    candidates = sorted(
        actions, key=lambda a: gumbel_noise[a] + priors_logit[a], reverse=True)[:m]
    num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1

    tot_depth = tot_boundary = tot_terminal = 0
    tot_reward_sum = tot_reward_sq = 0.0
    sims_run = 0

    backend = NNBackend(network, cap=nn_batch_size)

    for _phase in range(num_phases):
        m_phase = len(candidates)
        n_per_cand = max(1, total_sims // (num_phases * m_phase))
        flat_jobs = [cand for cand in candidates for _ in range(n_per_cand)]
        job_idx = 0

        while job_idx < len(flat_jobs):
            pending: list[_GLeafReq] = []
            leaf_to_slot: dict[int, int] = {}
            backend.reset()

            while len(pending) < nn_batch_size and job_idx < len(flat_jobs):
                cand = flat_jobs[job_idx]
                job_idx += 1
                sim_env = env.clone()
                path, leaf, leaf_reward, depth, boundary, reward_sum, reward_sq = \
                    _gumbel_descend(root, cand, sim_env, c_visit, c_scale, discount)
                legal = sim_env.legal_actions()
                terminal = not legal
                if use_vl:
                    _apply_visit_loss(path)
                if terminal:
                    pending.append(_GLeafReq(
                        path, leaf, leaf_reward, sim_env, [],
                        None, False, True, depth, boundary, reward_sum, reward_sq))
                    continue
                key = id(leaf)
                if key in leaf_to_slot:
                    pending.append(_GLeafReq(
                        path, leaf, leaf_reward, sim_env, legal,
                        leaf_to_slot[key], True, False,
                        depth, boundary, reward_sum, reward_sq))
                else:
                    slot_id = backend.add(sim_env.get_features(), sim_env.current_qubit)
                    leaf_to_slot[key] = slot_id
                    pending.append(_GLeafReq(
                        path, leaf, leaf_reward, sim_env, legal,
                        slot_id, False, False,
                        depth, boundary, reward_sum, reward_sq))

            backend.compute_blocking()

            for req in pending:
                if use_vl:
                    _revert_visit_loss(req.path)
                if req.terminal:
                    _expand_node(req.leaf, [], None, req.leaf_reward,
                                 sim_env=req.sim_env, config=mcts_cfg)
                    _backpropagate(req.path, 0.0, discount, None)
                else:
                    out = backend.get(req.slot_id)
                    if not req.is_collision:
                        _expand_node(req.leaf, req.legal, out, req.leaf_reward,
                                     sim_env=req.sim_env, config=mcts_cfg)
                    _backpropagate(req.path, out.value, discount, None)
                tot_depth += req.depth
                tot_boundary += req.boundary
                tot_terminal += int(req.terminal)
                tot_reward_sum += req.reward_sum
                tot_reward_sq += req.reward_sq
                sims_run += 1

        if m_phase <= 1:
            break
        scores = _gumbel_halving_scores(
            root, candidates, c_visit, c_scale, discount, gumbel_noise, priors_logit)
        candidates = sorted(
            candidates, key=lambda a: scores[a], reverse=True)[:max(1, m_phase // 2)]

    final_scores = _gumbel_halving_scores(
        root, candidates, c_visit, c_scale, discount, gumbel_noise, priors_logit)
    winner = max(candidates, key=lambda a: final_scores[a])

    env._shared_cost_cache = None
    root._gumbel_policy = _gumbel_improved_policy(root, c_visit, c_scale, discount)

    n = max(sims_run, 1)
    reward_mean = tot_reward_sum / n
    reward_var = max(tot_reward_sq / n - reward_mean ** 2, 0.0)
    root._mcts_avg_depth = tot_depth / n
    root._mcts_reward_frac = 0.0
    root._mcts_reward_sum_mean = reward_mean
    root._mcts_reward_sum_std = math.sqrt(reward_var)
    root._mcts_reward_abs_sum_mean = 0.0
    root._mcts_boundary_reach_frac = tot_boundary / n
    root._mcts_reached_terminal_frac = tot_terminal / n
    root._mcts_sign_changes_mean = 0.0
    root._mcts_nn_requests = backend.total_requests
    root._mcts_nn_batches = backend.total_batches
    root._mcts_nn_max_batch = backend.max_batch_seen

    return winner


def play_game(game: Game, mcts_cfg, network, *,
              add_exploration_noise: bool = True,
              deterministic: bool = False,
              temperature_override: float | None = None,
              cache=None) -> Game:
    """fast_gumbel backend: batched Gumbel AlphaZero play loop."""
    assert (getattr(mcts_cfg, 'gumbel', None) is not None
            and mcts_cfg.gumbel.enabled), \
        "fast_gumbel backend requires gumbel.enabled=True"
    nn_batch_size = int(getattr(mcts_cfg, 'nn_batch_size', 1))
    virtual_loss = float(getattr(mcts_cfg, 'virtual_loss', 0.0))
    assert nn_batch_size >= 1
    assert virtual_loss >= 0.0

    while not game.terminal() and len(game.history) < mcts_cfg.max_moves:
        root = Node(0)
        obs = game.make_observation(-1)
        with torch.no_grad():
            out = network.inference(obs, aslist=True)
        _expand_node(root, game.legal_actions(), out, reward=0,
                     sim_env=game.environment, config=mcts_cfg)
        # Seed root with its own V estimate (mirrors classic Gumbel path).
        root.visit_count = 1
        root.value_sum = root.network_value

        winner = run_gumbel_batched(
            mcts_cfg, root, game.environment, network,
            nn_batch_size=nn_batch_size,
            virtual_loss=virtual_loss,
        )
        game.cache_observation()
        game.apply(winner)
        game.store_search_statistics(root)
    game.cache_observation()
    return game
