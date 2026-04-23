"""Batched MCTS (Phase 1) — lc0-style leaf gather + virtual loss.

Semantics: reuses `neutral_atoms.mcts` primitives verbatim (`Node`,
`MinMaxStats`, `_select_child`, `_expand_node`, `_backpropagate`,
`_add_exploration_noise`, `_select_action`, `get_temperature`) so leaf
semantics are identical to classic. Only the control flow differs:

  - Gather up to `nn_batch_size` leaves per iteration
  - Apply virtual loss on each pending path (single-player max: subtract vl)
  - Fire one NN call, revert vl, expand + backup each path

Parity contract:
  - `nn_batch_size=1`, `virtual_loss=0.0`, `prior_mix_weight=0.0` →
    byte-identical to classic `play_game`.

Knobs (read from `cfg.mcts` with getattr fallback so `config.py` stays untouched):
  - `nn_batch_size: int = 1`
  - `virtual_loss: float = 0.0`
"""

from __future__ import annotations
import math
import torch

from neutral_atoms.game import Game
from neutral_atoms.mcts import (
    Node, MinMaxStats,
    _select_child, _expand_node, _backpropagate, _add_exploration_noise,
    _select_action, get_temperature,
)

from .nn_backend import NNBackend


def _cfg(mcts_cfg, name, default):
    return getattr(mcts_cfg, name, default)


def _apply_virtual_loss(path, vl: float):
    if vl == 0.0:
        return
    for node in path:
        node.visit_count += 1
        node.value_sum -= vl


def _revert_virtual_loss(path, vl: float):
    if vl == 0.0:
        return
    for node in path:
        node.visit_count -= 1
        node.value_sum += vl


def _descend(root: Node, sim_env, mcts_cfg, min_max_stats):
    """Select a leaf. Returns (path, leaf_node, last_reward, per-sim stats)."""
    node = root
    path = [node]
    last_reward = 0.0
    sim_reward_sum = 0.0
    sim_abs_reward_sum = 0.0
    sim_boundary_reached = False
    sim_sign_changes = 0
    prev_nonzero_sign = 0
    while node.expanded():
        layer_before = sim_env.tasks_done
        action, child = _select_child(mcts_cfg, node, min_max_stats)
        result = sim_env.step(action, skip_obs=True)
        path.append(child)
        node = child
        last_reward = result.reward
        sim_reward_sum += result.reward
        sim_abs_reward_sum += abs(result.reward)
        if result.info.get('tasks_done', 0) > layer_before:
            sim_boundary_reached = True
        sign = 1 if result.reward > 1e-6 else -1 if result.reward < -1e-6 else 0
        if sign != 0:
            if prev_nonzero_sign != 0 and sign != prev_nonzero_sign:
                sim_sign_changes += 1
            prev_nonzero_sign = sign
    stats = {
        'depth': len(path) - 1,
        'reward_sum': sim_reward_sum,
        'abs_reward_sum': sim_abs_reward_sum,
        'boundary': sim_boundary_reached,
        'sign_changes': sim_sign_changes,
    }
    return path, node, last_reward, stats


class _LeafReq:
    __slots__ = ('path', 'leaf', 'leaf_reward', 'sim_env', 'legal',
                 'slot_id', 'is_collision', 'stats', 'terminal')

    def __init__(self, path, leaf, leaf_reward, sim_env, legal,
                 slot_id, is_collision, stats, terminal):
        self.path = path
        self.leaf = leaf
        self.leaf_reward = leaf_reward
        self.sim_env = sim_env
        self.legal = legal
        self.slot_id = slot_id
        self.is_collision = is_collision
        self.stats = stats
        self.terminal = terminal


def run_mcts_batched(mcts_cfg, root, history, network, min_max_stats, env,
                     *, nn_batch_size: int, virtual_loss: float):
    num_sims = mcts_cfg.num_simulations
    assert num_sims > 0
    backend = NNBackend(network, cap=nn_batch_size)

    # per-sim telemetry (mirror classic)
    tot_depth = 0
    tot_reward_sum = 0.0
    tot_reward_sq_sum = 0.0
    tot_abs_reward_sum = 0.0
    tot_nonzero_reward = 0
    tot_boundary = 0
    tot_terminal = 0
    tot_sign_changes = 0
    sims_done = 0

    while sims_done < num_sims:
        leaf_to_slot: dict[int, int] = {}
        pending: list[_LeafReq] = []
        # --- gather ---
        # Parity rule: count ALL leaves (terminals + NN-backed) toward batch cap.
        # If we only counted NN-backed, a run of terminal leaves would starve the
        # backup phase — visit counts wouldn't update between descents and UCB
        # would keep re-picking the same child (see bug in phase 1 dev).
        while (len(pending) < nn_batch_size
               and sims_done + len(pending) < num_sims):
            sim_env = env.clone()
            path, leaf, leaf_reward, stats = _descend(
                root, sim_env, mcts_cfg, min_max_stats)
            legal = sim_env.legal_actions()
            terminal = not legal
            _apply_virtual_loss(path, virtual_loss)
            if terminal:
                pending.append(_LeafReq(path, leaf, leaf_reward, sim_env, [],
                                        slot_id=None, is_collision=False,
                                        stats=stats, terminal=True))
                continue
            key = id(leaf)
            if key in leaf_to_slot:
                pending.append(_LeafReq(path, leaf, leaf_reward, sim_env, legal,
                                        slot_id=leaf_to_slot[key],
                                        is_collision=True,
                                        stats=stats, terminal=False))
            else:
                slot_id = backend.add(sim_env.get_features(),
                                       sim_env.current_qubit)
                leaf_to_slot[key] = slot_id
                pending.append(_LeafReq(path, leaf, leaf_reward, sim_env, legal,
                                        slot_id=slot_id, is_collision=False,
                                        stats=stats, terminal=False))
        # --- compute ---
        backend.compute_blocking()
        # --- commit in submission order ---
        for req in pending:
            if req.terminal:
                _revert_virtual_loss(req.path, virtual_loss)
                # Classic calls _expand_node on terminals too — it early-returns
                # on empty actions but not before setting node.reward = leaf_reward.
                # That reward is load-bearing in _backpropagate (`value = node.reward
                # + discount * value`), so we must replicate the side effect here.
                _expand_node(req.leaf, [], None, req.leaf_reward,
                             sim_env=req.sim_env, config=mcts_cfg)
                _backpropagate(req.path, 0.0, mcts_cfg.discount, min_max_stats)
                tot_terminal += 1
                leaf_value = 0.0
            else:
                out = backend.get(req.slot_id)
                _revert_virtual_loss(req.path, virtual_loss)
                if not req.is_collision:
                    _expand_node(req.leaf, req.legal, out, req.leaf_reward,
                                 sim_env=req.sim_env, config=mcts_cfg)
                _backpropagate(req.path, out.value, mcts_cfg.discount,
                               min_max_stats)
                leaf_value = out.value
            # telemetry accumulate
            s = req.stats
            tot_depth += s['depth']
            tot_reward_sum += s['reward_sum']
            tot_reward_sq_sum += s['reward_sum'] ** 2
            tot_abs_reward_sum += s['abs_reward_sum']
            tot_sign_changes += s['sign_changes']
            if s['boundary']:
                tot_boundary += 1
            if abs(s['reward_sum']) > 1e-6:
                tot_nonzero_reward += 1
            sims_done += 1
        backend.reset()

    reward_mean = tot_reward_sum / num_sims
    reward_var = max(tot_reward_sq_sum / num_sims - reward_mean ** 2, 0.0)
    root._mcts_avg_depth = tot_depth / num_sims
    root._mcts_reward_frac = tot_nonzero_reward / num_sims
    root._mcts_reward_sum_mean = reward_mean
    root._mcts_reward_sum_std = math.sqrt(reward_var)
    root._mcts_reward_abs_sum_mean = tot_abs_reward_sum / num_sims
    root._mcts_boundary_reach_frac = tot_boundary / num_sims
    root._mcts_reached_terminal_frac = tot_terminal / num_sims
    root._mcts_sign_changes_mean = tot_sign_changes / num_sims
    # backend-level telemetry (useful for bench, not game-persistent)
    root._mcts_nn_requests = backend.total_requests
    root._mcts_nn_batches = backend.total_batches
    root._mcts_nn_max_batch = backend.max_batch_seen


def play_game(game: Game, mcts_cfg, network,
              *, add_exploration_noise: bool = True,
              deterministic: bool = False,
              temperature_override: float | None = None) -> Game:
    nn_batch_size = int(_cfg(mcts_cfg, 'nn_batch_size', 1))
    virtual_loss = float(_cfg(mcts_cfg, 'virtual_loss', 0.0))
    assert nn_batch_size >= 1
    assert virtual_loss >= 0.0
    # Phase 1 scope limit: prior-mix lookahead not yet batched. Fail fast
    # rather than silently diverge.
    assert getattr(mcts_cfg, 'prior_mix_weight', 0.0) == 0.0, (
        "fast_mcts Phase 1 does not support prior_mix_weight > 0 yet "
        "(use backend='classic')"
    )

    while not game.terminal() and len(game.history) < mcts_cfg.max_moves:
        min_max_stats = MinMaxStats(mcts_cfg.known_bounds)
        root = Node(0)
        obs = game.make_observation(-1)
        with torch.no_grad():
            out = network.inference(obs, aslist=True)
        _expand_node(root, game.legal_actions(), out, reward=0,
                     sim_env=game.environment, config=mcts_cfg)
        _backpropagate([root], out.value, mcts_cfg.discount, min_max_stats)
        if add_exploration_noise:
            _add_exploration_noise(mcts_cfg, root)

        run_mcts_batched(mcts_cfg, root, game.history, network,
                         min_max_stats, game.environment,
                         nn_batch_size=nn_batch_size,
                         virtual_loss=virtual_loss)
        action = _select_action(
            len(game.history), root, mcts_cfg, game.action_space_size,
            deterministic=deterministic,
            temperature_override=temperature_override,
        )
        game.cache_observation()
        game.apply(action)
        game.store_search_statistics(root)
    game.cache_observation()
    return game
