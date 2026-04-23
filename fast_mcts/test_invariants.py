"""Structural MCTS + game invariants.

Backend-agnostic: anything satisfying the `play_game(game, mcts_cfg, network, ...)`
contract must produce trees and games that uphold these asserts. Phase 1 `fast`
backend will be parametrized in once it registers.

Covers:
  - per-search-tree (root.visit_count, child-visit sum, legal children)
  - per-move stats (policy target sums to 1)
  - full-game (completion, history length, action validity)
"""

from __future__ import annotations
import math
import pytest
import torch

from neutral_atoms.game import Game
from neutral_atoms import mcts as mcts_mod

from ._common import (
    seed_all, build_cfg, build_env_spec, make_network, play_once,
)


# ---- direct run_mcts tree invariants ----

def _run_single_mcts(cfg, net, tasks, ip, *, seed: int,
                     add_exploration_noise: bool = False):
    """Replicate play_game's first-move scaffolding so we can inspect the root."""
    seed_all(seed)
    game = Game(cfg, tasks, ip)
    root = mcts_mod.Node(0)
    obs = game.make_observation(-1)
    with torch.no_grad():
        out = net.inference(obs, aslist=True)
    legal = game.legal_actions()
    mcts_mod._expand_node(root, legal, out, reward=0,
                          sim_env=game.environment, config=cfg.mcts)
    mcts_mod._backpropagate(
        [root], out.value, cfg.mcts.discount,
        mcts_mod.MinMaxStats(cfg.mcts.known_bounds),
    )
    if add_exploration_noise:
        mcts_mod._add_exploration_noise(cfg.mcts, root)
    mm = mcts_mod.MinMaxStats(cfg.mcts.known_bounds)
    mcts_mod.run_mcts(cfg.mcts, root, game.history, net, mm, game.environment)
    return root, legal, game


def _assert_tree_invariants(root, legal_at_root, num_sims: int, label: str = ''):
    # Root visit count: 1 seed backup + N sim backups
    assert root.visit_count == 1 + num_sims, (
        f"[{label}] root.visit_count={root.visit_count} "
        f"expected {1 + num_sims}"
    )
    # Direct-child visits must sum to num_sims (each sim traverses exactly one)
    direct_sum = sum(c.visit_count for c in root.children.values())
    assert direct_sum == num_sims, (
        f"[{label}] sum(child.visit_count)={direct_sum} expected {num_sims}"
    )
    # Children must be a subset of legal actions at root
    child_actions = set(root.children.keys())
    assert child_actions.issubset(set(legal_at_root)), (
        f"[{label}] illegal child actions: {child_actions - set(legal_at_root)}"
    )
    # Priors in [0, 1], sum ≈ 1
    priors = [c.prior for c in root.children.values()]
    assert all(0.0 <= p <= 1.0 + 1e-6 for p in priors), \
        f"[{label}] prior out of [0,1]"
    assert math.isclose(sum(priors), 1.0, abs_tol=1e-5), \
        f"[{label}] sum(priors)={sum(priors)}"
    # value() defined since visit_count > 0
    _ = root.value()


@pytest.mark.parametrize('map_num,sims', [(0, 10), (1, 20), (2, 25)])
def test_mcts_tree_invariants_fake(map_num, sims):
    cfg = build_cfg(map_num=map_num, sims=sims, use_fake=True)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=1)
    root, legal, _ = _run_single_mcts(cfg, net, tasks, ip, seed=1)
    _assert_tree_invariants(root, legal, sims,
                            label=f'map{map_num}/fake/{sims}sims')


def test_mcts_tree_invariants_real():
    cfg = build_cfg(map_num=0, sims=20, use_fake=False)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=7)
    root, legal, _ = _run_single_mcts(cfg, net, tasks, ip, seed=7)
    _assert_tree_invariants(root, legal, 20, label='map0/real/20sims')


def test_mcts_tree_invariants_noise():
    cfg = build_cfg(map_num=2, sims=25, use_fake=True)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=3)
    root, legal, _ = _run_single_mcts(cfg, net, tasks, ip, seed=3,
                                      add_exploration_noise=True)
    _assert_tree_invariants(root, legal, 25, label='map2/noise')


def test_mcts_tree_invariants_prior_mix():
    cfg = build_cfg(map_num=2, sims=25, use_fake=True, prior_mix_weight=0.5)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=9)
    root, legal, _ = _run_single_mcts(cfg, net, tasks, ip, seed=9)
    _assert_tree_invariants(root, legal, 25, label='map2/prior-mix')


# ---- full-game invariants ----

def _assert_game_invariants(game: Game, cfg, label: str = ''):
    env = game.environment
    assert game.done, f"[{label}] game not done"
    assert env.tasks_done == env.num_tasks, \
        f"[{label}] tasks_done={env.tasks_done} num_tasks={env.num_tasks}"
    assert len(game.history) == env.episode_length, \
        f"[{label}] len(history)={len(game.history)} ep_len={env.episode_length}"
    # action validity
    action_space = cfg.network.num_actions
    for a in game.history:
        assert 0 <= a < action_space, f"[{label}] action {a} out of space"
    # policy target rows sum to ~1 (each row is normalized visits)
    for i, row in enumerate(game.child_visits):
        s = sum(row)
        assert math.isclose(s, 1.0, abs_tol=1e-5), \
            f"[{label}] child_visits[{i}] sum={s}"
    # all stored structures aligned
    assert len(game.rewards) == len(game.history)
    assert len(game.root_values) == len(game.history)
    assert len(game.child_visits) == len(game.history)


@pytest.mark.parametrize('backend', ['classic', 'fast'])
@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_full_game_invariants(backend, map_num):
    cfg = build_cfg(map_num=map_num, sims=15, use_fake=True)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=11)
    g = play_once(backend, cfg, net, tasks, ip, seed=11,
                  deterministic=True, add_exploration_noise=False)
    _assert_game_invariants(g, cfg, label=f'{backend}/map{map_num}')


def test_full_game_invariants_real_net():
    cfg = build_cfg(map_num=0, sims=10, use_fake=False)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=13)
    g = play_once('classic', cfg, net, tasks, ip, seed=13,
                  deterministic=True, add_exploration_noise=False)
    _assert_game_invariants(g, cfg, label='classic/map0/real')


# --- fast-backend non-parity paths: virtual loss + batch > 1 ---
# These break byte-parity with classic by design. Only functional invariants
# apply. We still expect a valid, completed game with legal actions throughout.

def _build_fast_cfg(**kw):
    cfg = build_cfg(**kw)
    # Mutate the mcts sub-config: add knobs the classic path ignores.
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = kw.pop('nn_batch_size', 8)
        cfg.mcts.virtual_loss = kw.pop('virtual_loss', 1.0)
    return cfg


@pytest.mark.parametrize('batch,vl', [(1, 0.0), (4, 0.0), (8, 1.0), (16, 2.0)])
def test_fast_full_game_invariants_batched(batch, vl):
    cfg = build_cfg(map_num=2, sims=50, use_fake=True)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = batch
        cfg.mcts.virtual_loss = vl
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=17)
    g = play_once('fast', cfg, net, tasks, ip, seed=17,
                  deterministic=True, add_exploration_noise=False)
    _assert_game_invariants(g, cfg, label=f'fast/map2/batch={batch}/vl={vl}')


def test_fast_tree_invariants_batched():
    # Larger batch + virtual loss: root should still accumulate num_sims visits
    # from child backups plus the 1 from seed init.
    cfg = build_cfg(map_num=2, sims=64, use_fake=True)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 16
        cfg.mcts.virtual_loss = 1.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=19)
    root, legal, _ = _run_single_mcts_via_fast(cfg, net, tasks, ip, seed=19)
    _assert_tree_invariants(root, legal, 64, label='fast/map2/batch16/vl1')


def _run_single_mcts_via_fast(cfg, net, tasks, ip, *, seed):
    import torch
    from fast_mcts.search import run_mcts_batched
    seed_all(seed)
    game = Game(cfg, tasks, ip)
    root = mcts_mod.Node(0)
    obs = game.make_observation(-1)
    with torch.no_grad():
        out = net.inference(obs, aslist=True)
    legal = game.legal_actions()
    mcts_mod._expand_node(root, legal, out, reward=0,
                          sim_env=game.environment, config=cfg.mcts)
    mcts_mod._backpropagate(
        [root], out.value, cfg.mcts.discount,
        mcts_mod.MinMaxStats(cfg.mcts.known_bounds),
    )
    mm = mcts_mod.MinMaxStats(cfg.mcts.known_bounds)
    run_mcts_batched(cfg.mcts, root, game.history, net, mm, game.environment,
                     nn_batch_size=int(getattr(cfg.mcts, 'nn_batch_size', 1)),
                     virtual_loss=float(getattr(cfg.mcts, 'virtual_loss', 0.0)))
    return root, legal, game


if __name__ == '__main__':
    for m, s in [(0, 10), (1, 20), (2, 25)]:
        cfg = build_cfg(map_num=m, sims=s, use_fake=True)
        tasks, ip = build_env_spec(cfg)
        net = make_network(cfg, seed=1)
        root, legal, _ = _run_single_mcts(cfg, net, tasks, ip, seed=1)
        _assert_tree_invariants(root, legal, s,
                                label=f'map{m}/fake/{s}sims')
        print(f"[OK] tree invariants map={m} sims={s}")
    for m in (0, 1, 2):
        cfg = build_cfg(map_num=m, sims=15, use_fake=True)
        tasks, ip = build_env_spec(cfg)
        net = make_network(cfg, seed=11)
        g = play_once('classic', cfg, net, tasks, ip, seed=11,
                      deterministic=True, add_exploration_noise=False)
        _assert_game_invariants(g, cfg, label=f'classic/map{m}')
        print(f"[OK] game invariants map={m}")
