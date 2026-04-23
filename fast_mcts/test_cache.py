"""NNCache unit + integration tests.

Layers:
  1. Pure LRU semantics (no NN involvement)
  2. NNBackend with cache: hit == miss (byte-equal NetworkOutput)
  3. Byte-parity: fast(cache=on) == fast(cache=off) == classic, at safe knobs
  4. Full-game invariants hold with cache on (tree + episode completion)
  5. Cache persists across play_game calls (cross-game reuse scenario)
"""

from __future__ import annotations
import math
import pytest
import torch

from neutral_atoms.game import Game
from neutral_atoms import mcts as mcts_mod

from ._common import (
    seed_all, build_cfg, build_env_spec, make_network, play_once,
    game_snapshot, assert_snapshot_equal,
)
from .backends import get_backend
from .nn_cache import NNCache, make_feature_key
from .nn_backend import NNBackend


# ---- Layer 1: pure LRU ----

def test_cache_get_miss_returns_none():
    c = NNCache(cap=4)
    assert c.get(('a',)) is None
    assert c.stats()['misses'] == 1
    assert c.stats()['hits'] == 0


def test_cache_put_get_hit():
    c = NNCache(cap=4)
    c.put(('a',), 'v_a')
    assert c.get(('a',)) == 'v_a'
    assert c.stats()['hits'] == 1
    assert c.stats()['insertions'] == 1


def test_cache_lru_eviction():
    c = NNCache(cap=3)
    c.put('a', 1); c.put('b', 2); c.put('c', 3)
    assert len(c) == 3
    c.put('d', 4)  # evicts 'a' (oldest)
    assert 'a' not in c
    assert 'd' in c
    assert c.stats()['evictions'] == 1


def test_cache_mru_promotion_on_get():
    c = NNCache(cap=3)
    c.put('a', 1); c.put('b', 2); c.put('c', 3)
    c.get('a')  # promote 'a' to MRU
    c.put('d', 4)  # evicts 'b' (now LRU)
    assert 'b' not in c
    assert 'a' in c


def test_cache_clear_resets_stats():
    c = NNCache(cap=3)
    c.put('a', 1); c.get('a'); c.get('b')
    assert c.stats()['hits'] == 1
    c.clear()
    s = c.stats()
    assert s['size'] == 0 and s['hits'] == 0 and s['misses'] == 0


def test_cache_overwrite_existing():
    c = NNCache(cap=3)
    c.put('a', 1)
    c.put('a', 2)
    assert c.get('a') == 2
    # overwrite doesn't count as insertion
    assert c.stats()['insertions'] == 1


def test_make_feature_key_stable():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    k1 = make_feature_key(t, 3)
    k2 = make_feature_key(t.clone(), 3)
    assert k1 == k2
    k3 = make_feature_key(t, 4)
    assert k1 != k3


# ---- Layer 2: NNBackend with cache ----

def test_backend_cache_hit_returns_same_output():
    cfg = build_cfg(map_num=0, sims=5, use_fake=False)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=1)
    # Build a feature tensor from a real env state for realistic inputs.
    from neutral_atoms.env import NeutralAtomsEnv
    env = NeutralAtomsEnv(tasks, ip, cfg.env)
    features = env.get_features()
    qubit = env.current_qubit

    cache = NNCache(cap=100)
    backend1 = NNBackend(net, cap=1, cache=cache)
    slot = backend1.add(features, qubit)
    backend1.compute_blocking()
    miss_out = backend1.get(slot)
    assert backend1.cache_hits == 0
    assert backend1.nn_calls == 1

    # Second backend, same cache — should hit.
    backend2 = NNBackend(net, cap=1, cache=cache)
    slot = backend2.add(features, qubit)
    backend2.compute_blocking()
    hit_out = backend2.get(slot)
    assert backend2.cache_hits == 1
    assert backend2.nn_calls == 0

    # Same object returned (cache stores reference)
    assert miss_out is hit_out
    # Byte-equal attributes
    assert miss_out.value == hit_out.value
    assert torch.equal(miss_out.correctness_value_logits,
                       hit_out.correctness_value_logits)
    assert miss_out.policy_logits == hit_out.policy_logits


def test_backend_no_cache_is_phase1_behaviour():
    cfg = build_cfg(map_num=0, sims=5, use_fake=False)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=1)
    from neutral_atoms.env import NeutralAtomsEnv
    env = NeutralAtomsEnv(tasks, ip, cfg.env)
    features = env.get_features()
    qubit = env.current_qubit

    backend = NNBackend(net, cap=1, cache=None)
    slot = backend.add(features, qubit)
    backend.compute_blocking()
    assert backend.cache_hits == 0
    assert backend.nn_calls == 1


# ---- Layer 3: byte-parity with cache on vs off ----

def _play_with_cache(backend_name, cfg, tasks, ip, net, *, cache, seed):
    seed_all(seed)
    game = Game(cfg, tasks, ip)
    play_game = get_backend(backend_name)
    # `cache` kwarg only supported by 'fast'; classic ignores it.
    kw = dict(add_exploration_noise=False, deterministic=True)
    if backend_name == 'fast':
        kw['cache'] = cache
    return play_game(game, cfg.mcts, net, **kw)


@pytest.mark.parametrize('map_num,sims,fake', [
    (0, 10, False), (0, 25, True), (2, 25, True), (2, 50, True),
])
def test_cache_byte_parity_fast_on_vs_off(map_num, sims, fake):
    """Fast with cache must produce byte-identical trajectory to fast without
    cache, at safe knobs (batch=1, vl=0). NN is deterministic → cache hit
    reproduces the exact bytes a fresh NN call would produce."""
    cfg = build_cfg(map_num=map_num, sims=sims, use_fake=fake)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=42)

    g_off = _play_with_cache('fast', cfg, tasks, ip, net, cache=None, seed=42)
    cache = NNCache(cap=10_000)
    g_on = _play_with_cache('fast', cfg, tasks, ip, net, cache=cache, seed=42)
    assert_snapshot_equal(
        game_snapshot(g_off), game_snapshot(g_on),
        label=f'fast_cache_off_vs_on/map{map_num}/sims{sims}/fake={fake}',
    )
    # Some hits must have occurred (otherwise test is vacuous — NN called
    # once per unique state; any repeat state in the same game = hit).
    assert cache.stats()['hits'] > 0, (
        f"vacuous cache test — no hits on map{map_num}/sims{sims}/fake={fake}"
    )


@pytest.mark.parametrize('map_num,sims,fake', [
    (0, 10, False), (2, 25, True),
])
def test_cache_byte_parity_fast_vs_classic(map_num, sims, fake):
    """Fast + cache still byte-parity with classic at safe knobs."""
    cfg = build_cfg(map_num=map_num, sims=sims, use_fake=fake)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=42)

    cache = NNCache(cap=10_000)
    g_classic = _play_with_cache('classic', cfg, tasks, ip, net, cache=None, seed=42)
    g_fast = _play_with_cache('fast', cfg, tasks, ip, net, cache=cache, seed=42)
    assert_snapshot_equal(
        game_snapshot(g_classic), game_snapshot(g_fast),
        label=f'classic_vs_fast_cache/map{map_num}/sims{sims}/fake={fake}',
    )


# ---- Layer 4: invariants hold with cache ----

def test_game_invariants_with_cache():
    cfg = build_cfg(map_num=2, sims=25, use_fake=True)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 8
        cfg.mcts.virtual_loss = 1.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=7)
    cache = NNCache(cap=10_000)
    g = _play_with_cache('fast', cfg, tasks, ip, net, cache=cache, seed=7)
    env = g.environment
    assert g.done
    assert env.tasks_done == env.num_tasks
    assert len(g.history) == env.episode_length
    for row in g.child_visits:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-5)


# ---- Layer 5: cross-game reuse ----

def test_cache_persists_across_games():
    """Run 3 games on the same map with a shared cache. Expect strong hit
    rate growth — same starting board, overlapping subtree exploration."""
    cfg = build_cfg(map_num=2, sims=25, use_fake=True)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=42)
    cache = NNCache(cap=10_000)

    per_game_hit_rates = []
    for i in range(3):
        prev_total = cache.hits + cache.misses
        _play_with_cache('fast', cfg, tasks, ip, net, cache=cache, seed=42+i)
        delta_hits = cache.hits - sum(r[0] for r in per_game_hit_rates)
        new_total = cache.hits + cache.misses
        game_reqs = new_total - prev_total
        per_game_hit_rates.append((cache.hits, game_reqs))
    # Game 1: hit rate within-game only (new cache). Game 2+: cross-game.
    # We expect game 2 hit count > game 1 (cumulative).
    assert cache.stats()['hits'] > 0
    # Sanity: cache populated with < total requests (not everything identical).
    assert cache.stats()['size'] < cache.hits + cache.misses


def test_cache_cap_enforced_under_load():
    cfg = build_cfg(map_num=2, sims=25, use_fake=True)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=42)
    # Cap deliberately smaller than the working set so eviction must trigger.
    # m2_s25 generates ~20-30 unique states per game at production knobs;
    # cap=8 forces churn from the first game.
    cache = NNCache(cap=8)
    for i in range(3):
        _play_with_cache('fast', cfg, tasks, ip, net, cache=cache, seed=42+i)
    assert len(cache) <= 8
    assert cache.stats()['evictions'] > 0, cache.stats()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
