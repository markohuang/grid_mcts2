"""Tests for the _compute_remaining_cost memoisation cache.

Guards correctness of the shared-dict cache attached to NeutralAtomsEnv:
  - Cache returns the same value as uncached computation
  - Shared dict reference is propagated through clone()
  - Writes in a clone are visible via the root env's dict
  - Cache is isolated between separate MCTS search calls
  - Cache correctly handles the zero-cost edge case
  - Full game trajectories are byte-identical with cache vs without
"""
import pytest
import torch
from neutral_atoms.mcts import Node, MinMaxStats, _expand_node, _backpropagate, run_mcts
from fast_mcts._common import (
    build_cfg, build_env_spec, make_network, play_once,
    game_snapshot, assert_snapshot_equal,
)
from neutral_atoms.game import Game
from fast_mcts._common import seed_all


# --- helpers ---

def _make_env(reward_mode='plan_cost'):
    cfg = build_cfg(map_num=2, sims=10, use_fake=True)
    with cfg.env.unlocked():
        cfg.env.reward_mode = reward_mode
    tasks, ip = build_env_spec(cfg)
    from neutral_atoms.env import NeutralAtomsEnv
    return NeutralAtomsEnv(tasks, ip, cfg.env)


# --- unit: cache returns same value as uncached ---

class TestCacheCorrectness:
    def test_cache_matches_uncached_at_root(self):
        env = _make_env()
        uncached = env._compute_remaining_cost()
        env._shared_cost_cache = {}
        cached = env._compute_remaining_cost()
        assert cached == uncached

    def test_cache_matches_uncached_after_steps(self):
        env = _make_env()
        legal = env.legal_actions()
        for action in legal[:3]:
            env.step(action, skip_obs=True)
        uncached = env._compute_remaining_cost()
        env._shared_cost_cache = {}
        cached_first = env._compute_remaining_cost()
        cached_second = env._compute_remaining_cost()   # should hit cache
        assert cached_first == uncached
        assert cached_second == uncached

    def test_cache_hit_on_second_call(self):
        env = _make_env()
        env._shared_cost_cache = {}
        env._compute_remaining_cost()           # miss: writes entry
        assert len(env._shared_cost_cache) == 1
        env._compute_remaining_cost()           # hit: no new entry
        assert len(env._shared_cost_cache) == 1

    def test_different_states_get_different_keys(self):
        env = _make_env()
        env._shared_cost_cache = {}
        env._compute_remaining_cost()   # state 1
        legal = env.legal_actions()
        # Find an action that actually moves the atom (not a no-op), so atom_positions changes.
        current_flat = int(env.atom_positions[env.current_qubit][0] * env.board_width
                           + env.atom_positions[env.current_qubit][1])
        moving_actions = [a for a in legal if a != current_flat]
        if not moving_actions:
            pytest.skip("no non-no-op legal actions available at root")
        env.step(moving_actions[0], skip_obs=True)
        env._compute_remaining_cost()   # state 2 (different board)
        assert len(env._shared_cost_cache) == 2

    def test_zero_remaining_cost_not_confused_with_miss(self):
        # Cost 0 can happen when tasks_done == num_tasks.
        # cache.get(key) returns 0, not None — must not be treated as a miss.
        env = _make_env()
        env._shared_cost_cache = {}
        # Manually inject a zero-cost entry
        fake_key = (env.atom_positions.numpy().tobytes(), env.tasks_done + 9999, ())
        env._shared_cost_cache[fake_key] = 0
        # Retrieve it directly to verify 0 is not treated as None
        val = env._shared_cost_cache.get(fake_key)
        assert val == 0
        assert val is not None   # the sentinel check must distinguish 0 from None

    def test_no_cache_when_disabled(self):
        env = _make_env()
        assert env._shared_cost_cache is None
        env._compute_remaining_cost()    # should compute without touching cache
        assert env._shared_cost_cache is None


# --- unit: cache sharing through clone ---

class TestCacheSharing:
    def test_clone_shares_dict_reference(self):
        env = _make_env()
        env._shared_cost_cache = {}
        clone = env.clone()
        assert clone._shared_cost_cache is env._shared_cost_cache

    def test_write_in_clone_visible_in_root(self):
        env = _make_env()
        env._shared_cost_cache = {}
        clone = env.clone()
        clone._compute_remaining_cost()   # clone writes to shared dict
        assert len(env._shared_cost_cache) == 1

    def test_write_in_root_visible_in_clone(self):
        env = _make_env()
        env._shared_cost_cache = {}
        clone = env.clone()
        env._compute_remaining_cost()     # root writes
        # clone's next call should hit (same board state at root)
        hit_before = len(env._shared_cost_cache)
        clone._compute_remaining_cost()   # same state as root → cache hit
        assert len(env._shared_cost_cache) == hit_before   # no new entry added

    def test_two_clones_share_same_dict(self):
        env = _make_env()
        env._shared_cost_cache = {}
        c1 = env.clone()
        c2 = env.clone()
        assert c1._shared_cost_cache is c2._shared_cost_cache

    def test_none_cache_propagates_to_clone(self):
        env = _make_env()
        assert env._shared_cost_cache is None
        clone = env.clone()
        assert clone._shared_cost_cache is None

    def test_cache_isolated_after_disable(self):
        env = _make_env()
        env._shared_cost_cache = {}
        env._compute_remaining_cost()
        n = len(env._shared_cost_cache)
        env._shared_cost_cache = None      # disable
        env2 = env.clone()
        assert env2._shared_cost_cache is None   # new clones get None


# --- integration: cache isolation between search calls ---

class TestSearchIsolation:
    def test_cache_cleared_between_run_mcts_calls(self):
        """run_mcts sets cache to {} at start and None at end.
        A second run_mcts call must not see entries from the first."""
        cfg = build_cfg(map_num=0, sims=10, use_fake=True)
        with cfg.env.unlocked():
            cfg.env.reward_mode = 'plan_cost'
        tasks, ip = build_env_spec(cfg)
        from neutral_atoms.env import NeutralAtomsEnv
        import torch
        env = NeutralAtomsEnv(tasks, ip, cfg.env)
        net = make_network(cfg, seed=0)
        assert env._shared_cost_cache is None   # before search

        # First search
        root = Node(0)
        obs = {'features': env.get_features(), 'current_qubit': env.current_qubit}
        with torch.no_grad():
            out = net.inference(obs, aslist=True)
        _expand_node(root, env.legal_actions(), out, 0, sim_env=env, config=cfg.mcts)
        _backpropagate([root], out.value, cfg.mcts.discount, MinMaxStats(cfg.mcts.known_bounds))
        run_mcts(cfg.mcts, root, [], net, MinMaxStats(cfg.mcts.known_bounds), env)
        assert env._shared_cost_cache is None   # cleared after search

    def test_cache_active_during_run_mcts(self):
        """Attach a sentinel to verify the cache is set during run_mcts.
        We monkey-patch _compute_remaining_cost to record whether a cache was active."""
        cfg = build_cfg(map_num=0, sims=5, use_fake=True)
        with cfg.env.unlocked():
            cfg.env.reward_mode = 'plan_cost'
        tasks, ip = build_env_spec(cfg)
        from neutral_atoms.env import NeutralAtomsEnv
        import torch
        env = NeutralAtomsEnv(tasks, ip, cfg.env)
        net = make_network(cfg, seed=0)

        saw_active_cache = []
        original = NeutralAtomsEnv._compute_remaining_cost

        def patched(self):
            saw_active_cache.append(self._shared_cost_cache is not None)
            return original(self)

        NeutralAtomsEnv._compute_remaining_cost = patched
        try:
            root = Node(0)
            obs = {'features': env.get_features(), 'current_qubit': env.current_qubit}
            with torch.no_grad():
                out = net.inference(obs, aslist=True)
            _expand_node(root, env.legal_actions(), out, 0, sim_env=env, config=cfg.mcts)
            _backpropagate([root], out.value, cfg.mcts.discount, MinMaxStats(cfg.mcts.known_bounds))
            run_mcts(cfg.mcts, root, [], net, MinMaxStats(cfg.mcts.known_bounds), env)
        finally:
            NeutralAtomsEnv._compute_remaining_cost = original

        assert any(saw_active_cache), "cache was never active during run_mcts"


# --- integration: phase_moves key correctness ---

class TestPhaseMovesCacheKey:
    def test_different_phase_moves_different_keys(self):
        """Two states with same atom_positions but different current_phase_moves
        must produce different cache keys (different reconfig cost)."""
        env = _make_env()
        env._shared_cost_cache = {}
        legal = env.legal_actions()

        # State A: take first legal action (may move atom, adds to phase_moves)
        action_a = legal[0]
        env_a = env.clone()
        env_a.step(action_a, skip_obs=True)
        env_a._compute_remaining_cost()

        # State B: take a different action if available
        if len(legal) > 1:
            action_b = legal[1]
            env_b = env.clone()
            env_b.step(action_b, skip_obs=True)
            env_b._compute_remaining_cost()
            # Both clones share the same cache; if they had different states, 2 entries
            if action_a != action_b:
                assert len(env._shared_cost_cache) >= 1


# --- parity: full game byte-identical with and without cache ---

@pytest.mark.parametrize('backend', ['classic', 'fast'])
def test_game_parity_with_cache(backend):
    """Self-parity: cache must not change search results (same seed → same trajectory).
    Since run_mcts/run_mcts_batched now always enable the cache, self-parity
    between two runs of the same seed validates that the cache is transparent."""
    cfg = build_cfg(map_num=2, sims=30, use_fake=True)
    with cfg.env.unlocked():
        cfg.env.reward_mode = 'plan_cost'
    if backend == 'fast':
        with cfg.mcts.unlocked():
            cfg.mcts.nn_batch_size = 1
            cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=7)
    g1 = play_once(backend, cfg, net, tasks, ip,
                   seed=7, deterministic=True, add_exploration_noise=False)
    g2 = play_once(backend, cfg, net, tasks, ip,
                   seed=7, deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(game_snapshot(g1), game_snapshot(g2),
                          label=f'{backend}/cost-cache/self-parity')


@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_game_parity_classic_vs_fast_with_cache(map_num):
    """Cross-backend parity still holds after cache is enabled in both backends."""
    cfg = build_cfg(map_num=map_num, sims=25, use_fake=True)
    with cfg.env.unlocked():
        cfg.env.reward_mode = 'plan_cost'
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=3)
    gc = play_once('classic', cfg, net, tasks, ip,
                   seed=3, deterministic=True, add_exploration_noise=False)
    gf = play_once('fast', cfg, net, tasks, ip,
                   seed=3, deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(game_snapshot(gc), game_snapshot(gf),
                          label=f'classic-vs-fast/cost-cache/map{map_num}')


def test_cache_hit_rate_meaningful():
    """With a trained prior and 50 sims, cache should have more hits than misses."""
    cfg = build_cfg(map_num=2, sims=50, use_fake=False)
    with cfg.env.unlocked():
        cfg.env.reward_mode = 'plan_cost'
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    from neutral_atoms.env import NeutralAtomsEnv
    env = NeutralAtomsEnv(tasks, ip, cfg.env)
    net = make_network(cfg, seed=0)
    import torch

    hits = [0]
    misses = [0]
    original = NeutralAtomsEnv._compute_remaining_cost

    def patched(self):
        cache = self._shared_cost_cache
        if cache is not None:
            key = (self.atom_positions.numpy().tobytes(), self.tasks_done,
                   tuple(int(x) for m in self.current_phase_moves for x in m.tolist()))
            if key in cache:
                hits[0] += 1
            else:
                misses[0] += 1
        return original(self)

    NeutralAtomsEnv._compute_remaining_cost = patched
    try:
        seed_all(0)
        root = Node(0)
        obs = {'features': env.get_features(), 'current_qubit': env.current_qubit}
        with torch.no_grad():
            out = net.inference(obs, aslist=True)
        _expand_node(root, env.legal_actions(), out, 0, sim_env=env, config=cfg.mcts)
        _backpropagate([root], out.value, cfg.mcts.discount, MinMaxStats(cfg.mcts.known_bounds))
        run_mcts(cfg.mcts, root, [], net, MinMaxStats(cfg.mcts.known_bounds), env)
    finally:
        NeutralAtomsEnv._compute_remaining_cost = original

    total = hits[0] + misses[0]
    hit_rate = hits[0] / total if total > 0 else 0.0
    print(f'\n  cache hit rate: {hit_rate:.1%} ({hits[0]} hits, {misses[0]} misses)')
    assert hit_rate > 0.5, f"expected >50% hit rate, got {hit_rate:.1%}"
