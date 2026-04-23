"""Statistical parity: mean cost, paired by seed.

Byte-parity (test_parity.py) only covers `batch=1, vl=0`. Production knobs
(`batch=16, vl=1.0`) break byte-identity by design — descent order diverges.
Structural invariants (test_invariants.py) still hold, but game *quality*
could silently drift.

This test runs N games per backend on the same seeds with exploration noise
on, computes paired cost differences, and asserts the mean diff is within
tolerance of zero.

A systemic degradation — e.g. virtual-loss too high starves deep search, or
gather-loop yields too early — shows up as `mean_fast - mean_classic > 0`
(higher cost = worse) exceeding the noise floor.
"""

from __future__ import annotations
import math
import statistics
import time
import pytest

from neutral_atoms.experiment import compute_solution_cost

from ._common import (
    build_cfg, build_env_spec, make_network, play_once, seed_all,
)


def _one_game_cost(backend: str, cfg, net, tasks, ip, *, seed: int) -> int:
    g = play_once(backend, cfg, net, tasks, ip,
                  seed=seed, deterministic=False,
                  add_exploration_noise=True)
    return compute_solution_cost(g)


def _paired_run(classic_cfg, fast_cfg, net, tasks, ip, *, seeds):
    c_costs, f_costs = [], []
    t0 = time.perf_counter()
    for s in seeds:
        c_costs.append(_one_game_cost('classic', classic_cfg, net, tasks, ip, seed=s))
        f_costs.append(_one_game_cost('fast',    fast_cfg,    net, tasks, ip, seed=s))
    wall = time.perf_counter() - t0
    return c_costs, f_costs, wall


def _paired_stats(classic_costs, fast_costs) -> dict:
    assert len(classic_costs) == len(fast_costs)
    n = len(classic_costs)
    diffs = [f - c for c, f in zip(classic_costs, fast_costs)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.pstdev(diffs) if n > 1 else 0.0
    stderr = std_diff / math.sqrt(n) if n > 1 else 0.0
    return {
        'n': n,
        'classic_mean': statistics.mean(classic_costs),
        'fast_mean': statistics.mean(fast_costs),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'stderr': stderr,
        'classic_costs': classic_costs,
        'fast_costs': fast_costs,
    }


def _assert_stat_parity(stats: dict, *, abs_tol: float, z_tol: float, label: str):
    n = stats['n']
    mean_diff = stats['mean_diff']
    stderr = stats['stderr']
    # Two-sided: accept either (a) mean diff within absolute tolerance (cost units),
    # or (b) within z_tol * stderr (allows wider tolerance when variance is high).
    within_abs = abs(mean_diff) <= abs_tol
    within_z = stderr > 0 and abs(mean_diff) <= z_tol * stderr
    ok = within_abs or within_z or stderr == 0.0 and mean_diff == 0.0
    msg = (f"[{label}] n={n} "
           f"classic_mean={stats['classic_mean']:.2f} "
           f"fast_mean={stats['fast_mean']:.2f} "
           f"mean_diff={mean_diff:+.3f} "
           f"stderr={stderr:.3f} "
           f"z={mean_diff/stderr if stderr>0 else float('nan'):.2f} "
           f"abs_tol={abs_tol} z_tol={z_tol}")
    assert ok, f"stat-parity violation: {msg}"
    return msg


# --- scenarios: (map, sims, fake, batch, vl, n_games, abs_tol, z_tol) ---
#
# Real net (informative priors): strict tolerance. This is the production signal.
# Fake net (uniform priors): loose tolerance — tree search is pure UCB, so
# batch staleness hurts more than it should in production. Kept as a sanity
# check that nothing catastrophic broke, but not a correctness gate.

SCENARIOS = [
    # --- real net (strict) ---
    dict(label='real_map0_sims25_batch8_vl1',
         map_num=0, sims=25, fake=False, batch=8, vl=1.0,
         n=20, abs_tol=1.5, z_tol=2.5),
    dict(label='real_map0_sims25_batch16_vl1',
         map_num=0, sims=25, fake=False, batch=16, vl=1.0,
         n=20, abs_tol=1.5, z_tol=2.5),
    dict(label='real_map0_sims25_batch16_vl2',
         map_num=0, sims=25, fake=False, batch=16, vl=2.0,
         n=20, abs_tol=1.5, z_tol=2.5),
    # --- fake net smoke (loose: uniform prior + batch staleness amplifies noise) ---
    # Known-noisy: with FakeNet, MCTS has no policy signal to distinguish actions,
    # so virtual-loss steering is approximate. Real-net tests above are the real
    # correctness gate. This scenario is kept as a smoke test only (large abs_tol).
    dict(label='fake_map2_sims25_batch8_vl1_smoke',
         map_num=2, sims=25, fake=True, batch=8, vl=1.0,
         n=30, abs_tol=4.0, z_tol=5.0),
]


@pytest.mark.parametrize('sc', SCENARIOS, ids=[s['label'] for s in SCENARIOS])
def test_stat_parity(sc):
    classic_cfg = build_cfg(map_num=sc['map_num'], sims=sc['sims'],
                            use_fake=sc['fake'])
    fast_cfg = build_cfg(map_num=sc['map_num'], sims=sc['sims'],
                         use_fake=sc['fake'])
    with fast_cfg.mcts.unlocked():
        fast_cfg.mcts.nn_batch_size = sc['batch']
        fast_cfg.mcts.virtual_loss = sc['vl']
    tasks, ip = build_env_spec(classic_cfg)
    seed_all(1000)  # net-init seed (shared across backends)
    net = make_network(classic_cfg, seed=1000)
    seeds = list(range(2000, 2000 + sc['n']))
    c_costs, f_costs, wall = _paired_run(
        classic_cfg, fast_cfg, net, tasks, ip, seeds=seeds)
    stats = _paired_stats(c_costs, f_costs)
    msg = _assert_stat_parity(
        stats, abs_tol=sc['abs_tol'], z_tol=sc['z_tol'],
        label=f"{sc['label']} wall={wall:.1f}s")
    print(msg)


if __name__ == '__main__':
    import sys
    for sc in SCENARIOS:
        classic_cfg = build_cfg(map_num=sc['map_num'], sims=sc['sims'],
                                use_fake=sc['fake'])
        fast_cfg = build_cfg(map_num=sc['map_num'], sims=sc['sims'],
                             use_fake=sc['fake'])
        with fast_cfg.mcts.unlocked():
            fast_cfg.mcts.nn_batch_size = sc['batch']
            fast_cfg.mcts.virtual_loss = sc['vl']
        tasks, ip = build_env_spec(classic_cfg)
        seed_all(1000)
        net = make_network(classic_cfg, seed=1000)
        seeds = list(range(2000, 2000 + sc['n']))
        c_costs, f_costs, wall = _paired_run(
            classic_cfg, fast_cfg, net, tasks, ip, seeds=seeds)
        stats = _paired_stats(c_costs, f_costs)
        try:
            msg = _assert_stat_parity(
                stats, abs_tol=sc['abs_tol'], z_tol=sc['z_tol'],
                label=f"{sc['label']} wall={wall:.1f}s")
            print(f"[OK]   {msg}")
        except AssertionError as e:
            print(f"[FAIL] {e}")
            sys.exit(1)
