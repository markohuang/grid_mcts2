"""Profile env.step breakdown and verify correctness vs vanilla (no-cache) formulation.

Usage:
    python scripts/profile_env.py --ckpt <path> [--sims 2000] [--map_num 2]
                                   [--n_parity 5] [--sims_parity 50]

Outputs:
    1. Parity check: n_parity games optimised vs vanilla (cache/UCB-constants disabled).
       Asserts byte-identical trajectories.
    2. cProfile tables (cumtime + tottime) for the optimised run.
    3. Cache hit-rate summary.

Baseline (M4, pre-optimisation, 2000 sims, fast backend, real net):
    wall=150s  _ucb_score calls=5.65M  _compute_remaining_cost calls=807k
    ml_collections overhead=18s (12%)  remaining_cost cumtime=39s (26%)
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fast_mcts._common import build_cfg, build_env_spec, seed_all, game_snapshot
from fast_mcts.backends import get_backend
from neutral_atoms.env import NeutralAtomsEnv
from neutral_atoms.game import Game
from neutral_atoms.network import Network


def load_net(cfg, ckpt_path, device='cpu'):
    net = Network(cfg.network, use_fake=False)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net


def _vanilla_compute_remaining_cost(self):
    """Original _compute_remaining_cost with no cache — reference implementation."""
    total = 0
    for layer_idx in range(self.tasks_done, self.num_tasks):
        total += self._compute_layer_cost_fast(layer_idx)
    return total


def run_parity_check(cfg, net, tasks, ip, backend, n_games, sims_parity):
    """Play n_games with optimised code and vanilla (no-cache) code.
    Asserts byte-identical game snapshots for every seed."""
    print(f'\n--- Parity check: optimised vs vanilla  '
          f'({n_games} games, sims={sims_parity}, backend={backend}) ---')

    parity_cfg = build_cfg(map_num=cfg.map_num, sims=sims_parity, use_fake=False)
    with parity_cfg.env.unlocked():
        parity_cfg.env.reward_mode = 'plan_cost'
    with parity_cfg.mcts.unlocked():
        # Use safe knobs so classic == fast byte-identical
        parity_cfg.mcts.nn_batch_size = 1
        parity_cfg.mcts.virtual_loss = 0.0

    play = get_backend(backend)
    orig_method = NeutralAtomsEnv._compute_remaining_cost
    failures = 0

    for seed in range(n_games):
        # --- optimised run (cache + extracted UCB constants) ---
        seed_all(seed)
        g_opt = play(Game(parity_cfg, tasks, ip), parity_cfg.mcts, net,
                     add_exploration_noise=False, deterministic=True)

        # --- vanilla run (cache disabled via monkeypatch) ---
        NeutralAtomsEnv._compute_remaining_cost = _vanilla_compute_remaining_cost
        try:
            seed_all(seed)
            g_van = play(Game(parity_cfg, tasks, ip), parity_cfg.mcts, net,
                         add_exploration_noise=False, deterministic=True)
        finally:
            NeutralAtomsEnv._compute_remaining_cost = orig_method

        snap_opt = game_snapshot(g_opt)
        snap_van = game_snapshot(g_van)
        if snap_opt == snap_van:
            print(f'  OK   seed={seed}')
        else:
            print(f'  FAIL seed={seed}')
            print(f'    optimised history[:5] = {snap_opt["history"][:5]}')
            print(f'    vanilla   history[:5] = {snap_van["history"][:5]}')
            failures += 1

    if failures == 0:
        print(f'Parity check PASSED  ({n_games}/{n_games} games byte-identical)')
    else:
        print(f'Parity check FAILED  ({failures}/{n_games} seeds differ)')
    return failures == 0


def measure_cache_hit_rate(cfg, net, tasks, ip, backend, sims):
    """Play one game, counting cache hits and misses via monkeypatch."""
    hits = [0]
    misses = [0]
    orig = NeutralAtomsEnv._compute_remaining_cost

    def counting_compute(self):
        cache = self._shared_cost_cache
        if cache is not None:
            key = (self.atom_positions.numpy().tobytes(), self.tasks_done,
                   tuple(int(x) for m in self.current_phase_moves for x in m.tolist()))
            if key in cache:
                hits[0] += 1
            else:
                misses[0] += 1
        return orig(self)

    NeutralAtomsEnv._compute_remaining_cost = counting_compute
    try:
        seed_all(0)
        play_cfg = build_cfg(map_num=cfg.map_num, sims=sims, use_fake=False)
        with play_cfg.env.unlocked():
            play_cfg.env.reward_mode = 'plan_cost'
        with play_cfg.mcts.unlocked():
            play_cfg.mcts.nn_batch_size = 32
            play_cfg.mcts.virtual_loss = 1.0
        play = get_backend(backend)
        play(Game(play_cfg, tasks, ip), play_cfg.mcts, net,
             add_exploration_noise=True, deterministic=False)
    finally:
        NeutralAtomsEnv._compute_remaining_cost = orig

    total = hits[0] + misses[0]
    rate = hits[0] / total if total > 0 else 0.0
    print(f'\n--- Cache hit rate ---')
    print(f'  hits={hits[0]}  misses={misses[0]}  total={total}  '
          f'hit_rate={rate:.1%}')
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sims', type=int, default=2000,
                    help='Sims for the profiling game')
    ap.add_argument('--map_num', type=int, default=2)
    ap.add_argument('--backend', default='fast')
    ap.add_argument('--nn_batch_size', type=int, default=32)
    ap.add_argument('--device', default='cpu',
                    help='Device for NN inference (cpu or cuda)')
    ap.add_argument('--gumbel', action='store_true',
                    help='Enable Gumbel search (uses backend=classic automatically)')
    ap.add_argument('--n_parity', type=int, default=5,
                    help='Number of games for parity check (0 to skip)')
    ap.add_argument('--sims_parity', type=int, default=50,
                    help='Sims per parity game (keep small for speed)')
    args = ap.parse_args()

    if args.gumbel:
        args.backend = 'classic'  # Gumbel runs on classic; fast_gumbel is future work

    cfg = build_cfg(map_num=args.map_num, sims=args.sims, use_fake=False)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = args.nn_batch_size
        cfg.mcts.virtual_loss = 1.0
        if args.gumbel:
            cfg.mcts.gumbel.enabled = True
    with cfg.env.unlocked():
        cfg.env.reward_mode = 'plan_cost'

    tasks, ip = build_env_spec(cfg)
    net = load_net(cfg, args.ckpt, device=args.device)

    print(f'profile_env  map={args.map_num}  sims={args.sims}  '
          f'backend={args.backend}  device={args.device}  '
          f'gumbel={args.gumbel}  ckpt={args.ckpt}')

    # 1. Parity check (skip for Gumbel — its own parity suite lives elsewhere)
    if args.n_parity > 0 and not args.gumbel:
        ok = run_parity_check(cfg, net, tasks, ip, args.backend,
                              args.n_parity, args.sims_parity)
        if not ok:
            print('\nAborting profiling — parity check failed.')
            sys.exit(1)

    # 2. Cache hit rate (skip for Gumbel — cache not active on classic path yet)
    if not args.gumbel:
        measure_cache_hit_rate(cfg, net, tasks, ip, args.backend, args.sims)

    # 3. cProfile of optimised run
    print(f'\n--- cProfile  sims={args.sims} ---')
    seed_all(0)
    game = Game(cfg, tasks, ip)
    play = get_backend(args.backend)

    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    play(game, cfg.mcts, net, add_exploration_noise=True, deterministic=False)
    pr.disable()
    elapsed = time.perf_counter() - t0
    baseline_note = '(M4 baseline: 150.0s)' if not args.gumbel else '(no baseline yet)'
    print(f'Wall time: {elapsed:.1f}s  {baseline_note}\n')

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    print('--- sorted by tottime (leaf cost) ---')
    print(s2.getvalue())


if __name__ == '__main__':
    main()
