"""Profile env.step and env.clone breakdown for one game.

Usage:
    python scripts/profile_env.py --ckpt <path> [--sims 2000] [--map_num 2]

Outputs:
    - cProfile table (top 40 by cumtime) to stdout
    - Per-call counts for key functions
    - Rough fraction: _compute_remaining_cost vs clone vs other env work
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

from fast_mcts._common import build_cfg, build_env_spec, seed_all
from fast_mcts.backends import get_backend
from neutral_atoms.game import Game
from neutral_atoms.network import Network


def load_net(cfg, ckpt_path):
    net = Network(cfg.network, use_fake=False)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sims', type=int, default=2000)
    ap.add_argument('--map_num', type=int, default=2)
    ap.add_argument('--backend', default='fast')
    ap.add_argument('--nn_batch_size', type=int, default=32)
    args = ap.parse_args()

    cfg = build_cfg(map_num=args.map_num, sims=args.sims, use_fake=False)
    with cfg.mcts.unlocked():
        cfg.mcts.backend = args.backend
        cfg.mcts.nn_batch_size = args.nn_batch_size
        cfg.mcts.virtual_loss = 1.0
    with cfg.env.unlocked():
        cfg.env.reward_mode = 'plan_cost'

    tasks, ip = build_env_spec(cfg)
    net = load_net(cfg, args.ckpt)

    seed_all(0)
    game = Game(cfg, tasks, ip)
    play = get_backend(args.backend)

    print(f'Profiling: map={args.map_num}  sims={args.sims}  backend={args.backend}  ckpt={args.ckpt}')
    print('Running one game ...')
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    play(game, cfg.mcts, net, add_exploration_noise=True, deterministic=False)
    pr.disable()
    elapsed = time.perf_counter() - t0
    print(f'Wall time: {elapsed:.1f}s\n')

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    print(s.getvalue())

    # Also print tottime-sorted view for leaf costs
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    print('--- sorted by tottime (leaf cost) ---')
    print(s2.getvalue())


if __name__ == '__main__':
    main()
