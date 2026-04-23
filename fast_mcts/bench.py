"""Phase 0 bench harness: time classic MCTS self-play.

Attributes wall-time to categories by monkey-patching hot-path entries with
a thin timing wrapper. Same harness will drive future 'fast' backend —
dispatch by --backend name.

Usage:
  python -m fast_mcts.bench --games 3 --sims 50 --map 2
  python -m fast_mcts.bench --games 2 --sims 25 --map 2 --fake
  python -m fast_mcts.bench --games 2 --sims 50 --map 2 --backend classic

Output: per-category ms totals + per-game averages + throughput (games/s).
"""

from __future__ import annotations
import argparse
import functools
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from neutral_atoms.config import (
    get_config, set_derived_config, get_map_data, atom_map_to_positions,
)
from neutral_atoms.env import NeutralAtomsEnv
from neutral_atoms.network import Network
from neutral_atoms.game import Game
from neutral_atoms import mcts as mcts_mod

from .backends import get_backend, available


# ---- timing accumulator ----

class Timings:
    def __init__(self):
        self.t: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)

    @contextmanager
    def bucket(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] += time.perf_counter() - t0
            self.n[name] += 1

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def inner(*a, **kw):
            with self.bucket(name):
                return fn(*a, **kw)
        return inner

    def table(self, total_wall: float) -> str:
        rows = [('category', 'calls', 'ms', 'ms/call', '% wall')]
        items = sorted(self.t.items(), key=lambda kv: -kv[1])
        for k, t in items:
            n = self.n[k]
            ms = t * 1000
            pct = 100 * t / total_wall if total_wall > 0 else 0
            rows.append((k, str(n), f'{ms:.1f}', f'{ms/max(n,1):.3f}', f'{pct:.1f}'))
        rows.append(('TOTAL (wall)', '-', f'{total_wall*1000:.1f}', '-', '100.0'))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        return '\n'.join(
            '  '.join(c.ljust(w) for c, w in zip(row, widths))
            for row in rows
        )


# ---- instrumentation (monkey-patch, reverted on __exit__) ----

@contextmanager
def instrument_classic(timings: Timings):
    patches = []

    def _patch(target, attr, new):
        old = getattr(target, attr)
        setattr(target, attr, new)
        patches.append((target, attr, old))

    # Env hot paths
    _patch(NeutralAtomsEnv, 'clone',
           timings.wrap('env.clone', NeutralAtomsEnv.clone))
    _patch(NeutralAtomsEnv, 'step',
           timings.wrap('env.step', NeutralAtomsEnv.step))
    _patch(NeutralAtomsEnv, 'get_features',
           timings.wrap('env.get_features', NeutralAtomsEnv.get_features))
    _patch(NeutralAtomsEnv, 'legal_actions',
           timings.wrap('env.legal_actions', NeutralAtomsEnv.legal_actions))

    # Network inference
    _patch(Network, 'inference',
           timings.wrap('network.inference', Network.inference))

    # MCTS internals (module-level functions — wrap on the module)
    _patch(mcts_mod, '_expand_node',
           timings.wrap('mcts._expand_node', mcts_mod._expand_node))
    _patch(mcts_mod, '_backpropagate',
           timings.wrap('mcts._backpropagate', mcts_mod._backpropagate))
    _patch(mcts_mod, '_select_child',
           timings.wrap('mcts._select_child', mcts_mod._select_child))
    _patch(mcts_mod, 'run_mcts',
           timings.wrap('mcts.run_mcts', mcts_mod.run_mcts))

    try:
        yield
    finally:
        for target, attr, old in reversed(patches):
            setattr(target, attr, old)


# ---- bench loop ----

def build_config(map_num: int, sims: int, use_fake: bool,
                 *, nn_batch_size: int = 1, virtual_loss: float = 0.0) -> 'ConfigDict':
    cfg = get_config()
    cfg.map_num = map_num
    cfg.use_fake = use_fake
    cfg.mcts.num_simulations = sims
    # Disable prior-mix probing (its extra clones distort the category totals;
    # separate bench run can measure its cost if needed).
    cfg.mcts.prior_mix_weight = 0.0
    set_derived_config(cfg)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = nn_batch_size
        cfg.mcts.virtual_loss = virtual_loss
    return cfg


def run_bench(*, backend_name: str, games: int, sims: int, map_num: int,
              use_fake: bool, seed: int, deterministic: bool,
              nn_batch_size: int = 1, virtual_loss: float = 0.0) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    cfg = build_config(map_num, sims, use_fake,
                       nn_batch_size=nn_batch_size,
                       virtual_loss=virtual_loss)
    map_data = get_map_data(cfg)
    tasks = map_data['tasks']
    initial_positions = atom_map_to_positions(
        map_data['atom_map'], cfg.env.board_width)

    net = Network(cfg.network, use_fake=use_fake)
    net.eval()

    play_game = get_backend(backend_name)
    timings = Timings()
    game_times = []
    best_cost = None
    total_steps = 0

    t0 = time.perf_counter()
    with instrument_classic(timings):
        for i in range(games):
            g = Game(cfg, tasks, initial_positions)
            tg0 = time.perf_counter()
            g = play_game(g, cfg.mcts, net,
                          add_exploration_noise=not deterministic,
                          deterministic=deterministic)
            game_times.append(time.perf_counter() - tg0)
            total_steps += len(g.history)
            # basic sanity
            from neutral_atoms.experiment import compute_solution_cost
            c = compute_solution_cost(g)
            best_cost = c if best_cost is None else min(best_cost, c)
    wall = time.perf_counter() - t0

    return {
        'backend': backend_name,
        'games': games,
        'sims_per_move': sims,
        'map_num': map_num,
        'use_fake': use_fake,
        'nn_batch_size': nn_batch_size,
        'virtual_loss': virtual_loss,
        'wall_s': wall,
        'games_per_s': games / wall if wall > 0 else 0.0,
        'avg_game_s': sum(game_times) / len(game_times),
        'total_steps': total_steps,
        'best_cost': best_cost,
        'timings': timings,
    }


def print_report(r: dict) -> None:
    print(f"\n=== fast_mcts bench: backend={r['backend']} ===")
    tag = (f"  map={r['map_num']} games={r['games']} sims/move={r['sims_per_move']} "
           f"fake={r['use_fake']}")
    if r['backend'] == 'fast':
        tag += f" nn_batch={r.get('nn_batch_size', 1)} vl={r.get('virtual_loss', 0.0)}"
    print(tag)
    print(f"  wall={r['wall_s']:.2f}s  throughput={r['games_per_s']:.2f} g/s  "
          f"avg_game={r['avg_game_s']:.2f}s  total_steps={r['total_steps']}  "
          f"best_cost={r['best_cost']}")
    print()
    print(r['timings'].table(r['wall_s']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='classic', choices=available())
    p.add_argument('--games', type=int, default=2)
    p.add_argument('--sims', type=int, default=50)
    p.add_argument('--map', type=int, default=2, dest='map_num')
    p.add_argument('--fake', action='store_true', help='use FakeNet')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--deterministic', action='store_true',
                   help='argmax selection, no Dirichlet noise')
    p.add_argument('--nn-batch-size', type=int, default=1,
                   dest='nn_batch_size')
    p.add_argument('--virtual-loss', type=float, default=0.0,
                   dest='virtual_loss')
    args = p.parse_args()

    r = run_bench(
        backend_name=args.backend, games=args.games, sims=args.sims,
        map_num=args.map_num, use_fake=args.fake, seed=args.seed,
        deterministic=args.deterministic,
        nn_batch_size=args.nn_batch_size,
        virtual_loss=args.virtual_loss,
    )
    print_report(r)


if __name__ == '__main__':
    main()
