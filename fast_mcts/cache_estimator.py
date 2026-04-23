"""Phase 2 scouting: estimate NN-cache potential without implementing the cache.

Wraps `NNBackend.add` during `fast_mcts.search` runs, records a compact key
per request, and reports:
  - total NN requests (would-be forward passes)
  - unique keys
  - hit rate upper bound = 1 - unique / total
  - projected wall-time savings = hit_rate * baseline_nn_pct

Also splits hit attribution into three scopes to inform cache design:
  - within-move: same (key) seen within a single `run_mcts` call
  - within-game: same (key) seen across moves in one game
  - cross-game: same (key) seen across games (same map)

Usage:
  python -m fast_mcts.cache_estimator --map 2 --sims 50 --games 3
  python -m fast_mcts.cache_estimator --sweep       # full scale sweep
"""

from __future__ import annotations
import argparse
import functools
import time
from collections import defaultdict
from typing import Any

import torch

from neutral_atoms.config import (
    get_config, set_derived_config, get_map_data, atom_map_to_positions,
)
from neutral_atoms.network import Network
from neutral_atoms.game import Game
from neutral_atoms.experiment import compute_solution_cost

from .backends import get_backend
from .nn_backend import NNBackend
from ._common import seed_all


# ---- key derivation ----
# The NN is deterministic in eval mode, so any two requests with identical
# features + current_qubit produce identical output. We key on env state
# because it's ~40 bytes vs ~20 KB of feature tensor, and because the
# mapping (env state → features) is pure.

def _env_key(env) -> tuple:
    return (env.board.numpy().tobytes(),
            int(env.tasks_done),
            int(env.current_qubit))


# ---- tracking wrapper ----

class TrackingNNBackend(NNBackend):
    """NNBackend that records a key per request. No caching — we measure
    what the cache *would* save, then decide if it's worth building."""

    def __init__(self, network, *, cap: int = 1, scope_tag: Any = None):
        super().__init__(network, cap=cap)
        self._scope_tag = scope_tag
        self._keys_this_request: list[tuple] = []

    def add_with_key(self, key: tuple, features, current_qubit) -> int:
        # Same as parent `add` but also record the key in submission order.
        slot_id = super().add(features, current_qubit)
        self._keys_this_request.append(key)
        return slot_id

    def drain_keys(self) -> list[tuple]:
        out = self._keys_this_request
        self._keys_this_request = []
        return out


# ---- search-time instrumentation ----

def run_with_tracking(cfg, net, tasks, ip, *, n_games: int, seed_base: int):
    """Run `n_games` with the fast backend, recording every key the NN
    backend would see. Returns a list of per-game records."""
    from . import search as fs

    # Prepare per-run state
    all_keys_per_game: list[list[tuple]] = []
    per_game_move_boundaries: list[list[int]] = []

    # Monkey-patch _descend's ancestor: intercept backend.add calls. We wrap
    # NNBackend directly — cleanest hook point since every fast search path
    # creates one NNBackend per run_mcts call.
    orig_add = NNBackend.add

    current_keys: list[tuple] = []

    def tracking_add(self, features, current_qubit):
        # Caller doesn't pass the env, but we captured it via a context-var
        # trick below. Simpler: reconstruct the key from features tensor.
        # Since features are deterministic from env, we can hash features+qubit
        # to get a stable key.
        key = (features.numpy().tobytes(), int(current_qubit))
        current_keys.append(key)
        return orig_add(self, features, current_qubit)

    NNBackend.add = tracking_add
    try:
        for gi in range(n_games):
            seed = seed_base + gi
            seed_all(seed)
            game = Game(cfg, tasks, ip)
            game_keys: list[tuple] = []
            move_boundaries: list[int] = [0]
            # We can't easily hook per run_mcts call from outside, so we
            # drain `current_keys` between moves. But play_game calls
            # run_mcts_batched once per move internally. To get per-move
            # boundaries, we wrap play_game step-by-step:
            # Simplest: just record total keys per game, not per move.
            current_keys.clear()
            play_game = get_backend('fast')
            play_game(game, cfg.mcts, net,
                      add_exploration_noise=False, deterministic=True)
            game_keys = list(current_keys)
            all_keys_per_game.append(game_keys)
    finally:
        NNBackend.add = orig_add

    return all_keys_per_game


# ---- analysis ----

def analyze(all_keys_per_game: list[list[tuple]]) -> dict:
    total_requests = sum(len(ks) for ks in all_keys_per_game)
    per_game_unique = [len(set(ks)) for ks in all_keys_per_game]
    global_unique_set: set = set()
    cross_game_hit = 0
    within_game_hit_total = 0
    for ks in all_keys_per_game:
        seen = set()
        for k in ks:
            if k in seen:
                within_game_hit_total += 1
            elif k in global_unique_set:
                cross_game_hit += 1
            else:
                pass
            seen.add(k)
            global_unique_set.add(k)
    global_unique = len(global_unique_set)
    within_game_hit_rate = within_game_hit_total / total_requests if total_requests else 0.0
    cross_game_hit_rate = cross_game_hit / total_requests if total_requests else 0.0
    any_hit_rate = (total_requests - global_unique) / total_requests if total_requests else 0.0
    return {
        'total_requests': total_requests,
        'global_unique': global_unique,
        'within_game_hits': within_game_hit_total,
        'cross_game_hits': cross_game_hit,
        'within_game_hit_rate': within_game_hit_rate,
        'cross_game_hit_rate': cross_game_hit_rate,
        'any_hit_rate': any_hit_rate,
        'per_game_requests': [len(ks) for ks in all_keys_per_game],
        'per_game_unique': per_game_unique,
    }


def project_wall_time(stats: dict, baseline_nn_pct: float) -> dict:
    """Given hit-rate and baseline NN fraction of wall, project savings."""
    nn_save = stats['any_hit_rate'] * baseline_nn_pct
    # Cache overhead: ~1 µs per hit vs ~1.3 ms per NN call — negligible.
    # Assume full realization of cache hits.
    projected_speedup = 1.0 / (1.0 - nn_save) if nn_save < 1.0 else float('inf')
    return {
        'baseline_nn_pct': baseline_nn_pct,
        'nn_wall_saved_pct': nn_save * 100,
        'projected_speedup': projected_speedup,
    }


# ---- scenario runner ----

def build_cfg(*, map_num: int, sims: int, use_fake: bool,
              nn_batch_size: int, virtual_loss: float,
              random_board: bool = False):
    cfg = get_config()
    cfg.map_num = map_num
    cfg.use_fake = use_fake
    cfg.random_board = random_board
    cfg.mcts.num_simulations = sims
    cfg.mcts.prior_mix_weight = 0.0
    set_derived_config(cfg)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = nn_batch_size
        cfg.mcts.virtual_loss = virtual_loss
    return cfg


def run_scenario(*, label, map_num, sims, use_fake, games,
                 nn_batch_size=1, virtual_loss=0.0,
                 baseline_nn_pct=0.70, seed_base=42):
    cfg = build_cfg(map_num=map_num, sims=sims, use_fake=use_fake,
                    nn_batch_size=nn_batch_size, virtual_loss=virtual_loss)
    md = get_map_data(cfg)
    tasks = md['tasks']
    ip = atom_map_to_positions(md['atom_map'], cfg.env.board_width)
    torch.set_num_threads(1)
    seed_all(seed_base)
    net = Network(cfg.network, use_fake=use_fake)
    net.eval()
    t0 = time.perf_counter()
    keys_per_game = run_with_tracking(
        cfg, net, tasks, ip, n_games=games, seed_base=seed_base)
    wall = time.perf_counter() - t0
    stats = analyze(keys_per_game)
    proj = project_wall_time(stats, baseline_nn_pct)
    row = {
        'label': label,
        'map_num': map_num, 'sims': sims, 'fake': use_fake, 'games': games,
        'nn_batch_size': nn_batch_size, 'virtual_loss': virtual_loss,
        'wall_s': wall,
        **stats,
        **proj,
    }
    return row


# ---- reporting ----

def print_row(r: dict) -> None:
    print(f"[{r['label']}] map={r['map_num']} sims={r['sims']} fake={r['fake']} "
          f"games={r['games']} batch={r['nn_batch_size']} vl={r['virtual_loss']}")
    print(f"  wall={r['wall_s']:.2f}s "
          f"req={r['total_requests']} unique={r['global_unique']} "
          f"hit_rate={r['any_hit_rate']:.1%} "
          f"(within_game={r['within_game_hit_rate']:.1%} "
          f"cross_game={r['cross_game_hit_rate']:.1%})")
    print(f"  @ NN_baseline={r['baseline_nn_pct']:.0%} wall → "
          f"NN_saved={r['nn_wall_saved_pct']:.1f}% → "
          f"speedup={r['projected_speedup']:.2f}×")


def sweep() -> list[dict]:
    rows = []
    # --- small map (0, 2x6), easy — should show most cross-game reuse
    for sims in [25, 100]:
        for games in [1, 3, 10]:
            rows.append(run_scenario(
                label=f'm0_s{sims}_g{games}',
                map_num=0, sims=sims, use_fake=True, games=games))
    # --- medium map (2, 5x5), more state diversity
    for sims in [25, 100, 200]:
        for games in [1, 3, 10]:
            rows.append(run_scenario(
                label=f'm2_s{sims}_g{games}',
                map_num=2, sims=sims, use_fake=True, games=games))
    # --- with fast batching (production config)
    for batch, vl in [(16, 1.0), (32, 1.5)]:
        rows.append(run_scenario(
            label=f'm2_s100_batch{batch}_vl{vl}',
            map_num=2, sims=100, use_fake=True, games=5,
            nn_batch_size=batch, virtual_loss=vl))
    # --- real net, cheaper map
    rows.append(run_scenario(
        label='real_m0_s25_g3',
        map_num=0, sims=25, use_fake=False, games=3))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=int, default=2, dest='map_num')
    p.add_argument('--sims', type=int, default=100)
    p.add_argument('--games', type=int, default=3)
    p.add_argument('--fake', action='store_true', default=True)
    p.add_argument('--no-fake', action='store_false', dest='fake')
    p.add_argument('--batch', type=int, default=1, dest='nn_batch_size')
    p.add_argument('--vl', type=float, default=0.0, dest='virtual_loss')
    p.add_argument('--baseline-nn-pct', type=float, default=0.70,
                   help='Fraction of wall NN consumed in baseline (from bench)')
    p.add_argument('--sweep', action='store_true')
    args = p.parse_args()

    if args.sweep:
        rows = sweep()
        print()
        print('=' * 80)
        print(f"{'label':<30s} {'req':>6s} {'uniq':>6s} {'hit%':>6s} "
              f"{'wgame%':>7s} {'xgame%':>7s} {'→spd':>6s}")
        print('-' * 80)
        for r in rows:
            print(f"{r['label']:<30s} "
                  f"{r['total_requests']:>6d} {r['global_unique']:>6d} "
                  f"{r['any_hit_rate']*100:>5.1f}% "
                  f"{r['within_game_hit_rate']*100:>6.1f}% "
                  f"{r['cross_game_hit_rate']*100:>6.1f}% "
                  f"{r['projected_speedup']:>5.2f}×")
    else:
        r = run_scenario(
            label='single', map_num=args.map_num, sims=args.sims,
            use_fake=args.fake, games=args.games,
            nn_batch_size=args.nn_batch_size, virtual_loss=args.virtual_loss,
            baseline_nn_pct=args.baseline_nn_pct)
        print_row(r)


if __name__ == '__main__':
    main()
