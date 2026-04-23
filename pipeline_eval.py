"""pipeline_eval.py — deterministic evaluation of a pipeline checkpoint.

Generates a fixed, reproducible set of random validation maps, then plays each
map with deterministic MCTS (no Dirichlet noise, argmax action selection).
Optionally sweeps across several sim counts so you can see the cost/compute
tradeoff that the trained prior enables.

Saves per-sim-count summary + optional best-per-sim atom-viz-compatible JSONs.

Usage:
  python pipeline_eval.py \
    --ckpt=/scratch/.../pipeline_v3a01/checkpoints/cycle_05.ckpt \
    --num_maps=50 --sim_counts=400,1600,6400 \
    --map_num=2 --seed_base=100000 \
    --output_dir=/scratch/.../pipeline_v3a01/eval/cycle_05 \
    --save_best_json \
    --preset=hpc
"""

import argparse
import json
import os
import time
import torch

from neutral_atoms.config import (
    set_derived_config, get_map_data, atom_map_to_positions,
)
from neutral_atoms.network import Network
from neutral_atoms.game import Game
from fast_mcts.backends import get_backend
from neutral_atoms.experiment import compute_solution_cost, game_to_solution
from neutral_atoms.map_generator import generate_random_map


def _apply_overrides(config, overrides):
    for o in overrides:
        k, v = o.split('=', 1)
        try:
            v = eval(v)
        except Exception:
            pass
        d = config
        parts = k.split('.')
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num_maps', type=int, default=50)
    ap.add_argument('--sim_counts', default='400,1600,6400',
                    help='Comma-separated sim counts to sweep.')
    ap.add_argument('--seed_base', type=int, default=100000,
                    help='Val map seeds = seed_base + map_idx. Keep disjoint from training seeds.')
    ap.add_argument('--map_num', type=int, default=2,
                    help='Map structure class to use (for board_dim / num_qubits / num_tasks).')
    ap.add_argument('--output_dir', default='',
                    help='Where to write eval_summary.json and (optional) solution JSONs.')
    ap.add_argument('--save_best_json', action='store_true',
                    help='Save one atom-viz JSON per (sim_count) of the best-cost game.')
    ap.add_argument('--preset', choices=['default', 'hpc'], default='hpc')
    ap.add_argument('--config_override', action='append', default=[])
    ap.add_argument('--device', default='auto',
                    help='auto | cpu | cuda')
    args = ap.parse_args()

    if args.preset == 'hpc':
        from neutral_atoms.config_hpc import get_config as _get_config
    else:
        from neutral_atoms.config import get_config as _get_config
    config = _get_config()
    config.map_num = args.map_num
    config.random_board = True
    _apply_overrides(config, args.config_override)
    set_derived_config(config)
    map_data = get_map_data(config)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    net = Network(config.network, use_fake=False)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    print(f"Loaded: {args.ckpt}")
    print(f"  run_id={ckpt.get('run_id','?')}  training_steps={ckpt.get('training_steps','?')}  "
          f"parent_ckpt={ckpt.get('parent_ckpt','')!r}")
    print(f"  device={device}  num_maps={args.num_maps}  sim_counts={args.sim_counts}")

    gates_per_layer = max(len(layer) for layer in map_data['tasks'])
    val_maps = []
    for i in range(args.num_maps):
        m = generate_random_map(
            map_data['board_dim'], map_data['num_qubits'], config.network.num_tasks,
            gates_per_layer=gates_per_layer, seed=args.seed_base + i,
        )
        val_maps.append((
            m['tasks'],
            atom_map_to_positions(m['atom_map'], config.env.board_width),
        ))

    sim_counts = [int(s) for s in args.sim_counts.split(',')]
    all_results = {'ckpt': args.ckpt, 'device': device, 'num_maps': args.num_maps,
                   'seed_base': args.seed_base, 'map_num': args.map_num,
                   'training_steps': ckpt.get('training_steps'),
                   'run_id': ckpt.get('run_id'), 'per_sim': {}}

    for n_sims in sim_counts:
        config.mcts.num_simulations = n_sims
        costs, plan_times, best_game = [], [], None
        best_cost = float('inf')
        t_total0 = time.time()
        for i, (tasks, ip) in enumerate(val_maps):
            g = Game(config, tasks, ip)
            t0 = time.time()
            g = get_backend(config.mcts.backend)(g, config.mcts, net,
                                                add_exploration_noise=False, deterministic=True)
            plan_times.append(time.time() - t0)
            c = compute_solution_cost(g)
            costs.append(c)
            if c < best_cost:
                best_cost = c
                best_game = g
        total_t = time.time() - t_total0

        n = len(costs)
        avg = sum(costs) / n
        sorted_costs = sorted(costs)
        p10 = sorted_costs[max(0, int(0.1 * n) - 1)]
        p50 = sorted_costs[n // 2]
        p90 = sorted_costs[min(n - 1, int(0.9 * n))]
        summary = {
            'n_sims': n_sims,
            'num_maps': n,
            'avg_cost': round(avg, 3),
            'min_cost': min(costs),
            'max_cost': max(costs),
            'p10_cost': p10, 'p50_cost': p50, 'p90_cost': p90,
            'avg_plan_time_s': round(sum(plan_times) / n, 3),
            'total_plan_time_s': round(total_t, 2),
            'costs': costs,
        }
        all_results['per_sim'][str(n_sims)] = summary
        print(f"  sims={n_sims:>5d}: avg={avg:.2f} min={min(costs)} "
              f"p10={p10} p50={p50} p90={p90} "
              f"avg_plan={summary['avg_plan_time_s']:.2f}s  total={total_t:.0f}s")

        if args.save_best_json and args.output_dir and best_game is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            sol = game_to_solution(best_game)
            out = os.path.join(args.output_dir, f'best_sims{n_sims}_cost{best_cost}.json')
            with open(out, 'w') as f:
                json.dump(sol, f, indent=2)
            print(f"    saved best solution: {out}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, 'eval_summary.json')
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Summary: {out}")


if __name__ == '__main__':
    main()
