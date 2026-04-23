"""Evaluate baseline policies on the held-out validation map set.

Generates the same 30 maps as pipeline_eval.py (map_num=2, seed_base=100000)
and reports avg/min/p10/p50/p90 cost for each baseline.

Usage:
    ../.venv/bin/python scripts/eval_baselines.py
    ../.venv/bin/python scripts/eval_baselines.py --num_maps=30 --seed_base=100000
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from neutral_atoms.config import get_config, set_derived_config, get_map_data, atom_map_to_positions
from neutral_atoms.map_generator import generate_random_map
from neutral_atoms.moves import count_groups
from neutral_atoms.tasks import gates_to_moves
from baselines.kohei_policy import plan as kohei_plan


def cost_of_atomviz_plan(plan_moves, tasks, initial_positions_rc, cols):
    """Compute reconfig+gate cost from an atom-viz plan.

    plan_moves: list of layers, each layer = list of {atom, from:{row,col}, to:{row,col}}
    tasks:      list of gate layers, each = list of (q0, q1) pairs
    initial_positions_rc: {qubit_id: (row, col)}
    """
    positions = {int(q): torch.tensor(list(rc), dtype=torch.long)
                 for q, rc in initial_positions_rc.items()}
    total_cost = 0
    for layer_idx, (layer_moves, gate_layer) in enumerate(zip(plan_moves, tasks)):
        # reconfig cost
        if layer_moves:
            move_tensors = []
            for m in layer_moves:
                fr, fc = m['from']['row'], m['from']['col']
                tr, tc = m['to']['row'], m['to']['col']
                move_tensors.append(torch.tensor([fr, fc, tr, tc], dtype=torch.long))
                # update positions
                q = int(m['atom'])
                positions[q] = torch.tensor([tr, tc], dtype=torch.long)
            total_cost += count_groups(torch.stack(move_tensors), canonicalize=False)
        # gate cost
        gate_moves = gates_to_moves(gate_layer, positions)
        if gate_moves.shape[0] > 0:
            total_cost += 2 * count_groups(gate_moves, canonicalize=True)
    return total_cost


def stats(costs):
    n = len(costs)
    s = sorted(costs)
    return {
        'n': n,
        'avg': round(sum(costs) / n, 3),
        'min': s[0],
        'p10': s[max(0, int(0.1 * n) - 1)],
        'p50': s[n // 2],
        'p90': s[min(n - 1, int(0.9 * n))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_maps', type=int, default=30)
    ap.add_argument('--seed_base', type=int, default=100000)
    ap.add_argument('--map_num', type=int, default=2)
    args = ap.parse_args()

    config = get_config()
    config.map_num = args.map_num
    config.random_board = True
    set_derived_config(config)
    map_data = get_map_data(config)
    rows, cols = config.env.board_height, config.env.board_width
    gates_per_layer = max(len(layer) for layer in map_data['tasks'])

    print(f"map_num={args.map_num}  board={rows}x{cols}  num_maps={args.num_maps}  seed_base={args.seed_base}")

    kohei_costs, kohei_times = [], []

    for i in range(args.num_maps):
        m = generate_random_map(
            map_data['board_dim'], map_data['num_qubits'], config.network.num_tasks,
            gates_per_layer=gates_per_layer, seed=args.seed_base + i,
        )
        positions_rc = {q: rc for q, rc in enumerate(atom_map_to_positions(m['atom_map'], cols))}

        t0 = time.time()
        result = kohei_plan(positions_rc, m['tasks'], rows, cols)
        elapsed = time.time() - t0

        cost = cost_of_atomviz_plan(result['plan'], m['tasks'], positions_rc, cols)
        kohei_costs.append(cost)
        kohei_times.append(elapsed)

    ks = stats(kohei_costs)
    avg_t = sum(kohei_times) / len(kohei_times)
    print(f"\nKohei greedy  ({args.num_maps} maps)")
    print(f"  avg={ks['avg']}  min={ks['min']}  p10={ks['p10']}  p50={ks['p50']}  p90={ks['p90']}")
    print(f"  avg_plan_time={avg_t*1000:.1f}ms")


if __name__ == '__main__':
    main()
