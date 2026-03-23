"""Quick benchmark: SMT vs kohei on a few test cases.

Usage:
    ../grid_mcts2/.venv/bin/python baselines/benchmark_smt.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.random_board import random_board
from baselines.kohei_policy import plan as kohei_plan
from baselines.smt_policy import plan as smt_plan, solve_layer


def cost_of_plan(result):
    """Sum of move count as a proxy; the real cost is computed by the env."""
    return result.get('smt_cost', None)


def run_case(label, initial_positions, tasks, rows, cols):
    print(f"\n{'='*60}")
    print(f"Case: {label}  ({rows}x{cols}, {len(initial_positions)} qubits, {len(tasks)} layers)")

    # Kohei
    t0 = time.time()
    kr = kohei_plan(initial_positions, tasks, rows, cols)
    kohei_ms = (time.time() - t0) * 1000
    print(f"  kohei:  {kohei_ms:.1f} ms")

    # SMT (layer by layer)
    t0 = time.time()
    sr = smt_plan(initial_positions, tasks, rows, cols, timeout_ms=60_000)
    smt_ms = (time.time() - t0) * 1000
    print(f"  smt:    {smt_ms:.1f} ms  (cost={sr['smt_cost']}, z3_time={sr['smt_elapsed_s']*1000:.1f}ms)")
    print(f"  speedup: {smt_ms/kohei_ms:.1f}x slower than kohei")


if __name__ == '__main__':
    from neutral_atoms.config import MAPS

    # Case 1: Map 1 (4x4, 8 qubits)
    m = MAPS[1]
    from neutral_atoms.config import atom_map_to_positions
    rows1, cols1 = m['board_dim']
    pos1 = {i: (r, c) for i, (r, c) in enumerate(atom_map_to_positions(m['atom_map'], cols1))}
    run_case("Map1 (4x4/8q)", pos1, m['tasks'], rows1, cols1)

    # Case 2: Map 2 (5x5, 12 qubits)
    m = MAPS[2]
    rows2, cols2 = m['board_dim']
    pos2 = {i: (r, c) for i, (r, c) in enumerate(atom_map_to_positions(m['atom_map'], cols2))}
    run_case("Map2 (5x5/12q)", pos2, m['tasks'], rows2, cols2)

    # Case 3: random small (3x4, 6 qubits, 2 layers, 2 gates/layer)
    rb = random_board(rows=3, cols=4, num_qubits=6, num_layers=2, gates_per_layer=2, seed=42)
    pos3 = {int(k): (v['row'], v['col']) for k, v in rb['board']['initialAtoms'].items()}
    run_case("Random 3x4/6q/2L", pos3, rb['circuit'], 3, 4)

    # Case 4: random 4x4, 8 qubits, 3 layers, 3 gates/layer
    rb = random_board(rows=4, cols=4, num_qubits=8, num_layers=3, gates_per_layer=3, seed=7)
    pos4 = {int(k): (v['row'], v['col']) for k, v in rb['board']['initialAtoms'].items()}
    run_case("Random 4x4/8q/3L", pos4, rb['circuit'], 4, 4)
