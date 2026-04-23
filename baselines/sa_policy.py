"""Simulated annealing baseline for neutral atom reconfiguration.

Solves the SAME per-layer problem as smt_policy and the MCTS environment:
  minimize  reconfig_groups + 2 * gate_groups  (per layer, greedy across layers)

Adapted from Enola's Fast-SA placement (arxiv 2405.15095), but with the
distance proxy replaced by the exact count_groups cost used by the env.

See docs/baselines/simulated_annealing.md for literature + design notes.

Public API:
    plan(initial_positions, tasks, rows, cols, **sa_kwargs) -> atom_viz_plan
"""
import sys, time, math, random
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from neutral_atoms.moves import count_groups
from neutral_atoms.tasks import gates_to_moves


# ─── cost ─────────────────────────────────────────────────────────────────────

def _layer_cost(prev_pos, new_pos, gate_layer, relevant):
    """Exact env cost for one layer given prev/new positions of relevant qubits."""
    reconfig_moves = []
    for q in relevant:
        pr, pc = prev_pos[q]
        nr, nc = new_pos[q]
        if (nr, nc) != (pr, pc):
            reconfig_moves.append([pr, pc, nr, nc])
    reconfig_cost = (count_groups(torch.tensor(reconfig_moves, dtype=torch.long))
                     if reconfig_moves else 0)

    n_qubits = max(max(new_pos), max(prev_pos)) + 1
    atom_positions = torch.zeros(n_qubits, 2, dtype=torch.long)
    for q, (r, c) in prev_pos.items():
        atom_positions[q] = torch.tensor([r, c])
    for q, (r, c) in new_pos.items():
        atom_positions[q] = torch.tensor([r, c])
    gate_moves = gates_to_moves(gate_layer, atom_positions)
    gate_cost = 2 * count_groups(gate_moves, canonicalize=True) if len(gate_moves) > 0 else 0
    return reconfig_cost + gate_cost


# ─── SA core ──────────────────────────────────────────────────────────────────

@dataclass
class SAConfig:
    steps: int = 2000
    T_init: float = 2.0
    T_final: float = 0.01
    swap_prob: float = 0.4       # prob of swap neighbor vs reassign neighbor
    seed: int | None = None


def _empty_cells(positions, rows, cols):
    occupied = set(positions.values())
    return [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in occupied]


def _neighbor(state, all_positions, rows, cols, rng, swap_prob):
    """Propose a neighbor: swap two relevant qubits, or reassign one to an empty cell."""
    relevant = list(state.keys())
    if len(relevant) == 0:
        return dict(state)
    if len(relevant) >= 2 and rng.random() < swap_prob:
        q1, q2 = rng.sample(relevant, 2)
        out = dict(state)
        out[q1], out[q2] = state[q2], state[q1]
        return out
    q = rng.choice(relevant)
    merged = {**all_positions, **state}
    empties = _empty_cells(merged, rows, cols)
    if not empties:
        return dict(state)
    out = dict(state)
    out[q] = rng.choice(empties)
    return out


def _anneal_layer(prev_pos, gate_layer, relevant, rows, cols, cfg: SAConfig):
    rng = random.Random(cfg.seed)
    state = {q: prev_pos[q] for q in relevant}
    cost = _layer_cost(prev_pos, state, gate_layer, relevant)
    best_state, best_cost = dict(state), cost

    if cfg.steps <= 0 or not relevant:
        return best_state, best_cost

    # geometric cooling: T_k = T_init * (T_final / T_init)^(k / steps)
    decay = (cfg.T_final / cfg.T_init) ** (1.0 / max(1, cfg.steps))
    T = cfg.T_init

    for _ in range(cfg.steps):
        cand = _neighbor(state, prev_pos, rows, cols, rng, cfg.swap_prob)
        cand_cost = _layer_cost(prev_pos, cand, gate_layer, relevant)
        dC = cand_cost - cost
        if dC <= 0 or rng.random() < math.exp(-dC / max(T, 1e-9)):
            state, cost = cand, cand_cost
            if cost < best_cost:
                best_state, best_cost = dict(state), cost
        T *= decay

    return best_state, best_cost


# ─── public API ───────────────────────────────────────────────────────────────

def plan(initial_positions: dict, tasks: list, rows: int, cols: int,
         steps: int = 2000, T_init: float = 2.0, T_final: float = 0.01,
         swap_prob: float = 0.4, seed: int | None = None) -> dict:
    """Run layer-by-layer simulated annealing and return atom-viz plan dict.

    Args:
        initial_positions: {qubit_id: (row, col)} or {qubit_id: cell_index}
        tasks: list of layers, each layer = list of (q0, q1) pairs
        rows, cols: board dimensions
        steps: SA iterations per layer
        T_init, T_final: temperature schedule endpoints
        swap_prob: probability of swap vs reassign neighbor
        seed: RNG seed (each layer advances the derived seed)

    Returns:
        {board, circuit, plan, sa_cost, sa_elapsed_s} in atom-viz format
    """
    # normalise positions to (row, col) tuples
    pos = {}
    for q, v in initial_positions.items():
        q = int(q)
        if isinstance(v, (int, float)):
            pos[q] = (int(v) // cols, int(v) % cols)
        else:
            pos[q] = (int(v[0]), int(v[1]))

    plan_layers = []
    total_cost = 0
    t0 = time.time()
    cur_pos = dict(pos)

    for layer_idx, gate_layer in enumerate(tasks):
        relevant = sorted({q for pair in gate_layer for q in pair})
        cfg = SAConfig(steps=steps, T_init=T_init, T_final=T_final,
                       swap_prob=swap_prob,
                       seed=None if seed is None else seed + layer_idx)
        new_state, layer_cost = _anneal_layer(cur_pos, gate_layer, relevant, rows, cols, cfg)

        layer_moves = []
        for q in relevant:
            pr, pc = cur_pos[q]
            nr, nc = new_state[q]
            if (nr, nc) != (pr, pc):
                layer_moves.append({
                    'atom': q,
                    'from': {'row': pr, 'col': pc},
                    'to':   {'row': nr, 'col': nc},
                })
        plan_layers.append(layer_moves)
        cur_pos = {**cur_pos, **new_state}
        total_cost += layer_cost

    initial_atoms = {str(q): {'row': r, 'col': c} for q, (r, c) in pos.items()}
    circuit = [[list(g) for g in layer] for layer in tasks]

    return {
        'board':         {'rows': rows, 'cols': cols, 'initialAtoms': initial_atoms},
        'circuit':       circuit,
        'plan':          plan_layers,
        'sa_cost':       total_cost,
        'sa_elapsed_s':  time.time() - t0,
    }
