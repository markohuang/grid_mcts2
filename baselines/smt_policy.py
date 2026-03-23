"""SMT-optimal baseline for neutral atom reconfiguration.

Solves the SAME problem as the MCTS environment:
  minimize  sum_over_layers( reconfig_groups + 2 * gate_groups )
where groups = minimum parallel batches under the AOD non-crossing constraint.

Strategy: layer-by-layer (greedy across layers, optimal within each layer).
Each layer solve uses Z3 Optimize. The non-crossing constraint logic is a
direct Z3 translation of the same condition in neutral_atoms/fast_moves.py.

Why layer-by-layer and not globally optimal:
  Global optimality requires solving all layers simultaneously, which has
  O(board_size^(num_qubits * num_layers)) search space. Layer-by-layer is
  already NP-hard per layer (chromatic number), but tractable for small boards.
  A globally optimal SMT would be slower still; this is the fair comparison.

Public API:
    plan(initial_positions, tasks, rows, cols) -> dict  (atom-viz format)
    solve_layer(prev_pos, gate_layer, rows, cols) -> LayerSolution
"""
import sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))

from z3 import Int, Bool, And, Or, Not, Implies, If, Optimize, sat


# ---------------------------------------------------------------------------
# Non-crossing constraint (Z3 translation of fast_moves._is_parallel_np)
# ---------------------------------------------------------------------------

def _can_parallel(fr0, fc0, tr0, tc0, fr1, fc1, tr1, tc1):
    """Z3 boolean expression: True iff the two moves can execute in parallel.

    Moves are (from_row, from_col) -> (to_row, to_col).
    Directly mirrors the logic in fast_moves._is_parallel_np — same math,
    symbolic Z3 expressions instead of concrete numpy values.
    """
    v1r = fr0 - fr1   # from_row difference
    v1c = fc0 - fc1   # from_col difference
    v2r = tr0 - tr1   # to_row difference
    v2c = tc0 - tc1   # to_col difference

    # Row dimension non-crossing: relative row ordering must be preserved
    r_ok = Or(
        And(v1r == 0, v2r == 0),       # both start and end in same relative row
        And(v1r > 0, v2r > 0),         # i above j throughout
        And(v1r < 0, v2r < 0),         # i below j throughout
    )
    # Col dimension non-crossing: relative col ordering must be preserved
    c_ok = Or(
        And(v1c == 0, v2c == 0),
        And(v1c > 0, v2c > 0),
        And(v1c < 0, v2c < 0),
    )
    # No two atoms end at the same cell
    no_collision = Or(v2r != 0, v2c != 0)

    return And(r_ok, c_ok, no_collision)


# ---------------------------------------------------------------------------
# Single-layer optimal solver
# ---------------------------------------------------------------------------

@dataclass
class LayerSolution:
    new_pos: dict          # {qubit_id: (row, col)}
    reconfig_cost: int     # greedy count_groups reconfig (consistent with env/atom-viz)
    gate_cost: int         # greedy 2 * count_groups gate (consistent with env/atom-viz)
    optimal_reconfig: int  # Z3's chromatic-optimal reconfig (provably minimum achievable)
    optimal_gate: int      # Z3's chromatic-optimal 2*gate (provably minimum achievable)
    elapsed_s: float


def solve_layer(
    prev_pos: dict,
    gate_layer: list,
    rows: int,
    cols: int,
    timeout_ms: int = 30_000,
) -> Optional[LayerSolution]:
    """Find optimal atom positions for one gate layer using Z3 Optimize.

    Args:
        prev_pos:   {qubit_id: (row, col)} for ALL qubits, before this layer.
        gate_layer: list of (q0, q1) gate pairs in this layer.
        rows, cols: board dimensions.
        timeout_ms: Z3 solver timeout in ms (default 30s).

    Returns:
        LayerSolution, or None if Z3 times out / returns unknown.
    """
    t0 = time.time()
    relevant = sorted({q for pair in gate_layer for q in pair})
    n_gates = len(gate_layer)

    opt = Optimize()
    opt.set('timeout', timeout_ms)

    # --- Position variables for relevant atoms ---
    row = {q: Int(f'r_{q}') for q in relevant}
    col = {q: Int(f'c_{q}') for q in relevant}

    for q in relevant:
        opt.add(row[q] >= 0, row[q] < rows)
        opt.add(col[q] >= 0, col[q] < cols)

    # No two relevant atoms at the same cell (also can't collide with fixed atoms)
    fixed_cells = {(v[0], v[1]) for k, v in prev_pos.items() if k not in relevant}
    for i, q in enumerate(relevant):
        # Can't land on a fixed atom's cell
        for (fr, fc) in fixed_cells:
            opt.add(Or(row[q] != fr, col[q] != fc))
        # Can't collide with other relevant atoms
        for q2 in relevant[i + 1:]:
            opt.add(Or(row[q] != row[q2], col[q] != col[q2]))

    # --- Reconfig cost ---
    # Only atoms that actually move (new_pos != prev_pos) contribute.
    # k_r = number of parallel reconfig groups (0 if no atoms move).
    k_r = Int('k_r')
    opt.add(k_r >= 0)

    gr = {q: Int(f'gr_{q}') for q in relevant}
    for q in relevant:
        opt.add(gr[q] >= 0)
        pr, pc = int(prev_pos[q][0]), int(prev_pos[q][1])
        moved_q = Or(row[q] != pr, col[q] != pc)
        # If moved: group must be < k_r  (equivalently, k_r > gr[q])
        opt.add(Implies(moved_q, k_r > gr[q]))
        # If not moved: assign dummy group 0 (doesn't constrain k_r)
        opt.add(Implies(Not(moved_q), gr[q] == 0))

    # Non-crossing constraint for reconfig: conflicting moves → different groups
    for i, q in enumerate(relevant):
        for q2 in relevant[i + 1:]:
            pr, pc = int(prev_pos[q][0]), int(prev_pos[q][1])
            pr2, pc2 = int(prev_pos[q2][0]), int(prev_pos[q2][1])
            conflict = Not(_can_parallel(pr, pc, row[q], col[q],
                                         pr2, pc2, row[q2], col[q2]))
            # Only enforce group separation if both actually move
            moved_q = Or(row[q] != pr, col[q] != pc)
            moved_q2 = Or(row[q2] != pr2, col[q2] != pc2)
            opt.add(Implies(And(moved_q, moved_q2, conflict), gr[q] != gr[q2]))

    # --- Gate execution cost ---
    # Each gate (q0, q1): one atom moves to the other's cell.
    # With canonicalize=True, we choose the direction per gate to minimize groups.
    k_g = Int('k_g')
    opt.add(k_g >= 1)   # at least one group if there are any gates

    gg = [Int(f'gg_{g}') for g in range(n_gates)]
    gd = [Bool(f'gd_{g}') for g in range(n_gates)]  # True = q0->q1, False = q1->q0

    for g in range(n_gates):
        opt.add(gg[g] >= 0, k_g > gg[g])

    for g in range(n_gates):
        q0, q1 = gate_layer[g]
        fr0 = If(gd[g], row[q0], row[q1])
        fc0 = If(gd[g], col[q0], col[q1])
        tr0 = If(gd[g], row[q1], row[q0])
        tc0 = If(gd[g], col[q1], col[q0])

        for g2 in range(g + 1, n_gates):
            q0b, q1b = gate_layer[g2]
            fr1 = If(gd[g2], row[q0b], row[q1b])
            fc1 = If(gd[g2], col[q0b], col[q1b])
            tr1 = If(gd[g2], row[q1b], row[q0b])
            tc1 = If(gd[g2], col[q1b], col[q0b])

            conflict = Not(_can_parallel(fr0, fc0, tr0, tc0, fr1, fc1, tr1, tc1))
            opt.add(Implies(conflict, gg[g] != gg[g2]))

    # --- Objective ---
    opt.minimize(k_r + 2 * k_g)

    result = opt.check()
    if result != sat:
        return None   # timeout or unknown

    m = opt.model()

    def val(expr):
        return m.eval(expr, model_completion=True).as_long()

    # Z3's chromatic-optimal cost (provably minimum achievable)
    z3_reconfig = val(k_r)
    z3_gate = 2 * val(k_g)

    new_pos = {**prev_pos}  # start with all prev positions, then update
    for q in relevant:
        new_pos[q] = (val(row[q]), val(col[q]))

    # Greedy cost on Z3's chosen positions (consistent with env/atom-viz evaluation)
    import torch
    from neutral_atoms.moves import count_groups
    from neutral_atoms.tasks import gates_to_moves

    n_qubits = max(new_pos) + 1
    atom_positions = torch.zeros(n_qubits, 2, dtype=torch.long)
    for q, (r, c) in new_pos.items():
        atom_positions[q] = torch.tensor([r, c])

    reconfig_moves = []
    for q in relevant:
        pr, pc = int(prev_pos[q][0]), int(prev_pos[q][1])
        nr, nc = new_pos[q]
        if (nr, nc) != (pr, pc):
            reconfig_moves.append(torch.tensor([pr, pc, nr, nc], dtype=torch.long))
    greedy_reconfig = count_groups(torch.stack(reconfig_moves)) if reconfig_moves else 0

    gate_moves = gates_to_moves(gate_layer, atom_positions)
    greedy_gate = 2 * count_groups(gate_moves, canonicalize=True) if len(gate_moves) > 0 else 0

    return LayerSolution(
        new_pos=new_pos,
        reconfig_cost=greedy_reconfig,
        gate_cost=greedy_gate,
        optimal_reconfig=z3_reconfig,
        optimal_gate=z3_gate,
        elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Full plan
# ---------------------------------------------------------------------------

def plan(initial_positions: dict, tasks: list, rows: int, cols: int,
         timeout_ms: int = 30_000) -> dict:
    """Run layer-by-layer SMT optimization and return an atom-viz plan dict.

    Args:
        initial_positions: {qubit_id: (row, col)} OR {qubit_id: cell_index}
        tasks:   list of gate layers, each layer = list of (q0, q1) pairs
        rows, cols: board dimensions
        timeout_ms: per-layer Z3 timeout

    Returns:
        atom-viz plan dict with keys: board, circuit, plan, smt_cost, smt_elapsed_s
    """
    # Normalise positions to (row, col) tuples
    pos = {}
    for q, v in initial_positions.items():
        q = int(q)
        if isinstance(v, (int, float)):
            pos[q] = (int(v) // cols, int(v) % cols)
        else:
            pos[q] = (int(v[0]), int(v[1]))

    plan_layers = []
    total_cost = 0
    total_elapsed = 0.0
    cur_pos = dict(pos)

    for layer_idx, gate_layer in enumerate(tasks):
        sol = solve_layer(cur_pos, gate_layer, rows, cols, timeout_ms=timeout_ms)

        if sol is None:
            plan_layers.append([])
            continue

        total_elapsed += sol.elapsed_s
        layer_moves = []
        for q, (nr, nc) in sol.new_pos.items():
            if q in cur_pos:
                pr, pc = cur_pos[q]
                if (nr, nc) != (pr, pc):
                    layer_moves.append({
                        'atom': q,
                        'from': {'row': pr, 'col': pc},
                        'to':   {'row': nr, 'col': nc},
                    })
        plan_layers.append(layer_moves)
        cur_pos = sol.new_pos
        total_cost += sol.optimal_reconfig + sol.optimal_gate

    # Build atom-viz board from initial positions
    initial_atoms = {str(q): {'row': r, 'col': c} for q, (r, c) in pos.items()}
    circuit = [[list(g) for g in layer] for layer in tasks]

    return {
        'board':        {'rows': rows, 'cols': cols, 'initialAtoms': initial_atoms},
        'circuit':      circuit,
        'plan':         plan_layers,
        'smt_cost':     total_cost,  # Z3 chromatic optimal — matches display (optimal_count_groups)
        'smt_elapsed_s': total_elapsed,
    }
