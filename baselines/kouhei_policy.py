"""Kouhei greedy policy baseline.

At each qubit placement step, picks the cell that maximises gate_gain_vector
(tiebreak: move_gain_vector). No MCTS, no network.

Public API:
    plan(initial_positions, tasks, rows, cols) -> atom_viz_plan dict
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from neutral_atoms.moves import count_groups, is_parallel_executable_batch


# ─── board helpers ────────────────────────────────────────────────────────────

def cell(r, c, cols): return r * cols + c
def rc(cell_idx, cols): return divmod(cell_idx, cols)


# ─── gain vector primitives ───────────────────────────────────────────────────

def _gate_moves_tensor(gates, positions, cols):
    """Returns tensor (N, 4) of [fr, fc, tr, tc] for each gate in `gates`,
    canonicalised so the lexicographically-smaller endpoint is first.
    Skips gates where a qubit is not in positions."""
    rows_list = []
    for g in gates:
        q0, q1 = g[0], g[1]
        if q0 not in positions or q1 not in positions:
            continue
        r0, c0 = rc(positions[q0], cols)
        r1, c1 = rc(positions[q1], cols)
        if (c0, r0) <= (c1, r1):
            rows_list.append([r0, c0, r1, c1])
        else:
            rows_list.append([r1, c1, r0, c0])
    if not rows_list:
        return torch.empty(0, 4, dtype=torch.long)
    return torch.tensor(rows_list, dtype=torch.long)


def _fits_in_groups(base_tensor, new_move_row):
    """True if adding new_move_row (shape (4,)) does NOT increase group count."""
    new_t = new_move_row.unsqueeze(0)
    if base_tensor.shape[0] == 0:
        return True  # sole move always fits in 1 group
    combined = torch.cat([base_tensor, new_t], dim=0)
    return count_groups(combined) <= count_groups(base_tensor)


def gate_gain_vector(player, positions, twoq_gates, cols, board_size):
    """For each cell, the marginal gain of placing `player` there for the
    current layer's gate execution parallelism.

    Returns list of length board_size with values in {-1, 0, 1}.
    """
    # Separate player's gate from others
    player_gate = None
    non_player_gates = []
    for g in twoq_gates:
        if player in (g[0], g[1]):
            player_gate = g
        else:
            non_player_gates.append(g)

    if player_gate is None:
        return [0] * board_size

    partner = player_gate[0] if player_gate[1] == player else player_gate[1]
    if partner not in positions:
        return [0] * board_size

    base_tensor = _gate_moves_tensor(non_player_gates, positions, cols)
    base_groups = count_groups(base_tensor) if base_tensor.shape[0] > 0 else 0

    pr, pc = rc(positions[partner], cols)
    occupied = {c for q, c in positions.items() if q != player}

    def _player_move_tensor(r, c):
        if (c, r) <= (pc, pr):
            return torch.tensor([[r, c, pr, pc]], dtype=torch.long)
        else:
            return torch.tensor([[pr, pc, r, c]], dtype=torch.long)

    # Original gain (player at current position)
    player_cell = positions.get(player)
    if player_cell is not None:
        cr, cc = rc(player_cell, cols)
        orig_t = _player_move_tensor(cr, cc)
        combined = torch.cat([base_tensor, orig_t], dim=0) if base_tensor.shape[0] > 0 else orig_t
        original_gain = 1 if count_groups(combined) <= base_groups else 0
    else:
        original_gain = 0

    results = [0] * board_size
    for cell_idx in range(board_size):
        if cell_idx in occupied:
            continue
        r, c = rc(cell_idx, cols)
        move_t = _player_move_tensor(r, c)
        combined = torch.cat([base_tensor, move_t], dim=0) if base_tensor.shape[0] > 0 else move_t
        move_gain = 1 if count_groups(combined) <= base_groups else 0
        results[cell_idx] = move_gain - original_gain

    if player_cell is not None:
        results[player_cell] = 0  # no-op → delta zero by definition
    return results


def move_gain_vector(player, positions, committed_moves, cols, board_size):
    """For each cell, whether placing player there fits into existing committed
    reconfig moves without adding a new parallel group.

    committed_moves: list of (from_cell, to_cell) for already-committed
    reconfig moves this layer.
    Returns list of length board_size with values in {-1, 0, 1}.
    """
    if not committed_moves:
        return [0] * board_size

    # Build tensor for committed moves
    rows_list = []
    for frm, to in committed_moves:
        fr, fc = rc(frm, cols)
        tr, tc = rc(to, cols)
        rows_list.append([fr, fc, tr, tc])
    base_tensor = torch.tensor(rows_list, dtype=torch.long)
    base_groups = count_groups(base_tensor)

    player_cell = positions.get(player)
    occupied = {c for q, c in positions.items() if q != player}

    # Original gain (staying put = no new move added, so best possible)
    original_gain = 1  # no additional move = fits trivially

    results = [0] * board_size
    for cell_idx in range(board_size):
        if cell_idx == player_cell or cell_idx in occupied:
            results[cell_idx] = 0
            continue
        fr, fc = rc(player_cell, cols) if player_cell is not None else (0, 0)
        tr, tc = rc(cell_idx, cols)
        new_move = torch.tensor([[fr, fc, tr, tc]], dtype=torch.long)
        combined = torch.cat([base_tensor, new_move], dim=0)
        move_gain = 1 if count_groups(combined) <= base_groups else 0
        results[cell_idx] = move_gain - original_gain

    if player_cell is not None:
        results[player_cell] = 0
    return results


# ─── greedy policy ────────────────────────────────────────────────────────────

def _greedy_place(player, positions, twoq_gates, committed_moves, rows, cols):
    """Pick the best legal cell for `player` and return (chosen_cell, positions)."""
    board_size = rows * cols
    gg = gate_gain_vector(player, positions, twoq_gates, cols, board_size)
    mg = move_gain_vector(player, positions, committed_moves, cols, board_size)

    player_cell = positions[player]
    occupied = {c for q, c in positions.items() if q != player}
    legal = [c for c in range(board_size) if c not in occupied or c == player_cell]

    best_cell = max(legal, key=lambda c: (gg[c], mg[c]))
    new_positions = {**positions, player: best_cell}
    return best_cell, new_positions




# ─── public API ───────────────────────────────────────────────────────────────

def plan(initial_positions: dict, tasks: list, rows: int, cols: int) -> dict:
    """Run greedy kouhei policy and return atom-viz plan dict.

    Args:
        initial_positions: {qubit_id: (row, col)} or {qubit_id: cell_index}
        tasks: list of layers, each layer = list of (q0, q1) tuples
        rows, cols: board dimensions

    Returns:
        {board, circuit, plan} in atom-viz format
    """
    # Normalise positions to cell indices
    def to_cell(v):
        if isinstance(v, (tuple, list)):
            return v[0] * cols + v[1]
        return v  # already a cell index

    positions = {int(q): to_cell(p) for q, p in initial_positions.items()}

    atom_viz_plan = []
    board_initial = {}
    for q, cell_idx in positions.items():
        r, c = divmod(cell_idx, cols)
        board_initial[str(q)] = {"row": r, "col": c}

    circuit = [[list(g) for g in layer] for layer in tasks]

    for layer_gates in tasks:
        relevant = sorted({q for g in layer_gates for q in g})
        positions_before = dict(positions)
        committed_moves = []  # (from_cell, to_cell) pairs committed so far

        for player in relevant:
            best_cell, positions = _greedy_place(
                player, positions, layer_gates, committed_moves, rows, cols)
            if best_cell != positions_before[player]:
                committed_moves.append((positions_before[player], best_cell))

        # Collect actual moves for this layer
        layer_moves = []
        for q in relevant:
            frm = positions_before[q]
            to = positions[q]
            if frm != to:
                fr, fc = divmod(frm, cols)
                tr, tc = divmod(to, cols)
                layer_moves.append({"atom": q,
                                    "from": {"row": fr, "col": fc},
                                    "to": {"row": tr, "col": tc}})
        atom_viz_plan.append(layer_moves)

    return {
        "board": {"rows": rows, "cols": cols, "initialAtoms": board_initial},
        "circuit": circuit,
        "plan": atom_viz_plan,
    }
