"""
surrogate/primitives.py — Core AOD compatibility, graph coloring, ground-truth costs.

These are the reference implementations used by tests and surrogates.
No approximations — these compute exact discrete values.
"""

import torch
from itertools import product as iterproduct


# ─────────────────────────────────────────────────────────────────────────────
# AOD compatibility (exact, discrete)
# ─────────────────────────────────────────────────────────────────────────────

def aod_compatible(mi, mj):
    """Check AOD compatibility between moves.

    Args:
        mi, mj: (..., 4) tensors of [src_row, src_col, dst_row, dst_col]

    Returns: (...) bool tensor. True = parallel-compatible.

    The AOD constraint requires that for two simultaneous moves:
    1. Column ordering is preserved (src and dst relative order match)
    2. Row ordering is preserved (same)
    3. Destinations are distinct (no collision)
    """
    sr_i, sc_i, dr_i, dc_i = mi[..., 0], mi[..., 1], mi[..., 2], mi[..., 3]
    sr_j, sc_j, dr_j, dc_j = mj[..., 0], mj[..., 1], mj[..., 2], mj[..., 3]

    # Column axis: sign(src_col_diff) must equal sign(dst_col_diff)
    dcs, dcd = sc_i - sc_j, dc_i - dc_j
    h_ok = ((dcs == 0) & (dcd == 0)) | (
        ~((dcs == 0) | (dcd == 0)) & (torch.sign(dcs) == torch.sign(dcd)))

    # Row axis: same condition
    drs, drd = sr_i - sr_j, dr_i - dr_j
    v_ok = ((drs == 0) & (drd == 0)) | (
        ~((drs == 0) | (drd == 0)) & (torch.sign(drs) == torch.sign(drd)))

    # No collision: distinct destinations
    no_col = ~((drd == 0) & (dcd == 0))

    return h_ok & v_ok & no_col


# ─────────────────────────────────────────────────────────────────────────────
# Graph coloring (exact, greedy)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_chromatic(moves):
    """Greedy graph coloring on AOD conflict graph.

    Args:
        moves: (N, 4) tensor of [src_row, src_col, dst_row, dst_col]

    Returns: int — chromatic number (number of parallel groups).
    """
    n = moves.shape[0]
    if n == 0: return 0
    if n == 1: return 1

    conflict = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i + 1, n):
            if not aod_compatible(moves[i], moves[j]):
                conflict[i, j] = conflict[j, i] = True

    colors = [-1] * n
    for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
        used = {colors[k] for k in range(n) if conflict[idx, k] and colors[k] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[idx] = c
    return max(colors) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth cost functions (exact, for validation)
# ─────────────────────────────────────────────────────────────────────────────

def true_gate_cost(cells, gates, H, W, canon=True):
    """Exact gate cost for concrete cell assignments.

    Args:
        cells: (N,) cell indices
        gates: list of (atom_a, atom_b)
        H, W: grid dims
        canon: minimize over direction assignments

    Returns: int — 2 × χ_gate
    """
    M = len(gates)
    if M == 0: return 0
    rows, cols = cells // W, cells % W

    def _groups(dirs):
        mvs = []
        for g, (a, b) in enumerate(gates):
            if dirs[g]: a, b = b, a
            mvs.append(torch.tensor([rows[a], cols[a], rows[b], cols[b]],
                                     dtype=torch.long))
        return greedy_chromatic(torch.stack(mvs))

    if not canon:
        return 2 * _groups([0] * M)
    return 2 * min(_groups([(d >> g) & 1 for g in range(M)])
                   for d in range(1 << M))


def true_reconfig_cost(src_cells, dst_cells, H, W):
    """Exact reconfig cost for concrete cell assignments.

    Only atoms that actually move contribute.
    """
    moves = []
    for q in range(len(src_cells)):
        s = src_cells[q].item() if isinstance(src_cells[q], torch.Tensor) else src_cells[q]
        d = dst_cells[q].item() if isinstance(dst_cells[q], torch.Tensor) else dst_cells[q]
        if s != d:
            moves.append(torch.tensor([s // W, s % W, d // W, d % W],
                                       dtype=torch.long))
    if not moves: return 0
    return greedy_chromatic(torch.stack(moves))


def true_layer_cost(prev_cells, curr_cells, gates, H, W):
    """Exact total cost for one layer: reconfig + gate."""
    rc = true_reconfig_cost(prev_cells, curr_cells, H, W)
    gc = true_gate_cost(curr_cells, gates, H, W, canon=True)
    return rc, gc, rc + gc


def true_total_cost(initial_cells, layer_cells_list, tasks, H, W):
    """Exact total cost across all layers.

    Args:
        initial_cells: (N,) initial cell indices
        layer_cells_list: list of (N,) cell indices per layer
        tasks: list of gate lists per layer

    Returns: total_cost, list of (reconfig, gate) per layer
    """
    total = 0
    prev = initial_cells
    breakdown = []
    for t, (cells_t, gates_t) in enumerate(zip(layer_cells_list, tasks)):
        rc, gc, lc = true_layer_cost(prev, cells_t, gates_t, H, W)
        total += lc
        breakdown.append((rc, gc))
        prev = cells_t
    return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_relevant_atoms(gates):
    """Sorted list of atom indices involved in a gate layer."""
    return sorted(set(a for pair in gates for a in pair))


def cells_to_rowcol(cells, W):
    """Convert cell indices to (row, col) pairs."""
    return cells // W, cells % W


def rowcol_to_cells(rows, cols, W):
    """Convert (row, col) to cell indices."""
    return rows * W + cols