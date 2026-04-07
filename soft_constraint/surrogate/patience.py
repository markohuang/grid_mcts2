"""
surrogate/patience.py — Cost surrogate via patience sorting (LDS computation).

The AOD constraint on each axis is equivalent to requiring the source→destination
mapping to be order-preserving. The minimum number of parallel groups on one axis
equals the Longest Decreasing Subsequence (LDS) length of the induced permutation,
by Dilworth's theorem.

This module computes:
  - Exact LDS for concrete (discrete) positions
  - Expected LDS for soft (distributional) positions via DP
  - Combined cost: max(LDS_col, LDS_row) + collision penalty

Complexity: O(N × S × V) per axis where S = C(V+N, N), V = board width/height.
For N=8, V=5: ~50K ops. For N=12, V=8: ~12M ops.
"""

import torch
import torch.nn.functional as F
from bisect import bisect_left


# ─────────────────────────────────────────────────────────────────────────────
# Discrete LDS (for one-hot / ranking evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def lds_length(sequence):
    """Longest Decreasing Subsequence length via patience sorting.

    Args:
        sequence: list or 1D tensor of integers

    Returns: int — LDS length = number of piles in patience sorting
    """
    if len(sequence) == 0:
        return 0
    # LDS of seq = LIS of negated seq
    # LIS via patience sorting with bisect
    neg = [-x for x in sequence]
    piles = []
    for val in neg:
        pos = bisect_left(piles, val)
        if pos == len(piles):
            piles.append(val)
        else:
            piles[pos] = val
    return len(piles)


def axis_lds(src_coords, dst_coords):
    """Compute LDS for one axis given concrete source and destination coordinates.

    Sort by source coordinate, read off destination coordinates.
    Atoms with same source coordinate must have same destination coordinate
    for AOD compatibility; inversions among them each count as a conflict.

    Actually: we need to handle the AOD "equality preservation" rule.
    If src_i == src_j, then we need dst_i == dst_j. If dst_i != dst_j,
    BOTH pairs contribute to conflicts. In patience sorting terms,
    ties in source with different destinations are treated as inversions.

    Args:
        src_coords: (N,) integer source coordinates
        dst_coords: (N,) integer destination coordinates

    Returns: int — number of parallel groups needed on this axis
    """
    N = len(src_coords)
    if N <= 1:
        return 1 if N == 1 else 0

    # Sort by source coordinate, break ties by destination (descending)
    # This ensures that atoms with same source but different destination
    # are counted as needing separate groups.
    indices = sorted(range(N), key=lambda i: (src_coords[i], -dst_coords[i]))
    sorted_dst = [dst_coords[i] for i in indices]

    # LDS of sorted_dst = number of groups needed
    # But we need to handle the equality case: same src, same dst = ok (same group)
    # same src, diff dst = conflict (need different groups)
    # Different src: standard LDS applies
    #
    # The trick: by sorting ties with descending dst, we ensure that
    # same-src-same-dst atoms are adjacent and in "increasing" order (ok),
    # while same-src-diff-dst atoms create a decrease (conflict).
    # Then standard LDS on the sorted sequence gives the right answer.
    #
    # Actually, for the AOD "one-zero case": if src_i == src_j but dst_i != dst_j,
    # this is an AOD conflict regardless of which is larger. We need both to be
    # in different groups. The descending tie-break ensures these pairs create
    # a "decrease" in the patience sorting.

    return lds_length(sorted_dst)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete chi estimation: max(LDS_col, LDS_row) + collision handling
# ─────────────────────────────────────────────────────────────────────────────

def discrete_axis_chi(src_cells, dst_cells, H, W):
    """Compute per-axis chi estimates for concrete cell assignments.

    Only considers atoms that actually move.

    Args:
        src_cells: (N,) source cell indices
        dst_cells: (N,) destination cell indices
        H, W: grid dims

    Returns: chi_col, chi_row, n_collisions, movers mask
    """
    N = len(src_cells)

    # Identify movers
    movers = []
    for i in range(N):
        s = src_cells[i].item() if isinstance(src_cells[i], torch.Tensor) else src_cells[i]
        d = dst_cells[i].item() if isinstance(dst_cells[i], torch.Tensor) else dst_cells[i]
        if s != d:
            movers.append(i)

    if len(movers) == 0:
        return 0, 0, 0, movers

    # Extract coordinates for movers
    src_rows = [((src_cells[i].item() if isinstance(src_cells[i], torch.Tensor)
                  else src_cells[i]) // W) for i in movers]
    src_cols = [((src_cells[i].item() if isinstance(src_cells[i], torch.Tensor)
                  else src_cells[i]) % W) for i in movers]
    dst_rows = [((dst_cells[i].item() if isinstance(dst_cells[i], torch.Tensor)
                  else dst_cells[i]) // W) for i in movers]
    dst_cols = [((dst_cells[i].item() if isinstance(dst_cells[i], torch.Tensor)
                  else dst_cells[i]) % W) for i in movers]

    chi_col = axis_lds(src_cols, dst_cols)
    chi_row = axis_lds(src_rows, dst_rows)

    # Count destination collisions
    dst_set = {}
    n_collisions = 0
    for i in movers:
        d = dst_cells[i].item() if isinstance(dst_cells[i], torch.Tensor) else dst_cells[i]
        if d in dst_set:
            n_collisions += 1
        dst_set[d] = True

    return chi_col, chi_row, n_collisions, movers


def discrete_gate_axis_chi(cells, gate_atoms, H, W):
    """Compute per-axis chi for gate moves with optimal direction.

    Tries all 2^M direction assignments, returns the best.

    Args:
        cells: (N_atoms,) cell indices
        gate_atoms: list of (a, b) pairs
        H, W: grid dims

    Returns: best_chi_col, best_chi_row, best_chi_combined, best_directions
    """
    M = len(gate_atoms)
    if M == 0:
        return 0, 0, 0, []

    rows = [c // W for c in (cells.tolist() if isinstance(cells, torch.Tensor) else cells)]
    cols = [c % W for c in (cells.tolist() if isinstance(cells, torch.Tensor) else cells)]

    best_combined = M + 1
    best_info = None

    for dir_idx in range(1 << M):
        dirs = [(dir_idx >> g) & 1 for g in range(M)]

        src_rows, src_cols, dst_rows, dst_cols = [], [], [], []
        for g, (a, b) in enumerate(gate_atoms):
            if dirs[g] == 0:
                src_rows.append(rows[a]); src_cols.append(cols[a])
                dst_rows.append(rows[b]); dst_cols.append(cols[b])
            else:
                src_rows.append(rows[b]); src_cols.append(cols[b])
                dst_rows.append(rows[a]); dst_cols.append(cols[a])

        chi_col = axis_lds(src_cols, dst_cols)
        chi_row = axis_lds(src_rows, dst_rows)

        # Collision check
        n_col = len(set(zip(dst_rows, dst_cols)))
        n_collisions = M - n_col

        combined = max(chi_col, chi_row) + n_collisions  # simple collision penalty

        if combined < best_combined:
            best_combined = combined
            best_info = (chi_col, chi_row, n_collisions, dirs)

    return best_info[0], best_info[1], best_combined, best_info[3]


# ─────────────────────────────────────────────────────────────────────────────
# Full discrete cost computation (for ranking evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def discrete_layer_cost(prev_cells, curr_cells, gate_atoms, H, W, collision_weight=1.0):
    """Compute surrogate layer cost for concrete positions.

    reconfig_cost = max(LDS_col, LDS_row) for movers (+ collision penalty)
    gate_cost = 2 × min_d max(LDS_col, LDS_row) for gates under direction d (+ collision penalty)

    Args:
        prev_cells: (N,) previous layer cell indices
        curr_cells: (N,) current layer cell indices
        gate_atoms: list of (a, b) gate pairs
        H, W: grid dims
        collision_weight: penalty per destination collision

    Returns: (total_cost, reconfig_cost, gate_cost, info_dict)
    """
    # Reconfig
    rc_col, rc_row, rc_collisions, rc_movers = discrete_axis_chi(
        prev_cells, curr_cells, H, W)

    if len(rc_movers) == 0:
        reconfig_cost = 0
    else:
        reconfig_cost = max(rc_col, rc_row) + collision_weight * rc_collisions

    # Gate
    gc_col, gc_row, gc_combined, gc_dirs = discrete_gate_axis_chi(
        curr_cells, gate_atoms, H, W)
    gate_cost = 2 * gc_combined

    total = reconfig_cost + gate_cost

    return total, reconfig_cost, gate_cost, {
        'rc_col': rc_col, 'rc_row': rc_row, 'rc_collisions': rc_collisions,
        'rc_movers': len(rc_movers),
        'gc_col': gc_col, 'gc_row': gc_row, 'gc_combined': gc_combined,
        'gc_dirs': gc_dirs,
    }


def discrete_total_cost(initial_cells, layer_cells_list, tasks, H, W,
                         collision_weight=1.0):
    """Compute surrogate total cost across all layers for concrete positions.

    Args:
        initial_cells: (N,) initial cell indices
        layer_cells_list: list of (N,) per-layer cell indices
        tasks: list of gate lists per layer
        H, W: grid dims

    Returns: (total_cost, per_layer_details)
    """
    total = 0
    prev = initial_cells
    details = []

    for t, (cells_t, gates_t) in enumerate(zip(layer_cells_list, tasks)):
        lc, rc, gc, info = discrete_layer_cost(
            prev, cells_t, gates_t, H, W, collision_weight)
        total += lc
        details.append({
            'layer': t,
            'total': lc,
            'reconfig': rc,
            'gate': gc,
            **info,
        })
        prev = cells_t

    return total, details


# ─────────────────────────────────────────────────────────────────────────────
# Soft patience sorting DP (for optimization with distributions)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_piles(pile_tops, V):
    """Encode a pile-top configuration as an integer index.

    pile_tops: tuple of ints, non-increasing, each in [0, V-1]
    The empty configuration is ().

    We use a combinatorial encoding: non-increasing sequences of length k
    over [0, V-1] biject to combinations with repetition C(V, k).
    """
    # Convert to a canonical form for hashing
    return pile_tops  # use tuple directly as dict key


def _patience_insert(pile_tops, value, V):
    """Insert a value into patience piles. Returns new pile tops.

    Rule: place on leftmost pile whose top >= value.
    If none exists, create new pile.

    pile_tops: tuple of ints, non-increasing
    value: int in [0, V-1]

    Returns: new tuple of pile tops (non-increasing)
    """
    tops = list(pile_tops)
    # Find leftmost pile with top >= value
    inserted = False
    for i in range(len(tops)):
        if tops[i] >= value:
            tops[i] = value
            inserted = True
            break

    if not inserted:
        tops.append(value)

    return tuple(tops)


def soft_lds_dp(coord_marginals, V):
    """Compute E[LDS] via patience sorting DP on soft coordinate marginals.

    Processes atoms one by one (assumed pre-sorted by source coordinate).
    For each atom, marginalizes over its random destination coordinate.

    Args:
        coord_marginals: list of (V,) tensors — destination coordinate
                         distribution for each atom, in source-sorted order
        V: number of coordinate values (W for columns, H for rows)

    Returns: scalar tensor — E[LDS]. Differentiable.
    """
    N = len(coord_marginals)
    if N == 0:
        return torch.tensor(0.0, dtype=coord_marginals[0].dtype if coord_marginals else torch.float64)

    device = coord_marginals[0].device
    dtype = coord_marginals[0].dtype

    # State = pile top configuration (tuple of ints, non-increasing)
    # state_probs: dict mapping state -> probability tensor
    state_probs = {(): torch.tensor(1.0, device=device, dtype=dtype)}

    for i in range(N):
        q = coord_marginals[i]  # (V,) distribution over dest coordinate
        new_state_probs = {}

        for state, prob in state_probs.items():
            if prob.item() < 1e-30:
                continue

            for v in range(V):
                p_v = q[v]
                if p_v.item() < 1e-30:
                    continue

                new_state = _patience_insert(state, v, V)
                transition_prob = prob * p_v

                if new_state in new_state_probs:
                    new_state_probs[new_state] = new_state_probs[new_state] + transition_prob
                else:
                    new_state_probs[new_state] = transition_prob

        state_probs = new_state_probs

    # Expected number of piles
    expected_lds = torch.tensor(0.0, device=device, dtype=dtype)
    for state, prob in state_probs.items():
        n_piles = len(state)
        expected_lds = expected_lds + prob * n_piles

    return expected_lds


def soft_axis_lds(src_coord_marginals, dst_coord_marginals, V):
    """Compute E[LDS] for one axis with soft source and destination coordinates.

    For soft sources, we need to handle the sorting order. When sources
    are deterministic (one-hot), sorting is fixed. When soft, we sort by
    expected source coordinate (approximation).

    For atoms sharing the same source coordinate: in patience sorting,
    we process them with descending tie-break on destination to correctly
    count same-source-different-destination conflicts.

    Args:
        src_coord_marginals: list of (V,) tensors — source coordinate dists
        dst_coord_marginals: list of (V,) tensors — destination coordinate dists
        V: number of coordinate values

    Returns: scalar tensor — E[LDS]. Differentiable.
    """
    N = len(src_coord_marginals)
    if N <= 1:
        if N == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        return torch.tensor(1.0, dtype=src_coord_marginals[0].dtype,
                           device=src_coord_marginals[0].device)

    # Sort by expected source coordinate
    expected_src = [sum(v * m[v].item() for v in range(V)) for m in src_coord_marginals]
    order = sorted(range(N), key=lambda i: expected_src[i])

    # For deterministic sources (one-hot marginals), this is exact.
    # For soft sources, this is an approximation.
    sorted_dst = [dst_coord_marginals[order[i]] for i in range(N)]

    return soft_lds_dp(sorted_dst, V)


# ─────────────────────────────────────────────────────────────────────────────
# Soft cost computation (for optimization)
# ─────────────────────────────────────────────────────────────────────────────

def _cell_to_coord_marginal(cell_dist, H, W, axis):
    """Extract coordinate marginal from cell distribution."""
    C = H * W
    V = W if axis == 'col' else H
    device, dtype = cell_dist.device, cell_dist.dtype
    cells = torch.arange(C, device=device)
    coords = (cells % W) if axis == 'col' else (cells // W)

    marginal = torch.zeros(V, device=device, dtype=dtype)
    marginal.scatter_add_(0, coords, cell_dist)
    return marginal


def soft_reconfig_cost(src_dists, dst_dists, H, W, mover_indices=None,
                        collision_weight=1.0):
    """Soft reconfig cost via patience sorting DP.

    Args:
        src_dists: (N, C) source distributions
        dst_dists: (N, C) destination distributions
        H, W: grid dims
        mover_indices: atoms to consider (default: all)
        collision_weight: penalty per expected collision

    Returns: (cost, info_dict)
    """
    N, C = src_dists.shape
    device, dtype = src_dists.device, src_dists.dtype

    if mover_indices is None:
        mover_indices = list(range(N))

    M = len(mover_indices)
    if M == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    # P(any atom moves)
    p_stay = torch.stack([(src_dists[q] * dst_dists[q]).sum() for q in mover_indices])
    p_move = 1.0 - p_stay
    p_any = 1.0 - torch.exp(torch.log(p_stay.clamp(min=1e-30)).sum())

    # Extract coordinate marginals for movers
    src_col_marg = [_cell_to_coord_marginal(src_dists[q], H, W, 'col') for q in mover_indices]
    dst_col_marg = [_cell_to_coord_marginal(dst_dists[q], H, W, 'col') for q in mover_indices]
    src_row_marg = [_cell_to_coord_marginal(src_dists[q], H, W, 'row') for q in mover_indices]
    dst_row_marg = [_cell_to_coord_marginal(dst_dists[q], H, W, 'row') for q in mover_indices]

    lds_col = soft_axis_lds(src_col_marg, dst_col_marg, W)
    lds_row = soft_axis_lds(src_row_marg, dst_row_marg, H)

    chi_axis = torch.max(lds_col, lds_row)

    # Collision penalty
    collision = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(M):
        for j in range(i + 1, M):
            qi, qj = mover_indices[i], mover_indices[j]
            collision = collision + p_move[i] * p_move[j] * (dst_dists[qi] * dst_dists[qj]).sum()

    cost = p_any * chi_axis + collision_weight * collision

    return cost, {
        'lds_col': lds_col,
        'lds_row': lds_row,
        'chi_axis': chi_axis,
        'collision': collision,
        'p_any': p_any,
        'n_movers_expected': p_move.sum(),
    }


def soft_gate_cost(gate_atoms, placement_dists, H, W, directions=None,
                    collision_weight=1.0):
    """Soft gate cost via patience sorting DP.

    For M=4 gates, enumerates 2^M=16 direction assignments and takes the best.
    For each direction, computes E[LDS] on each axis.

    Args:
        gate_atoms: list of (a, b) per gate
        placement_dists: (N, C) distributions
        H, W: grid dims
        directions: (M,) soft directions in (0,1). If None, enumerate all.
        collision_weight: penalty per expected collision

    Returns: (cost, info_dict)
    """
    M = len(gate_atoms)
    device, dtype = placement_dists.device, placement_dists.dtype

    if M == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    if directions is not None:
        # Soft directions: mixture of forward/reverse marginals
        src_col, dst_col, src_row, dst_row = [], [], [], []
        for g, (a, b) in enumerate(gate_atoms):
            d = directions[g]
            src_col.append((1-d) * _cell_to_coord_marginal(placement_dists[a], H, W, 'col') +
                           d * _cell_to_coord_marginal(placement_dists[b], H, W, 'col'))
            dst_col.append((1-d) * _cell_to_coord_marginal(placement_dists[b], H, W, 'col') +
                           d * _cell_to_coord_marginal(placement_dists[a], H, W, 'col'))
            src_row.append((1-d) * _cell_to_coord_marginal(placement_dists[a], H, W, 'row') +
                           d * _cell_to_coord_marginal(placement_dists[b], H, W, 'row'))
            dst_row.append((1-d) * _cell_to_coord_marginal(placement_dists[b], H, W, 'row') +
                           d * _cell_to_coord_marginal(placement_dists[a], H, W, 'row'))

        lds_col = soft_axis_lds(src_col, dst_col, W)
        lds_row = soft_axis_lds(src_row, dst_row, H)
        chi = torch.max(lds_col, lds_row)

        # Collision penalty
        collision = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(M):
            for j in range(i + 1, M):
                # P(same destination cell) under soft directions
                # Approximate: use the mixed destination distributions
                dst_i = ((1-directions[i]) * placement_dists[gate_atoms[i][1]] +
                         directions[i] * placement_dists[gate_atoms[i][0]])
                dst_j = ((1-directions[j]) * placement_dists[gate_atoms[j][1]] +
                         directions[j] * placement_dists[gate_atoms[j][0]])
                collision = collision + (dst_i * dst_j).sum()

        cost = 2.0 * (chi + collision_weight * collision)
        return cost, {'lds_col': lds_col, 'lds_row': lds_row, 'chi': chi,
                      'collision': collision, 'directions': directions}

    # Enumerate all 2^M directions (discrete), take best
    best_cost = None
    best_info = None

    for dir_idx in range(1 << M):
        dirs_discrete = [(dir_idx >> g) & 1 for g in range(M)]

        src_col, dst_col, src_row, dst_row = [], [], [], []
        for g, (a, b) in enumerate(gate_atoms):
            if dirs_discrete[g] == 0:
                src_col.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'col'))
                dst_col.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'col'))
                src_row.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'row'))
                dst_row.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'row'))
            else:
                src_col.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'col'))
                dst_col.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'col'))
                src_row.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'row'))
                dst_row.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'row'))

        lds_col = soft_axis_lds(src_col, dst_col, W)
        lds_row = soft_axis_lds(src_row, dst_row, H)
        chi = torch.max(lds_col, lds_row)

        # Collision
        collision = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(M):
            for j in range(i + 1, M):
                a_i, b_i = gate_atoms[i]
                a_j, b_j = gate_atoms[j]
                dst_i = placement_dists[b_i if dirs_discrete[i] == 0 else a_i]
                dst_j = placement_dists[b_j if dirs_discrete[j] == 0 else a_j]
                collision = collision + (dst_i * dst_j).sum()

        total = chi + collision_weight * collision

        if best_cost is None or total.item() < best_cost.item():
            best_cost = total
            best_info = {'lds_col': lds_col, 'lds_row': lds_row, 'chi': chi,
                         'collision': collision, 'dirs': dirs_discrete}

    # Re-run best direction for gradient flow
    dirs_best = best_info['dirs']
    src_col, dst_col, src_row, dst_row = [], [], [], []
    for g, (a, b) in enumerate(gate_atoms):
        if dirs_best[g] == 0:
            src_col.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'col'))
            dst_col.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'col'))
            src_row.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'row'))
            dst_row.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'row'))
        else:
            src_col.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'col'))
            dst_col.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'col'))
            src_row.append(_cell_to_coord_marginal(placement_dists[b], H, W, 'row'))
            dst_row.append(_cell_to_coord_marginal(placement_dists[a], H, W, 'row'))

    lds_col = soft_axis_lds(src_col, dst_col, W)
    lds_row = soft_axis_lds(src_row, dst_row, H)
    chi = torch.max(lds_col, lds_row)

    collision = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(M):
        for j in range(i + 1, M):
            dst_i = placement_dists[gate_atoms[i][1] if dirs_best[i] == 0 else gate_atoms[i][0]]
            dst_j = placement_dists[gate_atoms[j][1] if dirs_best[j] == 0 else gate_atoms[j][0]]
            collision = collision + (dst_i * dst_j).sum()

    cost = 2.0 * (chi + collision_weight * collision)

    return cost, {'lds_col': lds_col, 'lds_row': lds_row, 'chi': chi,
                  'collision': collision, 'dirs': dirs_best}


# ─────────────────────────────────────────────────────────────────────────────
# Combined layer and total cost
# ─────────────────────────────────────────────────────────────────────────────

def soft_layer_cost(prev_dists, curr_dists, gate_atoms, H, W,
                     mover_indices=None, directions=None, collision_weight=1.0):
    """Combined soft layer cost: reconfig + 2 × gate."""
    rc, r_info = soft_reconfig_cost(
        prev_dists, curr_dists, H, W, mover_indices, collision_weight)
    gc, g_info = soft_gate_cost(
        gate_atoms, curr_dists, H, W, directions, collision_weight)
    return rc + gc, r_info, g_info


def soft_total_cost(initial_dists, layer_dists_list, tasks, H, W,
                     mover_indices_list=None, directions_list=None,
                     collision_weight=1.0):
    """Total soft cost across all layers."""
    total = torch.tensor(0.0, device=initial_dists.device, dtype=initial_dists.dtype)
    prev = initial_dists
    infos = []

    for t, (dists_t, gates_t) in enumerate(zip(layer_dists_list, tasks)):
        movers = mover_indices_list[t] if mover_indices_list else None
        dirs = directions_list[t] if directions_list else None
        lc, r_info, g_info = soft_layer_cost(
            prev, dists_t, gates_t, H, W, movers, dirs, collision_weight)
        total = total + lc
        infos.append((r_info, g_info))
        prev = dists_t

    return total, infos