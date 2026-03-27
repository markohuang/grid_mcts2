"""
surrogate/feasibility.py — Unified feasibility-based cost surrogate.

Both gate cost and reconfig cost are instances of the same computation:
checking whether a set of moves is coordinate-wise order-preserving.

Gate cost:    2 + λ_g · (-log F_gate)     where F_gate = max_d ∏ P(compat_ij | d)
Reconfig cost: P(any move) + λ_r · (-log F_reconfig)  where F_reconfig = ∏ P(compat_ij)

Each pairwise P(compat) is computed from coordinate marginals in O(V^4).
Total complexity: O(2^M · M^2 · V^4) for gates, O(N^2 · V^4) for reconfig.
"""

import torch
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# Core: pairwise compatibility probability from coordinate marginals
# ─────────────────────────────────────────────────────────────────────────────

def _coord_marginal(dist, H, W, axis):
    """Extract marginal over row or column coordinate values.

    Args:
        dist: (C,) distribution over cells
        H, W: grid dims
        axis: 'col' or 'row'

    Returns: (V,) marginal where V = W (col) or H (row)
    """
    C = H * W
    device, dtype = dist.device, dist.dtype
    V = W if axis == 'col' else H
    cells = torch.arange(C, device=device)
    coords = (cells % W) if axis == 'col' else (cells // W)

    # Scatter-add: marginal[v] = sum of dist[c] where coord(c) == v
    marginal = torch.zeros(V, device=device, dtype=dtype)
    marginal.scatter_add_(0, coords, dist)
    return marginal


def pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, axis):
    """P(axis-compatible) for one pair of moves on one axis.

    Move i: src_i → dst_i, Move j: src_j → dst_j.
    Compatible iff column (or row) ordering is preserved.

    Computed from coordinate marginals. O(V^4) where V = W or H.

    Args:
        src_i, dst_i: (C,) source/dest distributions for move i
        src_j, dst_j: (C,) source/dest distributions for move j
        H, W: grid dims
        axis: 'col' or 'row'

    Returns: scalar — P(compatible on this axis). Differentiable.
    """
    si = _coord_marginal(src_i, H, W, axis)
    di = _coord_marginal(dst_i, H, W, axis)
    sj = _coord_marginal(src_j, H, W, axis)
    dj = _coord_marginal(dst_j, H, W, axis)

    V = si.shape[0]
    v = torch.arange(V, device=si.device, dtype=si.dtype)

    # (V, V, V, V) tensor of differences
    ds = v[:, None, None, None] - v[None, :, None, None]  # src_i_coord - src_j_coord
    dd = v[None, None, :, None] - v[None, None, None, :]  # dst_i_coord - dst_j_coord

    # AOD axis compatibility: sign(ds) == sign(dd), with edge cases for zeros
    both_zero = (ds == 0) & (dd == 0)           # same src coord AND same dst coord: ok
    one_zero = (ds == 0) ^ (dd == 0)            # exactly one is zero: NOT ok
    same_sign = torch.sign(ds) == torch.sign(dd) # same ordering direction: ok
    neither_zero = (ds != 0) & (dd != 0)

    ok = both_zero | (neither_zero & same_sign)
    # Note: one_zero is already excluded by the above (it's neither both_zero nor neither_zero with same_sign)

    # Joint probability over all 4 coordinate values
    prob = (si[:, None, None, None] * sj[None, :, None, None] *
            di[None, None, :, None] * dj[None, None, None, :])

    return (prob * ok.to(prob.dtype)).sum()


def pairwise_no_collision_prob(dst_i, dst_j):
    """P(distinct destinations) for two moves.

    Args:
        dst_i, dst_j: (C,) destination distributions

    Returns: scalar — P(dst_i ≠ dst_j). Differentiable.
    """
    return 1.0 - (dst_i * dst_j).sum()


def pairwise_full_compat_prob(src_i, dst_i, src_j, dst_j, H, W):
    """P(fully AOD-compatible) for one pair of moves.

    Combines both axes and collision check.
    Assumes axis independence (mild bias, see theory doc Section 4.2).

    Returns: scalar — P(compatible). Differentiable.
    """
    p_col = pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, 'col')
    p_row = pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, 'row')
    p_nocol = pairwise_no_collision_prob(dst_i, dst_j)
    return p_col * p_row * p_nocol


# ─────────────────────────────────────────────────────────────────────────────
# Gate cost surrogate
# ─────────────────────────────────────────────────────────────────────────────

def gate_feasibility(gate_atoms, placement_dists, H, W):
    """Compute gate feasibility probability under best direction assignment.

    For each of 2^M direction assignments, computes the product of pairwise
    compatibility probabilities (edge-independence approximation). Returns
    the maximum over directions.

    Args:
        gate_atoms: list of (atom_a, atom_b) per gate
        placement_dists: (N, C) soft placement distributions
        H, W: grid dims

    Returns:
        best_F: scalar — max_d ∏ P(compat_ij | d). Differentiable.
        info: dict with per-direction details
    """
    M = len(gate_atoms)
    C = H * W
    device, dtype = placement_dists.device, placement_dists.dtype

    if M == 0:
        return torch.tensor(1.0, device=device, dtype=dtype), {}

    best_log_F = None
    best_dir = None

    all_log_Fs = []

    for dir_idx in range(1 << M):
        dirs = [(dir_idx >> g) & 1 for g in range(M)]

        # Build source/dest distributions for each gate under this direction
        gate_src = []
        gate_dst = []
        for g, (a, b) in enumerate(gate_atoms):
            if dirs[g] == 0:
                gate_src.append(placement_dists[a])
                gate_dst.append(placement_dists[b])
            else:
                gate_src.append(placement_dists[b])
                gate_dst.append(placement_dists[a])

        # Compute log-product of pairwise compat probabilities
        log_F = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(M):
            for j in range(i + 1, M):
                p = pairwise_full_compat_prob(
                    gate_src[i], gate_dst[i],
                    gate_src[j], gate_dst[j], H, W)
                log_F = log_F + torch.log(p.clamp(min=1e-30))

        all_log_Fs.append(log_F)

        if best_log_F is None or log_F.item() > best_log_F.item():
            best_log_F = log_F
            best_dir = dirs

    # Differentiable max: straight-through (differentiate through the argmax)
    # or soft-max over directions
    best_idx = max(range(len(all_log_Fs)), key=lambda i: all_log_Fs[i].item())
    best_log_F = all_log_Fs[best_idx]

    return torch.exp(best_log_F), {
        'log_F': best_log_F,
        'best_dir': best_dir,
        'all_log_Fs': [lf.item() for lf in all_log_Fs],
    }


def gate_cost_surrogate(gate_atoms, placement_dists, H, W,
                        lambda_g=1.0, lambda_aux=0.0):
    """Differentiable gate cost surrogate.

    cost = 2 + λ_g · (-log F_gate) + λ_aux · conflict_count

    The -log F term provides exact signal at the feasibility boundary (F=1).
    The conflict_count term provides gradient proportional to the number of
    violated constraints, giving better signal in the deep infeasible region.

    Args:
        gate_atoms: list of (atom_a, atom_b) per gate
        placement_dists: (N, C) distributions
        H, W: grid dims
        lambda_g: weight for feasibility penalty (-log F)
        lambda_aux: weight for conflict count auxiliary loss

    Returns: (cost, info_dict)
    """
    F_gate, info = gate_feasibility(gate_atoms, placement_dists, H, W)
    neg_log_F = -info['log_F']
    cost = 2.0 + lambda_g * neg_log_F

    # Auxiliary: conflict count under best direction
    if lambda_aux > 0:
        conflict_count = info.get('conflict_count', torch.tensor(0.0))
        cost = cost + lambda_aux * conflict_count

    info['F_gate'] = F_gate
    info['neg_log_F'] = neg_log_F
    return cost, info


# ─────────────────────────────────────────────────────────────────────────────
# Reconfig cost surrogate
# ─────────────────────────────────────────────────────────────────────────────

def reconfig_feasibility(src_dists, dst_dists, H, W, mover_indices=None):
    """Compute reconfig feasibility probability.

    For atoms that might move (P(stay) < 1), computes the product of
    pairwise compatibility probabilities, weighted by P(both move).

    No direction freedom — moves go from src to dst.

    Args:
        src_dists: (N, C) source distributions (previous layer)
        dst_dists: (N, C) destination distributions (current layer)
        H, W: grid dims
        mover_indices: which atoms to consider (default: all)

    Returns:
        F_reconfig: scalar feasibility probability
        info: dict with details
    """
    N, C = src_dists.shape
    device, dtype = src_dists.device, src_dists.dtype

    if mover_indices is None:
        mover_indices = list(range(N))

    M = len(mover_indices)
    if M < 2:
        return torch.tensor(1.0, device=device, dtype=dtype), {'n_movers': torch.tensor(0.0)}

    # P(atom moves) per candidate mover
    p_stay = torch.stack([(src_dists[q] * dst_dists[q]).sum()
                           for q in mover_indices])
    p_move = 1.0 - p_stay
    n_movers = p_move.sum()

    # Log-product of pairwise compat, weighted by P(both move)
    log_F = torch.tensor(0.0, device=device, dtype=dtype)
    conflict_count = torch.tensor(0.0, device=device, dtype=dtype)
    n_active_pairs = 0

    for ii in range(M):
        for jj in range(ii + 1, M):
            qi, qj = mover_indices[ii], mover_indices[jj]

            # P(both actually move)
            p_both = p_move[ii] * p_move[jj]

            if p_both.item() < 1e-10:
                continue  # neither moves with appreciable probability

            n_active_pairs += 1

            p = pairwise_full_compat_prob(
                src_dists[qi], dst_dists[qi],
                src_dists[qj], dst_dists[qj], H, W)

            # Weighted contribution
            effective_p = (1.0 - p_both) + p_both * p
            log_F = log_F + torch.log(effective_p.clamp(min=1e-30))

            # Conflict count: expected number of conflicting mover pairs
            conflict_count = conflict_count + p_both * (1.0 - p)

    F_reconfig = torch.exp(log_F)

    return F_reconfig, {
        'log_F': log_F,
        'n_movers': n_movers,
        'n_active_pairs': n_active_pairs,
        'p_move': p_move,
        'conflict_count': conflict_count,
    }


def reconfig_cost_surrogate(src_dists, dst_dists, H, W,
                             mover_indices=None, lambda_r=1.0, lambda_aux=0.0):
    """Differentiable reconfig cost surrogate.

    cost = P(any atom moves) · (1 + λ_r · (-log F_reconfig))

    When no atoms move: cost = 0.
    When atoms move and all compatible: cost ≈ 1.
    When atoms move and some conflict: cost > 1.

    Args:
        src_dists: (N, C) source distributions
        dst_dists: (N, C) destination distributions
        H, W: grid dims
        mover_indices: atoms to consider
        lambda_r: penalty weight

    Returns: (cost, info_dict)
    """
    F_reconfig, info = reconfig_feasibility(
        src_dists, dst_dists, H, W, mover_indices)

    n_movers = info['n_movers']
    # Soft indicator: any atom moves (smooth 0→1 transition)
    p_any_move = 1.0 - torch.exp(
        torch.log((1.0 - info['p_move']).clamp(min=1e-30)).sum()
    ) if 'p_move' in info and len(info['p_move']) > 0 else torch.tensor(0.0)
    # Above: P(at least one moves) = 1 - ∏ P(stay_i)

    neg_log_F = -info['log_F']
    cost = p_any_move * (1.0 + lambda_r * neg_log_F)

    info['F_reconfig'] = F_reconfig
    info['neg_log_F'] = neg_log_F
    info['p_any_move'] = p_any_move
    return cost, info


# ─────────────────────────────────────────────────────────────────────────────
# Combined layer cost
# ─────────────────────────────────────────────────────────────────────────────

def layer_cost_surrogate(prev_dists, curr_dists, gate_atoms, H, W,
                          mover_indices=None,
                          lambda_g=1.0, lambda_r=1.0, lambda_aux=0.0):
    """Combined surrogate cost for one layer.

    cost = reconfig_cost(prev → curr) + gate_cost(curr)

    Args:
        prev_dists: (N, C) previous layer distributions (or initial)
        curr_dists: (N, C) current layer distributions
        gate_atoms: list of (atom_a, atom_b) for this layer's gates
        H, W: grid dims
        mover_indices: atoms considered for reconfig (default: all)
        lambda_g, lambda_r: feasibility penalty weights
        lambda_aux: conflict count auxiliary loss weight

    Returns: (total_cost, gate_info, reconfig_info)
    """
    gc, g_info = gate_cost_surrogate(gate_atoms, curr_dists, H, W,
                                      lambda_g, lambda_aux)
    rc, r_info = reconfig_cost_surrogate(prev_dists, curr_dists, H, W,
                                          mover_indices, lambda_r, lambda_aux)
    return gc + rc, g_info, r_info


def total_cost_surrogate(initial_dists, layer_dists_list, tasks, H, W,
                          mover_indices_list=None,
                          lambda_g=1.0, lambda_r=1.0, lambda_aux=0.0):
    """Total surrogate cost across all layers.

    Args:
        initial_dists: (N, C) initial position distributions
        layer_dists_list: list of (N, C) per-layer distributions
        tasks: list of gate lists per layer
        H, W: grid dims
        mover_indices_list: per-layer mover indices (default: all atoms)
        lambda_g, lambda_r: feasibility penalty weights
        lambda_aux: conflict count auxiliary loss weight

    Returns: (total_cost, list of (gate_info, reconfig_info) per layer)
    """
    total = torch.tensor(0.0, device=initial_dists.device, dtype=initial_dists.dtype)
    prev = initial_dists
    layer_infos = []

    for t, (dists_t, gates_t) in enumerate(zip(layer_dists_list, tasks)):
        movers = mover_indices_list[t] if mover_indices_list else None
        lc, g_info, r_info = layer_cost_surrogate(
            prev, dists_t, gates_t, H, W, movers, lambda_g, lambda_r, lambda_aux)
        total = total + lc
        layer_infos.append((g_info, r_info))
        prev = dists_t

    return total, layer_infos