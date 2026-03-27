"""
surrogate/feasibility.py — Unified feasibility-based cost surrogate (vectorized).

Both gate cost and reconfig cost are instances of the same computation:
checking whether a set of moves is coordinate-wise order-preserving.

Gate cost:    2 + λ_g · (-log F_gate)     where F_gate = max_d ∏ P(compat_ij | d)
Reconfig cost: P(any move) + λ_r · (-log F_reconfig)  where F_reconfig = ∏ P(compat_ij)

Each pairwise P(compat) is computed from coordinate marginals in O(V^4).
All pairs and directions are batched — no Python for-loops in hot paths.
"""

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Caches for precomputed tensors (keyed by (V, device))
# ─────────────────────────────────────────────────────────────────────────────

_OK_CACHE = {}    # (V, device) -> (V², V²) float ok-mask
_COORD_CACHE = {} # (H, W, axis, device) -> (C,) long coord indices


def _get_ok_flat(V, device, dtype):
    """Get cached (V², V²) compatibility mask for axis dimension V."""
    key = (V, device)
    if key not in _OK_CACHE:
        v = torch.arange(V, device=device)
        ds = v[:, None] - v[None, :]  # (V, V) src differences
        dd = v[:, None] - v[None, :]  # (V, V) dst differences
        ds_flat = ds.reshape(-1, 1)   # (V², 1)
        dd_flat = dd.reshape(1, -1)   # (1, V²)
        both_zero = (ds_flat == 0) & (dd_flat == 0)
        neither_zero = (ds_flat != 0) & (dd_flat != 0)
        same_sign = torch.sign(ds_flat) == torch.sign(dd_flat)
        ok = both_zero | (neither_zero & same_sign)  # (V², V²)
        _OK_CACHE[key] = ok.float()
    return _OK_CACHE[key].to(dtype)


def _get_coord_idx(H, W, axis, device):
    """Get cached coordinate index tensor: (C,) mapping cell -> coord value."""
    key = (H, W, axis, device)
    if key not in _COORD_CACHE:
        C = H * W
        cells = torch.arange(C, device=device)
        _COORD_CACHE[key] = (cells % W) if axis == 'col' else (cells // W)
    return _COORD_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Batched primitives
# ─────────────────────────────────────────────────────────────────────────────

def _batch_coord_marginals(dists, H, W, axis):
    """Extract coordinate marginals for a batch of distributions.

    Args:
        dists: (n, C) distributions over cells
        H, W: grid dims
        axis: 'col' or 'row'

    Returns: (n, V) marginals where V = W (col) or H (row)
    """
    n, C = dists.shape
    V = W if axis == 'col' else H
    device, dtype = dists.device, dists.dtype
    coords = _get_coord_idx(H, W, axis, device)           # (C,)
    coords_exp = coords.unsqueeze(0).expand(n, -1)        # (n, C)
    marginals = torch.zeros(n, V, device=device, dtype=dtype)
    marginals.scatter_add_(1, coords_exp, dists)
    return marginals


def _batch_pairwise_axis_compat(si, sj, di, dj, V, device, dtype):
    """Batched axis compatibility for P pairs.

    P(axis-compatible) = Σ_{vi,vj,ui,uj} si[vi]*sj[vj]*di[ui]*dj[uj]*ok[vi,vj,ui,uj]

    Reshaped as: ((src_outer @ ok_flat) * dst_outer).sum(dim=1)

    Args:
        si, sj, di, dj: (P, V) marginals
        V: axis dimension
        device, dtype: tensor properties

    Returns: (P,) compatibility probabilities
    """
    ok_flat = _get_ok_flat(V, device, dtype)                # (V², V²)
    src_outer = (si.unsqueeze(2) * sj.unsqueeze(1)).reshape(-1, V * V)  # (P, V²)
    dst_outer = (di.unsqueeze(2) * dj.unsqueeze(1)).reshape(-1, V * V)  # (P, V²)
    return (torch.mm(src_outer, ok_flat) * dst_outer).sum(dim=1)        # (P,)


def _batch_pairwise_no_collision(dst_i, dst_j):
    """Batched P(distinct destinations) for P pairs.

    Args:
        dst_i, dst_j: (P, C) destination distributions

    Returns: (P,) probabilities
    """
    return 1.0 - (dst_i * dst_j).sum(dim=1)


def _batch_pairwise_full_compat(src_i, dst_i, src_j, dst_j, H, W):
    """Batched P(fully AOD-compatible) for P pairs.

    Args:
        src_i, dst_i, src_j, dst_j: (P, C) distributions

    Returns: (P,) compatibility probabilities
    """
    device, dtype = src_i.device, src_i.dtype
    # Column axis
    si_col = _batch_coord_marginals(src_i, H, W, 'col')
    di_col = _batch_coord_marginals(dst_i, H, W, 'col')
    sj_col = _batch_coord_marginals(src_j, H, W, 'col')
    dj_col = _batch_coord_marginals(dst_j, H, W, 'col')
    p_col = _batch_pairwise_axis_compat(si_col, sj_col, di_col, dj_col, W, device, dtype)
    # Row axis
    si_row = _batch_coord_marginals(src_i, H, W, 'row')
    di_row = _batch_coord_marginals(dst_i, H, W, 'row')
    sj_row = _batch_coord_marginals(src_j, H, W, 'row')
    dj_row = _batch_coord_marginals(dst_j, H, W, 'row')
    p_row = _batch_pairwise_axis_compat(si_row, sj_row, di_row, dj_row, H, device, dtype)
    # Collision
    p_nocol = _batch_pairwise_no_collision(dst_i, dst_j)
    return p_col * p_row * p_nocol


# ─────────────────────────────────────────────────────────────────────────────
# Legacy scalar functions (kept for test cross-validation)
# ─────────────────────────────────────────────────────────────────────────────

def _coord_marginal(dist, H, W, axis):
    """Extract marginal over row or column coordinate (single distribution)."""
    C = H * W
    V = W if axis == 'col' else H
    device, dtype = dist.device, dist.dtype
    coords = _get_coord_idx(H, W, axis, device)
    marginal = torch.zeros(V, device=device, dtype=dtype)
    marginal.scatter_add_(0, coords, dist)
    return marginal


def pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, axis):
    """P(axis-compatible) for one pair of moves on one axis (scalar version)."""
    si = _coord_marginal(src_i, H, W, axis)
    di = _coord_marginal(dst_i, H, W, axis)
    sj = _coord_marginal(src_j, H, W, axis)
    dj = _coord_marginal(dst_j, H, W, axis)
    V = si.shape[0]
    device, dtype = si.device, si.dtype
    ok_flat = _get_ok_flat(V, device, dtype)
    src_outer = (si[:, None] * sj[None, :]).reshape(1, V * V)
    dst_outer = (di[:, None] * dj[None, :]).reshape(1, V * V)
    return (torch.mm(src_outer, ok_flat) * dst_outer).sum()


def pairwise_no_collision_prob(dst_i, dst_j):
    """P(distinct destinations) for two moves (scalar version)."""
    return 1.0 - (dst_i * dst_j).sum()


def pairwise_full_compat_prob(src_i, dst_i, src_j, dst_j, H, W):
    """P(fully AOD-compatible) for one pair of moves (scalar version)."""
    p_col = pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, 'col')
    p_row = pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, 'row')
    p_nocol = pairwise_no_collision_prob(dst_i, dst_j)
    return p_col * p_row * p_nocol


# ─────────────────────────────────────────────────────────────────────────────
# Loss modes for pairwise compatibility
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_loss(p_compat, mode='log', delta=0.1):
    """Compute per-pair loss from compatibility probabilities.

    Modes:
        'log':      -log P_ij (original feasibility). Precise near optimum,
                    but 1/P_ij gradient blows up for infeasible pairs.
        'linear':   (1 - P_ij) (conflict count). Bounded, democratic gradients,
                    but only first-order accurate near optimum.
        'huber':    Huber-log — log for P > δ, linear for P ≤ δ.
                    Best of both: precise near optimum, bounded far away.

    Args:
        p_compat: (...) tensor of pairwise compat probabilities in [0, 1]
        mode: 'log', 'linear', or 'huber'
        delta: threshold for huber mode (default 0.1 → max 10× amplification)

    Returns: (...) per-pair loss values (higher = more infeasible)
    """
    if mode == 'log':
        return -torch.log(p_compat.clamp(min=1e-30))
    elif mode == 'linear':
        return 1.0 - p_compat
    elif mode == 'huber':
        log_part = -torch.log(p_compat.clamp(min=delta))
        lin_part = -torch.log(torch.tensor(delta, device=p_compat.device, dtype=p_compat.dtype)) + (delta - p_compat) / delta
        return torch.where(p_compat > delta, log_part, lin_part)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Gate cost surrogate (vectorized over directions × pairs)
# ─────────────────────────────────────────────────────────────────────────────

def gate_feasibility(gate_atoms, placement_dists, H, W, loss_mode='log', delta=0.1):
    """Compute gate feasibility probability under best direction assignment.

    Vectorized: all 2^M directions and M(M-1)/2 pairs computed in one batch.

    Args:
        gate_atoms: list of (atom_a, atom_b) per gate
        placement_dists: (N, C) soft placement distributions
        H, W: grid dims
        loss_mode: 'log', 'linear', or 'huber' (see _pairwise_loss)
        delta: threshold for huber mode

    Returns:
        best_F: scalar — max_d ∏ P(compat_ij | d). Differentiable.
        info: dict with loss, conflict_count, best_dir, p_compat
    """
    M = len(gate_atoms)
    C = H * W
    device, dtype = placement_dists.device, placement_dists.dtype

    if M == 0:
        return torch.tensor(1.0, device=device, dtype=dtype), {
            'log_F': torch.tensor(0.0, device=device, dtype=dtype),
            'conflict_count': torch.tensor(0.0, device=device, dtype=dtype),
        }

    D = 1 << M  # number of direction assignments
    atom_a = torch.tensor([a for a, b in gate_atoms], device=device, dtype=torch.long)  # (M,)
    atom_b = torch.tensor([b for a, b in gate_atoms], device=device, dtype=torch.long)  # (M,)

    # All direction assignments: dir_bits[d, g] = bit g of integer d
    d_idx = torch.arange(D, device=device).unsqueeze(1)  # (D, 1)
    g_idx = torch.arange(M, device=device).unsqueeze(0)  # (1, M)
    dir_bits = ((d_idx >> g_idx) & 1).bool()              # (D, M)

    # Select src/dst atom indices per direction: src=a if dir=0, src=b if dir=1
    src_atom_idx = torch.where(dir_bits, atom_b, atom_a)  # (D, M)
    dst_atom_idx = torch.where(dir_bits, atom_a, atom_b)  # (D, M)

    # Gather distributions: (D, M, C)
    src_dists = placement_dists[src_atom_idx]  # advanced indexing
    dst_dists = placement_dists[dst_atom_idx]

    if M < 2:
        # Single gate — always feasible (no pairs to check)
        return torch.tensor(1.0, device=device, dtype=dtype), {
            'log_F': torch.tensor(0.0, device=device, dtype=dtype),
            'best_dir': [0] * M,
            'conflict_count': torch.tensor(0.0, device=device, dtype=dtype),
            'all_log_Fs': [0.0] * D,
        }

    # Pair indices: (P, 2) where P = M*(M-1)/2
    pair_idx = torch.combinations(torch.arange(M, device=device), r=2)  # (P, 2)
    pi, pj = pair_idx[:, 0], pair_idx[:, 1]  # each (P,)
    P = pi.shape[0]

    # Gather pair distributions: (D, P, C)
    src_i = src_dists[:, pi, :]  # (D, P, C)
    dst_i = dst_dists[:, pi, :]
    src_j = src_dists[:, pj, :]
    dst_j = dst_dists[:, pj, :]

    # Flatten to (D*P, C) for batched computation
    DP = D * P
    src_i_flat = src_i.reshape(DP, C)
    dst_i_flat = dst_i.reshape(DP, C)
    src_j_flat = src_j.reshape(DP, C)
    dst_j_flat = dst_j.reshape(DP, C)

    # Batched pairwise compatibility: (D*P,) -> (D, P)
    p_compat = _batch_pairwise_full_compat(
        src_i_flat, dst_i_flat, src_j_flat, dst_j_flat, H, W
    ).reshape(D, P)

    # Per-pair loss under the chosen mode: (D, P)
    pair_loss = _pairwise_loss(p_compat, mode=loss_mode, delta=delta)

    # Total loss per direction: sum over pairs
    total_loss_all = pair_loss.sum(dim=1)  # (D,)

    # Best direction = minimum total loss (straight-through argmin)
    best_idx = total_loss_all.detach().argmin()
    best_loss = total_loss_all[best_idx]

    # Also compute log_F for the best direction (always useful for monitoring)
    log_F_all = torch.log(p_compat.clamp(min=1e-30)).sum(dim=1)
    best_log_F = log_F_all[best_idx]

    # Conflict count under best direction
    conflict_count = (1.0 - p_compat[best_idx]).sum()

    # Best direction as list (for info)
    best_dir_int = best_idx.item()
    best_dir = [(best_dir_int >> g) & 1 for g in range(M)]

    return torch.exp(best_log_F), {
        'loss': best_loss,
        'log_F': best_log_F,
        'best_dir': best_dir,
        'conflict_count': conflict_count,
        'all_log_Fs': log_F_all.detach().tolist(),
    }


def gate_cost_surrogate(gate_atoms, placement_dists, H, W,
                        lambda_g=1.0, lambda_aux=0.0,
                        loss_mode='log', delta=0.1):
    """Differentiable gate cost surrogate.

    cost = 2 + λ_g · gate_loss + λ_aux · conflict_count

    where gate_loss depends on loss_mode:
        'log':    -log F_gate (original)
        'linear': conflict_count (same as λ_aux with λ_g=0)
        'huber':  Huber-log — log near optimum, linear far away

    Args:
        gate_atoms: list of (atom_a, atom_b) per gate
        placement_dists: (N, C) distributions
        H, W: grid dims
        lambda_g: weight for the main loss term
        lambda_aux: weight for conflict count auxiliary loss
        loss_mode: 'log', 'linear', or 'huber'
        delta: threshold for huber mode

    Returns: (cost, info_dict)
    """
    F_gate, info = gate_feasibility(gate_atoms, placement_dists, H, W,
                                     loss_mode=loss_mode, delta=delta)
    cost = 2.0 + lambda_g * info['loss']

    if lambda_aux > 0:
        cost = cost + lambda_aux * info['conflict_count']

    info['F_gate'] = F_gate
    info['neg_log_F'] = -info['log_F']
    return cost, info


# ─────────────────────────────────────────────────────────────────────────────
# Reconfig cost surrogate (vectorized over pairs)
# ─────────────────────────────────────────────────────────────────────────────

def reconfig_feasibility(src_dists, dst_dists, H, W, mover_indices=None,
                          loss_mode='log', delta=0.1):
    """Compute reconfig feasibility probability (vectorized).

    Args:
        src_dists: (N, C) source distributions (previous layer)
        dst_dists: (N, C) destination distributions (current layer)
        H, W: grid dims
        mover_indices: which atoms to consider (default: all)
        loss_mode: 'log', 'linear', or 'huber'
        delta: threshold for huber mode

    Returns:
        F_reconfig: scalar feasibility probability
        info: dict with details
    """
    N, C = src_dists.shape
    device, dtype = src_dists.device, src_dists.dtype

    if mover_indices is None:
        mover_indices = list(range(N))

    M = len(mover_indices)
    _zero = torch.tensor(0.0, device=device, dtype=dtype)
    if M < 2:
        return torch.tensor(1.0, device=device, dtype=dtype), {
            'loss': _zero, 'log_F': _zero,
            'n_movers': _zero, 'n_active_pairs': 0,
            'p_move': torch.zeros(max(M, 1), device=device, dtype=dtype),
            'conflict_count': _zero,
        }

    mover_idx = torch.tensor(mover_indices, device=device, dtype=torch.long)  # (M,)

    # P(atom moves) per mover — vectorized
    p_stay = (src_dists[mover_idx] * dst_dists[mover_idx]).sum(dim=1)  # (M,)
    p_move = 1.0 - p_stay
    n_movers = p_move.sum()

    # All pair indices
    pair_idx = torch.combinations(torch.arange(M, device=device), r=2)  # (P, 2)
    pi, pj = pair_idx[:, 0], pair_idx[:, 1]

    # P(both move) per pair
    p_both = p_move[pi] * p_move[pj]  # (P,)

    # Filter active pairs
    active = p_both > 1e-10
    n_active_pairs = active.sum().item()

    if n_active_pairs == 0:
        return torch.tensor(1.0, device=device, dtype=dtype), {
            'loss': _zero, 'log_F': _zero,
            'n_movers': n_movers, 'n_active_pairs': 0,
            'p_move': p_move, 'conflict_count': _zero,
        }

    # Gather distributions for active pairs
    qi = mover_idx[pi[active]]  # global atom indices for active pair-i
    qj = mover_idx[pj[active]]  # global atom indices for active pair-j
    p_both_active = p_both[active]  # (A,)

    # Batched pairwise compat
    p_compat = _batch_pairwise_full_compat(
        src_dists[qi], dst_dists[qi],
        src_dists[qj], dst_dists[qj], H, W
    )  # (A,)

    # Weighted contribution: effective_p accounts for P(not both move)
    effective_p = (1.0 - p_both_active) + p_both_active * p_compat
    log_F = torch.log(effective_p.clamp(min=1e-30)).sum()

    # Per-pair loss weighted by P(both move)
    pair_loss_vals = _pairwise_loss(p_compat, mode=loss_mode, delta=delta)
    weighted_loss = (p_both_active * pair_loss_vals).sum()

    # Conflict count
    conflict_count = (p_both_active * (1.0 - p_compat)).sum()

    F_reconfig = torch.exp(log_F)

    return F_reconfig, {
        'loss': weighted_loss,
        'log_F': log_F,
        'n_movers': n_movers,
        'n_active_pairs': n_active_pairs,
        'p_move': p_move,
        'conflict_count': conflict_count,
    }


def reconfig_cost_surrogate(src_dists, dst_dists, H, W,
                             mover_indices=None, lambda_r=1.0, lambda_aux=0.0,
                             loss_mode='log', delta=0.1):
    """Differentiable reconfig cost surrogate.

    cost = P(any atom moves) · (1 + λ_r · reconfig_loss)
         + λ_aux · conflict_count

    where reconfig_loss depends on loss_mode (see _pairwise_loss).

    Args:
        src_dists: (N, C) source distributions
        dst_dists: (N, C) destination distributions
        H, W: grid dims
        mover_indices: atoms to consider
        lambda_r: penalty weight
        lambda_aux: conflict count auxiliary loss weight
        loss_mode: 'log', 'linear', or 'huber'
        delta: threshold for huber mode

    Returns: (cost, info_dict)
    """
    F_reconfig, info = reconfig_feasibility(
        src_dists, dst_dists, H, W, mover_indices, loss_mode=loss_mode, delta=delta)

    # Soft indicator: P(at least one atom moves) = 1 - ∏ P(stay_i)
    p_move = info['p_move']
    if len(p_move) > 0:
        p_any_move = 1.0 - torch.exp(
            torch.log((1.0 - p_move).clamp(min=1e-30)).sum())
    else:
        p_any_move = torch.tensor(0.0, device=src_dists.device, dtype=src_dists.dtype)

    cost = p_any_move * (1.0 + lambda_r * info['loss'])

    if lambda_aux > 0:
        cost = cost + lambda_aux * info['conflict_count']

    info['F_reconfig'] = F_reconfig
    info['neg_log_F'] = -info['log_F']
    info['p_any_move'] = p_any_move
    return cost, info


# ─────────────────────────────────────────────────────────────────────────────
# Combined layer cost
# ─────────────────────────────────────────────────────────────────────────────

def layer_cost_surrogate(prev_dists, curr_dists, gate_atoms, H, W,
                          mover_indices=None,
                          lambda_g=1.0, lambda_r=1.0, lambda_aux=0.0,
                          loss_mode='log', delta=0.1):
    """Combined surrogate cost for one layer.

    cost = reconfig_cost(prev → curr) + gate_cost(curr)
    """
    gc, g_info = gate_cost_surrogate(gate_atoms, curr_dists, H, W,
                                      lambda_g, lambda_aux, loss_mode, delta)
    rc, r_info = reconfig_cost_surrogate(prev_dists, curr_dists, H, W,
                                          mover_indices, lambda_r, lambda_aux,
                                          loss_mode, delta)
    return gc + rc, g_info, r_info


def total_cost_surrogate(initial_dists, layer_dists_list, tasks, H, W,
                          mover_indices_list=None,
                          lambda_g=1.0, lambda_r=1.0, lambda_aux=0.0,
                          loss_mode='log', delta=0.1):
    """Total surrogate cost across all layers."""
    total = torch.tensor(0.0, device=initial_dists.device, dtype=initial_dists.dtype)
    prev = initial_dists
    layer_infos = []

    for t, (dists_t, gates_t) in enumerate(zip(layer_dists_list, tasks)):
        movers = mover_indices_list[t] if mover_indices_list else None
        lc, g_info, r_info = layer_cost_surrogate(
            prev, dists_t, gates_t, H, W, movers, lambda_g, lambda_r, lambda_aux,
            loss_mode, delta)
        total = total + lc
        layer_infos.append((g_info, r_info))
        prev = dists_t

    return total, layer_infos
