"""
surrogate/batched.py — Batch-vectorized cost surrogate for shared task structure.

When all B samples in a batch share the same tasks (e.g., single_map or same-structure
streaming), we can batch the pairwise compatibility computations across samples,
replacing the Python B-loop with a single fused matmul.

Gate:   (B, N, C) → flatten to (B*D*P, C) → one batched compat → reshape → per-sample argmin
Reconfig: (B, N, C) × (B, N, C) → flatten to (B*A, C) → one batched compat → reshape
"""

import torch
import torch.nn.functional as F
from .feasibility import (
    _batch_pairwise_full_compat, _pairwise_loss,
    _batch_coord_marginals, _batch_pairwise_axis_compat,
    _batch_pairwise_no_collision,
)


def batched_gate_cost(gate_atoms, placement_dists, H, W,
                      lambda_g=1.0, loss_mode='huber', delta=0.1):
    """Batched gate cost surrogate for B samples with shared gate structure.

    Args:
        gate_atoms: list of (atom_a, atom_b) — same for all samples
        placement_dists: (B, N, C) soft placement distributions
        H, W: grid dims

    Returns: (B,) per-sample gate costs (differentiable)
    """
    M = len(gate_atoms)
    B, N, C = placement_dists.shape
    device, dtype = placement_dists.device, placement_dists.dtype

    base_cost = torch.full((B,), 2.0, device=device, dtype=dtype)
    if M < 2:
        return base_cost

    D = 1 << M
    atom_a = torch.tensor([a for a, b in gate_atoms], device=device, dtype=torch.long)
    atom_b = torch.tensor([b for a, b in gate_atoms], device=device, dtype=torch.long)

    # Direction bits: (D, M)
    d_idx = torch.arange(D, device=device).unsqueeze(1)
    g_idx = torch.arange(M, device=device).unsqueeze(0)
    dir_bits = ((d_idx >> g_idx) & 1).bool()

    # src/dst atom indices per direction: (D, M)
    src_atom_idx = torch.where(dir_bits, atom_b, atom_a)
    dst_atom_idx = torch.where(dir_bits, atom_a, atom_b)

    # Gather: (B, D, M, C)
    src_dists = placement_dists[:, src_atom_idx]  # (B, D, M, C)
    dst_dists = placement_dists[:, dst_atom_idx]

    # Pair indices: (P, 2)
    pair_idx = torch.combinations(torch.arange(M, device=device), r=2)
    pi, pj = pair_idx[:, 0], pair_idx[:, 1]
    P = pi.shape[0]

    # Gather pairs: (B, D, P, C)
    src_i = src_dists[:, :, pi]
    dst_i = dst_dists[:, :, pi]
    src_j = src_dists[:, :, pj]
    dst_j = dst_dists[:, :, pj]

    # Flatten to (B*D*P, C) for batched pairwise compat
    BDP = B * D * P
    p_compat = _batch_pairwise_full_compat(
        src_i.reshape(BDP, C), dst_i.reshape(BDP, C),
        src_j.reshape(BDP, C), dst_j.reshape(BDP, C), H, W
    ).reshape(B, D, P)

    # Per-pair loss: (B, D, P)
    pair_loss = _pairwise_loss(p_compat, mode=loss_mode, delta=delta)

    # Total loss per direction per sample: (B, D)
    total_loss_per_dir = pair_loss.sum(dim=2)

    # Best direction per sample (straight-through argmin)
    best_idx = total_loss_per_dir.detach().argmin(dim=1)  # (B,)
    best_loss = total_loss_per_dir[torch.arange(B, device=device), best_idx]  # (B,)

    return base_cost + lambda_g * best_loss


def batched_reconfig_cost(src_dists, dst_dists, H, W, mover_indices=None,
                           lambda_r=1.0, loss_mode='huber', delta=0.1):
    """Batched reconfig cost surrogate for B samples with shared mover structure.

    Args:
        src_dists: (B, N, C) source distributions (previous layer)
        dst_dists: (B, N, C) destination distributions (current layer)
        H, W: grid dims
        mover_indices: list of atom indices to consider (shared across batch)

    Returns: (B,) per-sample reconfig costs (differentiable)
    """
    B, N, C = src_dists.shape
    device, dtype = src_dists.device, src_dists.dtype

    if mover_indices is None:
        mover_indices = list(range(N))
    M = len(mover_indices)

    if M < 2:
        # P(any move) only
        if M == 0:
            return torch.zeros(B, device=device, dtype=dtype)
        mi = mover_indices[0]
        p_stay = (src_dists[:, mi] * dst_dists[:, mi]).sum(dim=1)
        return 1.0 - p_stay  # P(this one atom moves)

    mover_idx = torch.tensor(mover_indices, device=device, dtype=torch.long)

    # P(atom moves) per mover per sample: (B, M)
    p_stay = (src_dists[:, mover_idx] * dst_dists[:, mover_idx]).sum(dim=2)
    p_move = 1.0 - p_stay

    # P(any move): 1 - prod(1-p_move_i)
    p_any_move = 1.0 - torch.exp(torch.log((1.0 - p_move).clamp(min=1e-30)).sum(dim=1))  # (B,)

    # Pair indices: (P, 2)
    pair_idx = torch.combinations(torch.arange(M, device=device), r=2)
    pi, pj = pair_idx[:, 0], pair_idx[:, 1]
    P = pi.shape[0]

    # P(both move) per pair: (B, P)
    p_both = p_move[:, pi] * p_move[:, pj]

    # Global atom indices for pairs
    qi, qj = mover_idx[pi], mover_idx[pj]  # (P,)

    # Batched pairwise compat: flatten (B, P) -> (B*P,)
    BP = B * P
    p_compat = _batch_pairwise_full_compat(
        src_dists[:, qi].reshape(BP, C), dst_dists[:, qi].reshape(BP, C),
        src_dists[:, qj].reshape(BP, C), dst_dists[:, qj].reshape(BP, C), H, W
    ).reshape(B, P)

    # Weighted loss per pair: (B, P)
    pair_loss_vals = _pairwise_loss(p_compat, mode=loss_mode, delta=delta)
    weighted_loss = (p_both * pair_loss_vals).sum(dim=1)  # (B,)

    return p_any_move * (1.0 + lambda_r * weighted_loss)


def batched_total_cost_surrogate(init_dists, layer_dists_list, tasks, H, W,
                                  lambda_g=1.0, lambda_r=1.0, loss_mode='huber', delta=0.1):
    """Batched total surrogate cost across all layers — shared task structure.

    Args:
        init_dists: (B, N, C) initial distributions
        layer_dists_list: list of T (B, N, C) distributions
        tasks: list of T gate-lists (SHARED across batch)
        H, W: grid dims

    Returns: (B,) per-sample total costs
    """
    B = init_dists.shape[0]
    device, dtype = init_dists.device, init_dists.dtype
    total = torch.zeros(B, device=device, dtype=dtype)
    prev = init_dists

    for t, (dists_t, gates_t) in enumerate(zip(layer_dists_list, tasks)):
        gc = batched_gate_cost(gates_t, dists_t, H, W,
                               lambda_g=lambda_g, loss_mode=loss_mode, delta=delta)
        rc = batched_reconfig_cost(prev, dists_t, H, W,
                                    lambda_r=lambda_r, loss_mode=loss_mode, delta=delta)
        total = total + gc + rc
        prev = dists_t

    return total
