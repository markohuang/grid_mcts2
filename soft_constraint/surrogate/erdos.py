"""
surrogate/erdos.py — Erdős/Potts differentiable cost surrogate.

Instead of predicting χ from the conflict matrix, this formulation
explicitly constructs a coloring (group assignment) and counts its cost.

Loss = Σ_t [ χ̂_R(t) + 2·χ̂_G(t) + λ_e·(L_R_conflict(t) + L_G_conflict(t)) ]

where:
  χ̂ = number of active groups (differentiable count)
  L_conflict = Potts energy (penalizes same-group conflicts)

Variables:
  - Atom placements p_q (from model)
  - Gate directions d_g (from model)
  - Reconfig group assignments s_i^R (auxiliary, jointly optimized)
  - Gate group assignments s_g^G (auxiliary, jointly optimized)

Complexity: O(M² · (V⁴ + K)) per layer — polynomial in everything.
"""

import torch
import torch.nn.functional as F

from .feasibility import pairwise_full_compat_prob


# ─────────────────────────────────────────────────────────────────────────────
# Conflict matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_reconfig_conflict_matrix(src_dists, dst_dists, H, W, mover_indices=None):
    """Build soft conflict matrix for reconfig moves.

    A_ij = P(both move) × P(AOD incompatible | both move)

    Args:
        src_dists: (N, C) source distributions
        dst_dists: (N, C) destination distributions
        H, W: grid dims
        mover_indices: atoms to consider (default: all)

    Returns:
        A: (M, M) conflict probability matrix (symmetric, zero diagonal)
        p_move: (M,) per-atom probability of moving
        p_any_move: scalar P(at least one atom moves)
    """
    N, C = src_dists.shape
    device, dtype = src_dists.device, src_dists.dtype

    if mover_indices is None:
        mover_indices = list(range(N))
    M = len(mover_indices)

    # P(atom moves)
    p_stay = torch.stack([(src_dists[q] * dst_dists[q]).sum()
                           for q in mover_indices])
    p_move = 1.0 - p_stay
    p_any_move = 1.0 - torch.exp(torch.log(p_stay.clamp(min=1e-30)).sum())

    # Pairwise conflict matrix
    A = torch.zeros(M, M, device=device, dtype=dtype)
    for i in range(M):
        for j in range(i + 1, M):
            qi, qj = mover_indices[i], mover_indices[j]
            p_both = p_move[i] * p_move[j]

            if p_both.item() < 1e-10:
                continue

            p_compat = pairwise_full_compat_prob(
                src_dists[qi], dst_dists[qi],
                src_dists[qj], dst_dists[qj], H, W)

            A[i, j] = p_both * (1.0 - p_compat)
            A[j, i] = A[i, j]

    return A, p_move, p_any_move


def build_gate_conflict_matrix(gate_atoms, placement_dists, H, W, directions):
    """Build soft conflict matrix for gate moves under soft directions.

    For soft direction d_g ∈ (0,1), the conflict between gates i,j is a
    weighted sum over the 4 direction combinations:

    A_ij = Σ_{di,dj} w(di) w(dj) P(incompatible | di, dj)

    Args:
        gate_atoms: list of (atom_a, atom_b) per gate
        placement_dists: (N, C) distributions
        H, W: grid dims
        directions: (M,) tensor of soft directions in (0, 1)

    Returns:
        A: (M, M) conflict probability matrix
    """
    M = len(gate_atoms)
    device, dtype = placement_dists.device, placement_dists.dtype

    A = torch.zeros(M, M, device=device, dtype=dtype)

    for i in range(M):
        for j in range(i + 1, M):
            a_i, b_i = gate_atoms[i]
            a_j, b_j = gate_atoms[j]

            # Weighted sum over 4 direction combos
            conflict_ij = torch.tensor(0.0, device=device, dtype=dtype)
            for di in [0, 1]:
                for dj in [0, 1]:
                    w = ((1 - directions[i]) if di == 0 else directions[i]) * \
                        ((1 - directions[j]) if dj == 0 else directions[j])

                    # Source/dest under this direction combo
                    src_i = placement_dists[a_i if di == 0 else b_i]
                    dst_i = placement_dists[b_i if di == 0 else a_i]
                    src_j = placement_dists[a_j if dj == 0 else b_j]
                    dst_j = placement_dists[b_j if dj == 0 else a_j]

                    p_compat = pairwise_full_compat_prob(
                        src_i, dst_i, src_j, dst_j, H, W)

                    conflict_ij = conflict_ij + w * (1.0 - p_compat)

            A[i, j] = conflict_ij
            A[j, i] = conflict_ij

    return A


# ─────────────────────────────────────────────────────────────────────────────
# Potts energy (conflict penalty)
# ─────────────────────────────────────────────────────────────────────────────

def potts_energy(A, S):
    """Compute Potts conflict energy.

    L = Σ_{i<j} A_ij · (s_i · s_j)

    Penalizes conflicting moves assigned to the same group.
    Zero iff every conflicting pair is in different groups.

    Args:
        A: (M, M) conflict probability matrix
        S: (M, K) soft group assignments (each row in Δ^K)

    Returns: scalar loss (differentiable)
    """
    # s_i · s_j for all pairs: (S @ S^T)_ij = Σ_k s_ik s_jk
    same_group = S @ S.T  # (M, M)

    # Multiply by conflict matrix, sum upper triangle
    # A is symmetric with zero diagonal, so sum all and divide by 2
    return (A * same_group).sum() / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Group counting
# ─────────────────────────────────────────────────────────────────────────────

def count_active_groups(S, beta=5.0, tau=0.5):
    """Differentiable count of active groups.

    χ̂ = Σ_k σ(β · (Σ_i s_ik) - τ)

    A group k is "active" when the total mass assigned to it exceeds
    the threshold τ.

    Args:
        S: (M, K) soft group assignments
        beta: sigmoid sharpness
        tau: activation threshold

    Returns: scalar (differentiable)
    """
    group_mass = S.sum(dim=0)  # (K,) total mass per group
    return torch.sigmoid(beta * (group_mass - tau)).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Layer cost
# ─────────────────────────────────────────────────────────────────────────────

def reconfig_layer_cost(A_reconfig, S_reconfig, p_any_move,
                         lambda_e=10.0, beta=5.0, tau=0.5):
    """Reconfig cost for one layer.

    cost = p_any_move × χ̂_R + λ_e × L_conflict

    Args:
        A_reconfig: (M_R, M_R) reconfig conflict matrix
        S_reconfig: (M_R, K_R) reconfig group assignments
        p_any_move: scalar P(at least one atom moves)
        lambda_e: conflict penalty weight
        beta, tau: group counting parameters

    Returns: (cost, info_dict)
    """
    conflict = potts_energy(A_reconfig, S_reconfig)
    groups = count_active_groups(S_reconfig, beta, tau)

    cost = p_any_move * groups + lambda_e * conflict

    return cost, {
        'conflict': conflict,
        'groups': groups,
        'p_any_move': p_any_move,
    }


def gate_layer_cost(A_gate, S_gate, lambda_e=10.0, beta=5.0, tau=0.5):
    """Gate cost for one layer.

    cost = 2 × χ̂_G + λ_e × L_conflict

    Args:
        A_gate: (M_G, M_G) gate conflict matrix
        S_gate: (M_G, K_G) gate group assignments
        lambda_e: conflict penalty weight
        beta, tau: group counting parameters

    Returns: (cost, info_dict)
    """
    conflict = potts_energy(A_gate, S_gate)
    groups = count_active_groups(S_gate, beta, tau)

    cost = 2.0 * groups + lambda_e * conflict

    return cost, {
        'conflict': conflict,
        'groups': groups,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full multi-layer cost
# ─────────────────────────────────────────────────────────────────────────────

class ErdosSurrogate:
    """Manages auxiliary variables and computes the full Erdős loss.

    The model provides: atom placements, gate directions.
    This class manages: group assignment logits (auxiliary variables).

    Usage:
        surrogate = ErdosSurrogate(tasks, N, H, W, K_R=4, K_G=4)
        loss, info = surrogate.compute_loss(initial_dists, layer_dists, directions)
        loss.backward()
        # Gradient flows to layer_dists, directions, AND surrogate.parameters()
    """

    def __init__(self, tasks, N, H, W, K_R=4, K_G=4,
                 lambda_e=10.0, beta=5.0, tau=0.5):
        """
        Args:
            tasks: list of gate lists per layer
            N: total number of atoms
            H, W: grid dims
            K_R: max reconfig groups
            K_G: max gate groups
            lambda_e: conflict penalty weight
            beta: sigmoid sharpness for group counting
            tau: group activation threshold
        """
        self.tasks = tasks
        self.N = N
        self.H = H
        self.W = W
        self.K_R = K_R
        self.K_G = K_G
        self.lambda_e = lambda_e
        self.beta = beta
        self.tau = tau

        self.T = len(tasks)

        # Determine mover counts per layer
        self.relevant = []
        for t in range(self.T):
            atoms = sorted(set(a for pair in tasks[t] for a in pair))
            self.relevant.append(atoms)

        # Initialize auxiliary logits
        self._init_aux_logits()

    def _init_aux_logits(self):
        """Initialize group assignment logits.

        Key insight: initialize with FEWER groups than needed, forcing
        the optimizer to find compatible placements. If we start with
        each move in its own group, the conflict energy is trivially zero
        and there's no gradient signal to improve placements.

        Strategy: initialize all moves in group 0 (1 group). This has
        maximum conflict energy but minimum group count. The optimizer
        must then either:
        - Adjust placements to make everything compatible (ideal), or
        - Split into more groups (fallback)
        """
        self.reconfig_logits = []
        self.gate_logits = []

        for t in range(self.T):
            M_R = len(self.relevant[t])
            M_G = len(self.tasks[t])

            # Initialize all moves in group 0
            rl = torch.zeros(M_R, self.K_R)
            rl.data[:, 0] += 2.0  # bias toward group 0

            gl = torch.zeros(M_G, self.K_G)
            gl.data[:, 0] += 2.0  # bias toward group 0

            rl.requires_grad_(True)
            gl.requires_grad_(True)

            self.reconfig_logits.append(rl)
            self.gate_logits.append(gl)

    def parameters(self):
        """Return all auxiliary parameters for the optimizer."""
        params = []
        for t in range(self.T):
            params.append(self.reconfig_logits[t])
            params.append(self.gate_logits[t])
        return params

    def compute_loss(self, initial_dists, layer_dists, gate_directions=None):
        """Compute the full Erdős loss across all layers.

        Args:
            initial_dists: (N, C) initial position distributions
            layer_dists: list of (N, C) per-layer distributions
            gate_directions: list of (M_t,) tensors of soft directions per layer
                             If None, uses default (all forward = 0.5)

        Returns:
            total_loss: scalar (differentiable w.r.t. all inputs + aux params)
            layer_infos: list of info dicts per layer
        """
        device = initial_dists.device
        dtype = initial_dists.dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        layer_infos = []
        prev_dists = initial_dists

        for t in range(self.T):
            curr_dists = layer_dists[t]
            gates = self.tasks[t]
            M_G = len(gates)

            # Soft group assignments
            S_R = F.softmax(self.reconfig_logits[t].to(device), dim=-1)
            S_G = F.softmax(self.gate_logits[t].to(device), dim=-1)

            # Gate directions
            if gate_directions is not None:
                dirs = gate_directions[t]
            else:
                dirs = torch.full((M_G,), 0.5, device=device, dtype=dtype)

            # Build conflict matrices
            A_R, p_move, p_any = build_reconfig_conflict_matrix(
                prev_dists, curr_dists, self.H, self.W, self.relevant[t])

            A_G = build_gate_conflict_matrix(
                gates, curr_dists, self.H, self.W, dirs)

            # Layer costs
            rc, r_info = reconfig_layer_cost(
                A_R, S_R, p_any, self.lambda_e, self.beta, self.tau)

            gc, g_info = gate_layer_cost(
                A_G, S_G, self.lambda_e, self.beta, self.tau)

            layer_cost = rc + gc
            total_loss = total_loss + layer_cost

            layer_infos.append({
                'reconfig': r_info,
                'gate': g_info,
                'A_R': A_R.detach(),
                'A_G': A_G.detach(),
                'layer_cost': layer_cost.item(),
            })

            prev_dists = curr_dists

        return total_loss, layer_infos