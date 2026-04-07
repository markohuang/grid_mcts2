"""
surrogate/partition.py — Partition-function differentiable chromatic number.

Z̃(A, q) = Σ_{σ ∈ [q]^M} ∏_{i<j : σ_i=σ_j} (1 - A_ij)

Counts (softly) valid q-colorings. At discrete A, equals the chromatic polynomial.

χ̂(A) = 1 + Σ_{q=1}^{q_max} σ(-α · log Z̃(A, q))

No auxiliary variables. Exact at discrete A. Unified for reconfig and gate.
Gradient ∂χ̂/∂A_ij captures "how much does reducing conflict (i,j) help."
"""

import torch
import torch.nn.functional as F
from .erdos import build_reconfig_conflict_matrix, build_gate_conflict_matrix

_COLORING_CACHE = {}


def _get_colorings(M, q, device='cpu'):
    key = (M, q)
    if key not in _COLORING_CACHE:
        total = q ** M
        idx = torch.arange(total)
        c = torch.empty(total, M, dtype=torch.int8)
        for d in range(M):
            c[:, d] = ((idx // (q ** d)) % q).to(torch.int8)
        _COLORING_CACHE[key] = c
    return _COLORING_CACHE[key].to(device)


def log_Z_tilde(A, q):
    """log Z̃(A, q) in log-space. Differentiable w.r.t. A."""
    M = A.shape[0]
    if M == 0:
        return torch.tensor(0.0, device=A.device, dtype=A.dtype)
    device, dtype = A.device, A.dtype
    colorings = _get_colorings(M, q, device)
    log_1mA = torch.log((1 - A).clamp(min=1e-30))
    pairs_i, pairs_j = torch.triu_indices(M, M, offset=1, device=device)
    num_c = colorings.shape[0]
    log_w = torch.zeros(num_c, device=device, dtype=dtype)
    for p in range(pairs_i.shape[0]):
        i, j = pairs_i[p].item(), pairs_j[p].item()
        same = (colorings[:, i] == colorings[:, j]).to(dtype)
        log_w = log_w + same * log_1mA[i, j]
    return torch.logsumexp(log_w, dim=0)


def chromatic_estimate(A, q_max=5, alpha=10.0):
    """χ̂(A) = 1 + Σ_{q=1}^{q_max} σ(-α · log Z̃(A, q))."""
    chi_hat = torch.ones(1, device=A.device, dtype=A.dtype)
    for q in range(1, q_max + 1):
        logZ = log_Z_tilde(A, q)
        chi_hat = chi_hat + torch.sigmoid(-alpha * logZ)
    return chi_hat.squeeze()


def chromatic_cost(A, q_max=5):
    """Non-saturating cost: -Σ_q log(Z̃(q)+1) weighted by 1/q².
    Lower = more colorable. Gradient doesn't vanish at extremes.
    Weight 1/q² prevents domination by large-q terms."""
    cost = torch.tensor(0.0, device=A.device, dtype=A.dtype)
    for q in range(1, q_max + 1):
        logZ = log_Z_tilde(A, q)
        # log(Z̃+1) = log(exp(logZ) + 1) = softplus(logZ)
        cost = cost - F.softplus(logZ) / (q * q)
    return cost


class PartitionSurrogate:
    """Full multi-layer cost using partition-function χ̂.

    cost(t) = p_any_move · χ̂_R(A_R) + 2 · χ̂_G(A_G)

    No auxiliary parameters — gradient flows directly to placements.
    """
    def __init__(self, tasks, N, H, W, q_max=5, alpha=10.0, mode='sigmoid'):
        """mode: 'sigmoid' (χ̂ via sigmoid threshold) or 'cost' (non-saturating -log(Z̃+1))"""
        self.tasks = tasks
        self.N, self.H, self.W = N, H, W
        self.q_max, self.alpha, self.mode = q_max, alpha, mode
        self.T = len(tasks)
        self.relevant = [sorted(set(a for pair in t for a in pair)) for t in tasks]

    def compute_loss(self, initial_dists, layer_dists, gate_directions=None):
        device, dtype = initial_dists.device, initial_dists.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        layer_infos = []
        prev_dists = initial_dists

        for t in range(self.T):
            curr_dists = layer_dists[t]
            gates = self.tasks[t]
            M_G = len(gates)
            dirs = gate_directions[t] if gate_directions is not None else \
                torch.full((M_G,), 0.5, device=device, dtype=dtype)

            A_R, p_move, p_any = build_reconfig_conflict_matrix(
                prev_dists, curr_dists, self.H, self.W, self.relevant[t])
            A_G = build_gate_conflict_matrix(
                gates, curr_dists, self.H, self.W, dirs)

            if self.mode == 'sigmoid':
                chi_R = chromatic_estimate(A_R, self.q_max, self.alpha)
                chi_G = chromatic_estimate(A_G, self.q_max, self.alpha)
                layer_cost = p_any * chi_R + 2.0 * chi_G
            else:  # 'cost' mode
                cost_R = chromatic_cost(A_R, self.q_max)
                cost_G = chromatic_cost(A_G, self.q_max)
                chi_R = chromatic_estimate(A_R, self.q_max, self.alpha)
                chi_G = chromatic_estimate(A_G, self.q_max, self.alpha)
                layer_cost = p_any * cost_R + 2.0 * cost_G
            total_loss = total_loss + layer_cost

            layer_infos.append({
                'chi_R': chi_R.item(), 'chi_G': chi_G.item(),
                'p_any_move': p_any.item(),
                'reconfig_cost': (p_any * chi_R).item(),
                'gate_cost': (2.0 * chi_G).item(),
                'A_R': A_R.detach(), 'A_G': A_G.detach(),
                'layer_cost': layer_cost.item(),
            })
            prev_dists = curr_dists

        return total_loss, layer_infos

    def parameters(self):
        return []  # No auxiliary variables
