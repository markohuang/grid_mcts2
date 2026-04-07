import torch
from neutral_atoms.moves import is_parallel_executable_batch, count_groups
from neutral_atoms.tasks import gates_to_moves


def _top_k_candidates(logits, top_k):
    """Top-K destinations with renormalized probabilities.
    Args:
        logits: (N, board_size)
    Returns:
        probs: (N, K) renormalized softmax probabilities (gradient-carrying)
        cells: (N, K) flat cell indices
    """
    probs_full = logits.softmax(dim=-1)
    top_probs, top_cells = probs_full.topk(top_k, dim=-1)
    probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    return probs, top_cells


def _expected_conflicts(moves, probs, group_ids):
    """Expected pairwise conflicts between candidate moves, weighted by probability.
    Conflict values are discrete (0/1) and detached — gradients flow only through probs.
    Args:
        moves: (M, 4) [src_row, src_col, dst_row, dst_col] candidate moves
        probs: (M,) probability per candidate (carries gradient)
        group_ids: (M,) which entity (qubit or gate) each candidate belongs to
    Returns:
        scalar — expected number of pairwise conflicts
    """
    if moves.shape[0] <= 1:
        return probs.new_tensor(0.0)
    # Compatibility check on CPU (greedy_color internals use Python loops),
    # but is_parallel_executable_batch is pure tensor ops — fine on any device.
    compat = is_parallel_executable_batch(moves.detach())
    conflict = (1.0 - compat.float()).to(probs.device)  # no grad through geometry
    weights = probs[:, None] * probs[None, :]
    same_group = group_ids[:, None] == group_ids[None, :]
    weights = weights * (~same_group).float()
    return (weights * conflict).triu(1).sum()


def _soft_reconfig(src_positions, probs, cells, board_w):
    """Expected pairwise conflict count for reconfiguration moves.
    Args:
        src_positions: (N, 2) current [row, col] per qubit
        probs: (N, K) renormalized probabilities
        cells: (N, K) flat destination indices
    """
    N, K = probs.shape
    if N <= 1:
        return probs.new_tensor(0.0)
    dst_rows = cells // board_w
    dst_cols = cells % board_w
    src_rows = src_positions[:, 0:1].expand(-1, K)
    src_cols = src_positions[:, 1:2].expand(-1, K)
    moves = torch.stack([
        src_rows.reshape(-1), src_cols.reshape(-1),
        dst_rows.reshape(-1), dst_cols.reshape(-1),
    ], dim=1)  # (N*K, 4)
    qubit_ids = torch.arange(N, device=probs.device).repeat_interleave(K)
    return _expected_conflicts(moves, probs.reshape(-1), qubit_ids)


def _soft_gate(probs, cells, local_gate_pairs, board_w):
    """Expected pairwise conflict count for gate execution moves.
    Gate move for pair (q1, q2): from q1's destination to q2's destination.
    Each gate has K^2 candidate moves (K options for each qubit).
    Args:
        probs: (N, K) renormalized probabilities
        cells: (N, K) flat destination indices
        local_gate_pairs: list of (local_q1, local_q2) with indices into 0..N-1
    """
    G = len(local_gate_pairs)
    if G <= 1:
        return probs.new_tensor(0.0)
    K = probs.shape[1]
    dst_rows = cells // board_w  # (N, K)
    dst_cols = cells % board_w
    q1_idx = torch.tensor([q1 for q1, q2 in local_gate_pairs], device=probs.device)
    q2_idx = torch.tensor([q2 for q1, q2 in local_gate_pairs], device=probs.device)
    # Gate move source = q1 destination, gate move target = q2 destination
    # Shape: each (G, K_q1, K_q2) -> flatten to (G*K*K,)
    q1_r = dst_rows[q1_idx][:, :, None].expand(-1, -1, K)
    q1_c = dst_cols[q1_idx][:, :, None].expand(-1, -1, K)
    q2_r = dst_rows[q2_idx][:, None, :].expand(-1, K, -1)
    q2_c = dst_cols[q2_idx][:, None, :].expand(-1, K, -1)
    gate_moves = torch.stack([
        q1_r.reshape(-1), q1_c.reshape(-1),
        q2_r.reshape(-1), q2_c.reshape(-1),
    ], dim=1)  # (G*K*K, 4)
    # Joint probability: P(q1->k1) * P(q2->k2) for each gate
    q1_p = probs[q1_idx][:, :, None].expand(-1, -1, K)
    q2_p = probs[q2_idx][:, None, :].expand(-1, K, -1)
    gate_probs = (q1_p * q2_p).reshape(-1)  # (G*K*K,)
    gate_ids = torch.arange(G, device=probs.device).repeat_interleave(K * K)
    return _expected_conflicts(gate_moves, gate_probs, gate_ids)


def _soft_collision(probs, cells):
    """Expected number of qubit-pair collisions (same destination cell).
    Args:
        probs: (N, K) renormalized probabilities
        cells: (N, K) flat destination indices
    """
    N, K = probs.shape
    if N <= 1:
        return probs.new_tensor(0.0)
    flat_probs = probs.reshape(-1)
    flat_cells = cells.reshape(-1)
    same_dst = (flat_cells[:, None] == flat_cells[None, :]).float()
    weights = flat_probs[:, None] * flat_probs[None, :]
    qubit_ids = torch.arange(N, device=probs.device).repeat_interleave(K)
    same_qubit = (qubit_ids[:, None] == qubit_ids[None, :]).float()
    weights = weights * (1.0 - same_qubit)
    return (weights * same_dst).triu(1).sum()


def soft_layer_cost(atom_positions, logits, relevant_qubits, gate_pairs,
                    board_w, top_k=3, collision_weight=10.0, reconfig_weight=1.0):
    """Differentiable proxy for layer cost (reconfig + 2*gate + collision penalty).
    Gradients flow through logits via probability weights on precomputed conflict tables.

    Raw pairwise conflicts scale as O(N^2) for reconfig and O(G^2) for gates, but the
    actual discrete cost (chromatic number) grows sublinearly. To prevent reconfig from
    dominating and collapsing the model to no-ops, both components are normalized by
    their number of entity pairs, putting them on a "conflict probability per pair" scale.

    Args:
        atom_positions: (Q, 2) current positions of ALL qubits
        logits: (N_rel, board_size) destination logits for relevant qubits only
        relevant_qubits: list[int] of length N_rel — global qubit indices
        gate_pairs: list of (q1, q2) with GLOBAL qubit indices
        board_w: int — board width
        top_k: int — candidates per qubit
        collision_weight: float — penalty multiplier for same-cell collisions
        reconfig_weight: float — relative weight of reconfig vs gate cost
    Returns:
        scalar (differentiable w.r.t. logits)
    """
    src = atom_positions[relevant_qubits]  # (N_rel, 2)
    q2local = {q: i for i, q in enumerate(relevant_qubits)}
    local_gates = [(q2local[q1], q2local[q2]) for q1, q2 in gate_pairs]
    probs, cells = _top_k_candidates(logits, top_k)
    N = len(relevant_qubits)
    G = len(gate_pairs)
    # Normalize by number of entity pairs so reconfig and gate are on the same scale.
    # Without this, reconfig (C(N,2) pairs) dominates gate (C(G,2) pairs) ~4-5x,
    # causing the model to collapse to no-ops.
    reconfig_pairs = max(N * (N - 1) // 2, 1)
    gate_pairs_count = max(G * (G - 1) // 2, 1)
    reconfig = _soft_reconfig(src, probs, cells, board_w) / reconfig_pairs
    gate = _soft_gate(probs, cells, local_gates, board_w) / gate_pairs_count
    collision = _soft_collision(probs, cells) / reconfig_pairs
    return reconfig_weight * reconfig + 2 * gate + collision_weight * collision


def hard_layer_cost(atom_positions, logits, relevant_qubits, gate_pairs, board_w):
    """Argmax placement -> actual discrete cost. For evaluation only (not differentiable).
    All computation on CPU since count_groups uses greedy coloring with Python loops."""
    atom_positions = atom_positions.cpu()
    logits = logits.cpu()
    dests = logits.argmax(dim=-1)  # (N_rel,)
    dst_rows, dst_cols = dests // board_w, dests % board_w
    src = atom_positions[relevant_qubits]  # (N_rel, 2)
    moved = (dst_rows != src[:, 0]) | (dst_cols != src[:, 1])
    reconfig_groups = 0
    if moved.any():
        reconfig_moves = torch.stack([
            src[moved, 0], src[moved, 1], dst_rows[moved], dst_cols[moved],
        ], dim=1)
        reconfig_groups = count_groups(reconfig_moves, canonicalize=False)
    new_positions = atom_positions.clone()
    for i, q in enumerate(relevant_qubits):
        new_positions[q] = torch.tensor([dst_rows[i], dst_cols[i]])
    gate_moves = gates_to_moves(gate_pairs, new_positions)
    gate_groups = count_groups(gate_moves, canonicalize=True) if len(gate_moves) > 0 else 0
    return reconfig_groups + 2 * gate_groups
