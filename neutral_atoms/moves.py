import torch
from .types import Moves

def is_parallel_executable_batch(moves: Moves) -> torch.Tensor:
    """Returns (N, N) bool tensor, True = can run in parallel."""
    if len(moves) == 0:
        return torch.empty(0, 0, dtype=torch.bool)
    frm, to = moves[:, :2], moves[:, 2:]
    v1 = frm.unsqueeze(0) - frm.unsqueeze(1)  # (N, N, 2)
    v2 = to.unsqueeze(0) - to.unsqueeze(1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    # horizontal: no crossing in x
    both_same_col = (v1_x == 0) & (v2_x == 0)
    same_x_dir = torch.sign(v1_x) == torch.sign(v2_x)
    either_same_col = (v1_x == 0) | (v2_x == 0)
    h_ok = both_same_col | (~either_same_col & same_x_dir)
    # vertical: no crossing in y
    both_same_row = (v1_y == 0) & (v2_y == 0)
    same_y_dir = torch.sign(v1_y) == torch.sign(v2_y)
    either_same_row = (v1_y == 0) | (v2_y == 0)
    v_ok = both_same_row | (~either_same_row & same_y_dir)
    # collision: same destination
    no_collision = ~((v2_x == 0) & (v2_y == 0))
    result = h_ok & v_ok & no_collision
    result.fill_diagonal_(True)
    return result

def greedy_color_from_adjacency(can_parallel: torch.Tensor) -> torch.Tensor:
    """Returns (N,) group assignments."""
    N = can_parallel.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long)
    conflicts = ~can_parallel
    conflicts.fill_diagonal_(False)
    colors = torch.full((N,), -1, dtype=torch.long)
    order = torch.argsort(conflicts.sum(dim=1), descending=True)
    for idx in order.tolist():
        neighbor_colors = colors[conflicts[idx]]
        neighbor_colors = neighbor_colors[neighbor_colors >= 0]
        if len(neighbor_colors) == 0:
            colors[idx] = 0
        else:
            used = torch.zeros(neighbor_colors.max().item() + 2, dtype=torch.bool)
            used[neighbor_colors] = True
            colors[idx] = (~used).nonzero()[0].item()
    return colors

def canonicalize_moves(moves: Moves) -> Moves:
    """
    Canonicalize gate moves to minimize pairwise conflicts by choosing consistent directions.
    For two-qubit gates where either atom can move to the other, this picks directions
    that maximize parallelizability. Uses greedy assignment.
    Note: Greedy is ~0.2% suboptimal for n=3, ~1% for n=4, but fast.
    """
    n = moves.shape[0]
    if n <= 1:
        return moves
    result = moves.clone()
    flipped = torch.stack([moves[:, 2], moves[:, 3], moves[:, 0], moves[:, 1]], dim=1)
    # Precompute all pairwise checks for both directions (2N, 2N)
    all_moves = torch.vstack([moves, flipped])
    all_parallel = is_parallel_executable_batch(all_moves)
    # Extract blocks: orig-orig, orig-flip, flip-orig, flip-flip
    oo, of = all_parallel[:n, :n], all_parallel[:n, n:]
    fo, ff = all_parallel[n:, :n], all_parallel[n:, n:]
    # Track which moves are flipped
    is_flipped = torch.zeros(n, dtype=torch.bool)
    # Canonicalize first move: prefer +dx, tie-break +dy
    dx0, dy0 = result[0, 2] - result[0, 0], result[0, 3] - result[0, 1]
    if (dx0 < 0) or (dx0 == 0 and dy0 < 0):
        is_flipped[0] = True
        result[0] = flipped[0]
    # Greedily canonicalize remaining
    for i in range(1, n):
        fwd_conflicts, rev_conflicts = 0, 0
        for j in range(i):
            if is_flipped[j]:
                fwd_conflicts += 0 if fo[i, j] else 1
                rev_conflicts += 0 if ff[i, j] else 1
            else:
                fwd_conflicts += 0 if oo[i, j] else 1
                rev_conflicts += 0 if of[i, j] else 1
        if rev_conflicts < fwd_conflicts:
            is_flipped[i] = True
            result[i] = flipped[i]
    return result

def parallel_groups(moves: Moves, canonicalize: bool = False) -> torch.Tensor:
    """
    Returns (N,) group assignments.
    Args:
        moves: (N, 4) tensor of [from_x, from_y, to_x, to_y]
        canonicalize: True for gate moves (direction flexible), False for reconfig
    """
    if len(moves) == 0:
        return torch.empty(0, dtype=torch.long)
    if len(moves) == 1:
        return torch.zeros(1, dtype=torch.long)
    if canonicalize:
        moves = canonicalize_moves(moves)
    return greedy_color_from_adjacency(is_parallel_executable_batch(moves))

def count_groups(moves: Moves, canonicalize: bool = False) -> int:
    if len(moves) == 0:
        return 0
    return parallel_groups(moves, canonicalize).max().item() + 1

def group_sizes(moves: Moves, canonicalize: bool = False) -> torch.Tensor:
    if len(moves) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.bincount(parallel_groups(moves, canonicalize))