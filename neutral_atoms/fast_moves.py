"""Numba-accelerated graph coloring for parallel move grouping.

Drop-in replacement for the hot path in moves.py: count_groups() and parallel_groups().
All internal computation uses numpy arrays + numba JIT. Torch tensors are converted
at the boundary (zero-copy via .numpy() for contiguous CPU tensors).
"""
import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def _is_parallel_np(moves, N):
    """Pairwise parallelizability check. Returns (N, N) bool array."""
    result = np.ones((N, N), dtype=numba.boolean)
    for i in range(N):
        for j in range(i + 1, N):
            v1_x = moves[i, 0] - moves[j, 0]
            v1_y = moves[i, 1] - moves[j, 1]
            v2_x = moves[i, 2] - moves[j, 2]
            v2_y = moves[i, 3] - moves[j, 3]
            # horizontal: no crossing in x
            both_same_col = (v1_x == 0) and (v2_x == 0)
            either_same_col = (v1_x == 0) or (v2_x == 0)
            if both_same_col:
                h_ok = True
            elif either_same_col:
                h_ok = False
            else:
                h_ok = (v1_x > 0) == (v2_x > 0)
            # vertical: no crossing in y
            both_same_row = (v1_y == 0) and (v2_y == 0)
            either_same_row = (v1_y == 0) or (v2_y == 0)
            if both_same_row:
                v_ok = True
            elif either_same_row:
                v_ok = False
            else:
                v_ok = (v1_y > 0) == (v2_y > 0)
            # collision: same destination
            no_collision = not (v2_x == 0 and v2_y == 0)
            ok = h_ok and v_ok and no_collision
            result[i, j] = ok
            result[j, i] = ok
    return result


@numba.jit(nopython=True, cache=True)
def _greedy_color(conflicts, N):
    """Greedy graph coloring. Returns (N,) int array of color assignments."""
    colors = np.full(N, -1, dtype=np.int64)
    # Compute degrees for ordering
    degrees = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for j in range(N):
            if conflicts[i, j]:
                degrees[i] += 1
    order = np.argsort(-degrees)
    for oi in range(N):
        idx = order[oi]
        # Find max color among conflicting neighbors
        max_used = -1
        for j in range(N):
            if conflicts[idx, j] and colors[j] >= 0:
                if colors[j] > max_used:
                    max_used = colors[j]
        if max_used < 0:
            colors[idx] = 0
        else:
            used = np.zeros(max_used + 2, dtype=numba.boolean)
            for j in range(N):
                if conflicts[idx, j] and colors[j] >= 0:
                    used[colors[j]] = True
            for c in range(max_used + 2):
                if not used[c]:
                    colors[idx] = c
                    break
    return colors


@numba.jit(nopython=True, cache=True)
def _canonicalize(moves, can_parallel_all, N):
    """Greedy canonicalization of gate moves. Returns (N,) bool: which to flip."""
    is_flipped = np.zeros(N, dtype=numba.boolean)
    # First move: prefer +dx, tie-break +dy
    dx0 = moves[0, 2] - moves[0, 0]
    dy0 = moves[0, 3] - moves[0, 1]
    if dx0 < 0 or (dx0 == 0 and dy0 < 0):
        is_flipped[0] = True
    # Greedy for remaining
    for i in range(1, N):
        fwd_conflicts = 0
        rev_conflicts = 0
        for j in range(i):
            if is_flipped[j]:
                if not can_parallel_all[i, N + j]:
                    fwd_conflicts += 1
                if not can_parallel_all[N + i, N + j]:
                    rev_conflicts += 1
            else:
                if not can_parallel_all[i, j]:
                    fwd_conflicts += 1
                if not can_parallel_all[N + i, j]:
                    rev_conflicts += 1
        if rev_conflicts < fwd_conflicts:
            is_flipped[i] = True
    return is_flipped


@numba.jit(nopython=True, cache=True)
def _count_groups_impl(moves_np, canonicalize):
    """Full pipeline: optional canonicalize → adjacency → greedy color → count."""
    N = moves_np.shape[0]
    if N == 0:
        return 0
    if N == 1:
        return 1
    if canonicalize:
        flipped = np.empty((N, 4), dtype=moves_np.dtype)
        for i in range(N):
            flipped[i, 0] = moves_np[i, 2]
            flipped[i, 1] = moves_np[i, 3]
            flipped[i, 2] = moves_np[i, 0]
            flipped[i, 3] = moves_np[i, 1]
        all_moves = np.empty((2 * N, 4), dtype=moves_np.dtype)
        all_moves[:N] = moves_np
        all_moves[N:] = flipped
        all_parallel = _is_parallel_np(all_moves, 2 * N)
        is_flipped = _canonicalize(moves_np, all_parallel, N)
        result = np.empty((N, 4), dtype=moves_np.dtype)
        for i in range(N):
            if is_flipped[i]:
                result[i] = flipped[i]
            else:
                result[i] = moves_np[i]
        moves_np = result
    can_parallel = _is_parallel_np(moves_np, N)
    conflicts = ~can_parallel
    for i in range(N):
        conflicts[i, i] = False
    colors = _greedy_color(conflicts, N)
    max_color = np.int64(0)
    for i in range(N):
        if colors[i] > max_color:
            max_color = colors[i]
    return max_color + 1


@numba.jit(nopython=True, cache=True)
def _parallel_groups_impl(moves_np, canonicalize):
    """Full pipeline returning group assignments instead of just count."""
    N = moves_np.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)
    if N == 1:
        return np.zeros(1, dtype=np.int64)
    if canonicalize:
        flipped = np.empty((N, 4), dtype=moves_np.dtype)
        for i in range(N):
            flipped[i, 0] = moves_np[i, 2]
            flipped[i, 1] = moves_np[i, 3]
            flipped[i, 2] = moves_np[i, 0]
            flipped[i, 3] = moves_np[i, 1]
        all_moves = np.empty((2 * N, 4), dtype=moves_np.dtype)
        all_moves[:N] = moves_np
        all_moves[N:] = flipped
        all_parallel = _is_parallel_np(all_moves, 2 * N)
        is_flipped = _canonicalize(moves_np, all_parallel, N)
        result = np.empty((N, 4), dtype=moves_np.dtype)
        for i in range(N):
            if is_flipped[i]:
                result[i] = flipped[i]
            else:
                result[i] = moves_np[i]
        moves_np = result
    can_parallel = _is_parallel_np(moves_np, N)
    conflicts = ~can_parallel
    for i in range(N):
        conflicts[i, i] = False
    return _greedy_color(conflicts, N)


def count_groups_fast(moves_torch, canonicalize=False):
    """Drop-in replacement for moves.count_groups using numba."""
    if len(moves_torch) == 0:
        return 0
    return int(_count_groups_impl(moves_torch.numpy(), canonicalize))


def parallel_groups_fast(moves_torch, canonicalize=False):
    """Drop-in replacement for moves.parallel_groups using numba."""
    import torch
    return torch.from_numpy(_parallel_groups_impl(moves_torch.numpy(), canonicalize))


def group_entropy_fast(moves_torch, canonicalize=False):
    """Entropy of the group-size distribution. 0 = all gates in one group (best); max = uniform spread."""
    import torch
    import math
    if len(moves_torch) == 0:
        return 0.0
    groups = torch.from_numpy(_parallel_groups_impl(moves_torch.numpy(), canonicalize))
    sizes = torch.bincount(groups).float()
    probs = sizes / sizes.sum()
    return -(probs * probs.log()).sum().item()


def warmup():
    """Call once at startup to trigger JIT compilation."""
    dummy = np.zeros((2, 4), dtype=np.int64)
    dummy[0] = [0, 0, 1, 0]
    dummy[1] = [0, 1, 1, 1]
    _count_groups_impl(dummy, True)
    _count_groups_impl(dummy, False)
    _parallel_groups_impl(dummy, True)
