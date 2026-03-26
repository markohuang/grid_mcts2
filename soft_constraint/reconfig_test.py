"""
Reconfig cost surrogate based on pairwise ordering inversions.

Theory:
  - AOD compatibility ≈ coordinate-wise monotonicity of the src→dst mapping
  - χ_reconfig = max(LDS_col, LDS_row) where LDS = Longest Decreasing Subsequence
  - LDS = 1 ⟺ zero inversions ⟺ all moves parallel-compatible ⟺ χ = 1
  - Soft inversion count I_col, I_row: differentiable, exact (no top-K), O(N²W⁴)

This script:
  1. Validates the axis-decomposition theory (does max(χ_col, χ_row) = χ_true?)
  2. Tests inversion count as a surrogate (monotonic correlation with χ?)
  3. Tests gradient quality (do gradients push toward χ=1?)
  4. Multi-layer planning test (can we jointly optimize gate + reconfig cost?)
"""

import torch
import torch.nn.functional as F
import time
import math
import random
from itertools import product as iterproduct
from collections import Counter

torch.set_default_dtype(torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# AOD primitives
# ─────────────────────────────────────────────────────────────────────────────

def aod_compatible(mi, mj):
    """Check AOD compatibility. Inputs: (..., 4) [sr, sc, dr, dc]."""
    sr_i, sc_i, dr_i, dc_i = mi[...,0], mi[...,1], mi[...,2], mi[...,3]
    sr_j, sc_j, dr_j, dc_j = mj[...,0], mj[...,1], mj[...,2], mj[...,3]
    dcs, dcd = sc_i - sc_j, dc_i - dc_j
    h_ok = ((dcs==0)&(dcd==0)) | (~((dcs==0)|(dcd==0)) & (torch.sign(dcs)==torch.sign(dcd)))
    drs, drd = sr_i - sr_j, dr_i - dr_j
    v_ok = ((drs==0)&(drd==0)) | (~((drs==0)|(drd==0)) & (torch.sign(drs)==torch.sign(drd)))
    no_col = ~((drd==0)&(dcd==0))
    return h_ok & v_ok & no_col


def greedy_chromatic(moves):
    """Greedy graph coloring on AOD conflict graph. Returns χ."""
    n = moves.shape[0]
    if n == 0: return 0
    if n == 1: return 1
    conflict = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i+1, n):
            if not aod_compatible(moves[i], moves[j]):
                conflict[i,j] = conflict[j,i] = True
    colors = [-1]*n
    for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
        used = {colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
        c = 0
        while c in used: c += 1
        colors[idx] = c
    return max(colors) + 1


def axis_chromatic(moves, axis):
    """Chromatic number restricted to one axis (col=0, row=1).
    This equals the LDS length of the induced permutation on that axis."""
    n = moves.shape[0]
    if n <= 1: return 1 if n == 1 else 0

    # For column axis: check sign(sc_i - sc_j) vs sign(dc_i - dc_j)
    # For row axis: check sign(sr_i - sr_j) vs sign(dr_i - dr_j)
    if axis == 0:  # column
        s_vals = moves[:, 1]  # source col
        d_vals = moves[:, 3]  # dest col
    else:  # row
        s_vals = moves[:, 0]  # source row
        d_vals = moves[:, 2]  # dest row

    # Build conflict graph for this axis only
    conflict = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i+1, n):
            ds = s_vals[i] - s_vals[j]
            dd = d_vals[i] - d_vals[j]
            if ds == 0 and dd == 0:
                ok = True
            elif ds == 0 or dd == 0:
                ok = False  # one shares, other doesn't
            else:
                ok = (ds > 0) == (dd > 0)  # same direction
            if not ok:
                conflict[i,j] = conflict[j,i] = True

    colors = [-1]*n
    for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
        used = {colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
        c = 0
        while c in used: c += 1
        colors[idx] = c
    return max(colors) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable inversion count
# ─────────────────────────────────────────────────────────────────────────────

def soft_inversion_count_axis(src_dists, dst_dists, H, W, axis='col'):
    """Compute expected number of pairwise inversions on one axis.

    Args:
        src_dists: (N, C) source placement distributions
        dst_dists: (N, C) destination placement distributions
        H, W: grid dimensions
        axis: 'col' or 'row'

    Returns: scalar, differentiable expected inversion count
    """
    N, C = src_dists.shape
    device = src_dists.device
    dtype = src_dists.dtype

    cells = torch.arange(C, device=device)
    if axis == 'col':
        coords = (cells % W).to(dtype)
        n_coords = W
    else:
        coords = (cells // W).to(dtype)
        n_coords = H

    # Compute marginal distributions over coordinate values
    # src_marginal[i, v] = P(atom i's source coordinate = v)
    src_marg = torch.zeros(N, n_coords, device=device, dtype=dtype)
    dst_marg = torch.zeros(N, n_coords, device=device, dtype=dtype)
    for v in range(n_coords):
        mask = (coords == v).to(dtype)  # (C,)
        src_marg[:, v] = (src_dists * mask[None, :]).sum(dim=1)
        dst_marg[:, v] = (dst_dists * mask[None, :]).sum(dim=1)

    # For each pair (i, j), compute P(inversion)
    # Inversion: sign(s_i - s_j) ≠ sign(d_i - d_j), excluding cases where either diff is 0
    #
    # P(inv_ij) = P(s_i > s_j, d_i < d_j) + P(s_i < s_j, d_i > d_j)
    #           + P(s_i = s_j, d_i ≠ d_j) + P(s_i ≠ s_j, d_i = d_j)
    #
    # Actually, the full AOD condition for one axis is:
    #   ok = (both_same AND both_same_dst) OR (neither_same AND same_direction)
    #   conflict = NOT ok
    #   conflict = (one_same_not_other) OR (neither_same AND different_direction)

    total_inv = torch.tensor(0.0, device=device, dtype=dtype)

    for i in range(N):
        for j in range(i+1, N):
            # P(s_i = v1, s_j = v2) for all v1, v2
            s_joint = src_marg[i, :, None] * src_marg[j, None, :]  # (V, V)
            d_joint = dst_marg[i, :, None] * dst_marg[j, None, :]  # (V, V)

            V = n_coords
            v1 = torch.arange(V, device=device, dtype=dtype)
            v2 = torch.arange(V, device=device, dtype=dtype)

            ds = v1[:, None] - v2[None, :]  # (V, V) source diffs
            dd = v1[:, None] - v2[None, :]  # (V, V) dest diffs

            # For each (sv1, sv2, dv1, dv2), check if it's a conflict
            # This is O(V^4) but V is small (5 for 5x5 board)
            # s_joint[sv1, sv2] * d_joint[dv1, dv2] * conflict(sv1-sv2, dv1-dv2)

            ds_4d = v1[None, :, None, None] - v2[None, None, :, None]  # src diff: (1, V, V, 1) -- wrong

            # Let me just do it cleanly with 4 indices
            sv1 = torch.arange(V, device=device)
            sv2 = torch.arange(V, device=device)
            dv1 = torch.arange(V, device=device)
            dv2 = torch.arange(V, device=device)

            # Differences
            ds = sv1[:, None, None, None] - sv2[None, :, None, None]  # (V, V, 1, 1)
            dd = dv1[None, None, :, None] - dv2[None, None, None, :]  # (1, 1, V, V)

            # Conflict conditions
            both_same_s = (ds == 0)
            both_same_d = (dd == 0)
            either_same_s = (ds == 0)
            either_same_d = (dd == 0)
            same_dir = (torch.sign(ds) == torch.sign(dd))

            ok = (both_same_s & both_same_d) | (~either_same_s & ~either_same_d & same_dir)
            conflict = ~ok

            # Also need: not same destination (collision check is separate, skip for reconfig axis)
            # Actually for reconfig, collision = same destination cell, not same coordinate.
            # We handle collision separately. Here we just check axis ordering.

            # Joint probability
            prob = (src_marg[i][sv1[:, None, None, None]] *
                    src_marg[j][sv2[None, :, None, None]] *
                    dst_marg[i][dv1[None, None, :, None]] *
                    dst_marg[j][dv2[None, None, None, :]])  # (V, V, V, V)

            inv_ij = (prob * conflict.to(dtype)).sum()
            total_inv = total_inv + inv_ij

    return total_inv


def soft_inversion_count(src_dists, dst_dists, H, W):
    """Total soft inversion count (both axes)."""
    i_col = soft_inversion_count_axis(src_dists, dst_dists, H, W, 'col')
    i_row = soft_inversion_count_axis(src_dists, dst_dists, H, W, 'row')
    return i_col, i_row


def soft_n_movers(src_dists, dst_dists):
    """Expected number of atoms that actually move.
    P(atom q moves) = 1 - Σ_c p_src(c) * p_dst(c)"""
    overlap = (src_dists * dst_dists).sum(dim=1)  # (N,) P(stay) per atom
    return (1.0 - overlap).sum()  # expected number of movers


def reconfig_surrogate(src_dists, dst_dists, H, W, lambda_inv=1.0):
    """Differentiable reconfig cost surrogate.

    Returns estimated number of parallel groups needed.
    - 0 if no atoms move
    - 1 if atoms move but no inversions (all compatible)
    - 1 + λ * max(I_col, I_row) if inversions exist
    """
    i_col, i_row = soft_inversion_count(src_dists, dst_dists, H, W)
    n_movers = soft_n_movers(src_dists, dst_dists)

    # Soft indicator that any atom moves
    # Use sigmoid to smooth the 0/1 transition
    any_moves = torch.sigmoid(n_movers * 10 - 0.5)  # ≈1 when n_movers > 0.05

    # Cost = (something moves) * (1 + λ * max(inversions per axis))
    max_inv = torch.max(i_col, i_row)
    cost = any_moves * (1.0 + lambda_inv * max_inv)

    return cost, {'i_col': i_col, 'i_row': i_row, 'n_movers': n_movers,
                  'any_moves': any_moves, 'max_inv': max_inv}


# ─────────────────────────────────────────────────────────────────────────────
# True reconfig cost
# ─────────────────────────────────────────────────────────────────────────────

def true_reconfig_cost(src_cells, dst_cells, H, W):
    """True reconfig cost for concrete cell assignments.
    Only counts atoms that actually move."""
    N = src_cells.shape[0]
    moves = []
    for q in range(N):
        if src_cells[q] != dst_cells[q]:
            sr, sc = src_cells[q] // W, src_cells[q] % W
            dr, dc = dst_cells[q] // W, dst_cells[q] % W
            moves.append(torch.tensor([sr, sc, dr, dc], dtype=torch.long))
    if len(moves) == 0:
        return 0
    return greedy_chromatic(torch.stack(moves))


# ─────────────────────────────────────────────────────────────────────────────
# Random test case generators
# ─────────────────────────────────────────────────────────────────────────────

def random_positions(N, H, W, device='cpu'):
    """Random non-overlapping positions for N atoms."""
    C = H * W
    cells = torch.randperm(C, device=device)[:N]
    return cells


def random_reconfig(N, H, W, n_movers=None, device='cpu'):
    """Generate random src → dst assignments."""
    C = H * W
    src = random_positions(N, H, W, device)
    dst = src.clone()

    if n_movers is None:
        n_movers = random.randint(1, N)

    movers = random.sample(range(N), min(n_movers, N))
    available = list(set(range(C)) - set(dst.tolist()))

    for q in movers:
        if available:
            new_cell = random.choice(available)
            available.remove(new_cell)
            available.append(dst[q].item())
            dst[q] = new_cell

    return src, dst


# ═════════════════════════════════════════════════════════════════════════════
# Setup
# ═════════════════════════════════════════════════════════════════════════════

H, W, C = 5, 5, 25
N_ATOMS = 8


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Axis decomposition validity
# Does max(χ_col, χ_row) = χ_true?
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 65)
print("TEST 1: Axis decomposition — does max(χ_col, χ_row) = χ_true?")
print("═" * 65)

torch.manual_seed(42)
random.seed(42)

n_trials = 2000
exact_matches = 0
max_axis_leq = 0  # max(χ_col, χ_row) <= χ_true (always true)
max_axis_lt = 0   # max(χ_col, χ_row) < χ_true (gap exists)
chi_histogram = Counter()
gap_histogram = Counter()

for trial in range(n_trials):
    n = random.randint(2, 8)
    src, dst = random_reconfig(n, H, W, n_movers=random.randint(2, n))

    # Only keep movers
    moved = src != dst
    if moved.sum() < 2:
        continue

    src_moved = src[moved]
    dst_moved = dst[moved]

    moves = torch.stack([
        src_moved // W, src_moved % W, dst_moved // W, dst_moved % W
    ], dim=-1)

    chi_true = greedy_chromatic(moves)
    chi_col = axis_chromatic(moves, 0)
    chi_row = axis_chromatic(moves, 1)
    chi_axis_max = max(chi_col, chi_row)

    chi_histogram[chi_true] += 1

    if chi_axis_max == chi_true:
        exact_matches += 1
    if chi_axis_max <= chi_true:
        max_axis_leq += 1
    if chi_axis_max < chi_true:
        max_axis_lt += 1
        gap = chi_true - chi_axis_max
        gap_histogram[gap] += 1

valid_trials = sum(chi_histogram.values())
print(f"  Trials: {valid_trials}")
print(f"  max(χ_col, χ_row) = χ_true:  {exact_matches}/{valid_trials} "
      f"({exact_matches/valid_trials:.1%})")
print(f"  max(χ_col, χ_row) ≤ χ_true:  {max_axis_leq}/{valid_trials} "
      f"({max_axis_leq/valid_trials:.1%})  (should be 100%)")
print(f"  max(χ_col, χ_row) < χ_true:  {max_axis_lt}/{valid_trials} "
      f"({max_axis_lt/valid_trials:.1%})  (gap cases)")
print(f"  χ_true distribution: {dict(sorted(chi_histogram.items()))}")
if gap_histogram:
    print(f"  Gap distribution: {dict(sorted(gap_histogram.items()))}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Inversion count vs true χ (monotonic correlation?)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 2: Inversion count vs true χ — monotonic correlation?")
print("═" * 65)

torch.manual_seed(42)
random.seed(42)

records = []

for trial in range(2000):
    n = random.randint(2, 8)
    src, dst = random_reconfig(n, H, W, n_movers=random.randint(2, n))

    moved = src != dst
    if moved.sum() < 2:
        continue

    src_m, dst_m = src[moved], dst[moved]
    moves = torch.stack([src_m//W, src_m%W, dst_m//W, dst_m%W], dim=-1)

    chi_true = greedy_chromatic(moves)

    # Compute concrete inversion counts per axis
    nm = moves.shape[0]
    inv_col, inv_row = 0, 0
    for i in range(nm):
        for j in range(i+1, nm):
            # Column axis
            ds_c = moves[i, 1] - moves[j, 1]  # src col diff
            dd_c = moves[i, 3] - moves[j, 3]  # dst col diff
            if ds_c == 0 and dd_c == 0:
                pass  # ok
            elif ds_c == 0 or dd_c == 0:
                inv_col += 1
            elif (ds_c > 0) != (dd_c > 0):
                inv_col += 1

            # Row axis
            ds_r = moves[i, 0] - moves[j, 0]
            dd_r = moves[i, 2] - moves[j, 2]
            if ds_r == 0 and dd_r == 0:
                pass
            elif ds_r == 0 or dd_r == 0:
                inv_row += 1
            elif (ds_r > 0) != (dd_r > 0):
                inv_row += 1

    max_inv = max(inv_col, inv_row)
    total_inv = inv_col + inv_row
    records.append((chi_true, inv_col, inv_row, max_inv, total_inv, nm))

# Analyze correlation
chis = torch.tensor([r[0] for r in records], dtype=torch.float64)
max_invs = torch.tensor([r[3] for r in records], dtype=torch.float64)
total_invs = torch.tensor([r[4] for r in records], dtype=torch.float64)

corr_max = torch.corrcoef(torch.stack([chis, max_invs]))[0, 1].item()
corr_total = torch.corrcoef(torch.stack([chis, total_invs]))[0, 1].item()

print(f"  Trials: {len(records)}")
print(f"  Pearson correlation (χ vs max(I_col, I_row)): {corr_max:.4f}")
print(f"  Pearson correlation (χ vs I_col + I_row):     {corr_total:.4f}")

# Check: I=0 ⟺ χ=1
zero_inv = [r for r in records if r[3] == 0]
chi1_cases = [r for r in records if r[0] == 1]
print(f"\n  I=0 cases: {len(zero_inv)}, of which χ=1: {sum(1 for r in zero_inv if r[0]==1)}")
print(f"  χ=1 cases: {len(chi1_cases)}, of which I=0: {sum(1 for r in chi1_cases if r[3]==0)}")

# Mean χ per inversion bucket
from collections import defaultdict
bucket = defaultdict(list)
for r in records:
    bucket[r[3]].append(r[0])
print(f"\n  Mean χ by max(I_col, I_row):")
for inv_val in sorted(bucket.keys())[:10]:
    vals = bucket[inv_val]
    print(f"    I={inv_val}: mean_χ={sum(vals)/len(vals):.2f}  n={len(vals)}  "
          f"range=[{min(vals)},{max(vals)}]")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Soft surrogate matches concrete for one-hot distributions
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 3: Soft surrogate matches concrete for one-hot distributions")
print("═" * 65)

torch.manual_seed(42)
random.seed(42)

match_count = 0
test_count = 0

for trial in range(200):
    n = random.randint(3, 8)
    src, dst = random_reconfig(n, H, W, n_movers=random.randint(2, n))

    # One-hot distributions
    src_dists = torch.zeros(n, C)
    dst_dists = torch.zeros(n, C)
    for q in range(n):
        src_dists[q, src[q]] = 1.0
        dst_dists[q, dst[q]] = 1.0

    i_col, i_row = soft_inversion_count(src_dists, dst_dists, H, W)

    # Concrete inversion count
    moved = src != dst
    if moved.sum() < 2:
        continue
    test_count += 1

    src_m, dst_m = src[moved], dst[moved]
    moves = torch.stack([src_m//W, src_m%W, dst_m//W, dst_m%W], dim=-1)
    nm = moves.shape[0]
    concrete_col, concrete_row = 0, 0
    for i in range(nm):
        for j in range(i+1, nm):
            ds_c = moves[i,1]-moves[j,1]; dd_c = moves[i,3]-moves[j,3]
            if (ds_c==0 and dd_c==0): pass
            elif (ds_c==0 or dd_c==0): concrete_col += 1
            elif (ds_c>0)!=(dd_c>0): concrete_col += 1

            ds_r = moves[i,0]-moves[j,0]; dd_r = moves[i,2]-moves[j,2]
            if (ds_r==0 and dd_r==0): pass
            elif (ds_r==0 or dd_r==0): concrete_row += 1
            elif (ds_r>0)!=(dd_r>0): concrete_row += 1

    # BUT: soft version counts ALL pairs (including non-movers)
    # while concrete only counts mover pairs. Need to compare correctly.
    # Non-movers have src=dst, so their "move" is (r,c)->(r,c).
    # Against a mover: ds might be nonzero, dd also nonzero, could be inversion.
    # Against another non-mover: ds could be nonzero but dd=same as ds, so no inversion.
    # So non-movers CAN create inversions with movers!
    # But in true reconfig cost, non-movers aren't counted as moves.
    #
    # This is a mismatch. The soft surrogate counts inversions between ALL atom pairs
    # including non-movers, but the true cost only counts actual moves.
    # We should only count inversions between MOVERS.
    #
    # Fix: weight by P(both atoms move)

    # For now, just check soft matches concrete on mover-only distributions
    src_movers = src_dists[moved]
    dst_movers = dst_dists[moved]
    i_col_m, i_row_m = soft_inversion_count(src_movers, dst_movers, H, W)

    col_match = abs(i_col_m.item() - concrete_col) < 0.01
    row_match = abs(i_row_m.item() - concrete_row) < 0.01
    if col_match and row_match:
        match_count += 1

print(f"  Tested: {test_count}")
print(f"  Exact match (movers only): {match_count}/{test_count} "
      f"({match_count/test_count:.1%})")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Edge cases
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 4: Edge cases")
print("═" * 65)

def test_case(name, src_cells, dst_cells, expected_chi):
    moves_list = []
    for i in range(len(src_cells)):
        if src_cells[i] != dst_cells[i]:
            sr, sc = src_cells[i] // W, src_cells[i] % W
            dr, dc = dst_cells[i] // W, dst_cells[i] % W
            moves_list.append(torch.tensor([sr, sc, dr, dc], dtype=torch.long))
    if len(moves_list) == 0:
        chi = 0
    else:
        chi = greedy_chromatic(torch.stack(moves_list))
    status = '✓' if chi == expected_chi else f'✗ (got {chi})'
    print(f"  {name}: χ={chi} {status}")
    return chi

# All atoms move in same direction (pure translation) → χ=1
test_case("Pure translation right",
          [0,1,2,5,6,7], [1,2,3,6,7,8], expected_chi=1)

# All atoms move to same row from different rows → might conflict
test_case("Converge to row 2",
          [0,5,10,15], [10,11,12,13], expected_chi=1)  # order preserved

# Two atoms swap positions → χ=2
test_case("Two atoms swap",
          [0, 4], [4, 0], expected_chi=2)

# Fully reversed order
test_case("4 atoms reverse col order",
          [0,1,2,3], [3,2,1,0], expected_chi=4)

# All stay → χ=0
test_case("All stay", [0,1,2,3], [0,1,2,3], expected_chi=0)

# One mover → χ=1
test_case("One mover", [0,1,2,3], [0,1,2,4], expected_chi=1)

# Cross pattern: (0,0)→(1,1) and (1,0)→(0,1) — row inversion
test_case("Cross pattern",
          [0, 5], [6, 1], expected_chi=2)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Gradient test — can gradients push inversions to zero?
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 5: Gradient optimization — push inversions to zero")
print("═" * 65)

torch.manual_seed(7)

n_atoms = 6
# Fixed source positions
src_cells = torch.tensor([0, 3, 6, 12, 18, 24])
src_dists = torch.zeros(n_atoms, C)
for q in range(n_atoms):
    src_dists[q, src_cells[q]] = 1.0

# Random initial destination logits (somewhat spread)
dst_logits = torch.randn(n_atoms, C) * 0.5
# Bias toward some positions that create inversions
dst_logits[0, 20] += 3  # atom 0: (0,0) → (4,0) — big move
dst_logits[1, 4] += 3   # atom 1: (0,3) → (0,4)
dst_logits[2, 15] += 3  # atom 2: (1,1) → (3,0) — crosses atom 0
dst_logits[3, 22] += 3  # atom 3: (2,2) → (4,2)
dst_logits[4, 1] += 3   # atom 4: (3,3) → (0,1) — big reverse
dst_logits[5, 14] += 3  # atom 5: (4,4) → (2,4)
dst_logits.requires_grad_(True)

opt = torch.optim.Adam([dst_logits], lr=0.1)

print("  Optimizing destination logits to minimize inversions...")
for step in range(200):
    opt.zero_grad()
    d = F.softmax(dst_logits, dim=-1)
    cost, info = reconfig_surrogate(src_dists, d, H, W, lambda_inv=1.0)
    cost.backward()
    opt.step()

    if step % 40 == 0 or step == 199:
        with torch.no_grad():
            hard_dst = F.softmax(dst_logits, dim=-1).argmax(dim=-1)
            true_chi = true_reconfig_cost(src_cells, hard_dst, H, W)
        print(f"    Step {step:3d}: surrogate={cost.item():.4f}  "
              f"I_col={info['i_col'].item():.3f}  I_row={info['i_row'].item():.3f}  "
              f"n_movers={info['n_movers'].item():.2f}  true_χ={true_chi}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Calibration — fit λ so surrogate ≈ true cost
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 6: Calibration — what λ makes surrogate ≈ true cost?")
print("═" * 65)

torch.manual_seed(42)
random.seed(42)

cal_records = []
for trial in range(1000):
    n = random.randint(3, 8)
    src, dst = random_reconfig(n, H, W, n_movers=random.randint(2, n))

    moved = src != dst
    if moved.sum() < 2:
        continue

    true_chi = true_reconfig_cost(src, dst, H, W)

    # Concrete inversions (movers only)
    src_m, dst_m = src[moved], dst[moved]
    moves = torch.stack([src_m//W, src_m%W, dst_m//W, dst_m%W], dim=-1)
    nm = moves.shape[0]
    inv_col, inv_row = 0, 0
    for i in range(nm):
        for j in range(i+1, nm):
            ds_c = (moves[i,1]-moves[j,1]).item()
            dd_c = (moves[i,3]-moves[j,3]).item()
            if ds_c==0 and dd_c==0: pass
            elif ds_c==0 or dd_c==0: inv_col += 1
            elif (ds_c>0)!=(dd_c>0): inv_col += 1

            ds_r = (moves[i,0]-moves[j,0]).item()
            dd_r = (moves[i,2]-moves[j,2]).item()
            if ds_r==0 and dd_r==0: pass
            elif ds_r==0 or dd_r==0: inv_row += 1
            elif (ds_r>0)!=(dd_r>0): inv_row += 1

    max_inv = max(inv_col, inv_row)
    cal_records.append((true_chi, max_inv, inv_col + inv_row, nm))

# Fit: true_χ ≈ α + β * max_inv (for cases where movers > 0)
chis = torch.tensor([r[0] for r in cal_records], dtype=torch.float64)
max_invs = torch.tensor([r[1] for r in cal_records], dtype=torch.float64)

# Simple linear regression
X = torch.stack([torch.ones_like(max_invs), max_invs], dim=1)
# Normal equation: β = (X^T X)^{-1} X^T y
beta = torch.linalg.lstsq(X, chis).solution
print(f"  Linear fit: χ ≈ {beta[0].item():.3f} + {beta[1].item():.3f} * max(I_col, I_row)")

# Residuals
predicted = X @ beta
residuals = chis - predicted
print(f"  R² = {1 - residuals.var() / chis.var():.4f}")
print(f"  Mean absolute error: {residuals.abs().mean():.3f}")
print(f"  Max absolute error:  {residuals.abs().max():.3f}")

# Also fit with total inversions
total_invs = torch.tensor([r[2] for r in cal_records], dtype=torch.float64)
X2 = torch.stack([torch.ones_like(total_invs), total_invs], dim=1)
beta2 = torch.linalg.lstsq(X2, chis).solution
predicted2 = X2 @ beta2
residuals2 = chis - predicted2
print(f"\n  Linear fit: χ ≈ {beta2[0].item():.3f} + {beta2[1].item():.3f} * (I_col + I_row)")
print(f"  R² = {1 - residuals2.var() / chis.var():.4f}")
print(f"  Mean absolute error: {residuals2.abs().mean():.3f}")

# Fit with both axes + n_movers
n_movers_t = torch.tensor([r[3] for r in cal_records], dtype=torch.float64)
inv_col_t = torch.tensor([r[1] for r in cal_records], dtype=torch.float64)  # using max for now
X3 = torch.stack([torch.ones_like(chis), max_invs, n_movers_t], dim=1)
beta3 = torch.linalg.lstsq(X3, chis).solution
predicted3 = X3 @ beta3
residuals3 = chis - predicted3
print(f"\n  Linear fit: χ ≈ {beta3[0].item():.3f} + {beta3[1].item():.3f}*max_inv "
      f"+ {beta3[2].item():.3f}*n_movers")
print(f"  R² = {1 - residuals3.var() / chis.var():.4f}")
print(f"  Mean absolute error: {residuals3.abs().mean():.3f}")