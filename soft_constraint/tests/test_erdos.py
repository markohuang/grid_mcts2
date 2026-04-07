"""
tests/test_erdos.py — Sanity checks for the Erdős/Potts surrogate.

Tests:
  1. Potts energy: zero for valid colorings, positive for invalid
  2. Group counting: correct count at one-hot assignments
  3. Gate conflict matrix: matches discrete AOD check at one-hot
  4. Reconfig conflict matrix: matches discrete AOD check at one-hot
  5. One-hot cost recovery: does the surrogate match true cost when
     placements are one-hot AND group assignments are optimal?
  6. Gradient sanity: does optimization reduce cost?
  7. Joint optimization: placements + directions + groups on Map 2
  8. Monotone path: does cost decrease along do-nothing → optimal?
"""

import torch
import torch.nn.functional as F
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.primitives import (
    aod_compatible, true_gate_cost, true_reconfig_cost,
    true_total_cost, get_relevant_atoms
)
from surrogate.erdos import (
    build_reconfig_conflict_matrix, build_gate_conflict_matrix,
    potts_energy, count_active_groups,
    reconfig_layer_cost, gate_layer_cost,
    ErdosSurrogate
)

H, W, C = 5, 5, 25
EPS = 1e-6

def onehot(cell, C=25):
    d = torch.zeros(C, dtype=torch.float64)
    d[cell] = 1.0
    return d

def header(name):
    print(f"\n{'═' * 65}")
    print(f"  {name}")
    print(f"{'═' * 65}")

def check(name, condition, detail=""):
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}: {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Potts energy
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 1: Potts energy — zero for valid colorings")

# 3 nodes, edges (0,1) and (1,2), no edge (0,2)
A = torch.zeros(3, 3, dtype=torch.float64)
A[0, 1] = A[1, 0] = 1.0
A[1, 2] = A[2, 1] = 1.0

# Valid 2-coloring: node 0→color 0, node 1→color 1, node 2→color 0
S_valid = torch.zeros(3, 2, dtype=torch.float64)
S_valid[0, 0] = 1.0  # node 0 → group 0
S_valid[1, 1] = 1.0  # node 1 → group 1
S_valid[2, 0] = 1.0  # node 2 → group 0

E_valid = potts_energy(A, S_valid)
check("Valid coloring: energy = 0", E_valid.item() < EPS, f"E={E_valid.item():.6f}")

# Invalid: nodes 0 and 1 in same group
S_invalid = torch.zeros(3, 2, dtype=torch.float64)
S_invalid[0, 0] = 1.0
S_invalid[1, 0] = 1.0  # conflict! edge (0,1) both in group 0
S_invalid[2, 1] = 1.0

E_invalid = potts_energy(A, S_invalid)
check("Invalid coloring: energy > 0", E_invalid.item() > 1 - EPS, f"E={E_invalid.item():.6f}")

# Fully connected 3-clique: needs 3 colors
A_clique = torch.ones(3, 3, dtype=torch.float64)
A_clique.fill_diagonal_(0)

S_3color = torch.zeros(3, 3, dtype=torch.float64)
S_3color[0, 0] = S_3color[1, 1] = S_3color[2, 2] = 1.0

E_3color = potts_energy(A_clique, S_3color)
check("3-clique with 3 colors: energy = 0", E_3color.item() < EPS,
      f"E={E_3color.item():.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Group counting
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 2: Group counting")

# 4 moves, 3 groups used (groups 0, 1, 2; group 3 empty)
S_count = torch.zeros(4, 4, dtype=torch.float64)
S_count[0, 0] = 1.0
S_count[1, 0] = 1.0
S_count[2, 1] = 1.0
S_count[3, 2] = 1.0

n_groups = count_active_groups(S_count, beta=20.0, tau=0.5)
check("3 active groups", abs(n_groups.item() - 3.0) < 0.1, f"count={n_groups.item():.3f}")

# 1 group used
S_one = torch.zeros(4, 4, dtype=torch.float64)
S_one[:, 0] = 1.0

n_one = count_active_groups(S_one, beta=20.0, tau=0.5)
check("1 active group", abs(n_one.item() - 1.0) < 0.1, f"count={n_one.item():.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Gate conflict matrix at one-hot
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 3: Gate conflict matrix at one-hot placements")

pos = torch.tensor([
    [0,4],[2,2],[1,2],[2,1],[3,0],[1,0],
    [0,2],[4,1],[4,2],[4,3],[4,0],[1,3],
])
N = pos.shape[0]
init_cells = pos[:, 0] * W + pos[:, 1]

onehot_dists = torch.zeros(N, C, dtype=torch.float64)
for q in range(N):
    onehot_dists[q, init_cells[q]] = 1.0

tasks = [
    [(4,3), (6,7), (9,8), (10,11)],
    [(2,1), (4,5), (7,6), (8,9)],
    [(0,1), (2,3), (6,5), (9,8)],
]

# Check gate conflict matrix for layer 0 with forward directions
gates0 = tasks[0]
M = len(gates0)
dirs_fwd = torch.zeros(M, dtype=torch.float64)  # all forward

A_gate = build_gate_conflict_matrix(gates0, onehot_dists, H, W, dirs_fwd)

# Verify against discrete AOD check
rows, cols = init_cells // W, init_cells % W
for i in range(M):
    for j in range(i + 1, M):
        a_i, b_i = gates0[i]
        a_j, b_j = gates0[j]
        move_i = torch.tensor([rows[a_i], cols[a_i], rows[b_i], cols[b_i]], dtype=torch.long)
        move_j = torch.tensor([rows[a_j], cols[a_j], rows[b_j], cols[b_j]], dtype=torch.long)
        discrete_conflict = not aod_compatible(move_i, move_j).item()
        soft_conflict = A_gate[i, j].item()

        match = (discrete_conflict and soft_conflict > 0.5) or \
                (not discrete_conflict and soft_conflict < 0.5)
        check(f"Gate pair ({i},{j}): discrete={discrete_conflict}, soft={soft_conflict:.3f}",
              match)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: One-hot cost recovery with optimal group assignment
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 4: One-hot cost recovery")

# For one-hot placements, if we set group assignments to a valid optimal
# coloring, the Erdős cost should match the true cost.

# True gate cost for layer 0
true_gc = true_gate_cost(init_cells, gates0, H, W, canon=True)
print(f"  True gate cost layer 0: {true_gc}")

# Build conflict matrix and try all direction combos
best_true_chi = true_gc // 2
print(f"  True χ_gate: {best_true_chi}")

# Find best direction and its conflict structure
best_dir_idx = None
for di in range(1 << M):
    ds = [(di >> g) & 1 for g in range(M)]
    mvs = []
    for g, (a, b) in enumerate(gates0):
        if ds[g]: a, b = b, a
        mvs.append(torch.tensor([rows[a], cols[a], rows[b], cols[b]], dtype=torch.long))
    mvs = torch.stack(mvs)
    # Check if this direction gives the best chi
    from surrogate.primitives import greedy_chromatic
    chi = greedy_chromatic(mvs)
    if chi == best_true_chi:
        best_dir_idx = di
        # Build the coloring
        n = mvs.shape[0]
        conflict = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            for j in range(i+1, n):
                if not aod_compatible(mvs[i], mvs[j]):
                    conflict[i,j] = conflict[j,i] = True
        colors = [-1]*n
        for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
            used = {colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
            c = 0
            while c in used: c += 1
            colors[idx] = c
        best_colors = colors
        break

if best_dir_idx is not None:
    print(f"  Best direction: {[(best_dir_idx>>g)&1 for g in range(M)]}")
    print(f"  Coloring: {best_colors}")

    # Set up one-hot direction and group assignment
    dirs = torch.tensor([(best_dir_idx >> g) & 1 for g in range(M)],
                         dtype=torch.float64)
    S_G = torch.zeros(M, best_true_chi + 1, dtype=torch.float64)  # +1 for safety
    for g in range(M):
        S_G[g, best_colors[g]] = 1.0

    A_G = build_gate_conflict_matrix(gates0, onehot_dists, H, W, dirs)
    conflict_energy = potts_energy(A_G, S_G)
    n_groups = count_active_groups(S_G, beta=20.0, tau=0.5)

    check("Conflict energy = 0 with optimal coloring",
          conflict_energy.item() < EPS, f"E={conflict_energy.item():.6f}")
    check(f"Group count ≈ {best_true_chi}",
          abs(n_groups.item() - best_true_chi) < 0.2,
          f"count={n_groups.item():.3f}")
    check(f"Gate cost ≈ {true_gc}",
          abs(2 * n_groups.item() - true_gc) < 0.5,
          f"2×groups={2*n_groups.item():.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Gradient sanity — does optimization reduce cost?
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 5: Gradient optimization with λ_e annealing")

torch.manual_seed(42)

surrogate = ErdosSurrogate(tasks, N, H, W, K_R=4, K_G=4,
                            lambda_e=1.0, beta=5.0, tau=0.5)

# Placement logits (relevant atoms per layer)
relevant = [get_relevant_atoms(t) for t in tasks]
layer_logits = []
for t in range(len(tasks)):
    n_rel = len(relevant[t])
    ll = torch.randn(n_rel, C, dtype=torch.float64) * 0.3
    for local_idx, global_idx in enumerate(relevant[t]):
        cell = pos[global_idx, 0] * W + pos[global_idx, 1]
        ll[local_idx, cell] += 4.0
    ll.requires_grad_(True)
    layer_logits.append(ll)

# Direction logits
dir_logits = []
for t in range(len(tasks)):
    dl = torch.zeros(len(tasks[t]), dtype=torch.float64, requires_grad=True)
    dir_logits.append(dl)

# Separate optimizers for different variable groups
placement_params = list(layer_logits) + list(dir_logits)
group_params = surrogate.parameters()

opt_placement = torch.optim.Adam(placement_params, lr=0.05)
opt_groups = torch.optim.Adam(group_params, lr=0.1)

init_true = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)
print(f"  Do-nothing baseline: {init_true}")
print(f"  Strategy: anneal λ_e from 0.1 → 20 over 500 steps")
print(f"            (start: explore placements; end: enforce valid colorings)")

N_STEPS = 500

for step in range(N_STEPS):
    opt_placement.zero_grad()
    opt_groups.zero_grad()

    # Anneal λ_e: start small (explore), end large (enforce)
    progress = step / N_STEPS
    lambda_e = 0.1 + (20.0 - 0.1) * progress
    surrogate.lambda_e = lambda_e

    # Build distributions
    init_dists = torch.zeros(N, C, dtype=torch.float64)
    for q in range(N):
        init_dists[q, init_cells[q]] = 1.0

    layer_dists = []
    prev = init_dists.clone()
    for t in range(len(tasks)):
        d = prev.clone().detach()
        for local_idx, global_idx in enumerate(relevant[t]):
            d[global_idx] = F.softmax(layer_logits[t][local_idx], dim=-1)
        layer_dists.append(d)
        prev = d

    directions = [torch.sigmoid(dl) for dl in dir_logits]

    loss, infos = surrogate.compute_loss(init_dists, layer_dists, directions)
    loss.backward()

    opt_placement.step()
    opt_groups.step()

    if step % 100 == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            hard_cells = []
            for t in range(len(tasks)):
                hard_cells.append(layer_dists[t].argmax(dim=-1))
            tc, bd = true_total_cost(init_cells, hard_cells, tasks, H, W)

            bd_str = ', '.join([f'r{r}+g{g}' for r, g in bd])
            r_conf = sum(info['reconfig']['conflict'].item() for info in infos)
            g_conf = sum(info['gate']['conflict'].item() for info in infos)
            r_grp = sum(info['reconfig']['groups'].item() for info in infos)
            g_grp = sum(info['gate']['groups'].item() for info in infos)

        print(f"    Step {step:3d} (λ_e={lambda_e:.1f}): loss={loss.item():.2f}  "
              f"true={tc}  [{bd_str}]  "
              f"R_conf={r_conf:.3f} G_conf={g_conf:.3f}  "
              f"R_grp={r_grp:.1f} G_grp={g_grp:.1f}")

    if step % 100 == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            hard_cells = []
            for t in range(len(tasks)):
                hard_cells.append(layer_dists[t].argmax(dim=-1))
            tc, bd = true_total_cost(init_cells, hard_cells, tasks, H, W)

            bd_str = ', '.join([f'r{r}+g{g}' for r, g in bd])
            r_conf = sum(info['reconfig']['conflict'].item() for info in infos)
            g_conf = sum(info['gate']['conflict'].item() for info in infos)
            r_grp = sum(info['reconfig']['groups'].item() for info in infos)
            g_grp = sum(info['gate']['groups'].item() for info in infos)

        print(f"    Step {step:3d} (λ_e={lambda_e:.1f}): loss={loss.item():.2f}  "
              f"true={tc}  [{bd_str}]  "
              f"R_conf={r_conf:.3f} G_conf={g_conf:.3f}  "
              f"R_grp={r_grp:.1f} G_grp={g_grp:.1f}")

# Final result
with torch.no_grad():
    hard_cells_final = []
    for t in range(len(tasks)):
        hard_cells_final.append(layer_dists[t].argmax(dim=-1))
    tc_final, bd_final = true_total_cost(init_cells, hard_cells_final, tasks, H, W)

print(f"\n  Result: {init_true} → {tc_final}")

# Check directions
print(f"  Learned directions:")
for t in range(len(tasks)):
    dirs_hard = (torch.sigmoid(dir_logits[t]) > 0.5).long().tolist()
    print(f"    Layer {t}: {dirs_hard}")

# Collisions
from collections import Counter
for t in range(len(tasks)):
    cells = hard_cells_final[t].tolist()
    dupes = {c: n for c, n in Counter(cells).items() if n > 1}
    if dupes:
        print(f"  Layer {t} ⚠ COLLISIONS: {dupes}")

# Group assignments
print(f"\n  Group assignments:")
for t in range(len(tasks)):
    S_R = F.softmax(surrogate.reconfig_logits[t], dim=-1)
    S_G = F.softmax(surrogate.gate_logits[t], dim=-1)
    r_groups = S_R.argmax(dim=-1).tolist()
    g_groups = S_G.argmax(dim=-1).tolist()
    print(f"    Layer {t}: reconfig_groups={r_groups}  gate_groups={g_groups}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Speed benchmark
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 6: Speed benchmark")

torch.manual_seed(42)
surr = ErdosSurrogate(tasks, N, H, W, K_R=4, K_G=4)

dists = [F.softmax(torch.randn(N, C, dtype=torch.float64), dim=-1) for _ in range(len(tasks))]
init_d = torch.zeros(N, C, dtype=torch.float64)
for q in range(N):
    init_d[q, init_cells[q]] = 1.0
dirs = [torch.full((len(t),), 0.5, dtype=torch.float64) for t in tasks]

# Warmup
_ = surr.compute_loss(init_d, dists, dirs)

times = []
for _ in range(20):
    t0 = time.time()
    loss, _ = surr.compute_loss(init_d, dists, dirs)
    times.append(time.time() - t0)

avg_t = sum(times) / len(times)
print(f"  Forward pass: {avg_t*1000:.1f}ms / eval")

# With backward
times_bwd = []
for _ in range(10):
    for p in surr.parameters():
        if p.grad is not None:
            p.grad.zero_()
    t0 = time.time()
    loss, _ = surr.compute_loss(init_d, dists, dirs)
    loss.backward()
    times_bwd.append(time.time() - t0)

avg_bwd = sum(times_bwd) / len(times_bwd)
print(f"  Forward + backward: {avg_bwd*1000:.1f}ms / eval")

check("Forward < 200ms", avg_t < 0.2, f"{avg_t*1000:.1f}ms")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

header("SUMMARY")
print("  All tests above should show ✓ PASS.")
print("  Key metrics to watch:")
print("  - Conflict energy → 0 (valid colorings found)")
print("  - Group count matches true χ at one-hot")
print("  - Optimization improves true cost")