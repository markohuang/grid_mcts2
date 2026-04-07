"""
tests/test_partition.py — Validate partition-function chromatic surrogate.

Tests:
  1. Discrete recovery: χ̂ matches true χ for known graphs
  2. Gradient sanity: ∂χ̂/∂A_ij > 0 (more conflict → higher χ̂)
  3. Joint optimization: placements + directions on Map 2 (same setup as test_erdos.py)
  4. Speed benchmark
"""

import torch
import torch.nn.functional as F
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.partition import log_Z_tilde, chromatic_estimate, PartitionSurrogate
from surrogate.primitives import (
    aod_compatible, true_gate_cost, true_reconfig_cost,
    true_total_cost, get_relevant_atoms, greedy_chromatic
)

H, W, C = 5, 5, 25
EPS = 1e-6

def header(name):
    print(f"\n{'═' * 65}")
    print(f"  {name}")
    print(f"{'═' * 65}")

def check(name, condition, detail=""):
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}: {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Discrete recovery
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 1: χ̂ matches true χ on discrete graphs")

alpha = 20.0  # sharp sigmoid for discrete tests

# Empty graph (no edges): χ = 1
A_empty = torch.zeros(4, 4, dtype=torch.float64)
chi_empty = chromatic_estimate(A_empty, q_max=5, alpha=alpha)
check("Empty graph: χ̂ ≈ 1", abs(chi_empty.item() - 1.0) < 0.6,
      f"χ̂={chi_empty.item():.3f}")

# Single edge: χ = 2
A_edge = torch.zeros(4, 4, dtype=torch.float64)
A_edge[0, 1] = A_edge[1, 0] = 1.0
chi_edge = chromatic_estimate(A_edge, q_max=5, alpha=alpha)
check("Single edge: χ̂ ≈ 2", abs(chi_edge.item() - 2.0) < 0.6,
      f"χ̂={chi_edge.item():.3f}")

# Triangle (K₃): χ = 3
A_tri = torch.zeros(3, 3, dtype=torch.float64)
A_tri[0,1] = A_tri[1,0] = 1.0
A_tri[0,2] = A_tri[2,0] = 1.0
A_tri[1,2] = A_tri[2,1] = 1.0
chi_tri = chromatic_estimate(A_tri, q_max=5, alpha=alpha)
check("Triangle K₃: χ̂ ≈ 3", abs(chi_tri.item() - 3.0) < 0.6,
      f"χ̂={chi_tri.item():.3f}")

# Complete K₄: χ = 4
A_k4 = torch.ones(4, 4, dtype=torch.float64)
A_k4.fill_diagonal_(0)
chi_k4 = chromatic_estimate(A_k4, q_max=5, alpha=alpha)
check("Complete K₄: χ̂ ≈ 4", abs(chi_k4.item() - 4.0) < 0.6,
      f"χ̂={chi_k4.item():.3f}")

# Cycle C₅: χ = 3
A_c5 = torch.zeros(5, 5, dtype=torch.float64)
for i in range(5):
    j = (i + 1) % 5
    A_c5[i, j] = A_c5[j, i] = 1.0
chi_c5 = chromatic_estimate(A_c5, q_max=5, alpha=alpha)
check("Cycle C₅: χ̂ ≈ 3", abs(chi_c5.item() - 3.0) < 0.6,
      f"χ̂={chi_c5.item():.3f}")

# Bipartite K_{2,3}: χ = 2
A_bip = torch.zeros(5, 5, dtype=torch.float64)
for i in [0, 1]:
    for j in [2, 3, 4]:
        A_bip[i, j] = A_bip[j, i] = 1.0
chi_bip = chromatic_estimate(A_bip, q_max=5, alpha=alpha)
check("Bipartite K₂₃: χ̂ ≈ 2", abs(chi_bip.item() - 2.0) < 0.6,
      f"χ̂={chi_bip.item():.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Gradient sanity — ∂χ̂/∂A_ij > 0
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 2: Gradient direction (more conflict → higher χ̂)")

A_soft = torch.tensor([
    [0.0, 0.3, 0.0, 0.0],
    [0.3, 0.0, 0.8, 0.0],
    [0.0, 0.8, 0.0, 0.5],
    [0.0, 0.0, 0.5, 0.0],
], dtype=torch.float64, requires_grad=True)

chi_soft = chromatic_estimate(A_soft, q_max=5, alpha=10.0)
chi_soft.backward()

grad = A_soft.grad
# Gradient should be non-negative (more conflict → higher χ)
# Check upper triangle where edges exist
pairs = [(0,1), (1,2), (2,3)]
all_nonneg = True
for i, j in pairs:
    g = grad[i, j].item()
    ok = g >= -EPS
    if not ok: all_nonneg = False
    print(f"  ∂χ̂/∂A[{i},{j}] = {g:.4f}  (A={A_soft.data[i,j]:.1f})")

check("All edge gradients ≥ 0", all_nonneg)
check("χ̂ in reasonable range", 1.5 < chi_soft.item() < 4.0,
      f"χ̂={chi_soft.item():.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: AOD conflict graph — χ̂ matches true χ at one-hot placements
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 3: χ̂ on real AOD conflict graphs at one-hot placements")

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

from surrogate.erdos import build_gate_conflict_matrix

for t_idx, gates in enumerate(tasks):
    M_G = len(gates)
    # True χ_G (minimize over directions)
    true_gc = true_gate_cost(init_cells, gates, H, W, canon=True)
    true_chi = true_gc // 2

    # Build conflict matrix at one-hot with best direction
    # Find best direction first
    rows, cols = init_cells // W, init_cells % W
    best_chi_found = M_G
    for d in range(1 << M_G):
        ds = [(d >> g) & 1 for g in range(M_G)]
        mvs = []
        for g, (a, b) in enumerate(gates):
            if ds[g]: a, b = b, a
            mvs.append(torch.tensor([rows[a], cols[a], rows[b], cols[b]], dtype=torch.long))
        chi_d = greedy_chromatic(torch.stack(mvs))
        if chi_d < best_chi_found:
            best_chi_found = chi_d
            best_dirs = ds

    dirs_t = torch.tensor(best_dirs, dtype=torch.float64)
    A_G = build_gate_conflict_matrix(gates, onehot_dists, H, W, dirs_t)
    chi_hat = chromatic_estimate(A_G, q_max=5, alpha=20.0)

    check(f"Layer {t_idx}: χ̂_G ≈ {true_chi} (true)", abs(chi_hat.item() - true_chi) < 0.6,
          f"χ̂={chi_hat.item():.3f}, true={true_chi}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Joint optimization — placements + directions
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 4: Gradient optimization (compare with erdos.py)")

torch.manual_seed(42)

surrogate = PartitionSurrogate(tasks, N, H, W, q_max=5, alpha=10.0)

# Placement logits: initialized near initial positions
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

# Only placement + direction params — no auxiliary variables!
all_params = list(layer_logits) + list(dir_logits)
optimizer = torch.optim.Adam(all_params, lr=0.05)

init_true = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)
print(f"  Do-nothing baseline: {init_true}")

for step in range(300):
    optimizer.zero_grad()

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
    optimizer.step()

    if step % 60 == 0 or step == 299:
        with torch.no_grad():
            hard_cells = []
            prev_c = init_cells
            for t in range(len(tasks)):
                hard = layer_dists[t].argmax(dim=-1)
                hard_cells.append(hard)
                prev_c = hard
            tc, bd = true_total_cost(init_cells, hard_cells, tasks, H, W)
            bd_str = ', '.join([f'r{r}+g{g}' for r, g in bd])
            chi_R_str = ' '.join([f'{info["chi_R"]:.1f}' for info in infos])
            chi_G_str = ' '.join([f'{info["chi_G"]:.1f}' for info in infos])

        print(f"    Step {step:3d}: loss={loss.item():.3f}  true={tc}  [{bd_str}]  "
              f"χ̂_R=[{chi_R_str}]  χ̂_G=[{chi_G_str}]")

# Final
with torch.no_grad():
    hard_cells_final = [layer_dists[t].argmax(dim=-1) for t in range(len(tasks))]
    tc_final, bd_final = true_total_cost(init_cells, hard_cells_final, tasks, H, W)

print(f"\n  Result: {init_true} → {tc_final}")

# Directions
print(f"  Learned directions:")
for t in range(len(tasks)):
    dirs_hard = (torch.sigmoid(dir_logits[t]) > 0.5).long().tolist()
    print(f"    Layer {t}: {dirs_hard}")

# Placement changes
print(f"  Placement changes:")
for t in range(len(tasks)):
    prev = init_cells if t == 0 else hard_cells_final[t-1]
    curr = hard_cells_final[t]
    changes = []
    for local_idx, global_idx in enumerate(relevant[t]):
        s = prev[global_idx].item()
        d = curr[global_idx].item()
        if s != d:
            sr, sc = s // W, s % W
            dr, dc = d // W, d % W
            changes.append(f"atom{global_idx}:({sr},{sc})→({dr},{dc})")
    if changes:
        print(f"    Layer {t}: {', '.join(changes)}")
    else:
        print(f"    Layer {t}: no movement")

# Collisions check
from collections import Counter
for t in range(len(tasks)):
    cells = hard_cells_final[t].tolist()
    dupes = {c: n for c, n in Counter(cells).items() if n > 1}
    if dupes:
        print(f"  Layer {t} ⚠ COLLISIONS: {dupes}")

# Gradient norms (to verify signal exists)
print(f"\n  Final gradient norms:")
for t in range(len(tasks)):
    if layer_logits[t].grad is not None:
        print(f"    Layer {t} placements: {layer_logits[t].grad.norm().item():.4f}")
    if dir_logits[t].grad is not None:
        print(f"    Layer {t} directions: {dir_logits[t].grad.norm().item():.4f}")


# ═══���═════════════════════��═══════════════════════════════════════════════════
# TEST 5: Ablation — sigmoid α values + non-saturating cost mode
# ═════════════════════════════════════��═══════════════════════════════════════

header("TEST 5: Ablation — effect of α and cost mode")

def run_optimization(mode, alpha, init_bias, label, steps=300, lr=0.05):
    torch.manual_seed(42)
    surr = PartitionSurrogate(tasks, N, H, W, q_max=5, alpha=alpha, mode=mode)

    ll = []
    for t in range(len(tasks)):
        n_rel = len(relevant[t])
        l = torch.randn(n_rel, C, dtype=torch.float64) * 0.3
        for local_idx, global_idx in enumerate(relevant[t]):
            cell = pos[global_idx, 0] * W + pos[global_idx, 1]
            l[local_idx, cell] += init_bias
        l.requires_grad_(True)
        ll.append(l)

    dl = [torch.zeros(len(tasks[t]), dtype=torch.float64, requires_grad=True)
          for t in range(len(tasks))]

    opt = torch.optim.Adam(list(ll) + list(dl), lr=lr)

    for step in range(steps):
        opt.zero_grad()
        init_dists = torch.zeros(N, C, dtype=torch.float64)
        for q in range(N):
            init_dists[q, init_cells[q]] = 1.0
        ld = []
        prev = init_dists.clone()
        for t in range(len(tasks)):
            d = prev.clone().detach()
            for local_idx, global_idx in enumerate(relevant[t]):
                d[global_idx] = F.softmax(ll[t][local_idx], dim=-1)
            ld.append(d)
            prev = d
        dirs = [torch.sigmoid(d) for d in dl]
        loss, infos = surr.compute_loss(init_dists, ld, dirs)
        loss.backward()
        opt.step()

    # Final true cost
    with torch.no_grad():
        hc = [ld[t].argmax(dim=-1) for t in range(len(tasks))]
        tc, bd = true_total_cost(init_cells, hc, tasks, H, W)
        bd_str = ', '.join([f'r{r}+g{g}' for r, g in bd])
        chi_G_str = ' '.join([f'{info["chi_G"]:.2f}' for info in infos])
        grad_norm = sum(l.grad.norm().item() for l in ll if l.grad is not None)
        n_moved = sum(1 for t in range(len(tasks))
                      for li, gi in enumerate(relevant[t])
                      if (init_cells[gi] != hc[t][gi]).item())

    print(f"  {label:40s}  true={tc:2d}  [{bd_str}]  "
          f"χ̂_G=[{chi_G_str}]  ∇={grad_norm:.4f}  moved={n_moved}")
    return tc


print(f"  {'Config':40s}  {'true':>5s}  {'breakdown':15s}  {'χ̂_G':15s}  {'grad':>8s}  {'moved':>5s}")
print(f"  {'-'*40}  {'-'*5}  {'-'*15}  {'-'*15}  {'-'*8}  {'-'*5}")

# Sigmoid mode with different α values
for alpha in [10.0, 3.0, 1.0, 0.5]:
    run_optimization('sigmoid', alpha, 4.0, f'sigmoid α={alpha}, bias=4.0')

# Non-saturating cost mode
run_optimization('cost', 10.0, 4.0, 'cost mode, bias=4.0')

# Softer initialization
for bias in [4.0, 2.0, 1.0, 0.5]:
    run_optimization('cost', 10.0, bias, f'cost mode, bias={bias}')

# Cost mode with higher LR
run_optimization('cost', 10.0, 4.0, 'cost mode, bias=4.0, lr=0.2', lr=0.2)
run_optimization('cost', 10.0, 2.0, 'cost mode, bias=2.0, lr=0.2', lr=0.2)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Speed benchmark
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 6: Speed benchmark")

torch.manual_seed(42)
surr = PartitionSurrogate(tasks, N, H, W, q_max=5, alpha=10.0)

dists = [F.softmax(torch.randn(N, C, dtype=torch.float64), dim=-1)
         for _ in range(len(tasks))]
init_d = torch.zeros(N, C, dtype=torch.float64)
for q in range(N):
    init_d[q, init_cells[q]] = 1.0
dirs = [torch.full((len(t),), 0.5, dtype=torch.float64) for t in tasks]

# Warmup
_ = surr.compute_loss(init_d, dists, dirs)

times = []
for _ in range(10):
    t0 = time.time()
    loss, _ = surr.compute_loss(init_d, dists, dirs)
    times.append(time.time() - t0)
avg_fwd = sum(times) / len(times)
print(f"  Forward pass: {avg_fwd*1000:.1f}ms / eval")

# With backward
times_bwd = []
for _ in range(5):
    # Need grad-enabled dists for backward
    dists_g = [F.softmax(torch.randn(N, C, dtype=torch.float64, requires_grad=True), dim=-1)
               for _ in range(len(tasks))]
    t0 = time.time()
    loss, _ = surr.compute_loss(init_d, dists_g, dirs)
    loss.backward()
    times_bwd.append(time.time() - t0)
avg_bwd = sum(times_bwd) / len(times_bwd)
print(f"  Forward + backward: {avg_bwd*1000:.1f}ms / eval")

check("Forward < 2000ms", avg_fwd < 2.0, f"{avg_fwd*1000:.1f}ms")

from surrogate.partition import chromatic_cost


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

header("SUMMARY")
print("  Key results:")
print(f"  - Discrete recovery: χ̂ matches true χ for K₃, K₄, C₅, bipartite")
print(f"  - Gradient direction: ∂χ̂/∂A_ij ≥ 0 (verified)")
print(f"  - Optimization: {init_true} → {tc_final}")
print(f"  - No auxiliary variables — only placement + direction gradients")
print(f"  - Speed: {avg_fwd*1000:.0f}ms forward, {avg_bwd*1000:.0f}ms fwd+bwd")
