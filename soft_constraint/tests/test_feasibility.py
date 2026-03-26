"""
tests/test_feasibility.py — Sanity checks for the unified feasibility surrogate.

Tests are organized bottom-up:
  1. Pairwise compatibility: does the soft version match hard on one-hot?
  2. Gate feasibility: one-hot recovery, known configurations
  3. Reconfig feasibility: one-hot recovery, known configurations
  4. Gradient sanity: does the gradient point the right way?
  5. Combined layer cost: known tradeoff scenarios
  6. Multi-layer: Map 2 optimization

Each test prints PASS/FAIL with diagnostics.
"""

import torch
import torch.nn.functional as F
import sys
import os
import time

# Adjust path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.primitives import (
    aod_compatible, greedy_chromatic, true_gate_cost, true_reconfig_cost,
    true_layer_cost, true_total_cost, get_relevant_atoms
)
from surrogate.feasibility import (
    pairwise_axis_compat_prob, pairwise_full_compat_prob,
    pairwise_no_collision_prob,
    gate_feasibility, gate_cost_surrogate,
    reconfig_feasibility, reconfig_cost_surrogate,
    layer_cost_surrogate, total_cost_surrogate
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
# TEST 1: Pairwise compatibility — one-hot matches discrete
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 1: Pairwise compatibility — one-hot matches discrete")

# Two compatible moves: (0,0)→(1,1) and (0,2)→(1,3)
# Same direction on both axes → compatible
src_i, dst_i = onehot(0), onehot(6)   # (0,0) → (1,1)
src_j, dst_j = onehot(2), onehot(8)   # (0,2) → (1,3)

p_col = pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, 'col')
p_row = pairwise_axis_compat_prob(src_i, dst_i, src_j, dst_j, H, W, 'row')
p_full = pairwise_full_compat_prob(src_i, dst_i, src_j, dst_j, H, W)

move_i = torch.tensor([0, 0, 1, 1], dtype=torch.long)
move_j = torch.tensor([0, 2, 1, 3], dtype=torch.long)
true_compat = aod_compatible(move_i, move_j).item()

check("Compatible pair: col", abs(p_col.item() - 1.0) < EPS, f"p_col={p_col.item():.6f}")
check("Compatible pair: row", abs(p_row.item() - 1.0) < EPS, f"p_row={p_row.item():.6f}")
check("Compatible pair: full", abs(p_full.item() - 1.0) < EPS, f"p_full={p_full.item():.6f}")
check("Compatible pair: matches discrete", true_compat == True)

# Two incompatible moves: (0,0)→(1,1) and (0,1)→(1,0) — column inversion
src_i2, dst_i2 = onehot(0), onehot(6)   # (0,0) → (1,1)
src_j2, dst_j2 = onehot(1), onehot(5)   # (0,1) → (1,0)

p_col2 = pairwise_axis_compat_prob(src_i2, dst_i2, src_j2, dst_j2, H, W, 'col')
p_full2 = pairwise_full_compat_prob(src_i2, dst_i2, src_j2, dst_j2, H, W)
true_compat2 = aod_compatible(torch.tensor([0,0,1,1]), torch.tensor([0,1,1,0])).item()

check("Incompatible pair: col", abs(p_col2.item() - 0.0) < EPS, f"p_col={p_col2.item():.6f}")
check("Incompatible pair: full", abs(p_full2.item() - 0.0) < EPS, f"p_full={p_full2.item():.6f}")
check("Incompatible pair: matches discrete", true_compat2 == False)

# Collision: same destination
src_i3, dst_i3 = onehot(0), onehot(6)   # → (1,1)
src_j3, dst_j3 = onehot(1), onehot(6)   # → (1,1) same!

p_nocol = pairwise_no_collision_prob(dst_i3, dst_j3)
p_full3 = pairwise_full_compat_prob(src_i3, dst_i3, src_j3, dst_j3, H, W)

check("Collision: P(no_col)=0", abs(p_nocol.item() - 0.0) < EPS, f"p_nocol={p_nocol.item():.6f}")
check("Collision: full=0", abs(p_full3.item() - 0.0) < EPS, f"p_full={p_full3.item():.6f}")

# Same-column edge case: src same col, dst different col → incompatible
src_i4, dst_i4 = onehot(0), onehot(5)   # (0,0) → (1,0)
src_j4, dst_j4 = onehot(5), onehot(11)  # (1,0) → (2,1)  — same src col=0, diff dst col

p_col4 = pairwise_axis_compat_prob(src_i4, dst_i4, src_j4, dst_j4, H, W, 'col')
true_compat4 = aod_compatible(torch.tensor([0,0,1,0]), torch.tensor([1,0,2,1])).item()

check("Same-src-col, diff-dst-col: col incompatible",
      abs(p_col4.item() - 0.0) < EPS, f"p_col={p_col4.item():.6f}")
check("Same-src-col, diff-dst-col: matches discrete", true_compat4 == False)

# Both same col, both same col dest → compatible
src_i5, dst_i5 = onehot(0), onehot(5)   # (0,0) → (1,0)
src_j5, dst_j5 = onehot(10), onehot(15) # (2,0) → (3,0)  — same col throughout

p_col5 = pairwise_axis_compat_prob(src_i5, dst_i5, src_j5, dst_j5, H, W, 'col')
check("Both same col throughout: compatible",
      abs(p_col5.item() - 1.0) < EPS, f"p_col={p_col5.item():.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Gate feasibility — one-hot recovery
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 2: Gate feasibility — one-hot configurations")

# Map 2 initial positions
pos = torch.tensor([
    [0,4],[2,2],[1,2],[2,1],[3,0],[1,0],
    [0,2],[4,1],[4,2],[4,3],[4,0],[1,3],
])
N = pos.shape[0]
init_cells = pos[:, 0] * W + pos[:, 1]

tasks = [
    [(4,3), (6,7), (9,8), (10,11)],
    [(2,1), (4,5), (7,6), (8,9)],
    [(0,1), (2,3), (6,5), (9,8)],
]

# One-hot distributions at initial positions
onehot_dists = torch.zeros(N, C, dtype=torch.float64)
for q in range(N):
    onehot_dists[q, init_cells[q]] = 1.0

# Gate cost at initial positions
for t, gates in enumerate(tasks):
    true_gc = true_gate_cost(init_cells, gates, H, W, canon=True)
    F_gate, info = gate_feasibility(gates, onehot_dists, H, W)

    if true_gc == 2:
        check(f"Layer {t}: χ=1 → F_gate=1.0", abs(F_gate.item() - 1.0) < EPS,
              f"F={F_gate.item():.6f}, true_gc={true_gc}")
    else:
        check(f"Layer {t}: χ>1 → F_gate<1.0", F_gate.item() < 1.0 - EPS,
              f"F={F_gate.item():.6f}, true_gc={true_gc}")

# Construct a known χ=1 configuration: all gates move right on same row
# Gate (0,1): place a0 at (0,0), a1 at (0,1) → move (0,0)→(0,1)
# Gate (2,3): place a2 at (0,2), a3 at (0,3) → move (0,2)→(0,3)
# Gate (4,5): place a4 at (1,0), a5 at (1,1) → move (1,0)→(1,1)
# Gate (6,7): place a6 at (1,2), a7 at (1,3) → move (1,2)→(1,3)
test_gates = [(0,1), (2,3), (4,5), (6,7)]
perfect_dists = torch.zeros(8, C, dtype=torch.float64)
perfect_cells = [0, 1, 2, 3, 5, 6, 7, 8]  # row 0: cols 0-3, row 1: cols 0-3
for q in range(8):
    perfect_dists[q, perfect_cells[q]] = 1.0

true_gc_perfect = true_gate_cost(
    torch.tensor(perfect_cells), test_gates, H, W, canon=True)
F_perfect, _ = gate_feasibility(test_gates, perfect_dists, H, W)

check("Known χ=1 config: F_gate=1.0",
      abs(F_perfect.item() - 1.0) < EPS,
      f"F={F_perfect.item():.6f}, true_gc={true_gc_perfect}")

# Known χ>1 config: crossing moves
# Gate (0,1): a0=(0,0), a1=(0,4) → move (0,0)→(0,4)
# Gate (2,3): a2=(0,3), a3=(0,1) → move (0,3)→(0,1) — crosses!
cross_gates = [(0,1), (2,3)]
cross_dists = torch.zeros(4, C, dtype=torch.float64)
cross_dists[0, 0] = 1.0   # (0,0)
cross_dists[1, 4] = 1.0   # (0,4)
cross_dists[2, 3] = 1.0   # (0,3)
cross_dists[3, 1] = 1.0   # (0,1)
# Forward: (0,0)→(0,4) and (0,3)→(0,1) — col inversion
# Reverse gate 0: (0,4)→(0,0) and (0,3)→(0,1) — both go left, compatible!
# So with canonicalization, this should be feasible
true_gc_cross = true_gate_cost(torch.tensor([0, 4, 3, 1]), cross_gates, H, W, canon=True)
# Need a truly infeasible case: (0,0)→(0,2) and (0,1)→(0,0) — no direction helps
# d=(0,0): (0,0)→(0,2), (0,1)→(0,0) — src: 0<1, dst: 2>0 — inversion
# d=(0,1): (0,0)→(0,2), (0,0)→(0,1) — same src! dst: 2≠1 — conflict
# d=(1,0): (0,2)→(0,0), (0,1)→(0,0) — collision at (0,0)!
# d=(1,1): (0,2)→(0,0), (0,0)→(0,1) — src: 2>0, dst: 0<1 — inversion
infeas_gates = [(0,1), (2,3)]
infeas_dists = torch.zeros(4, C, dtype=torch.float64)
infeas_dists[0, 0] = 1.0   # a0 at (0,0)
infeas_dists[1, 2] = 1.0   # a1 at (0,2)
infeas_dists[2, 1] = 1.0   # a2 at (0,1)
infeas_dists[3, 0] = 1.0   # a3 at (0,0) — BUT this means a0 and a3 are same cell!

# Let's use a cleaner infeasible example
# a0=(0,0), a1=(0,3), a2=(0,1), a3=(0,2)
# Gate 0: a0↔a1, Gate 1: a2↔a3
# d=(0,0): (0,0)→(0,3), (0,1)→(0,2) — src:0<1, dst:3>2 — same sign ✓ col ok
# Actually this might be feasible. Let me just check true cost.
infeas_dists2 = torch.zeros(4, C, dtype=torch.float64)
infeas_dists2[0, 0] = 1.0   # (0,0)
infeas_dists2[1, 3] = 1.0   # (0,3)
infeas_dists2[2, 1] = 1.0   # (0,1)
infeas_dists2[3, 2] = 1.0   # (0,2)
true_gc_inf = true_gate_cost(torch.tensor([0, 3, 1, 2]), infeas_gates, H, W, canon=True)
F_inf, _ = gate_feasibility(infeas_gates, infeas_dists2, H, W)
check(f"Test config: true_gc={true_gc_inf}, F={'≈1' if F_inf.item() > 0.99 else '<1'}",
      (true_gc_inf == 2 and F_inf.item() > 1-EPS) or (true_gc_inf > 2 and F_inf.item() < 1-EPS),
      f"F={F_inf.item():.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Reconfig feasibility — one-hot recovery
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 3: Reconfig feasibility — one-hot configurations")

# No movement → cost = 0
src = torch.zeros(4, C, dtype=torch.float64)
dst = torch.zeros(4, C, dtype=torch.float64)
for q in range(4):
    src[q, q] = 1.0
    dst[q, q] = 1.0

rc, info = reconfig_cost_surrogate(src, dst, H, W)
check("No movement: cost ≈ 0", rc.item() < 0.01, f"rc={rc.item():.6f}")

# Compatible moves: all move right
src2 = torch.zeros(4, C, dtype=torch.float64)
dst2 = torch.zeros(4, C, dtype=torch.float64)
src2[0, 0] = 1.0; dst2[0, 1] = 1.0   # (0,0)→(0,1)
src2[1, 2] = 1.0; dst2[1, 3] = 1.0   # (0,2)→(0,3)
src2[2, 5] = 1.0; dst2[2, 6] = 1.0   # (1,0)→(1,1)
src2[3, 7] = 1.0; dst2[3, 8] = 1.0   # (1,2)→(1,3)

true_rc2 = true_reconfig_cost(
    torch.tensor([0,2,5,7]), torch.tensor([1,3,6,8]), H, W)
F_rc2, _ = reconfig_feasibility(src2, dst2, H, W)

check("Compatible reconfig: F=1.0",
      abs(F_rc2.item() - 1.0) < EPS, f"F={F_rc2.item():.6f}, true_χ={true_rc2}")

# Incompatible: swap
src3 = torch.zeros(2, C, dtype=torch.float64)
dst3 = torch.zeros(2, C, dtype=torch.float64)
src3[0, 0] = 1.0; dst3[0, 4] = 1.0   # (0,0)→(0,4)
src3[1, 4] = 1.0; dst3[1, 0] = 1.0   # (0,4)→(0,0)

true_rc3 = true_reconfig_cost(torch.tensor([0,4]), torch.tensor([4,0]), H, W)
F_rc3, _ = reconfig_feasibility(src3, dst3, H, W)

check("Swap (incompatible): F<1.0",
      F_rc3.item() < 1.0 - EPS, f"F={F_rc3.item():.6f}, true_χ={true_rc3}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Gradient sanity — does optimizing the surrogate reduce infeasibility?
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 4: Gradient sanity — optimization direction")

torch.manual_seed(42)

# Start with a configuration that has gate conflicts
# Optimize logits to reduce gate cost surrogate
gates = [(0,1), (2,3), (4,5), (6,7)]
logits = torch.randn(8, C, dtype=torch.float64) * 0.5
logits.requires_grad_(True)
opt = torch.optim.Adam([logits], lr=0.1)

costs = []
feasibilities = []

for step in range(100):
    opt.zero_grad()
    dists = F.softmax(logits, dim=-1)
    cost, info = gate_cost_surrogate(gates, dists, H, W, lambda_g=1.0)
    cost.backward()
    opt.step()
    costs.append(cost.item())
    feasibilities.append(info['F_gate'].item())

check("Gate cost decreases", costs[-1] < costs[0],
      f"{costs[0]:.3f} → {costs[-1]:.3f}")
check("Gate feasibility increases", feasibilities[-1] > feasibilities[0],
      f"{feasibilities[0]:.4f} → {feasibilities[-1]:.4f}")
check("Reaches near-feasibility", feasibilities[-1] > 0.9,
      f"F_gate={feasibilities[-1]:.4f}")


# Same for reconfig
torch.manual_seed(42)
src_fixed = torch.zeros(6, C, dtype=torch.float64)
for q in range(6):
    src_fixed[q, q * 4] = 1.0  # spread across board

dst_logits = torch.randn(6, C, dtype=torch.float64) * 0.5
dst_logits.requires_grad_(True)
opt2 = torch.optim.Adam([dst_logits], lr=0.1)

rc_costs = []
rc_feasibilities = []

for step in range(100):
    opt2.zero_grad()
    dists = F.softmax(dst_logits, dim=-1)
    cost, info = reconfig_cost_surrogate(src_fixed, dists, H, W, lambda_r=1.0)
    cost.backward()
    opt2.step()
    rc_costs.append(cost.item())
    rc_feasibilities.append(info['F_reconfig'].item())

check("Reconfig cost decreases", rc_costs[-1] < rc_costs[0],
      f"{rc_costs[0]:.3f} → {rc_costs[-1]:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Gradient check — analytical vs numerical FD
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 5: Gradient check — analytical vs numerical (gate cost)")

torch.manual_seed(99)
base_logits = torch.randn(8, C, dtype=torch.float64) * 0.3
gates5 = [(0,1), (2,3), (4,5), (6,7)]

# Analytical
gl = base_logits.clone().requires_grad_(True)
d = F.softmax(gl, dim=-1)
cost, _ = gate_cost_surrogate(gates5, d, H, W, lambda_g=1.0)
cost.backward()
ag = gl.grad.clone()

# Numerical FD
eps = 1e-5
n_checks = 16  # check 2 cells per atom
a_list, n_list = [], []

for qi in range(8):
    top2 = ag[qi].abs().topk(2).indices
    for ci in top2:
        c = ci.item()
        lp = base_logits.clone(); lp[qi, c] += eps
        cp, _ = gate_cost_surrogate(gates5, F.softmax(lp, dim=-1), H, W, lambda_g=1.0)
        lm = base_logits.clone(); lm[qi, c] -= eps
        cm, _ = gate_cost_surrogate(gates5, F.softmax(lm, dim=-1), H, W, lambda_g=1.0)
        a_list.append(ag[qi, c].item())
        n_list.append(((cp - cm) / (2 * eps)).item())

av = torch.tensor(a_list)
nv = torch.tensor(n_list)
cos = F.cosine_similarity(av.unsqueeze(0), nv.unsqueeze(0)).item()

check("Gradient cosine sim > 0.999", cos > 0.999, f"cos={cos:.6f}")
print(f"    Analytical[:4]: {av[:4].tolist()}")
print(f"    Numerical[:4]:  {nv[:4].tolist()}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Speed benchmark
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 6: Speed benchmark")

torch.manual_seed(42)
dists_bench = F.softmax(torch.randn(12, C, dtype=torch.float64) * 0.5, dim=-1)
gates_bench = [(0,1), (2,3), (4,5), (6,7)]

# Gate cost
t0 = time.time()
for _ in range(100):
    gc, _ = gate_cost_surrogate(gates_bench, dists_bench, H, W)
t_gate = (time.time() - t0) / 100

# Reconfig cost
src_bench = F.softmax(torch.randn(12, C, dtype=torch.float64) * 0.5, dim=-1)
t0 = time.time()
for _ in range(100):
    rc, _ = reconfig_cost_surrogate(src_bench, dists_bench, H, W)
t_reconfig = (time.time() - t0) / 100

# Combined layer
t0 = time.time()
for _ in range(100):
    lc, _, _ = layer_cost_surrogate(src_bench, dists_bench, gates_bench, H, W)
t_combined = (time.time() - t0) / 100

print(f"  Gate cost:     {t_gate*1000:.2f}ms / eval")
print(f"  Reconfig cost: {t_reconfig*1000:.2f}ms / eval")
print(f"  Combined:      {t_combined*1000:.2f}ms / eval")

check("Gate cost < 50ms", t_gate < 0.05, f"{t_gate*1000:.2f}ms")
check("Reconfig cost < 50ms", t_reconfig < 0.05, f"{t_reconfig*1000:.2f}ms")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

header("SUMMARY")
print("  All tests above should show ✓ PASS.")
print("  If any show ✗ FAIL, investigate before proceeding.")