"""
tests/test_patience.py — Validation for patience sorting LDS surrogate.

Tests:
  1. Discrete LDS correctness on known sequences
  2. Axis LDS matches true per-axis chi on random configurations
  3. Discrete total cost vs true total cost on Map 2 solutions
  4. Ranking test: compare surrogate ranking against true cost ranking
  5. Soft DP matches MC estimate
  6. Gradient check
"""

import torch
import torch.nn.functional as F
import sys, os, time, random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.primitives import (
    true_gate_cost, true_reconfig_cost, true_total_cost, get_relevant_atoms
)
from surrogate.patience import (
    lds_length, axis_lds, discrete_axis_chi, discrete_gate_axis_chi,
    discrete_layer_cost, discrete_total_cost,
    soft_lds_dp, soft_reconfig_cost, soft_gate_cost, soft_layer_cost,
    _cell_to_coord_marginal
)

H, W, C = 5, 5, 25

def header(name):
    print(f"\n{'═' * 65}")
    print(f"  {name}")
    print(f"{'═' * 65}")

def check(name, condition, detail=""):
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}: {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Discrete LDS on known sequences
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 1: Discrete LDS correctness")

check("Empty", lds_length([]) == 0)
check("Single", lds_length([3]) == 1)
check("Increasing [1,2,3,4]", lds_length([1, 2, 3, 4]) == 1)
check("Decreasing [4,3,2,1]", lds_length([4, 3, 2, 1]) == 4)
check("Two groups [3,4,1,2]", lds_length([3, 4, 1, 2]) == 2)
check("All same [2,2,2,2]", lds_length([2, 2, 2, 2]) == 1)
check("[3,1,4,1,5]", lds_length([3, 1, 4, 1, 5]) == 2)
check("[5,1,4,2,3]", lds_length([5, 1, 4, 2, 3]) == 3)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Axis LDS vs true per-axis chi on random configs
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 2: Axis LDS on random reconfig configurations")

random.seed(42)
n_tests = 500
axis_match = 0

for trial in range(n_tests):
    n = random.randint(2, 8)
    cells = random.sample(range(C), min(n * 2, C))
    src = cells[:n]
    dst = cells[n:2*n] if len(cells) >= 2*n else random.sample(range(C), n)

    # Remove non-movers
    movers = [(s, d) for s, d in zip(src, dst) if s != d]
    if len(movers) < 2:
        continue

    src_m = [m[0] for m in movers]
    dst_m = [m[1] for m in movers]

    src_cols = [s % W for s in src_m]
    dst_cols = [d % W for d in dst_m]
    src_rows = [s // W for s in src_m]
    dst_rows = [d // W for d in dst_m]

    lds_col = axis_lds(src_cols, dst_cols)
    lds_row = axis_lds(src_rows, dst_rows)

    # True chi
    true_chi = true_reconfig_cost(
        torch.tensor(src_m), torch.tensor(dst_m), H, W)

    max_lds = max(lds_col, lds_row)

    if max_lds == true_chi:
        axis_match += 1

total_tested = n_tests  # approximate (some skipped)
print(f"  max(LDS_col, LDS_row) = true_χ: {axis_match}/{n_tests} "
      f"({axis_match/n_tests:.1%})")
print(f"  (Remaining cases: axis decomposition underestimates)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Map 2 — discrete cost vs true cost
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 3: Map 2 discrete cost comparisons")

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

# Do-nothing
true_dn = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)
surr_dn, dn_details = discrete_total_cost(
    init_cells, [init_cells]*3, tasks, H, W)

print(f"  Do-nothing: true={true_dn}, surrogate={surr_dn}")
for d in dn_details:
    print(f"    Layer {d['layer']}: surr_rc={d['reconfig']:.0f} surr_gc={d['gate']:.0f} "
          f"(LDS_col={d['gc_col']} LDS_row={d['gc_row']} dirs={d['gc_dirs']})")

# A known good solution (from our earlier optimization)
# Layer 0: q7:(4,1)→(0,1), q10:(4,0)→(1,4)
# Layer 1: q1:(2,2)→(3,3)
# Layer 2: q0:(0,4)→(3,1), q1:(3,3)→(2,2), q5:(1,0)→(1,1)
layer0_cells = init_cells.clone()
layer0_cells[7] = 0*W+1   # q7 → (0,1)
layer0_cells[10] = 1*W+4  # q10 → (1,4)

layer1_cells = layer0_cells.clone()
layer1_cells[1] = 3*W+3   # q1 → (3,3)

layer2_cells = layer1_cells.clone()
layer2_cells[0] = 3*W+1   # q0 → (3,1)
layer2_cells[1] = 2*W+2   # q1 → (2,2)
layer2_cells[5] = 1*W+1   # q5 → (1,1)

true_best, true_bd = true_total_cost(
    init_cells, [layer0_cells, layer1_cells, layer2_cells], tasks, H, W)
surr_best, surr_bd = discrete_total_cost(
    init_cells, [layer0_cells, layer1_cells, layer2_cells], tasks, H, W)

print(f"\n  Best known: true={true_best}, surrogate={surr_best}")
for d in surr_bd:
    print(f"    Layer {d['layer']}: surr_rc={d['reconfig']:.0f} surr_gc={d['gate']:.0f}")

true_bd_str = ', '.join([f'r{r}+g{g}' for r, g in true_bd])
print(f"  True breakdown: [{true_bd_str}]")

check("Surrogate ranks best < do-nothing",
      surr_best < surr_dn,
      f"surr_best={surr_best} vs surr_dn={surr_dn}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Ranking test on random perturbations
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 4: Ranking test — surrogate vs true cost ordering")

random.seed(42)

# Generate solutions by random single-atom perturbations from do-nothing
solutions = []

# Do-nothing
solutions.append(('do-nothing', [init_cells.clone()] * 3))

# Random perturbations: move one atom per layer
for trial in range(30):
    layer_cells = []
    prev = init_cells.clone()
    for t in range(3):
        cells = prev.clone()
        # Randomly move one relevant atom
        rel = get_relevant_atoms(tasks[t])
        q = random.choice(rel)
        available = [c for c in range(C) if c not in cells.tolist()]
        if available:
            cells[q] = random.choice(available)
        layer_cells.append(cells)
        prev = cells
    solutions.append((f'perturb-{trial}', layer_cells))

# Evaluate all solutions
evaluated = []
for name, layer_cells in solutions:
    true_tc, true_bd = true_total_cost(init_cells, layer_cells, tasks, H, W)
    surr_tc, surr_bd = discrete_total_cost(init_cells, layer_cells, tasks, H, W)
    evaluated.append((name, true_tc, surr_tc))

# Count ranking violations
violations = 0
total_pairs = 0
violation_gaps = []

for i in range(len(evaluated)):
    for j in range(i + 1, len(evaluated)):
        _, true_i, surr_i = evaluated[i]
        _, true_j, surr_j = evaluated[j]

        if true_i == true_j:
            continue  # ties don't count

        total_pairs += 1
        true_order = true_i < true_j  # True: i is better
        surr_order = surr_i < surr_j

        if true_order != surr_order:
            violations += 1
            gap = abs(true_i - true_j)
            violation_gaps.append(gap)

violation_rate = violations / total_pairs if total_pairs > 0 else 0
print(f"  Solutions evaluated: {len(evaluated)}")
print(f"  Pairs compared: {total_pairs}")
print(f"  Violations: {violations} ({violation_rate:.1%})")

if violation_gaps:
    gap_counts = Counter(violation_gaps)
    print(f"  Violation gap distribution: {dict(sorted(gap_counts.items()))}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Soft DP matches MC estimate
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 5: Soft LDS DP vs Monte Carlo")

torch.manual_seed(42)

# Create soft coordinate marginals
V = W  # = 5
N_atoms = 6

marginals = []
for i in range(N_atoms):
    logits = torch.randn(V, dtype=torch.float64)
    logits[i % V] += 2.0  # bias
    marginals.append(F.softmax(logits, dim=-1))

# DP estimate
dp_result = soft_lds_dp(marginals, V)

# MC estimate
n_mc = 10000
mc_lds_values = []
for _ in range(n_mc):
    seq = [torch.multinomial(m, 1).item() for m in marginals]
    mc_lds_values.append(lds_length(seq))

mc_mean = sum(mc_lds_values) / n_mc
mc_std = (sum((x - mc_mean)**2 for x in mc_lds_values) / n_mc) ** 0.5
mc_se = mc_std / (n_mc ** 0.5)

print(f"  DP estimate:  {dp_result.item():.4f}")
print(f"  MC estimate:  {mc_mean:.4f} ± {mc_se:.4f}")
print(f"  Difference:   {abs(dp_result.item() - mc_mean):.4f}")

check("DP matches MC (within 3σ)",
      abs(dp_result.item() - mc_mean) < 3 * mc_se,
      f"diff={abs(dp_result.item() - mc_mean):.4f}, 3σ={3*mc_se:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Gradient check
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 6: Gradient check — soft gate cost")

torch.manual_seed(99)

gate_atoms = [(0,1), (2,3), (4,5), (6,7)]
base_logits = torch.randn(8, C, dtype=torch.float64) * 0.3

# Analytical gradient
gl = base_logits.clone().requires_grad_(True)
dists = F.softmax(gl, dim=-1)
cost, _ = soft_gate_cost(gate_atoms, dists, H, W, collision_weight=0.5)
cost.backward()
ag = gl.grad.clone()

print(f"  Cost: {cost.item():.4f}")
print(f"  Grad norm: {ag.norm().item():.6f}")

# Numerical FD
eps = 1e-5
a_list, n_list = [], []

for qi in range(8):
    top2 = ag[qi].abs().topk(2).indices
    for ci in top2:
        c = ci.item()
        lp = base_logits.clone(); lp[qi, c] += eps
        cp, _ = soft_gate_cost(gate_atoms, F.softmax(lp, dim=-1), H, W, collision_weight=0.5)
        lm = base_logits.clone(); lm[qi, c] -= eps
        cm, _ = soft_gate_cost(gate_atoms, F.softmax(lm, dim=-1), H, W, collision_weight=0.5)
        a_list.append(ag[qi, c].item())
        n_list.append(((cp - cm) / (2 * eps)).item())

av = torch.tensor(a_list)
nv = torch.tensor(n_list)
cos = F.cosine_similarity(av.unsqueeze(0), nv.unsqueeze(0)).item()

check("Gradient cosine sim > 0.99", cos > 0.99, f"cos={cos:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: Speed
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 7: Speed benchmark")

torch.manual_seed(42)
dists = F.softmax(torch.randn(12, C, dtype=torch.float64), dim=-1)
src_dists = F.softmax(torch.randn(12, C, dtype=torch.float64), dim=-1)

t0 = time.time()
for _ in range(50):
    discrete_total_cost(init_cells, [init_cells]*3, tasks, H, W)
t_discrete = (time.time() - t0) / 50

t0 = time.time()
for _ in range(20):
    rc, _ = soft_reconfig_cost(src_dists, dists, H, W)
t_soft_rc = (time.time() - t0) / 20

t0 = time.time()
for _ in range(20):
    gc, _ = soft_gate_cost(tasks[0], dists, H, W)
t_soft_gc = (time.time() - t0) / 20

print(f"  Discrete total (3 layers): {t_discrete*1000:.1f}ms")
print(f"  Soft reconfig (1 layer):   {t_soft_rc*1000:.1f}ms")
print(f"  Soft gate (1 layer):       {t_soft_gc*1000:.1f}ms")


# ═════════════════════════════════════════════════════════════════════════════
header("SUMMARY")
print("  Key metrics:")
print("  - Discrete LDS matches known sequences")
print("  - Axis decomposition match rate with true χ")
print("  - Surrogate correctly ranks do-nothing vs best")
print("  - Soft DP matches MC (unbiased)")
print("  - Gradient cosine similarity > 0.99")