"""
Ablation study: does the conflict-count auxiliary loss help?

Compares three configurations across multiple random seeds on Map 2:
  A: feasibility only (λ_aux = 0)
  B: feasibility + auxiliary (λ_aux = 0.5)
  C: auxiliary only (λ_g = 0, λ_r = 0, λ_aux = 1.0)

Measures: true cost, convergence speed, seed sensitivity.
Also measures the overestimation bias: does the surrogate declare
feasibility when the true configuration is infeasible?
"""

import torch
import torch.nn.functional as F
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.primitives import (
    true_gate_cost, true_reconfig_cost, true_total_cost, get_relevant_atoms
)
from surrogate.feasibility import (
    gate_cost_surrogate, reconfig_cost_surrogate, gate_feasibility
)

torch.set_default_dtype(torch.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Map 2
# ─────────────────────────────────────────────────────────────────────────────

H, W, C = 5, 5, 25
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
relevant = [get_relevant_atoms(t) for t in tasks]

# Baseline
donothing = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)


def run_optimization(seed, lambda_g, lambda_r, lambda_aux, n_steps=400, lr=0.03):
    """Run single-instance optimization and return results."""
    torch.manual_seed(seed)

    # Logits for relevant atoms per layer
    layer_logits = []
    for t in range(len(tasks)):
        n_rel = len(relevant[t])
        logits_t = torch.randn(n_rel, C) * 0.3
        for local_idx, global_idx in enumerate(relevant[t]):
            cell = pos[global_idx, 0] * W + pos[global_idx, 1]
            logits_t[local_idx, cell] += 4.0
        logits_t.requires_grad_(True)
        layer_logits.append(logits_t)

    optimizer = torch.optim.Adam(layer_logits, lr=lr)

    # Initial one-hot
    init_dists = torch.zeros(N, C)
    for q in range(N):
        init_dists[q, init_cells[q]] = 1.0

    # Track convergence
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Build distributions
        layer_dists = []
        prev_dist = init_dists.clone()
        for t in range(len(tasks)):
            dist_t = prev_dist.clone().detach()
            for local_idx, global_idx in enumerate(relevant[t]):
                dist_t[global_idx] = F.softmax(layer_logits[t][local_idx], dim=-1)
            layer_dists.append(dist_t)
            prev_dist = dist_t

        # Compute loss
        total_loss = torch.tensor(0.0)
        for t in range(len(tasks)):
            gc, g_info = gate_cost_surrogate(
                tasks[t], layer_dists[t], H, W,
                lambda_g=lambda_g, lambda_aux=lambda_aux)
            src_d = init_dists if t == 0 else layer_dists[t-1]
            rc, r_info = reconfig_cost_surrogate(
                src_d, layer_dists[t], H, W,
                mover_indices=relevant[t], lambda_r=lambda_r, lambda_aux=lambda_aux)
            total_loss = total_loss + gc + rc

        total_loss.backward()
        optimizer.step()

        # Periodic evaluation
        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                hard_cells_list = []
                prev = init_cells
                for t in range(len(tasks)):
                    d = layer_dists[t]
                    hard = d.argmax(dim=-1)
                    hard_cells_list.append(hard)
                    prev = hard

                true_tc, breakdown = true_total_cost(
                    init_cells, hard_cells_list, tasks, H, W)

                history.append({
                    'step': step,
                    'surrogate': total_loss.item(),
                    'true_cost': true_tc,
                    'breakdown': breakdown,
                })

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────────────────────────────────────

configs = {
    'A: feas only':         {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 0.0},
    'B: feas + aux(0.5)':   {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 0.5},
    'C: feas + aux(1.0)':   {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 1.0},
    'D: feas + aux(2.0)':   {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 2.0},
    'E: aux only(1.0)':     {'lambda_g': 0.0, 'lambda_r': 0.0, 'lambda_aux': 1.0},
    'F: aux only(2.0)':     {'lambda_g': 0.0, 'lambda_r': 0.0, 'lambda_aux': 2.0},
}

N_SEEDS = 10
N_STEPS = 400

print(f"Map 2: {H}×{W}, {N} atoms, {len(tasks)} layers")
print(f"Do-nothing baseline: {donothing}")
print(f"Seeds: {N_SEEDS}, Steps: {N_STEPS}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Run ablation
# ─────────────────────────────────────────────────────────────────────────────

results = {}

for config_name, params in configs.items():
    print(f"{'═' * 65}")
    print(f"  {config_name}: λ_g={params['lambda_g']}, λ_r={params['lambda_r']}, "
          f"λ_aux={params['lambda_aux']}")
    print(f"{'═' * 65}")

    seed_results = []
    t_start = time.time()

    for seed in range(N_SEEDS):
        history = run_optimization(
            seed=seed * 100 + 42,
            n_steps=N_STEPS,
            **params
        )

        final = history[-1]
        mid = history[len(history)//2] if len(history) > 1 else history[0]

        seed_results.append({
            'seed': seed,
            'final_true': final['true_cost'],
            'final_surr': final['surrogate'],
            'mid_true': mid['true_cost'],
            'breakdown': final['breakdown'],
        })

    t_elapsed = time.time() - t_start

    # Aggregate
    final_costs = [r['final_true'] for r in seed_results]
    mid_costs = [r['mid_true'] for r in seed_results]

    best = min(final_costs)
    worst = max(final_costs)
    mean = sum(final_costs) / len(final_costs)
    n_optimal = sum(1 for c in final_costs if c <= 12)  # <= best known

    print(f"  Final costs: {final_costs}")
    print(f"  Best: {best}  Worst: {worst}  Mean: {mean:.1f}")
    print(f"  ≤12: {n_optimal}/{N_SEEDS}")
    print(f"  Mid-training costs: {mid_costs}")
    print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/N_SEEDS:.1f}s/seed)")

    # Show best solution
    best_idx = final_costs.index(best)
    bd = seed_results[best_idx]['breakdown']
    print(f"  Best breakdown: {['r'+str(r)+'+g'+str(g) for r,g in bd]}")
    print()

    results[config_name] = {
        'final_costs': final_costs,
        'best': best,
        'worst': worst,
        'mean': mean,
        'n_optimal': n_optimal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary comparison
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 65}")
print("  ABLATION SUMMARY")
print(f"{'═' * 65}")
print(f"  {'Config':<25s} {'Best':>5s} {'Worst':>6s} {'Mean':>6s} {'≤12':>4s}")
print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*6} {'-'*4}")
for name, res in results.items():
    print(f"  {name:<25s} {res['best']:>5d} {res['worst']:>6d} "
          f"{res['mean']:>6.1f} {res['n_optimal']:>3d}/{N_SEEDS}")


# ─────────────────────────────────────────────────────────────────────────────
# Overestimation bias check
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 65}")
print("  OVERESTIMATION BIAS CHECK")
print(f"{'═' * 65}")
print("  Does the surrogate ever declare F≈1 when true χ > 1?")

torch.manual_seed(42)
n_bias_tests = 500
false_feasible = 0

for trial in range(n_bias_tests):
    # Random soft distributions
    logits = torch.randn(N, C) * 1.0
    for q in range(N):
        logits[q, init_cells[q]] += 2.0
    dists = F.softmax(logits, dim=-1)

    for t, gates in enumerate(tasks):
        F_gate, info = gate_feasibility(gates, dists, H, W)

        # Hard assignment
        hard = dists.argmax(dim=-1)
        true_gc = true_gate_cost(hard, gates, H, W, canon=True)

        if F_gate.item() > 0.5 and true_gc > 2:
            false_feasible += 1

total_checks = n_bias_tests * len(tasks)
print(f"  Checked {total_checks} random (dist, layer) pairs")
print(f"  False feasible (F>0.5 but χ>1): {false_feasible} ({false_feasible/total_checks:.1%})")