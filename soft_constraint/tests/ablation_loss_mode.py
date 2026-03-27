"""
Ablation: loss landscape study — log vs linear vs huber.

Tests the hypothesis that:
- log (-log F) is precise near optimum but creates rigid trajectories
- linear (conflict_count) is smooth but only first-order accurate
- huber combines both: log precision near optimum, linear stability far away

Also tests rebalanced combinations (λ_g << λ_aux) and varying δ.
"""

import torch
import torch.nn.functional as F
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, set_derived_config, get_map_data, atom_map_to_positions
from surrogate import (
    gate_cost_surrogate, reconfig_cost_surrogate, gate_feasibility,
    true_gate_cost, true_total_cost, get_relevant_atoms,
)

torch.set_default_dtype(torch.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

base_config = get_config()
set_derived_config(base_config)
map_data = get_map_data(base_config)

H = base_config.env.board_height
W = base_config.env.board_width
C = base_config.env.board_size
N = base_config.env.num_qubits
tasks = map_data['tasks']

positions = atom_map_to_positions(map_data['atom_map'], W)
pos = torch.tensor(positions, dtype=torch.long)
init_cells = pos[:, 0] * W + pos[:, 1]
relevant = [get_relevant_atoms(t) for t in tasks]

donothing = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)


def run_optimization(seed, lambda_g, lambda_r, lambda_aux, loss_mode, delta,
                     n_steps=400, lr=0.03):
    torch.manual_seed(seed)

    layer_logits = []
    for t in range(len(tasks)):
        n_rel = len(relevant[t])
        logits_t = torch.randn(n_rel, C) * base_config.optimizer.logit_init_scale
        for local_idx, global_idx in enumerate(relevant[t]):
            cell = (pos[global_idx, 0] * W + pos[global_idx, 1]).item()
            logits_t[local_idx, cell] += base_config.optimizer.logit_bias
        logits_t.requires_grad_(True)
        layer_logits.append(logits_t)

    optimizer = torch.optim.Adam(layer_logits, lr=lr)

    init_dists = torch.zeros(N, C)
    init_dists.scatter_(1, init_cells.unsqueeze(1), 1.0)

    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        layer_dists = []
        prev_dist = init_dists.clone()
        for t in range(len(tasks)):
            dist_t = prev_dist.clone().detach()
            for local_idx, global_idx in enumerate(relevant[t]):
                dist_t[global_idx] = F.softmax(layer_logits[t][local_idx], dim=-1)
            layer_dists.append(dist_t)
            prev_dist = dist_t

        total_loss = torch.tensor(0.0)
        for t in range(len(tasks)):
            gc, _ = gate_cost_surrogate(
                tasks[t], layer_dists[t], H, W,
                lambda_g=lambda_g, lambda_aux=lambda_aux,
                loss_mode=loss_mode, delta=delta)
            src_d = init_dists if t == 0 else layer_dists[t-1]
            rc, _ = reconfig_cost_surrogate(
                src_d, layer_dists[t], H, W,
                mover_indices=relevant[t], lambda_r=lambda_r, lambda_aux=lambda_aux,
                loss_mode=loss_mode, delta=delta)
            total_loss = total_loss + gc + rc

        total_loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                hard_cells_list = [layer_dists[t].argmax(dim=-1) for t in range(len(tasks))]
                true_tc, breakdown = true_total_cost(
                    init_cells, hard_cells_list, tasks, H, W)
                history.append({
                    'step': step, 'surrogate': total_loss.item(),
                    'true_cost': true_tc, 'breakdown': breakdown,
                })

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configs
# ─────────────────────────────────────────────────────────────────────────────

configs = {
    # Baselines from previous ablation
    'log (λ_g=1)':           {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 0.0, 'loss_mode': 'log',    'delta': 0.1},
    'linear (λ_aux=2)':      {'lambda_g': 0.0, 'lambda_r': 0.0, 'lambda_aux': 2.0, 'loss_mode': 'log',    'delta': 0.1},

    # Huber at different δ
    'huber δ=0.05':           {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 0.0, 'loss_mode': 'huber',  'delta': 0.05},
    'huber δ=0.1':            {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 0.0, 'loss_mode': 'huber',  'delta': 0.1},
    'huber δ=0.3':            {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 0.0, 'loss_mode': 'huber',  'delta': 0.3},

    # Rebalanced: strong aux + weak log (addresses the λ magnitude imbalance)
    'log(0.1)+aux(2)':        {'lambda_g': 0.1, 'lambda_r': 0.1, 'lambda_aux': 2.0, 'loss_mode': 'log',    'delta': 0.1},

    # Huber + aux
    'huber(0.1)+aux(1)':      {'lambda_g': 1.0, 'lambda_r': 1.0, 'lambda_aux': 1.0, 'loss_mode': 'huber',  'delta': 0.1},

    # Stronger aux
    'linear (λ_aux=3)':      {'lambda_g': 0.0, 'lambda_r': 0.0, 'lambda_aux': 3.0, 'loss_mode': 'log',    'delta': 0.1},
    'linear (λ_aux=4)':      {'lambda_g': 0.0, 'lambda_r': 0.0, 'lambda_aux': 4.0, 'loss_mode': 'log',    'delta': 0.1},
}

N_SEEDS = 10
N_STEPS = 400
LR = base_config.optimizer.lr

print(f"Map {base_config.map_num}: {H}×{W}, {N} atoms, {len(tasks)} layers")
print(f"Do-nothing baseline: {donothing}")
print(f"Seeds: {N_SEEDS}, Steps: {N_STEPS}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

results = {}

for config_name, params in configs.items():
    print(f"{'═' * 70}")
    mode = params['loss_mode']
    delta = params['delta']
    print(f"  {config_name}: mode={mode} δ={delta} λ_g={params['lambda_g']} "
          f"λ_r={params['lambda_r']} λ_aux={params['lambda_aux']}")
    print(f"{'═' * 70}")

    seed_results = []
    t_start = time.time()

    for seed_idx in range(N_SEEDS):
        seed = seed_idx * 100 + base_config.experiment.seed
        history = run_optimization(seed=seed, n_steps=N_STEPS, lr=LR, **params)

        final = history[-1]
        mid = history[len(history)//2] if len(history) > 1 else history[0]

        seed_results.append({
            'seed': seed,
            'final_true': final['true_cost'],
            'mid_true': mid['true_cost'],
            'breakdown': final['breakdown'],
        })

    t_elapsed = time.time() - t_start

    final_costs = [r['final_true'] for r in seed_results]
    mid_costs = [r['mid_true'] for r in seed_results]

    best = min(final_costs)
    worst = max(final_costs)
    mean = sum(final_costs) / len(final_costs)
    n_good = sum(1 for c in final_costs if c <= 12)

    print(f"  Final: {final_costs}")
    print(f"  Best: {best}  Worst: {worst}  Mean: {mean:.1f}  ≤12: {n_good}/{N_SEEDS}")
    print(f"  Mid: {mid_costs}")
    print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/N_SEEDS:.1f}s/seed)")

    best_idx = final_costs.index(best)
    bd = seed_results[best_idx]['breakdown']
    print(f"  Best: {['r'+str(r)+'+g'+str(g) for r,g in bd]}")
    print()

    results[config_name] = {
        'final_costs': final_costs, 'best': best, 'worst': worst,
        'mean': mean, 'n_good': n_good,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═' * 70}")
print("  LOSS MODE ABLATION SUMMARY")
print(f"{'═' * 70}")
print(f"  {'Config':<25s} {'Best':>5s} {'Worst':>6s} {'Mean':>6s} {'≤12':>4s}")
print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*6} {'-'*4}")
for name, res in results.items():
    print(f"  {name:<25s} {res['best']:>5d} {res['worst']:>6d} "
          f"{res['mean']:>6.1f} {res['n_good']:>3d}/{N_SEEDS}")
